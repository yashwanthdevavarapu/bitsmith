"""One call to calibrate and quantize a whole model.

Two calibration orders are supported (see ADR 0009):

* ``sequential=True`` (default): layers are grouped by transformer block (the module path
  up to the first numeric index, for example ``model.layers.3``). Blocks are calibrated and
  quantized in order, and each block's statistics are collected with every earlier block
  already quantized, so later layers learn to compensate for earlier errors. Each
  calibration pass stops as soon as the current block has run.
* ``sequential=False``: statistics for every layer are captured once from the full
  precision model ("one-shot", ADR 0005). Faster, slightly less accurate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn

from bitsmith.calib.awq import awq_search
from bitsmith.calib.capture import LayerStats, collect_activation_stats
from bitsmith.calib.gptq import gptq_quantize, gptq_quantize_table
from bitsmith.calib.smoothquant import smooth_weight, smoothing_factors
from bitsmith.nn.codecs import TABLE_SCHEMES, TableCodec, make_codec
from bitsmith.nn.convert import DEFAULT_EXCLUDE, ConversionReport, _excluded, linear_layers, quantize_model
from bitsmith.quant import QTensor

Method = Literal["rtn", "awq", "gptq", "smoothquant"]
METHODS: tuple[str, ...] = ("rtn", "awq", "gptq", "smoothquant")


def block_of(name: str) -> str:
    """Module path up to and including the first numeric component, or ``""`` if none."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return ""


def group_by_block(names: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group layer names by block, keeping model order. Unblocked layers come last."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(block_of(name), []).append(name)
    tail = groups.pop("", None)
    ordered = list(groups.items())
    if tail:
        ordered.append(("", tail))
    return ordered


GPTQ_SCHEMES = ("int4", "int4-sym", *TABLE_SCHEMES)


def calibrate_and_quantize(
    model: nn.Module,
    calib_inputs: Sequence[Tensor] | Iterable[Tensor],
    *,
    method: Method = "rtn",
    scheme: str = "int4",
    group_size: int = 128,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    act_bits: int | None = None,
    alpha: float = 0.5,
    damp: float = 0.01,
    act_order: bool = True,
    sequential: bool = True,
) -> ConversionReport:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    exclude = tuple(exclude)
    if method == "rtn":
        return quantize_model(model, scheme, group_size, exclude=exclude, act_bits=act_bits)
    if method == "gptq" and scheme not in GPTQ_SCHEMES:
        raise ValueError(f"gptq supports the {', '.join(GPTQ_SCHEMES)} schemes")

    calib = list(calib_inputs)
    names = [n for n in linear_layers(model) if not _excluded(n, exclude)]
    groups = group_by_block(names) if sequential else [("", names)]
    report = ConversionReport(scheme=f"{scheme}+{method}")
    for block, layer_names in groups:
        stop = model.get_submodule(block) if (sequential and block) else None
        stats = collect_activation_stats(
            model,
            calib,
            layer_names,
            track_hessian=method == "gptq",
            keep_samples=method == "awq",
            stop_after=stop,
        )
        overrides, qtensors, encoded = _calibrate_layers(
            model, stats, method, scheme, group_size, alpha=alpha, damp=damp, act_order=act_order
        )
        report.merge(
            quantize_model(
                model,
                scheme,
                group_size,
                exclude=exclude,
                act_bits=act_bits,
                overrides=overrides,
                qtensors=qtensors,
                encoded=encoded,
                include=set(layer_names),
            )
        )
        del stats
    report.skipped = [n for n in linear_layers(model) if _excluded(n, exclude)]
    return report


def _calibrate_layers(
    model: nn.Module,
    stats: dict[str, LayerStats],
    method: str,
    scheme: str,
    group_size: int,
    *,
    alpha: float,
    damp: float,
    act_order: bool,
) -> tuple[dict[str, tuple[Tensor, Tensor | None]], dict[str, QTensor], dict[str, dict[str, Tensor]]]:
    overrides: dict[str, tuple[Tensor, Tensor | None]] = {}
    qtensors: dict[str, QTensor] = {}
    encoded: dict[str, dict[str, Tensor]] = {}
    for name, st in stats.items():
        linear = model.get_submodule(name)
        assert isinstance(linear, nn.Linear)
        if st.n_rows == 0:
            continue
        if method == "smoothquant":
            s = smoothing_factors(st.act_absmax, linear.weight, alpha)
            w, inv = smooth_weight(linear.weight, s)
            overrides[name] = (w, inv.to(torch.float16))
        elif method == "awq":
            res = awq_search(linear.weight, st.abs_mean, st.samples, scheme=scheme, group_size=group_size)
            scaled = (linear.weight.float() * res.scale).to(linear.weight.dtype)
            overrides[name] = (scaled, (1.0 / res.scale).to(torch.float16))
        elif scheme in TABLE_SCHEMES:  # gptq onto the NF4 or MX grid
            codec = cast(TableCodec, make_codec(scheme, group_size))
            res_t = gptq_quantize_table(linear.weight, st.hessian, codec, damp=damp, act_order=act_order)
            encoded[name] = res_t.buffers
        else:  # gptq
            res_g = gptq_quantize(
                linear.weight,
                st.hessian,
                group_size=group_size,
                symmetric=scheme == "int4-sym",
                damp=damp,
                act_order=act_order,
            )
            qtensors[name] = res_g.qtensor
    return overrides, qtensors, encoded
