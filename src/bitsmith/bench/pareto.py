"""Quality versus size sweep over quantization configurations, with a Pareto frontier."""

from __future__ import annotations

import copy
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from bitsmith.calib import calibrate_and_quantize
from bitsmith.eval.metrics import sqnr_db
from bitsmith.eval.perplexity import sliding_window_perplexity
from bitsmith.nn import QLinear, linear_layers
from bitsmith.nn.convert import DEFAULT_EXCLUDE, _excluded

BASELINE = "baseline"


@dataclass(frozen=True)
class SweepConfig:
    label: str
    scheme: str  # a codec scheme, or "baseline" for the unquantized model
    method: str = "rtn"
    group_size: int = 128
    act_bits: int | None = None


@dataclass
class SweepPoint:
    label: str
    scheme: str
    method: str
    bits_per_weight: float
    weight_mib: float
    compression: float
    perplexity: float
    delta_pct: float
    weight_sqnr_db: float
    ms_per_forward: float
    on_frontier: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_configs(group_size: int = 64) -> list[SweepConfig]:
    g = group_size
    return [
        SweepConfig("fp32", BASELINE),
        SweepConfig("int8", "int8"),
        SweepConfig("fp8-e4m3", "fp8-e4m3"),
        SweepConfig("w8a8", "int8", act_bits=8),
        SweepConfig("w8a8+smoothquant", "int8", "smoothquant", act_bits=8),
        SweepConfig(f"int4-g{g}", "int4", group_size=g),
        SweepConfig(f"int4-g{g}+awq", "int4", "awq", group_size=g),
        SweepConfig(f"int4-g{g}+gptq", "int4", "gptq", group_size=g),
        SweepConfig("sparse24", "sparse24"),
        SweepConfig("ternary", "ternary"),
    ]


def pareto_front(points: Sequence[SweepPoint]) -> set[str]:
    """Labels of points not dominated in (bits_per_weight, perplexity); lower is better for both."""
    front = set()
    for p in points:
        dominated = any(
            q.bits_per_weight <= p.bits_per_weight
            and q.perplexity <= p.perplexity
            and (q.bits_per_weight < p.bits_per_weight or q.perplexity < p.perplexity)
            for q in points
        )
        if not dominated:
            front.add(p.label)
    return front


def _sync(x: Tensor) -> None:
    if x.is_cuda:
        torch.cuda.synchronize(x.device)


def time_forward(model: nn.Module, x: Tensor, repeats: int = 5, warmup: int = 1) -> float:
    """Median wall time of ``model(x)`` in milliseconds (CUDA synchronized)."""
    times = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            _sync(x)
            t0 = time.perf_counter()
            model(x)
            _sync(x)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _linear_weights(model: nn.Module, exclude: Sequence[str]) -> dict[str, Tensor]:
    return {n: m.weight.detach().clone() for n, m in linear_layers(model).items() if not _excluded(n, exclude)}


def run_sweep(
    model: nn.Module,
    eval_tokens: Tensor,
    calib_inputs: Sequence[Tensor],
    configs: Sequence[SweepConfig] | None = None,
    *,
    logits_fn: Callable[[nn.Module, Tensor], Tensor] | None = None,
    window: int = 128,
    stride: int = 64,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    timing_input: Tensor | None = None,
    progress: Callable[[str], None] | None = None,
    on_model: Callable[[str, nn.Module], None] | None = None,
) -> list[SweepPoint]:
    """Quantize a fresh copy of ``model`` for every config and measure it.

    ``on_model(label, model)`` is called with each quantized model for extra measurements.
    """
    configs = list(configs or default_configs())
    device = next(model.parameters()).device
    eval_tokens = eval_tokens.to(device)
    calib_inputs = [c.to(device) for c in calib_inputs]
    get_logits = logits_fn or (lambda m, ids: m(ids))
    ref_weights = _linear_weights(model, exclude)
    n_weights = sum(w.numel() for w in ref_weights.values())
    base_bytes = sum(w.numel() * w.element_size() for w in ref_weights.values())
    timing_input = (timing_input if timing_input is not None else eval_tokens[:window].unsqueeze(0)).to(device)

    points: list[SweepPoint] = []
    base_ppl: float | None = None
    for cfg in configs:
        if progress:
            progress(cfg.label)
        m = copy.deepcopy(model).eval()
        if cfg.scheme == BASELINE:
            qbytes, sqnr = base_bytes, float("inf")
        else:
            report = calibrate_and_quantize(
                m,
                calib_inputs,
                method=cfg.method,  # type: ignore[arg-type]
                scheme=cfg.scheme,
                group_size=cfg.group_size,
                exclude=exclude,
                act_bits=cfg.act_bits,
            )
            qbytes = report.quantized_bytes
            sqnrs = []
            for name, w in ref_weights.items():
                layer = m.get_submodule(name)
                assert isinstance(layer, QLinear)
                sqnrs.append(sqnr_db(w, layer.effective_weight()))
            sqnr = statistics.mean(sqnrs)

        def logits(ids: Tensor, m: nn.Module = m) -> Tensor:
            return get_logits(m, ids)

        ppl = sliding_window_perplexity(logits, eval_tokens, window=window, stride=stride)
        if base_ppl is None:
            base_ppl = ppl.perplexity
        points.append(
            SweepPoint(
                label=cfg.label,
                scheme=cfg.scheme,
                method=cfg.method,
                bits_per_weight=8.0 * qbytes / n_weights,
                weight_mib=qbytes / 2**20,
                compression=base_bytes / qbytes,
                perplexity=ppl.perplexity,
                delta_pct=100.0 * (ppl.perplexity / base_ppl - 1.0),
                weight_sqnr_db=sqnr,
                ms_per_forward=time_forward(m, timing_input),
            )
        )
        if on_model:
            on_model(cfg.label, m)
        del m
    front = pareto_front(points)
    for p in points:
        p.on_frontier = p.label in front
    return points


SWEEP_HEADERS = (
    "config",
    "bits/w",
    "weights MiB",
    "compression",
    "perplexity",
    "delta %",
    "SQNR dB",
    "ms/fwd",
    "pareto",
)


def sweep_rows(points: Sequence[SweepPoint]) -> list[list[object]]:
    return [
        [
            p.label,
            p.bits_per_weight,
            p.weight_mib,
            p.compression,
            p.perplexity,
            p.delta_pct,
            p.weight_sqnr_db,
            p.ms_per_forward,
            "yes" if p.on_frontier else "",
        ]
        for p in points
    ]
