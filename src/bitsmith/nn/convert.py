"""Swap every ``nn.Linear`` in a model for a :class:`QLinear`, in place."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch

from torch import Tensor, nn

from bitsmith.nn.qlinear import QLinear
from bitsmith.quant import QTensor

DEFAULT_EXCLUDE = ("lm_head",)


@dataclass
class ConversionReport:
    scheme: str
    converted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    original_bytes: int = 0
    quantized_bytes: int = 0
    n_weights: int = 0

    @property
    def compression(self) -> float:
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes else 1.0

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.quantized_bytes / self.n_weights if self.n_weights else 0.0

    def merge(self, other: ConversionReport) -> None:
        self.converted += other.converted
        self.skipped += other.skipped
        self.original_bytes += other.original_bytes
        self.quantized_bytes += other.quantized_bytes
        self.n_weights += other.n_weights

    def summary(self) -> str:
        return (
            f"{self.scheme}: {len(self.converted)} layers converted, {len(self.skipped)} skipped, "
            f"{self.original_bytes / 2**20:.2f} MiB -> {self.quantized_bytes / 2**20:.2f} MiB "
            f"({self.compression:.2f}x, {self.bits_per_weight:.2f} bits/weight)"
        )


def _excluded(name: str, patterns: Iterable[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return any(fnmatch(name, p) or leaf == p for p in patterns)


def linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def quantize_model(
    model: nn.Module,
    scheme: str = "int4",
    group_size: int = 128,
    *,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    act_bits: int | None = None,
    overrides: Mapping[str, tuple[Tensor, Tensor | None]] | None = None,
    qtensors: Mapping[str, QTensor] | None = None,
    encoded: Mapping[str, dict[str, Tensor]] | None = None,
    include: Collection[str] | None = None,
) -> ConversionReport:
    """Replace linear layers with ``QLinear`` and return a report.

    ``overrides`` maps a layer name to ``(weight, input_scale)``, used by SmoothQuant and
    AWQ. ``qtensors`` maps a layer name to precomputed GPTQ codes, and ``encoded`` to ready
    codec buffers (GPTQ for NF4 and MX). If ``include`` is given,
    only those layers are considered (used for block by block calibration).
    """
    exclude = tuple(exclude)
    overrides = overrides or {}
    qtensors = qtensors or {}
    encoded = encoded or {}
    report = ConversionReport(scheme=scheme)
    for name, linear in linear_layers(model).items():
        if include is not None and name not in include:
            continue
        if _excluded(name, exclude):
            report.skipped.append(name)
            continue
        weight_bytes = linear.weight.numel() * linear.weight.element_size()
        if name in qtensors:
            q = QLinear.from_qtensor(linear, qtensors[name], act_bits=act_bits)
        elif name in encoded:
            q = QLinear.from_buffers(linear, scheme, group_size, encoded[name], act_bits=act_bits)
        else:
            weight, input_scale = overrides.get(name, (None, None))
            q = QLinear.from_linear(
                linear, scheme, group_size, weight=weight, input_scale=input_scale, act_bits=act_bits
            )
        _set_submodule(model, name, q)
        report.converted.append(name)
        report.original_bytes += weight_bytes
        report.quantized_bytes += q.weight_bytes()
        report.n_weights += linear.weight.numel()
    return report


def set_backend(model: nn.Module, backend: str) -> int:
    """Switch every compatible ``QLinear`` in ``model`` to ``backend``; returns how many changed."""
    count = 0
    for module in model.modules():
        if isinstance(module, QLinear) and (backend == "reference" or module.supports_triton()):
            module.set_backend(backend)
            count += 1
    return count


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)
