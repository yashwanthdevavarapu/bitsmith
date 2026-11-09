"""Uniform integer quantization: affine and symmetric, at tensor, channel or group granularity.

All math runs in float32 regardless of the input dtype, so results are identical for
fp32, fp16 and bf16 inputs that hold the same values.

Conventions
-----------
* Affine (asymmetric): codes in ``[0, 2**b - 1]`` with an integer zero point ``z``.
  The range is widened to always contain 0, so 0.0 is represented exactly
  (important for padding and for ReLU-like activations).
* Symmetric: codes in ``[-(2**(b-1) - 1), 2**(b-1) - 1]`` with ``z = 0``. The most
  negative code is left unused so the grid is symmetric around zero (see ADR 0002).
* Granularity ``group`` splits the last dimension into groups of ``group_size``.
  A trailing partial group is zero padded internally and sliced away on dequantize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Granularity = Literal["tensor", "channel", "group"]

_EPS = 1e-12


@dataclass(frozen=True)
class QuantSpec:
    """Describes how a tensor is quantized."""

    bits: int = 8
    symmetric: bool = True
    granularity: Granularity = "channel"
    group_size: int | None = None

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")
        if self.granularity not in ("tensor", "channel", "group"):
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.granularity == "group" and (self.group_size is None or self.group_size < 2):
            raise ValueError("group granularity requires group_size >= 2")

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1) - 1) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.symmetric else 2**self.bits - 1


@dataclass
class QTensor:
    """Integer codes plus the parameters needed to reconstruct real values.

    ``codes`` has shape ``(rows, n_groups, group)``; ``scale`` and ``zero`` have shape
    ``(rows, n_groups, 1)``. ``shape`` is the original tensor shape.
    """

    codes: Tensor
    scale: Tensor
    zero: Tensor
    spec: QuantSpec
    shape: torch.Size

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        x = (self.codes.to(torch.float32) - self.zero) * self.scale
        return _from_groups(x, self.shape).to(dtype)

    @property
    def n_groups(self) -> int:
        return int(self.codes.shape[1])


def _to_groups(x: Tensor, spec: QuantSpec) -> Tensor:
    """Reshape ``x`` to ``(rows, n_groups, group)`` according to the spec."""
    x = x.to(torch.float32)
    if spec.granularity == "tensor":
        return x.reshape(1, 1, -1)
    rows = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x.reshape(1, -1)
    if spec.granularity == "channel":
        return rows.unsqueeze(1)
    assert spec.group_size is not None  # validated in QuantSpec
    g = spec.group_size
    cols = rows.shape[1]
    pad = (-cols) % g
    if pad:
        rows = torch.nn.functional.pad(rows, (0, pad))
    return rows.reshape(rows.shape[0], -1, g)


def _from_groups(x: Tensor, shape: torch.Size) -> Tensor:
    """Inverse of :func:`_to_groups`: drop padding and restore the original shape."""
    if len(shape) == 0:
        return x.reshape(())
    cols = shape[-1]
    rows = shape.numel() // cols if cols else 0
    if x.shape[0] == 1 and rows != 1:  # tensor granularity: one flat group
        return x.reshape(-1)[: shape.numel()].reshape(shape)
    return x.reshape(rows, -1)[:, :cols].reshape(shape)


def compute_qparams(groups: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Return ``(scale, zero)`` of shape ``(rows, n_groups, 1)`` for grouped data."""
    if spec.symmetric:
        absmax = groups.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / spec.qmax).clamp_min(_EPS)
        zero = torch.zeros_like(scale)
        return scale, zero
    lo = groups.amin(dim=-1, keepdim=True).clamp_max(0.0)
    hi = groups.amax(dim=-1, keepdim=True).clamp_min(0.0)
    scale = ((hi - lo) / (spec.qmax - spec.qmin)).clamp_min(_EPS)
    zero = (torch.round(-lo / scale) + spec.qmin).clamp(spec.qmin, spec.qmax)
    return scale, zero


def quantize_with(groups: Tensor, scale: Tensor, zero: Tensor, spec: QuantSpec) -> Tensor:
    """Map grouped real values to integer codes with fixed parameters."""
    q = torch.round(groups / scale) + zero
    return q.clamp(spec.qmin, spec.qmax)


def _code_dtype(spec: QuantSpec) -> torch.dtype:
    return torch.int8 if spec.symmetric else torch.uint8


def quantize(x: Tensor, spec: QuantSpec) -> QTensor:
    """Quantize ``x`` and return a :class:`QTensor`."""
    groups = _to_groups(x, spec)
    scale, zero = compute_qparams(groups, spec)
    codes = quantize_with(groups, scale, zero, spec).to(_code_dtype(spec))
    return QTensor(codes=codes, scale=scale, zero=zero, spec=spec, shape=x.shape)


def fake_quantize(x: Tensor, spec: QuantSpec) -> Tensor:
    """Quantize then dequantize, returning a tensor of the input dtype."""
    return quantize(x, spec).dequantize(x.dtype)


def quantize_activations_per_token(x: Tensor, bits: int = 8) -> Tensor:
    """Dynamic symmetric per-token fake quantization used for W8A8 style simulation."""
    spec = QuantSpec(bits=bits, symmetric=True, granularity="channel")
    return fake_quantize(x, spec)
