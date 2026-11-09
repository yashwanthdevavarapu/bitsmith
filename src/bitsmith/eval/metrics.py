"""Tensor level error metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def mse(ref: Tensor, approx: Tensor) -> float:
    return float((ref.to(torch.float64) - approx.to(torch.float64)).pow(2).mean())


def sqnr_db(ref: Tensor, approx: Tensor) -> float:
    """Signal to quantization noise ratio in decibels. Higher is better; +inf when exact."""
    r = ref.to(torch.float64)
    noise = (r - approx.to(torch.float64)).pow(2).sum()
    if noise == 0:
        return math.inf
    return float(10.0 * torch.log10(r.pow(2).sum() / noise))


def max_abs_error(ref: Tensor, approx: Tensor) -> float:
    return float((ref.to(torch.float64) - approx.to(torch.float64)).abs().max())


def cosine_similarity(ref: Tensor, approx: Tensor) -> float:
    a = ref.flatten().to(torch.float64)
    b = approx.flatten().to(torch.float64)
    denom = a.norm() * b.norm()
    return float(a @ b / denom) if denom > 0 else 1.0


def outlier_channels(x: Tensor, threshold: float = 6.0) -> list[int]:
    """Channels (last dim) whose absmax exceeds ``threshold`` times the median channel absmax."""
    per_channel = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).to(torch.float32)
    median = per_channel.median().clamp_min(1e-12)
    return torch.nonzero(per_channel > threshold * median).flatten().tolist()


@dataclass(frozen=True)
class ErrorReport:
    sqnr_db: float
    mse: float
    max_abs_error: float
    cosine: float

    @classmethod
    def compare(cls, ref: Tensor, approx: Tensor) -> ErrorReport:
        return cls(
            sqnr_db=sqnr_db(ref, approx),
            mse=mse(ref, approx),
            max_abs_error=max_abs_error(ref, approx),
            cosine=cosine_similarity(ref, approx),
        )
