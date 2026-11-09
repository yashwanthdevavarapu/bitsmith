"""Ternary weight quantization in the BitNet b1.58 style.

Each output row is scaled by the mean absolute value of its weights (``absmean``) and
then rounded to ``{-1, 0, +1}``. The reconstruction is ``t * gamma``.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def ternarize(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(trits, gamma)`` where trits is int8 in {-1, 0, 1} and gamma is ``(rows, 1)``."""
    w = weight.to(torch.float32)
    gamma = w.abs().mean(dim=-1, keepdim=True).clamp_min(_EPS)
    trits = torch.round(w / gamma).clamp(-1, 1).to(torch.int8)
    return trits, gamma


def ternary_fake_quantize(weight: Tensor) -> Tensor:
    trits, gamma = ternarize(weight)
    return (trits.to(torch.float32) * gamma).to(weight.dtype)
