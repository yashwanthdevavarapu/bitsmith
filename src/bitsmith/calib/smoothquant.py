"""SmoothQuant: move activation outliers into the weights with a per-channel scale.

For input channel ``j``::

    s_j = max|X_j| ** alpha / max|W_:,j| ** (1 - alpha)
    Y = X W^T = (X / s) (W * s)^T          (exact in real arithmetic)

Dividing activations by ``s`` flattens outlier channels so per-token INT8 activation
quantization loses less, while the weights absorb a range they can easily afford.
"""

from __future__ import annotations

import torch
from torch import Tensor

_FLOOR = 1e-5


def smoothing_factors(act_absmax: Tensor, weight: Tensor, alpha: float = 0.5) -> Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    w_absmax = weight.detach().abs().amax(dim=0).to(torch.float32)
    a = act_absmax.to(torch.float32).clamp_min(_FLOOR)
    s = a.pow(alpha) / w_absmax.clamp_min(_FLOOR).pow(1.0 - alpha)
    return s.clamp_min(_FLOOR)


def smooth_weight(weight: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(weight * s, 1 / s)``; feed activations through the second factor."""
    smoothed = (weight.to(torch.float32) * s.unsqueeze(0)).to(weight.dtype)
    return smoothed, 1.0 / s
