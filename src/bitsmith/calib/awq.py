"""Activation-aware weight quantization (AWQ) scale search.

Input channels that see large activations matter more. AWQ scales those weight columns
up before quantization (so they get finer resolution relative to the group range) and
divides the activations by the same factor::

    s = mean|X| ** alpha,   normalized so that sqrt(max(s) * min(s)) = 1
    loss(alpha) = || X W^T - (X / s) Q(W * s)^T ||^2

``alpha`` is chosen by grid search on calibration samples. ``alpha = 0`` is plain
round-to-nearest, so the search can never do worse than the baseline on its own data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from bitsmith.nn.codecs import make_codec


@dataclass
class AWQResult:
    scale: Tensor
    alpha: float
    loss: float
    baseline_loss: float
    grid: list[tuple[float, float]]

    @property
    def improvement(self) -> float:
        return self.baseline_loss / self.loss if self.loss > 0 else float("inf")


def _fake_quant(weight: Tensor, scheme: str, group_size: int) -> Tensor:
    codec = make_codec(scheme, group_size)
    out, inp = weight.shape
    return codec.decode(codec.encode(weight), out, inp)


@torch.no_grad()
def awq_search(
    weight: Tensor,
    act_mean: Tensor,
    x: Tensor,
    *,
    scheme: str = "int4",
    group_size: int = 128,
    n_grid: int = 20,
) -> AWQResult:
    w = weight.detach().to(torch.float32)
    x = x.to(torch.float32)
    y_ref = x @ w.T
    base = act_mean.to(torch.float32).clamp_min(1e-4)
    grid: list[tuple[float, float]] = []
    best = (float("inf"), 0.0, torch.ones_like(base))
    for k in range(n_grid + 1):
        alpha = k / n_grid
        s = base.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()
        wq = _fake_quant(w * s, scheme, group_size)
        loss = float(((x / s) @ wq.T - y_ref).pow(2).mean())
        grid.append((alpha, loss))
        if loss < best[0]:
            best = (loss, alpha, s)
    return AWQResult(scale=best[2], alpha=best[1], loss=best[0], baseline_loss=grid[0][1], grid=grid)
