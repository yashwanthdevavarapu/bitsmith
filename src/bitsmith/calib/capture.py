"""Collect per-layer input statistics with forward hooks.

Statistics live on the same device as the activations, so calibration on a GPU never
copies activations back to the host.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


class _StopForwardError(Exception):
    """Raised by a hook to end a calibration forward pass early."""


@dataclass
class LayerStats:
    """Running statistics of the inputs seen by one linear layer."""

    in_features: int
    track_hessian: bool = True
    keep_samples: bool = True
    max_samples: int = 512
    n_rows: int = 0
    absmax: Tensor | None = field(init=False, default=None)
    abs_sum: Tensor | None = field(init=False, default=None)
    xtx: Tensor | None = field(init=False, default=None)
    _samples: list[Tensor] = field(init=False, default_factory=list)

    def _init(self, device: torch.device) -> None:
        self.absmax = torch.zeros(self.in_features, device=device)
        self.abs_sum = torch.zeros(self.in_features, dtype=torch.float64, device=device)
        if self.track_hessian:
            self.xtx = torch.zeros(self.in_features, self.in_features, device=device)

    def update(self, x: Tensor) -> None:
        rows = x.detach().reshape(-1, self.in_features).to(torch.float32)
        if self.absmax is None:
            self._init(rows.device)
        assert self.absmax is not None and self.abs_sum is not None
        self.n_rows += rows.shape[0]
        self.absmax = torch.maximum(self.absmax, rows.abs().amax(dim=0))
        self.abs_sum += rows.abs().sum(dim=0, dtype=torch.float64)
        if self.xtx is not None:
            self.xtx += rows.T @ rows
        if not self.keep_samples:
            return
        self._samples.append(rows)
        n = sum(s.shape[0] for s in self._samples)
        if n > 2 * self.max_samples:  # keep memory bounded: decimate by two
            self._samples = [torch.cat(self._samples)[::2]]

    def _require(self) -> tuple[Tensor, Tensor]:
        if self.absmax is None or self.abs_sum is None:
            raise RuntimeError("no activations were recorded for this layer")
        return self.absmax, self.abs_sum

    @property
    def act_absmax(self) -> Tensor:
        return self._require()[0]

    @property
    def abs_mean(self) -> Tensor:
        return (self._require()[1] / max(self.n_rows, 1)).to(torch.float32)

    @property
    def hessian(self) -> Tensor:
        """``H = 2 X^T X / n``, the Hessian of the layer-wise squared error."""
        if not self.track_hessian:
            raise RuntimeError("hessian tracking was disabled for this layer")
        if self.xtx is None:
            raise RuntimeError("no activations were recorded for this layer")
        return 2.0 * self.xtx / max(self.n_rows, 1)

    @property
    def samples(self) -> Tensor:
        if not self._samples:
            return torch.zeros(0, self.in_features)
        x = torch.cat(self._samples)
        if x.shape[0] > self.max_samples:
            idx = torch.linspace(0, x.shape[0] - 1, self.max_samples, device=x.device).long()
            x = x[idx]
        return x


@torch.no_grad()
def collect_activation_stats(
    model: nn.Module,
    inputs: Iterable[Tensor],
    layer_names: Iterable[str],
    *,
    track_hessian: bool = True,
    keep_samples: bool = True,
    max_samples: int = 512,
    stop_after: nn.Module | None = None,
) -> dict[str, LayerStats]:
    """Run ``model(x)`` for every ``x`` in ``inputs`` and record each named layer's input.

    If ``stop_after`` is given, each forward pass ends as soon as that module has run,
    which skips the rest of the network when only an early block needs statistics.
    """
    stats: dict[str, LayerStats] = {}
    handles = []
    for name in layer_names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")
        st = LayerStats(layer.in_features, track_hessian, keep_samples, max_samples)
        stats[name] = st
        handles.append(layer.register_forward_pre_hook(lambda _m, args, st=st: st.update(args[0])))
    if stop_after is not None:

        def _stop(*_: object) -> None:
            raise _StopForwardError

        handles.append(stop_after.register_forward_hook(_stop))
    try:
        for x in inputs:
            with contextlib.suppress(_StopForwardError):
                model(x)
    finally:
        for h in handles:
            h.remove()
    return stats
