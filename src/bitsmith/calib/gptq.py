"""GPTQ: quantize one column at a time and push the error onto the remaining columns.

With ``H = 2 X^T X / n`` the layer-wise objective ``||X W^T - X Q^T||^2`` has the closed
form update from Optimal Brain Surgeon. After quantizing column ``i`` with error
``e = (w_i - q_i) / [H^-1]_ii`` every later column ``j`` is corrected by
``w_j -= e * [H^-1]_ij``. Using the upper Cholesky factor of ``H^-1`` makes each step a
row slice, and updates are applied lazily per block for speed.

Group parameters (scale, zero)
------------------------------
* Default: computed when the first column of a group is reached, on the already corrected
  weights. ``block_size`` is rounded to a multiple of ``group_size`` so a group never
  straddles two lazy-update blocks.
* ``act_order=True``: columns are processed in order of decreasing ``diag(H)`` (the inputs
  with the most energy first, while the most columns are left to absorb their error).
  Group parameters are then computed up front from the original weights ("static
  groups"), so the storage layout stays contiguous groups and needs no index map.

See ADR 0005 and ADR 0009.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from bitsmith.nn.codecs import round_scale
from bitsmith.quant import QTensor, QuantSpec, compute_qparams


@dataclass
class GPTQResult:
    qtensor: QTensor
    proxy_loss: float  # sum of (w - q)^2 / [H^-1]_ii / 2, the OBS objective


def damped_inverse_cholesky(hessian: Tensor, damp: float = 0.01) -> tuple[Tensor, Tensor]:
    """Return ``(upper Cholesky factor of H^-1, dead-column mask)``."""
    h = hessian.to(torch.float64).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    h += damp * torch.diag(h).mean() * torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    lower = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_inverse(lower)
    return torch.linalg.cholesky(h_inv, upper=True).to(torch.float32), dead


def _group_params(w: Tensor, group_size: int, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Static ``(scale, zero)`` of shape ``(rows, n_groups)`` from the given weights."""
    rows, cols = w.shape
    n_groups = -(-cols // group_size)
    padded = torch.nn.functional.pad(w, (0, n_groups * group_size - cols))
    s, z = compute_qparams(padded.reshape(rows, n_groups, group_size), spec)
    return round_scale(s[..., 0]), z[..., 0]


@torch.no_grad()
def gptq_quantize(
    weight: Tensor,
    hessian: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    block_size: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> GPTQResult:
    spec = QuantSpec(bits=bits, symmetric=symmetric, granularity="group", group_size=group_size)
    w = weight.detach().to(torch.float32).clone()
    h = hessian.to(device=w.device, dtype=torch.float32)
    rows, cols = w.shape
    dead = torch.diag(h) == 0
    w[:, dead] = 0.0

    n_groups = -(-cols // group_size)
    scales = torch.zeros(rows, n_groups, device=w.device)
    zeros = torch.zeros(rows, n_groups, device=w.device)
    perm = torch.arange(cols, device=w.device)
    if act_order:
        scales, zeros = _group_params(w, group_size, spec)
        perm = torch.argsort(torch.diag(h), descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]
    hinv, _ = damped_inverse_cholesky(h, damp)

    block = max(group_size, (block_size // group_size) * group_size)
    codes = torch.zeros(rows, cols, device=w.device)
    loss = torch.zeros(rows, device=w.device)

    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            g = int(perm[col]) // group_size
            if not act_order and col % group_size == 0:
                s, z = compute_qparams(w1[:, i : i + group_size].unsqueeze(1), spec)
                scales[:, g], zeros[:, g] = round_scale(s[:, 0, 0]), z[:, 0, 0]
            s, z = scales[:, g], zeros[:, g]
            col_w = w1[:, i]
            q = (torch.round(col_w / s) + z).clamp(spec.qmin, spec.qmax)
            dq = (q - z) * s
            d = hinv1[i, i]
            codes[:, col] = q
            loss += (col_w - dq).pow(2) / d.pow(2) / 2
            err = (col_w - dq) / d
            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    if act_order:  # back to the original column order
        unperm = torch.empty_like(perm)
        unperm[perm] = torch.arange(cols, device=w.device)
        codes = codes[:, unperm]

    # Padded columns (beyond ``cols``) get the zero point so they dequantize to exactly 0.
    pad = n_groups * group_size - cols
    codes = torch.nn.functional.pad(codes, (0, pad))
    if pad:
        codes[:, cols:] = zeros[:, -1:].expand(rows, pad)
    dtype = torch.int8 if symmetric else torch.uint8
    qt = QTensor(
        codes=codes.reshape(rows, n_groups, group_size).to(dtype),
        scale=scales.unsqueeze(-1),
        zero=zeros.unsqueeze(-1),
        spec=spec,
        shape=weight.shape,
    )
    return GPTQResult(qtensor=qt, proxy_loss=float(loss.sum()))


def output_error(weight: Tensor, approx: Tensor, hessian: Tensor) -> float:
    """``tr((W - Q) H (W - Q)^T)``: expected squared output error under the calibration data."""
    d = (weight.to(torch.float32) - approx.to(torch.float32)).to(torch.float64)
    h = hessian.to(device=d.device, dtype=torch.float64)
    return float(torch.einsum("ij,jk,ik->", d, h, d))
