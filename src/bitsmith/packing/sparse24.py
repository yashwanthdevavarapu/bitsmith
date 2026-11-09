"""2:4 structured sparsity: keep the 2 largest magnitudes in every group of 4.

Storage follows the layout used by sparse tensor cores:

* ``values``: the kept elements, shape ``(rows, cols / 2)``, in their original order.
* ``meta``: for each group, the two kept column indices (0..3) as 2-bit fields form a
  4-bit nibble ``idx0 | idx1 << 2``. Two groups share one byte (low nibble first), so
  ``meta`` has shape ``(rows, cols / 8)``.

Metadata costs 4 bits per 4 weights, so with fp16 values the cost is
``(2 * 16 + 4) / 4 = 9`` bits per weight.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mask_24(weight: Tensor) -> Tensor:
    """Boolean mask keeping the two largest-magnitude entries of each group of 4.

    Ties break toward the lower column index so the result is deterministic.
    """
    if weight.shape[-1] % 4:
        raise ValueError(f"last dimension must be a multiple of 4, got {weight.shape[-1]}")
    g = weight.reshape(*weight.shape[:-1], -1, 4).abs().to(torch.float32)
    order = torch.argsort(g, dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(g, dtype=torch.bool)
    keep.scatter_(-1, order[..., :2], True)
    return keep.reshape(weight.shape)


def prune_24(weight: Tensor) -> Tensor:
    """Return a copy of ``weight`` with the 2:4 pattern applied."""
    return weight * mask_24(weight).to(weight.dtype)


def pack_24(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Compress a 2-D weight into ``(values, meta)``. ``cols`` must be a multiple of 8."""
    if weight.dim() != 2 or weight.shape[-1] % 8:
        raise ValueError("pack_24 expects a 2-D tensor whose columns are a multiple of 8")
    rows, cols = weight.shape
    keep = mask_24(weight).reshape(rows, -1, 4)
    idx = torch.arange(4, device=weight.device).expand_as(keep)
    kept_idx = idx[keep].reshape(rows, -1, 2)  # ascending column order per group
    groups = weight.reshape(rows, -1, 4)
    values = torch.gather(groups, -1, kept_idx).reshape(rows, cols // 2)
    nibbles = (kept_idx[..., 0] | (kept_idx[..., 1] << 2)).to(torch.uint8)
    meta = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return values, meta


def unpack_24(values: Tensor, meta: Tensor, cols: int) -> Tensor:
    """Rebuild the dense matrix (zeros at pruned positions)."""
    rows = values.shape[0]
    m = meta.to(torch.int16)
    nibbles = torch.stack((m & 0x0F, (m >> 4) & 0x0F), dim=-1).reshape(rows, -1)
    idx = torch.stack((nibbles & 0x3, (nibbles >> 2) & 0x3), dim=-1).to(torch.long)
    dense = torch.zeros(rows, cols // 4, 4, dtype=values.dtype, device=values.device)
    dense.scatter_(-1, idx, values.reshape(rows, -1, 2))
    return dense.reshape(rows, cols)
