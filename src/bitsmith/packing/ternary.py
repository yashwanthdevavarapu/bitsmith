"""Two ways to store ternary weights.

``pack_ternary_2bit``: four trits per byte (2.0 bits per weight). Each trit ``t`` is
stored as ``t + 1`` in two bits, element ``4k + i`` at bit offset ``2i``. Unpacking is a
shift and mask, which maps directly to SIMD lanes.

``pack_ternary_base3``: five trits per byte (1.6 bits per weight), using the fact that
``3**5 = 243 <= 256``. Each byte holds ``sum((t_i + 1) * 3**i)``. Denser, but unpacking
needs integer division, so it is slower on most hardware. See ADR 0003.
"""

from __future__ import annotations

import torch
from torch import Tensor

_POW3 = (1, 3, 9, 27, 81)


def _check_trits(trits: Tensor) -> Tensor:
    t = trits.to(torch.int16)
    if t.numel() and (int(t.min()) < -1 or int(t.max()) > 1):
        raise ValueError("ternary values must be in {-1, 0, 1}")
    return t + 1


def pack_ternary_2bit(trits: Tensor) -> Tensor:
    if trits.shape[-1] % 4:
        raise ValueError(f"last dimension must be a multiple of 4, got {trits.shape[-1]}")
    u = _check_trits(trits).to(torch.uint8)
    return u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)


def unpack_ternary_2bit(packed: Tensor) -> Tensor:
    p = packed.to(torch.int16)
    parts = [((p >> (2 * i)) & 0x3) - 1 for i in range(4)]
    return torch.stack(parts, dim=-1).flatten(-2).to(torch.int8)


def pack_ternary_base3(trits: Tensor) -> Tensor:
    if trits.shape[-1] % 5:
        raise ValueError(f"last dimension must be a multiple of 5, got {trits.shape[-1]}")
    u = _check_trits(trits)
    total = sum(u[..., i::5] * _POW3[i] for i in range(5))
    return torch.as_tensor(total).to(torch.uint8)


def unpack_ternary_base3(packed: Tensor) -> Tensor:
    p = packed.to(torch.int16)
    parts = []
    for _ in range(5):
        parts.append((p % 3) - 1)
        p = p // 3
    return torch.stack(parts, dim=-1).flatten(-2).to(torch.int8)
