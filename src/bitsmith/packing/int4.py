"""Pack 4-bit codes two per byte.

Layout: element ``2k`` goes to the low nibble and element ``2k + 1`` to the high nibble::

    byte = (v[2k] & 0x0F) | ((v[2k + 1] & 0x0F) << 4)

Signed values are stored in two's complement and sign extended on unpack with
``(n ^ 8) - 8``, which maps 0..7 to 0..7 and 8..15 to -8..-1 without branches.
"""

from __future__ import annotations

import torch
from torch import Tensor


def pack_int4(codes: Tensor) -> Tensor:
    """Pack integer codes in ``[-8, 15]`` (last dim even) into a uint8 tensor of half width."""
    if codes.shape[-1] % 2:
        raise ValueError(f"last dimension must be even, got {codes.shape[-1]}")
    c = codes.to(torch.int16)
    if c.numel() and (int(c.min()) < -8 or int(c.max()) > 15):
        raise ValueError("int4 codes must lie in [-8, 15]")
    nib = (c & 0x0F).to(torch.uint8)
    return nib[..., 0::2] | (nib[..., 1::2] << 4)


def unpack_int4(packed: Tensor, signed: bool) -> Tensor:
    """Inverse of :func:`pack_int4`. Returns int8 codes with twice the last dimension."""
    p = packed.to(torch.int16)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    if signed:
        lo = (lo ^ 8) - 8
        hi = (hi ^ 8) - 8
    out = torch.stack((lo, hi), dim=-1).flatten(-2)
    return out.to(torch.int8)
