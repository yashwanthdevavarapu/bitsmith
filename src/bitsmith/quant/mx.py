"""OCP Microscaling (MX) formats: MXFP4 and MXINT8.

An MX block holds 32 elements that share one 8-bit scale. The scale is E8M0: an unsigned
exponent ``e`` meaning ``2^(e - 127)``, with ``0xFF`` reserved for NaN. Following the
OCP MX v1.0 specification, the shared exponent of a block is::

    shared = floor(log2(absmax)) - emax_elem

where ``emax_elem`` is the exponent of the largest normal element value (2 for FP4 E2M1,
whose largest value is 6 = 1.5 * 2^2, and 0 for MXINT8). Elements are then quantized
from ``x / 2^shared`` with saturation. Because the scale is a power of two, the block
absmax lands in ``[4, 8)`` for FP4, so values above 6 saturate: that clipping is part of
the specification, not a bug.

Element formats:

* FP4 E2M1: sign, 2 exponent bits (bias 1), 1 mantissa bit. Magnitudes by code:
  0, 0.5, 1, 1.5, 2, 3, 4, 6. No infinities or NaN.
* INT8: two's complement with an implicit scale of 2^-6, so codes -127..127 cover
  about ``[-2, 2)``. ``-128`` is not produced, which keeps the range symmetric.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import torch
from torch import Tensor

from bitsmith.quant.fp8 import round_to_table

MXFormat = Literal["fp4", "int8"]
BLOCK = 32
SCALE_BIAS = 127
SCALE_NAN = 0xFF
EMAX = {"fp4": 2, "int8": 0}
INT8_MAX = 127
INT8_STEP = 2.0**-6


@cache
def e2m1_table() -> Tensor:
    """Float32 value of each of the 16 FP4 E2M1 codes (bit 3 is the sign)."""
    values = []
    for code in range(16):
        sign = -1.0 if code & 0b1000 else 1.0
        exp, man = (code >> 1) & 0b11, code & 0b1
        mag = (man / 2) if exp == 0 else (1 + man / 2) * 2.0 ** (exp - 1)
        values.append(sign * mag)
    return torch.tensor(values, dtype=torch.float32)


@cache
def _table_on(device: torch.device) -> Tensor:
    return e2m1_table().to(device)


def fp4_encode(x: Tensor) -> Tensor:
    """Encode real values as uint8 E2M1 codes in ``[0, 15]`` (saturating, round to nearest even)."""
    x = x.to(torch.float32)
    mags = _table_on(x.device)[:8]
    code = round_to_table(x.abs(), mags)
    return (code | (torch.signbit(x).to(torch.long) << 3)).to(torch.uint8)


def fp4_decode(codes: Tensor) -> Tensor:
    return _table_on(codes.device)[codes.to(torch.long)]


def shared_exponent(blocks: Tensor, fmt: MXFormat) -> Tensor:
    """E8M0 scale code (uint8) for each block along the last dimension."""
    absmax = blocks.to(torch.float32).abs().amax(dim=-1)
    _, e = torch.frexp(absmax)  # absmax = m * 2^e with m in [0.5, 1), so floor(log2) = e - 1
    shared = (e.to(torch.int32) - 1 - EMAX[fmt]).clamp(-SCALE_BIAS, SCALE_BIAS)
    shared = torch.where(absmax > 0, shared, torch.full_like(shared, -SCALE_BIAS))
    return (shared + SCALE_BIAS).to(torch.uint8)


def scale_value(scale_codes: Tensor) -> Tensor:
    """Decode E8M0 codes to float32 powers of two (NaN for 0xFF)."""
    e = scale_codes.to(torch.int32) - SCALE_BIAS
    v = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    return torch.where(scale_codes == SCALE_NAN, torch.full_like(v, float("nan")), v)


def element_encode(x: Tensor, fmt: MXFormat) -> Tensor:
    """Quantize already scaled values to element codes.

    FP4 codes are uint8 in ``[0, 15]``; INT8 codes are int8 in ``[-127, 127]``.
    """
    if fmt == "fp4":
        return fp4_encode(x)
    return torch.round(x.to(torch.float32) / INT8_STEP).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def element_decode(codes: Tensor, fmt: MXFormat) -> Tensor:
    """Float32 value of each element code, before the shared scale."""
    return fp4_decode(codes) if fmt == "fp4" else codes.to(torch.float32) * INT8_STEP


def mx_encode(blocks: Tensor, fmt: MXFormat) -> tuple[Tensor, Tensor]:
    """Quantize ``(..., n_blocks, 32)`` values. Returns element codes and E8M0 scale codes."""
    if blocks.shape[-1] != BLOCK:
        raise ValueError(f"MX blocks hold {BLOCK} elements, got {blocks.shape[-1]}")
    scale = shared_exponent(blocks, fmt)
    return element_encode(blocks.to(torch.float32) / scale_value(scale).unsqueeze(-1), fmt), scale


def mx_decode(codes: Tensor, scale: Tensor, fmt: MXFormat) -> Tensor:
    """Inverse of :func:`mx_encode`, returning float32 values of shape ``(..., n_blocks, 32)``."""
    return element_decode(codes, fmt) * scale_value(scale).unsqueeze(-1)
