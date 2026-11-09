"""Bit-exact software emulation of the OCP 8-bit floating point formats.

* ``e4m3``  (E4M3FN): bias 7, 3 mantissa bits, no infinities, max 448, NaN = S.1111.111
* ``e5m2``: bias 15, 2 mantissa bits, IEEE-like infinities and NaNs, max 57344

Encoding saturates finite values to the largest representable magnitude and rounds to
nearest with ties to even. Because the bit patterns of non-negative finite values are
monotonic in their value, encoding is a sorted search over a 1-D table of magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch
from torch import Tensor

FP8Format = Literal["e4m3", "e5m2"]


@dataclass(frozen=True)
class FP8Spec:
    exp_bits: int
    man_bits: int
    bias: int
    max_code: int  # largest positive finite code
    nan_code: int  # canonical positive NaN code

    @property
    def max_value(self) -> float:
        return float(_decode_table(self)[self.max_code])


FORMATS: dict[str, FP8Spec] = {
    "e4m3": FP8Spec(exp_bits=4, man_bits=3, bias=7, max_code=0x7E, nan_code=0x7F),
    "e5m2": FP8Spec(exp_bits=5, man_bits=2, bias=15, max_code=0x7B, nan_code=0x7F),
}


def _decode_code(code: int, spec: FP8Spec) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> spec.man_bits) & ((1 << spec.exp_bits) - 1)
    man = code & ((1 << spec.man_bits) - 1)
    max_exp = (1 << spec.exp_bits) - 1
    if spec.exp_bits == 4 and exp == max_exp and man == (1 << spec.man_bits) - 1:
        return float("nan")
    if spec.exp_bits == 5 and exp == max_exp:
        return sign * float("inf") if man == 0 else float("nan")
    if exp == 0:  # subnormal
        return sign * (man / (1 << spec.man_bits)) * 2.0 ** (1 - spec.bias)
    return sign * (1.0 + man / (1 << spec.man_bits)) * 2.0 ** (exp - spec.bias)


@cache
def _decode_table(spec: FP8Spec) -> Tensor:
    return torch.tensor([_decode_code(c, spec) for c in range(256)], dtype=torch.float32)


def fp8_decode(codes: Tensor, fmt: FP8Format = "e4m3") -> Tensor:
    """Decode uint8 bit patterns into float32 values."""
    table = _decode_table(FORMATS[fmt]).to(codes.device)
    return table[codes.to(torch.long)]


def fp8_encode(x: Tensor, fmt: FP8Format = "e4m3") -> Tensor:
    """Encode real values into uint8 FP8 bit patterns (saturating, round to nearest even)."""
    spec = FORMATS[fmt]
    mags = _decode_table(spec)[: spec.max_code + 1].to(x.device)  # non-negative finite magnitudes, ascending
    x = x.to(torch.float32)
    a = x.abs().clamp(max=float(mags[-1]))
    hi = torch.searchsorted(mags, a.contiguous()).clamp(max=spec.max_code)
    lo = (hi - 1).clamp(min=0)
    d_lo = a - mags[lo]
    d_hi = mags[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(pick_hi, hi, lo)
    sign = (torch.signbit(x)).to(torch.long) << 7
    code = code | sign
    code = torch.where(torch.isnan(x), torch.full_like(code, spec.nan_code), code)
    return code.to(torch.uint8)


def fp8_fake_quantize(x: Tensor, fmt: FP8Format = "e4m3") -> Tensor:
    """Round ``x`` through the FP8 grid and return it in its original dtype."""
    return fp8_decode(fp8_encode(x, fmt), fmt).to(x.dtype)
