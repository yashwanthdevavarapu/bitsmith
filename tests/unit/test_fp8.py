from __future__ import annotations

import math

import pytest
import torch

from bitsmith.quant import FORMATS, fp8_decode, fp8_encode, fp8_fake_quantize, ternarize, ternary_fake_quantize

TORCH_DTYPES = {"e4m3": getattr(torch, "float8_e4m3fn", None), "e5m2": getattr(torch, "float8_e5m2", None)}


@pytest.mark.parametrize(("fmt", "expected"), [("e4m3", 448.0), ("e5m2", 57344.0)])
def test_max_values(fmt: str, expected: float) -> None:
    assert FORMATS[fmt].max_value == expected


@pytest.mark.parametrize(("fmt", "smallest_subnormal"), [("e4m3", 2.0**-9), ("e5m2", 2.0**-16)])
def test_smallest_subnormal(fmt: str, smallest_subnormal: float) -> None:
    assert float(fp8_decode(torch.tensor([1]), fmt)) == smallest_subnormal  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_every_finite_code_round_trips(fmt: str) -> None:
    codes = torch.arange(256, dtype=torch.uint8)
    values = fp8_decode(codes, fmt)  # type: ignore[arg-type]
    finite = torch.isfinite(values)
    again = fp8_encode(values[finite], fmt)  # type: ignore[arg-type]
    assert torch.equal(again, codes[finite])


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_bit_exact_against_torch(fmt: str) -> None:
    dtype = TORCH_DTYPES[fmt]
    if dtype is None:
        pytest.skip("this torch build has no float8 dtypes")
    limit = FORMATS[fmt].max_value
    x = torch.cat([torch.randn(20_000) * 3, torch.randn(20_000) * limit / 4, torch.randn(5_000) * 1e-3])
    x = x.clamp(-limit, limit)
    ours = fp8_encode(x, fmt)  # type: ignore[arg-type]
    ref = x.to(dtype).view(torch.uint8)
    assert torch.equal(ours, ref)


def test_ties_round_to_even_mantissa() -> None:
    # 1.0 = 0x38 and 1.125 = 0x39 in e4m3; the midpoint must go to the even code 0x38.
    assert int(fp8_encode(torch.tensor([1.0625]), "e4m3")) == 0x38
    # 1.125 and 1.25 (0x3A): midpoint 1.1875 goes to 0x3A.
    assert int(fp8_encode(torch.tensor([1.1875]), "e4m3")) == 0x3A


def test_saturation_and_nan() -> None:
    codes = fp8_encode(torch.tensor([1e6, -1e6, math.nan]), "e4m3")
    values = fp8_decode(codes, "e4m3")
    assert values[0] == 448.0
    assert values[1] == -448.0
    assert math.isnan(float(values[2]))


def test_e5m2_has_infinity_code() -> None:
    assert math.isinf(float(fp8_decode(torch.tensor([0x7C]), "e5m2")))


def test_fake_quantize_keeps_dtype() -> None:
    x = torch.randn(16, dtype=torch.bfloat16)
    assert fp8_fake_quantize(x).dtype == torch.bfloat16


def test_relative_error_bound_e4m3() -> None:
    x = torch.rand(10_000) * 400 + 0.02  # normal range
    rel = ((fp8_fake_quantize(x) - x).abs() / x).max()
    assert rel <= 2.0**-4 + 1e-6  # half an ulp with 3 mantissa bits


def test_ternarize_values_and_scale() -> None:
    w = torch.tensor([[0.9, -0.1, -1.2, 0.4]])
    trits, gamma = ternarize(w)
    assert set(trits.flatten().tolist()) <= {-1, 0, 1}
    assert torch.isclose(gamma, w.abs().mean(dim=-1, keepdim=True)).all()
    assert trits.tolist() == [[1, 0, -1, 1]]
    assert ternary_fake_quantize(w).shape == w.shape
