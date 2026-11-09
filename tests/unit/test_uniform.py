from __future__ import annotations

import pytest
import torch

from bitsmith.eval import sqnr_db
from bitsmith.quant import QuantSpec, fake_quantize, quantize, quantize_activations_per_token
from bitsmith.quant.uniform import _from_groups

SPECS = [
    QuantSpec(bits=b, symmetric=s, granularity=g, group_size=32 if g == "group" else None)
    for b in (2, 4, 8)
    for s in (True, False)
    for g in ("tensor", "channel", "group")
]


@pytest.mark.parametrize("spec", SPECS, ids=str)
def test_error_is_at_most_half_a_step(spec: QuantSpec, weight: torch.Tensor) -> None:
    qt = quantize(weight, spec)
    err = (weight - qt.dequantize()).abs()
    step = _from_groups(qt.scale.expand(qt.codes.shape), weight.shape)
    assert torch.all(err <= step / 2 + 1e-6)


@pytest.mark.parametrize("spec", SPECS, ids=str)
def test_codes_stay_in_range(spec: QuantSpec, weight: torch.Tensor) -> None:
    codes = quantize(weight * 5, spec).codes.to(torch.int32)
    assert int(codes.min()) >= spec.qmin
    assert int(codes.max()) <= spec.qmax


def test_affine_represents_zero_exactly() -> None:
    x = torch.tensor([[0.0, 0.3, 0.7, 1.1, 2.0]])
    out = fake_quantize(x, QuantSpec(bits=4, symmetric=False, granularity="tensor"))
    assert out[0, 0] == 0.0


def test_affine_handles_all_positive_range() -> None:
    x = torch.linspace(1.0, 2.0, 64).reshape(1, -1)
    qt = quantize(x, QuantSpec(bits=8, symmetric=False, granularity="channel"))
    # range is widened to include 0, so the grid spans [0, 2]
    assert qt.zero.item() == 0
    assert (x - qt.dequantize()).abs().max() <= qt.scale.item() / 2 + 1e-6


def test_symmetric_grid_is_symmetric() -> None:
    spec = QuantSpec(bits=4, symmetric=True)
    assert (spec.qmin, spec.qmax) == (-7, 7)
    x = torch.tensor([[-1.0, 1.0]])
    out = fake_quantize(x, spec)
    assert torch.equal(out, x)


def test_more_bits_means_less_error(weight: torch.Tensor) -> None:
    sqnrs = [sqnr_db(weight, fake_quantize(weight, QuantSpec(bits=b, granularity="channel"))) for b in (2, 3, 4, 6, 8)]
    assert sqnrs == sorted(sqnrs)
    # roughly 6 dB per bit for a uniform quantizer
    assert 4.5 < (sqnrs[-1] - sqnrs[2]) / 4 < 7.0


def test_smaller_groups_isolate_outliers(weight: torch.Tensor) -> None:
    w = weight.clone()
    w[:, 7] *= 50
    coarse = sqnr_db(w, fake_quantize(w, QuantSpec(bits=4, granularity="channel")))
    fine = sqnr_db(w, fake_quantize(w, QuantSpec(bits=4, granularity="group", group_size=32)))
    assert fine > coarse + 3


@pytest.mark.parametrize("cols", [1, 31, 33, 100])
def test_group_padding_restores_shape(cols: int) -> None:
    x = torch.randn(5, cols)
    spec = QuantSpec(bits=4, symmetric=False, granularity="group", group_size=32)
    qt = quantize(x, spec)
    assert qt.codes.shape == (5, -(-cols // 32), 32)
    assert qt.dequantize().shape == x.shape


@pytest.mark.parametrize("shape", [(17,), (2, 3, 40), (4, 1)])
def test_arbitrary_rank(shape: tuple[int, ...]) -> None:
    x = torch.randn(*shape)
    for g in ("tensor", "channel"):
        out = fake_quantize(x, QuantSpec(bits=8, granularity=g))  # type: ignore[arg-type]
        assert out.shape == x.shape
        assert sqnr_db(x, out) > 30


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_dtype_is_preserved(dtype: torch.dtype, weight: torch.Tensor) -> None:
    x = weight.to(dtype)
    out = fake_quantize(x, QuantSpec(bits=8))
    assert out.dtype == dtype
    ref = fake_quantize(x.float(), QuantSpec(bits=8))
    torch.testing.assert_close(out.float(), ref.to(dtype).float())


def test_all_zero_tensor_is_stable() -> None:
    x = torch.zeros(4, 8)
    for s in (True, False):
        out = fake_quantize(x, QuantSpec(bits=4, symmetric=s))
        assert torch.equal(out, x)


def test_per_token_activation_quant() -> None:
    x = torch.randn(3, 5, 64)
    out = quantize_activations_per_token(x, 8)
    assert out.shape == x.shape
    assert sqnr_db(x, out) > 35


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 1},
        {"bits": 9},
        {"granularity": "group"},
        {"granularity": "row"},
        {"granularity": "group", "group_size": 1},
    ],
)
def test_invalid_specs_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"bits|granularity|group_size"):
        QuantSpec(**kwargs)  # type: ignore[arg-type]
