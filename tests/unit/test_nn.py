from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitsmith.eval import sqnr_db
from bitsmith.nn import SCHEMES, QLinear, make_codec, quantize_model
from bitsmith.quant import QuantSpec, quantize

# Minimum weight SQNR (dB) each scheme must reach on a Gaussian weight.
MIN_SQNR = {
    "int8": 38,
    "int4": 17,
    "int4-sym": 14,
    "ternary": 4,
    "ternary-b3": 4,
    "sparse24": 7,
    "fp8-e4m3": 28,
    "fp8-e5m2": 22,
}

# Expected bits per weight for a (64, 256) layer with group size 128.
EXPECTED_BPW = {
    "int8": 8 + 16 / 256,
    "int4": 4 + (16 + 8) / 128,
    "int4-sym": 4 + 16 / 128,
    "ternary": 2 + 16 / 256,
    "ternary-b3": 8 * 52 / 256 + 16 / 256,  # 256 trits -> 52 bytes (padded to 260)
    "sparse24": 9.0,
    "fp8-e4m3": 8 + 32 / 256,
    "fp8-e5m2": 8 + 32 / 256,
}


def _linear(out_f: int = 64, in_f: int = 256, bias: bool = True) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
    return lin


@pytest.mark.parametrize("scheme", SCHEMES)
def test_codec_round_trip_quality(scheme: str, weight: torch.Tensor) -> None:
    codec = make_codec(scheme, 128)
    approx = codec.decode(codec.encode(weight), *weight.shape)
    assert approx.shape == weight.shape
    assert sqnr_db(weight, approx) > MIN_SQNR[scheme]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_qlinear_forward_matches_effective_weight(scheme: str) -> None:
    lin = _linear()
    q = QLinear.from_linear(lin, scheme, 128)
    x = torch.randn(3, 5, 256)
    expected = F.linear(x, q.effective_weight(), lin.bias)
    torch.testing.assert_close(q(x), expected)
    # and it stays close to the float layer
    assert sqnr_db(lin(x), q(x)) > MIN_SQNR[scheme] - 3


@pytest.mark.parametrize("scheme", SCHEMES)
def test_bits_per_weight_accounting(scheme: str) -> None:
    q = QLinear.from_linear(_linear(), scheme, 128)
    assert q.bits_per_weight() == pytest.approx(EXPECTED_BPW[scheme])


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("in_f", [100, 36, 9])
def test_awkward_input_widths(scheme: str, in_f: int) -> None:
    lin = _linear(8, in_f)
    q = QLinear.from_linear(lin, scheme, 32)
    assert q(torch.randn(2, in_f)).shape == (2, 8)
    assert q.dequantized_weight().shape == (8, in_f)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_state_dict_round_trip(scheme: str) -> None:
    src = QLinear.from_linear(_linear(), scheme, 128)
    dst = QLinear.from_linear(_linear(), scheme, 128)
    dst.load_state_dict(src.state_dict())
    x = torch.randn(4, 256)
    assert torch.equal(src(x), dst(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_layer_with_bias(dtype: torch.dtype) -> None:
    lin = _linear().to(dtype)
    q = QLinear.from_linear(lin, "int4", 64)
    x = torch.randn(2, 256, dtype=dtype)
    y = q(x)
    assert y.dtype == dtype
    assert q.bias is not None and q.bias.dtype == dtype


def test_input_scale_and_activation_quant() -> None:
    lin = _linear()
    s = torch.rand(256) + 0.5
    q = QLinear.from_linear(lin, "int8", weight=lin.weight * s, input_scale=1 / s, act_bits=8)
    x = torch.randn(4, 256)
    assert sqnr_db(lin(x), q(x)) > 30
    assert "act_bits=8" in repr(q)


def test_from_qtensor_uses_codes_verbatim() -> None:
    lin = _linear()
    spec = QuantSpec(bits=4, symmetric=False, granularity="group", group_size=64)
    qt = quantize(lin.weight.detach(), spec)
    q = QLinear.from_qtensor(lin, qt)
    torch.testing.assert_close(q.dequantized_weight(), qt.dequantize(), rtol=1e-3, atol=1e-5)


def test_from_qtensor_rejects_other_specs() -> None:
    qt = quantize(torch.randn(4, 8), QuantSpec(bits=8))
    with pytest.raises(ValueError, match="group-wise 4-bit"):
        QLinear.from_qtensor(_linear(4, 8), qt)


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="unknown scheme"):
        make_codec("int3")
    with pytest.raises(ValueError, match="even"):
        make_codec("int4", group_size=7)
    with pytest.raises(ValueError, match="shape"):
        QLinear.from_linear(_linear(), "int8", weight=torch.zeros(3, 3))


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.lm_head = nn.Linear(32, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.encoder(x))


def test_quantize_model_replaces_layers_and_reports() -> None:
    model = _Toy()
    report = quantize_model(model, "int4", group_size=32)
    assert isinstance(model.encoder[0], QLinear)
    assert isinstance(model.encoder[2], QLinear)
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, QLinear)
    assert report.converted == ["encoder.0", "encoder.2"]
    assert report.skipped == ["lm_head"]
    assert report.n_weights == 2 * 32 * 64
    assert report.compression > 5
    assert "2 layers converted" in report.summary()
    assert model(torch.randn(2, 32)).shape == (2, 10)


def test_quantize_model_glob_exclude() -> None:
    model = _Toy()
    report = quantize_model(model, "int8", exclude=["encoder.2", "lm_head"])
    assert report.converted == ["encoder.0"]


def test_backend_switching() -> None:
    from bitsmith.nn import set_backend

    model = _Toy()
    quantize_model(model, "int4", group_size=32)
    assert set_backend(model, "triton") == 2
    assert "backend=triton" in repr(model.encoder[0])
    x = torch.randn(2, 32)
    # on CPU the triton backend falls back to the reference path
    assert model(x).shape == (2, 10)
    assert set_backend(model, "reference") == 2
    q8 = QLinear.from_linear(_linear(), "int8")
    with pytest.raises(ValueError, match="int4 weights only"):
        q8.set_backend("triton")
    with pytest.raises(ValueError, match="backend must be"):
        q8.set_backend("cuda")
