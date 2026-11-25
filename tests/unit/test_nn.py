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
    "int3": 11,
    "int2": 4,
    "nf4": 17,
    "mxfp4": 16,
    "mxint8": 38,
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
    "int3": 3 + (16 + 8) / 128,
    "int2": 2 + (16 + 8) / 128,
    "nf4": 4 + 16 / 128,
    "mxfp4": 4 + 8 / 32,
    "mxint8": 8 + 8 / 32,
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
@pytest.mark.usefixtures("cpu_half_matmul")
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
    with pytest.raises(ValueError, match="group-wise 4-, 3- and 2-bit"):
        QLinear.from_qtensor(_linear(4, 8), qt)


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="unknown scheme"):
        make_codec("int5")
    with pytest.raises(ValueError, match="multiple of 2"):
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
    with pytest.raises(ValueError, match="needs int4, int3 or int2 weights"):
        q8.set_backend("triton")
    with pytest.raises(ValueError, match="backend must be"):
        q8.set_backend("cuda")


def test_int3_and_int2_codec_validation() -> None:
    from bitsmith.nn.codecs import IntCodec

    assert make_codec("int3").name == "int3" and make_codec("int2").name == "int2"
    with pytest.raises(ValueError, match="multiple of 8"):
        make_codec("int3", group_size=12)
    with pytest.raises(ValueError, match="asymmetric only"):
        IntCodec(bits=3, symmetric=True)
    with pytest.raises(ValueError, match="bits must be"):
        IntCodec(bits=5)
    assert QLinear.from_linear(_linear(), "int3", 128).supports_triton()  # fused kernels, ADR 0021
    assert QLinear.from_linear(_linear(), "int2", 64).supports_triton()
    assert not QLinear.from_linear(_linear(), "int3", 16).supports_triton()  # group too small
    assert not QLinear.from_linear(_linear(), "int4", 32, act_bits=8).supports_triton()


@pytest.mark.parametrize("bits", [3, 2])
def test_from_qtensor_low_bit(bits: int) -> None:
    lin = _linear()
    spec = QuantSpec(bits=bits, symmetric=False, granularity="group", group_size=64)
    qt = quantize(lin.weight.detach(), spec)
    q = QLinear.from_qtensor(lin, qt)
    assert q.codec.name == f"int{bits}"
    torch.testing.assert_close(q.dequantized_weight(), qt.dequantize(), rtol=1e-3, atol=1e-5)


def test_int8_backend_matches_simulated_w8a8() -> None:
    """The integer matmul uses the same codes as the simulated W8A8 path."""
    from bitsmith.nn import set_backend

    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    q = QLinear.from_linear(lin, "int8", act_bits=8)
    x = torch.randn(3, 7, 64)
    ref = q(x)
    assert q.supports("int8") and not q.supports("triton")
    q.set_backend("int8")
    assert "backend=int8" in repr(q)
    torch.testing.assert_close(q(x), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q(x[0, :1]), ref[0, :1], rtol=1e-5, atol=1e-5)  # one row, padded inside

    model = _Toy()
    quantize_model(model, "int8", act_bits=8)
    assert set_backend(model, "int8") == 2
    with pytest.raises(ValueError, match="int8 weights with act_bits=8"):
        QLinear.from_linear(_linear(), "int8").set_backend("int8")  # weight-only int8
    with pytest.raises(ValueError, match="int8 weights with act_bits=8"):
        QLinear.from_linear(_linear(), "int4", 32, act_bits=8).set_backend("int8")


def test_int8_backend_applies_input_scale_inside() -> None:
    """A smoothed W8A8 layer gives the same result on the int8 backend as on the reference."""
    from bitsmith.kernels.int8_mm import int8_linear, quantize_rows_int8

    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    s = torch.rand(64) + 0.5
    q = QLinear.from_linear(lin, "int8", weight=lin.weight * s, input_scale=1 / s, act_bits=8)
    x = torch.randn(5, 64)
    ref = q(x)
    q.set_backend("int8")
    torch.testing.assert_close(q(x), ref, rtol=1e-5, atol=1e-5)
    codes, scale = quantize_rows_int8(x, 1 / s)
    codes_ref, scale_ref = quantize_rows_int8(x * (1 / s))
    assert torch.equal(codes, codes_ref) and torch.equal(scale, scale_ref)
    bufs = q.packed_buffers()
    with pytest.raises(ValueError, match="input_scale"):
        int8_linear(x, bufs["qweight"], bufs["scale"], input_scale=torch.ones(63))


def test_int8_linear_validation() -> None:
    from bitsmith.kernels.int8_mm import int8_linear, int8_mm_supported

    w = torch.randint(-127, 128, (16, 32), dtype=torch.int8)
    s = torch.ones(16, 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="int8"):
        int8_linear(torch.randn(2, 32), w.float(), s)
    with pytest.raises(ValueError, match="input features"):
        int8_linear(torch.randn(2, 31), w, s)
    with pytest.raises(ValueError, match="divisible by 8"):
        int8_linear(torch.randn(2, 30), w[:, :30], s)
    assert not int8_mm_supported(30, 16)
    x = torch.randn(20, 32)
    y = int8_linear(x, w, s, bias=torch.full((16,), 2.0))
    assert y.shape == (20, 16) and y.dtype == x.dtype
