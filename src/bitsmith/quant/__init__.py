"""Quantization primitives: uniform integer, FP8 emulation and ternary."""

from bitsmith.quant.fp8 import FORMATS, fp8_decode, fp8_encode, fp8_fake_quantize
from bitsmith.quant.ternary import ternarize, ternary_fake_quantize
from bitsmith.quant.uniform import (
    QTensor,
    QuantSpec,
    compute_qparams,
    fake_quantize,
    quantize,
    quantize_activations_per_token,
    quantize_with,
)

__all__ = [
    "FORMATS",
    "QTensor",
    "QuantSpec",
    "compute_qparams",
    "fake_quantize",
    "fp8_decode",
    "fp8_encode",
    "fp8_fake_quantize",
    "quantize",
    "quantize_activations_per_token",
    "quantize_with",
    "ternarize",
    "ternary_fake_quantize",
]
