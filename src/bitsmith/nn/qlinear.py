"""``QLinear``: a drop-in replacement for ``nn.Linear`` that stores packed weights.

Forward pass (reference runtime, see ADR 0004)::

    x  -> x * input_scale            (optional, from SmoothQuant or AWQ)
       -> per-token fake quant       (optional, act_bits)
    W  =  codec.decode(buffers)      (unpack + dequantize to x.dtype)
    y  =  x @ W.T + bias

The dense weight is rebuilt on every call. That keeps the code readable and exact, but it
is slower than a fused kernel that dequantizes inside the matmul.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bitsmith.nn.codecs import Int4Codec, Int8Codec, WeightCodec, make_codec
from bitsmith.quant import QTensor, quantize_activations_per_token

BACKENDS = ("reference", "triton", "int8")


class QLinear(nn.Module):
    input_scale: Tensor | None
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codec: WeightCodec,
        buffers: dict[str, Tensor],
        bias: Tensor | None = None,
        input_scale: Tensor | None = None,
        act_bits: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codec = codec
        self.act_bits = act_bits
        self.backend = "reference"
        self._buffer_names = tuple(buffers)
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor.contiguous())
        # Keep the bias in the original dtype so half-precision models stay consistent.
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
        self.register_buffer("input_scale", input_scale.detach().clone() if input_scale is not None else None)

    # construction ---------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        scheme: str = "int4",
        group_size: int = 128,
        *,
        weight: Tensor | None = None,
        input_scale: Tensor | None = None,
        act_bits: int | None = None,
    ) -> QLinear:
        """Quantize ``linear`` (or an explicit ``weight`` override, e.g. after smoothing)."""
        codec = make_codec(scheme, group_size)
        w = (weight if weight is not None else linear.weight).detach()
        if w.shape != linear.weight.shape:
            raise ValueError(f"weight shape {tuple(w.shape)} != {tuple(linear.weight.shape)}")
        buffers = {k: v.to(linear.weight.device) for k, v in codec.encode(w).items()}
        return cls(
            linear.in_features,
            linear.out_features,
            codec,
            buffers,
            bias=linear.bias,
            input_scale=input_scale,
            act_bits=act_bits,
        )

    @classmethod
    def from_qtensor(
        cls,
        linear: nn.Linear,
        qt: QTensor,
        *,
        input_scale: Tensor | None = None,
        act_bits: int | None = None,
    ) -> QLinear:
        """Wrap integer codes produced elsewhere (GPTQ) without re-quantizing them."""
        if qt.spec.bits not in (2, 3, 4) or qt.spec.granularity != "group":
            raise ValueError("from_qtensor supports group-wise 4-, 3- and 2-bit tensors")
        scheme = f"int{qt.spec.bits}" + ("-sym" if qt.spec.symmetric else "")
        codec = make_codec(scheme, int(qt.spec.group_size or 0))
        buffers = {k: v.to(linear.weight.device) for k, v in codec.from_qtensor(qt).items()}  # type: ignore[attr-defined]
        return cls(
            linear.in_features,
            linear.out_features,
            codec,
            buffers,
            bias=linear.bias,
            input_scale=input_scale,
            act_bits=act_bits,
        )

    @classmethod
    def from_buffers(
        cls,
        linear: nn.Linear,
        scheme: str,
        group_size: int,
        buffers: dict[str, Tensor],
        *,
        act_bits: int | None = None,
    ) -> QLinear:
        """Wrap buffers already produced by the codec of ``scheme`` (GPTQ on grid formats)."""
        codec = make_codec(scheme, group_size)
        moved = {k: v.to(linear.weight.device) for k, v in buffers.items()}
        return cls(linear.in_features, linear.out_features, codec, moved, bias=linear.bias, act_bits=act_bits)

    # runtime --------------------------------------------------------------------------
    def packed_buffers(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in self._buffer_names}

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> Tensor:
        w = self.codec.decode(self.packed_buffers(), self.out_features, self.in_features)
        return w.to(dtype)

    def effective_weight(self) -> Tensor:
        """The float32 matrix this layer actually applies, input scaling folded in."""
        w = self.dequantized_weight()
        if self.input_scale is not None:
            w = w * self.input_scale.to(torch.float32).unsqueeze(0)
        return w

    def supports_triton(self) -> bool:
        return isinstance(self.codec, Int4Codec) and self.act_bits is None and self.codec.group_size in (32, 64, 128)

    def supports_int8(self) -> bool:
        """INT8 weights with INT8 activations (W8A8) on ``torch._int_mm``."""
        from bitsmith.kernels.int8_mm import int8_mm_supported

        shape_ok = int8_mm_supported(self.in_features, self.out_features)
        return isinstance(self.codec, Int8Codec) and self.act_bits == 8 and shape_ok

    def supports(self, backend: str) -> bool:
        return {"reference": True, "triton": self.supports_triton(), "int8": self.supports_int8()}.get(backend, False)

    def set_backend(self, backend: str) -> None:
        """``"reference"`` (dequantize then matmul), ``"triton"`` (fused INT4, INT3 and INT2
        kernels, ADR 0021) or ``"int8"`` (integer W8A8 matmul)."""
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if not self.supports(backend):
            needs = {
                "triton": "int4, int3 or int2 weights with group size 32, 64 or 128",
                "int8": "int8 weights with act_bits=8",
            }[backend]
            raise ValueError(f"the {backend} backend needs {needs}, not {self.codec.name}")
        self.backend = backend

    def forward(self, x: Tensor) -> Tensor:
        if self.backend == "int8":  # the input scale is applied inside the quantization pass
            from bitsmith.kernels.int8_mm import int8_linear

            bufs = self.packed_buffers()
            return int8_linear(x, bufs["qweight"], bufs["scale"], self.bias, input_scale=self.input_scale)
        if self.input_scale is not None:
            x = x * self.input_scale.to(x.dtype)
        if self.backend == "triton" and x.is_cuda:
            from bitsmith.kernels import int4_linear

            codec = self.codec
            assert isinstance(codec, Int4Codec)
            bufs = self.packed_buffers()
            return int4_linear(
                x,
                bufs["qweight"],
                bufs["scale"],
                None if codec.symmetric else bufs["zero"],
                self.in_features,
                codec.group_size,
                self.bias,
                bits=codec.bits,
            )
        if self.act_bits is not None:
            x = quantize_activations_per_token(x, self.act_bits)
        w = self.dequantized_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    # accounting -----------------------------------------------------------------------
    def weight_bytes(self) -> int:
        """Bytes used by the packed weight, its scales and metadata (bias excluded)."""
        total = sum(t.numel() * t.element_size() for t in self.packed_buffers().values())
        if self.input_scale is not None:
            total += self.input_scale.numel() * self.input_scale.element_size()
        return total

    def bits_per_weight(self) -> float:
        return 8.0 * self.weight_bytes() / (self.in_features * self.out_features)

    def extra_repr(self) -> str:
        act = f", act_bits={self.act_bits}" if self.act_bits else ""
        backend = f", backend={self.backend}" if self.backend != "reference" else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.codec.name}, bits_per_weight={self.bits_per_weight():.3f}{act}{backend}"
        )
