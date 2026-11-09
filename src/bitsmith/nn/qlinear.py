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

from bitsmith.nn.codecs import Int4Codec, WeightCodec, make_codec
from bitsmith.quant import QTensor, quantize_activations_per_token


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
        if qt.spec.bits != 4 or qt.spec.granularity != "group":
            raise ValueError("from_qtensor currently supports group-wise 4-bit tensors")
        scheme = "int4-sym" if qt.spec.symmetric else "int4"
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
        return isinstance(self.codec, Int4Codec) and self.act_bits is None

    def set_backend(self, backend: str) -> None:
        """``"reference"`` (dequantize then matmul) or ``"triton"`` (fused INT4 kernel)."""
        if backend not in ("reference", "triton"):
            raise ValueError("backend must be 'reference' or 'triton'")
        if backend == "triton" and not self.supports_triton():
            raise ValueError(f"the triton backend supports int4 weights only, not {self.codec.name}")
        self.backend = backend

    def forward(self, x: Tensor) -> Tensor:
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
        backend = ", backend=triton" if self.backend == "triton" else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.codec.name}, bits_per_weight={self.bits_per_weight():.3f}{act}{backend}"
        )
