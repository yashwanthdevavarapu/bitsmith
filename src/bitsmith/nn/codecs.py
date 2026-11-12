"""Weight codecs: each one turns a dense ``(out, in)`` weight into compact buffers and back.

A codec is stateless apart from its configuration. ``encode`` returns a dict of tensors
that :class:`bitsmith.nn.QLinear` registers as buffers, so the packed form is what ends
up in ``state_dict``. ``decode`` rebuilds a float32 ``(out, in)`` matrix.

Scales are stored in fp16. To keep encode and decode consistent, scales are rounded to
fp16 *before* the integer codes are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from bitsmith.packing import (
    pack_24,
    pack_int4,
    pack_ternary_2bit,
    pack_ternary_base3,
    unpack_24,
    unpack_int4,
    unpack_ternary_2bit,
    unpack_ternary_base3,
)
from bitsmith.quant import (
    QTensor,
    QuantSpec,
    compute_qparams,
    fp8_decode,
    fp8_encode,
    mx_decode,
    nf4_decode,
    nf4_encode,
    ternarize,
)
from bitsmith.quant.fp8 import FORMATS
from bitsmith.quant.mx import BLOCK as MX_BLOCK
from bitsmith.quant.mx import element_decode, element_encode, scale_value, shared_exponent
from bitsmith.quant.uniform import _to_groups, quantize_with

SCALE_DTYPE = torch.float16


def round_scale(scale: Tensor) -> Tensor:
    """Round a float32 scale to the storage dtype and back."""
    return scale.to(SCALE_DTYPE).to(torch.float32).clamp_min(2.0**-24)


def _pad_cols(w: Tensor, multiple: int) -> Tensor:
    pad = (-w.shape[-1]) % multiple
    return F.pad(w, (0, pad)) if pad else w


class WeightCodec(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, weight: Tensor) -> dict[str, Tensor]: ...

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor: ...


@dataclass(frozen=True)
class Int8Codec:
    """Symmetric per-output-channel INT8."""

    name: ClassVar[str] = "int8"

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        spec = QuantSpec(bits=8, symmetric=True, granularity="channel")
        groups = _to_groups(weight, spec)
        scale, zero = compute_qparams(groups, spec)
        scale = round_scale(scale)
        codes = quantize_with(groups, scale, zero, spec).to(torch.int8)
        return {"qweight": codes.reshape(weight.shape), "scale": scale.reshape(-1, 1).to(SCALE_DTYPE)}

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        return buffers["qweight"].to(torch.float32) * buffers["scale"].to(torch.float32)


@dataclass(frozen=True)
class Int4Codec:
    """Group-wise INT4, asymmetric by default, packed two codes per byte."""

    group_size: int = 128
    symmetric: bool = False
    name: ClassVar[str] = "int4"

    def __post_init__(self) -> None:
        if self.group_size < 2 or self.group_size % 2:
            raise ValueError("int4 group_size must be an even number >= 2")

    @property
    def spec(self) -> QuantSpec:
        return QuantSpec(bits=4, symmetric=self.symmetric, granularity="group", group_size=self.group_size)

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        groups = _to_groups(weight, self.spec)
        scale, zero = compute_qparams(groups, self.spec)
        scale = round_scale(scale)
        codes = quantize_with(groups, scale, zero, self.spec)
        qt = QTensor(codes=codes, scale=scale, zero=zero, spec=self.spec, shape=weight.shape)
        return self.from_qtensor(qt)

    def from_qtensor(self, qt: QTensor) -> dict[str, Tensor]:
        """Pack an already quantized tensor (for example the output of GPTQ)."""
        if qt.spec != self.spec:
            raise ValueError(f"QTensor spec {qt.spec} does not match codec spec {self.spec}")
        rows = qt.codes.shape[0]
        bufs = {
            "qweight": pack_int4(qt.codes.reshape(rows, -1).to(torch.int16)),
            "scale": qt.scale.reshape(rows, -1).to(SCALE_DTYPE),
        }
        if not self.symmetric:
            bufs["zero"] = qt.zero.reshape(rows, -1).to(torch.uint8)
        return bufs

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        codes = unpack_int4(buffers["qweight"], signed=self.symmetric).to(torch.float32)
        codes = codes.reshape(out_features, -1, self.group_size)
        scale = buffers["scale"].to(torch.float32).unsqueeze(-1)
        if self.symmetric:
            w = codes * scale
        else:
            w = (codes - buffers["zero"].to(torch.float32).unsqueeze(-1)) * scale
        return w.reshape(out_features, -1)[:, :in_features]


@dataclass(frozen=True)
class TernaryCodec:
    """BitNet b1.58 style ternary weights with a per-row absmean scale."""

    packing: str = "2bit"

    @property
    def name(self) -> str:
        return "ternary" if self.packing == "2bit" else "ternary-b3"

    def __post_init__(self) -> None:
        if self.packing not in ("2bit", "base3"):
            raise ValueError("packing must be '2bit' or 'base3'")

    @property
    def _multiple(self) -> int:
        return 4 if self.packing == "2bit" else 5

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        trits, gamma = ternarize(weight)
        trits = _pad_cols(trits, self._multiple)
        pack = pack_ternary_2bit if self.packing == "2bit" else pack_ternary_base3
        return {"qweight": pack(trits), "scale": gamma.to(SCALE_DTYPE)}

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        unpack = unpack_ternary_2bit if self.packing == "2bit" else unpack_ternary_base3
        trits = unpack(buffers["qweight"])[:, :in_features].to(torch.float32)
        return trits * buffers["scale"].to(torch.float32)


@dataclass(frozen=True)
class Sparse24Codec:
    """2:4 structured sparsity with fp16 values and 4-bit index metadata per group."""

    name: ClassVar[str] = "sparse24"

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        values, meta = pack_24(_pad_cols(weight.to(torch.float32), 8))
        return {"values": values.to(torch.float16), "meta": meta}

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        cols = in_features + (-in_features) % 8
        dense = unpack_24(buffers["values"].to(torch.float32), buffers["meta"], cols)
        return dense[:, :in_features]


@dataclass(frozen=True)
class FP8Codec:
    """OCP FP8 weights with a float32 per-output-channel scale mapping absmax to the format max."""

    fmt: str = "e4m3"

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown fp8 format {self.fmt!r}")

    @property
    def name(self) -> str:
        return f"fp8-{self.fmt}"

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        w = weight.to(torch.float32)
        absmax = w.abs().amax(dim=-1, keepdim=True)
        # float32 scale: e5m2 scales can fall below the fp16 normal range
        scale = (absmax / FORMATS[self.fmt].max_value).clamp_min(1e-30)
        codes = fp8_encode(w / scale, self.fmt)  # type: ignore[arg-type]
        return {"qweight": codes, "scale": scale}

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        values = fp8_decode(buffers["qweight"], self.fmt)  # type: ignore[arg-type]
        return values * buffers["scale"].to(torch.float32)


class TableCodec(WeightCodec, Protocol):
    """A codec whose weights are one stored scale per block times a value from a fixed grid.

    Splitting encoding into "pick the scales" and "encode with these scales" lets GPTQ fix
    the scales from the original weights and then round each column onto the grid itself.
    """

    @property
    def block_size(self) -> int: ...

    def scale_buffer(self, weight: Tensor) -> Tensor:
        """Stored scale of each block, shape ``(rows, n_blocks)``."""
        ...

    def scale_values(self, scale: Tensor) -> Tensor:
        """Float32 multiplier for each stored scale."""
        ...

    def snap(self, x: Tensor) -> Tensor:
        """Nearest grid value for each already scaled value."""
        ...

    def encode_with_scale(self, weight: Tensor, scale: Tensor) -> dict[str, Tensor]: ...


def _blocks(weight: Tensor, size: int) -> Tensor:
    return _pad_cols(weight.to(torch.float32), size).reshape(weight.shape[0], -1, size)


@dataclass(frozen=True)
class NF4Codec:
    """NormalFloat4: per-block absmax scale (fp16) and 4-bit indices into the NF4 table."""

    group_size: int = 64
    name: ClassVar[str] = "nf4"

    def __post_init__(self) -> None:
        if self.group_size < 2 or self.group_size % 2:
            raise ValueError("nf4 group_size must be an even number >= 2")

    @property
    def block_size(self) -> int:
        return self.group_size

    def scale_buffer(self, weight: Tensor) -> Tensor:
        return round_scale(_blocks(weight, self.group_size).abs().amax(dim=-1)).to(SCALE_DTYPE)

    def scale_values(self, scale: Tensor) -> Tensor:
        return scale.to(torch.float32)

    def snap(self, x: Tensor) -> Tensor:
        return nf4_decode(nf4_encode(x))

    def encode_with_scale(self, weight: Tensor, scale: Tensor) -> dict[str, Tensor]:
        x = _blocks(weight, self.group_size) / self.scale_values(scale).unsqueeze(-1)
        return {"qweight": pack_int4(nf4_encode(x).reshape(weight.shape[0], -1)), "scale": scale}

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        return self.encode_with_scale(weight, self.scale_buffer(weight))

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        levels = nf4_decode(unpack_int4(buffers["qweight"], signed=False))
        w = levels.reshape(out_features, -1, self.group_size) * buffers["scale"].to(torch.float32).unsqueeze(-1)
        return w.reshape(out_features, -1)[:, :in_features]


@dataclass(frozen=True)
class MXCodec:
    """OCP microscaling: blocks of 32 elements sharing one E8M0 power-of-two scale."""

    fmt: str = "fp4"

    def __post_init__(self) -> None:
        if self.fmt not in ("fp4", "int8"):
            raise ValueError("mx element format must be 'fp4' or 'int8'")

    @property
    def name(self) -> str:
        return f"mx{self.fmt}"

    @property
    def block_size(self) -> int:
        return MX_BLOCK

    def scale_buffer(self, weight: Tensor) -> Tensor:
        return shared_exponent(_blocks(weight, MX_BLOCK), self.fmt)  # type: ignore[arg-type]

    def scale_values(self, scale: Tensor) -> Tensor:
        return scale_value(scale)

    def snap(self, x: Tensor) -> Tensor:
        return element_decode(element_encode(x, self.fmt), self.fmt)  # type: ignore[arg-type]

    def encode_with_scale(self, weight: Tensor, scale: Tensor) -> dict[str, Tensor]:
        x = _blocks(weight, MX_BLOCK) / self.scale_values(scale).unsqueeze(-1)
        codes = element_encode(x, self.fmt).reshape(weight.shape[0], -1)  # type: ignore[arg-type]
        return {"qweight": pack_int4(codes) if self.fmt == "fp4" else codes, "scale": scale}

    def encode(self, weight: Tensor) -> dict[str, Tensor]:
        return self.encode_with_scale(weight, self.scale_buffer(weight))

    def decode(self, buffers: dict[str, Tensor], out_features: int, in_features: int) -> Tensor:
        codes = buffers["qweight"]
        if self.fmt == "fp4":
            codes = unpack_int4(codes, signed=False)
        blocks = codes.reshape(out_features, -1, MX_BLOCK)
        w = mx_decode(blocks, buffers["scale"], self.fmt)  # type: ignore[arg-type]
        return w.reshape(out_features, -1)[:, :in_features]


SCHEMES = (
    "int8",
    "int4",
    "int4-sym",
    "nf4",
    "mxfp4",
    "mxint8",
    "ternary",
    "ternary-b3",
    "sparse24",
    "fp8-e4m3",
    "fp8-e5m2",
)


# schemes whose codec is a TableCodec, so GPTQ can round onto their grid
TABLE_SCHEMES = ("nf4", "mxfp4", "mxint8")


def make_codec(scheme: str, group_size: int = 128) -> WeightCodec:
    """Build a codec from a scheme name. See :data:`SCHEMES`.

    ``group_size`` applies to ``int4``, ``int4-sym`` and ``nf4``. MX formats always use
    blocks of 32, as the specification requires.
    """
    if scheme == "int8":
        return Int8Codec()
    if scheme == "int4":
        return Int4Codec(group_size=group_size)
    if scheme == "int4-sym":
        return Int4Codec(group_size=group_size, symmetric=True)
    if scheme == "nf4":
        return NF4Codec(group_size=group_size)
    if scheme == "mxfp4":
        return MXCodec("fp4")
    if scheme == "mxint8":
        return MXCodec("int8")
    if scheme == "ternary":
        return TernaryCodec("2bit")
    if scheme == "ternary-b3":
        return TernaryCodec("base3")
    if scheme == "sparse24":
        return Sparse24Codec()
    if scheme.startswith("fp8-"):
        return FP8Codec(scheme.removeprefix("fp8-"))
    raise ValueError(f"unknown scheme {scheme!r}; choose from {', '.join(SCHEMES)}")
