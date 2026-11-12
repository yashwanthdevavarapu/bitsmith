"""Known values for NF4 and the OCP MX formats."""

from __future__ import annotations

import pytest
import torch

from bitsmith.nn import make_codec
from bitsmith.quant import mx_decode, mx_encode, nf4_decode, nf4_encode, nf4_table
from bitsmith.quant.mx import e2m1_table, fp4_decode, fp4_encode, scale_value, shared_exponent

# NF4 levels as published with QLoRA (bitsandbytes), stored as float32 there.
PUBLISHED_NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


# NF4 --------------------------------------------------------------------------------
def test_nf4_table_matches_published_values() -> None:
    torch.testing.assert_close(nf4_table(), torch.tensor(PUBLISHED_NF4), rtol=0, atol=1e-6)


def test_nf4_table_has_exact_zero_and_ends() -> None:
    t = nf4_table()
    assert t[0] == -1.0 and t[7] == 0.0 and t[15] == 1.0
    assert bool((t[1:] > t[:-1]).all())


def test_nf4_every_level_round_trips() -> None:
    codes = torch.arange(16, dtype=torch.uint8)
    assert torch.equal(nf4_encode(nf4_decode(codes)), codes)


def test_nf4_picks_nearest_level_and_saturates() -> None:
    t = nf4_table()
    x = torch.tensor([-5.0, 0.01, 0.06, 0.9, 3.0])
    expected = torch.tensor([0, 7, 8, 15, 15], dtype=torch.uint8)
    assert torch.equal(nf4_encode(x), expected)
    mid = (t[8] + t[9]) / 2
    assert int(nf4_encode(mid.reshape(1))) == 8  # exact tie goes to the lower level


def test_nf4_codec_reproduces_block_absmax_exactly() -> None:
    w = torch.randn(4, 128) * 0.02
    codec = make_codec("nf4", 64)
    approx = codec.decode(codec.encode(w), 4, 128)
    blocks, approx_blocks = w.reshape(4, 2, 64), approx.reshape(4, 2, 64)
    absmax = blocks.abs().amax(-1)
    got = approx_blocks.abs().amax(-1)
    torch.testing.assert_close(got, absmax.half().float(), rtol=0, atol=0)


def test_nf4_codec_rejects_odd_group() -> None:
    with pytest.raises(ValueError, match="even"):
        make_codec("nf4", group_size=5)


# FP4 E2M1 ---------------------------------------------------------------------------
def test_e2m1_table() -> None:
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert e2m1_table().tolist() == mags + [-m for m in mags]


def test_fp4_round_to_nearest_even_and_saturation() -> None:
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0, -0.3, -2.6])
    # ties: 0.25 -> 0 (code 0), 0.75 -> 1.0 (code 2), 1.25 -> 1.0, 1.75 -> 2.0,
    # 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0; above 6 saturates
    expected = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -0.5, -3.0]
    assert fp4_decode(fp4_encode(x)).tolist() == expected


def test_fp4_negative_zero_keeps_sign_bit() -> None:
    assert int(fp4_encode(torch.tensor([-0.0]))) == 0b1000


# MX scales --------------------------------------------------------------------------
def test_shared_exponent_follows_the_spec() -> None:
    blocks = torch.zeros(3, 32)
    blocks[0, 0] = 6.0  # floor(log2 6) = 2, minus emax 2 -> 2^0
    blocks[1, 5] = -0.1  # floor(log2 0.1) = -4 -> 2^-6
    # block 2 is all zero
    assert (shared_exponent(blocks, "fp4").to(torch.int32) - 127).tolist() == [0, -6, -127]
    assert (shared_exponent(blocks, "int8").to(torch.int32) - 127).tolist() == [2, -4, -127]


def test_scale_value_decodes_powers_of_two_and_nan() -> None:
    v = scale_value(torch.tensor([127, 130, 0, 254, 255], dtype=torch.uint8))
    assert v[:4].tolist() == [1.0, 8.0, 2.0**-127, 2.0**127]
    assert torch.isnan(v[4])


# MX blocks --------------------------------------------------------------------------
def test_mxfp4_block_known_values() -> None:
    block = torch.zeros(1, 32)
    block[0, :4] = torch.tensor([6.0, 3.0, -1.5, 0.4])
    codes, scale = mx_encode(block, "fp4")
    assert int(scale) == 127
    assert mx_decode(codes, scale, "fp4")[0, :4].tolist() == [6.0, 3.0, -1.5, 0.5]


def test_mxfp4_clips_values_above_six_times_scale() -> None:
    block = torch.full((1, 32), 0.5)
    block[0, 0] = 7.5  # absmax in [4, 8) after scaling by 2^0: 7.5 saturates to 6
    codes, scale = mx_encode(block, "fp4")
    assert mx_decode(codes, scale, "fp4")[0, 0] == 6.0


def test_mxint8_block_known_values() -> None:
    block = torch.zeros(1, 32)
    block[0, :3] = torch.tensor([1.0, -0.5, 1.999])
    codes, scale = mx_encode(block, "int8")
    assert int(scale) == 127  # floor(log2 1.999) = 0
    assert codes[0, :3].tolist() == [64, -32, 127]  # 1.999 * 64 = 127.9 saturates
    torch.testing.assert_close(mx_decode(codes, scale, "int8")[0, :3], torch.tensor([1.0, -0.5, 127 / 64]))


@pytest.mark.parametrize("fmt", ["fp4", "int8"])
def test_mx_scale_invariance(fmt: str) -> None:
    """Multiplying a block by a power of two only moves its shared exponent."""
    x = torch.randn(8, 32)
    c1, s1 = mx_encode(x, fmt)  # type: ignore[arg-type]
    c2, s2 = mx_encode(x * 2.0**-5, fmt)  # type: ignore[arg-type]
    assert torch.equal(c1, c2)
    assert torch.equal(s1.to(torch.int32) - 5, s2.to(torch.int32))


def test_mx_rejects_wrong_block_size() -> None:
    with pytest.raises(ValueError, match="32 elements"):
        mx_encode(torch.zeros(2, 16), "fp4")


@pytest.mark.parametrize(("scheme", "nbytes"), [("mxfp4", 16 + 1), ("mxint8", 32 + 1)])
def test_mx_codec_storage_per_block(scheme: str, nbytes: int) -> None:
    bufs = make_codec(scheme).encode(torch.randn(1, 32))
    assert sum(t.numel() * t.element_size() for t in bufs.values()) == nbytes


def test_mx_codec_names_and_validation() -> None:
    from bitsmith.nn.codecs import MXCodec

    assert make_codec("mxfp4").name == "mxfp4"
    assert make_codec("mxint8").name == "mxint8"
    with pytest.raises(ValueError, match="fp4' or 'int8"):
        MXCodec("fp8")
