from __future__ import annotations

import pytest
import torch

from bitsmith.packing import (
    mask_24,
    pack_24,
    pack_int4,
    pack_ternary_2bit,
    pack_ternary_base3,
    prune_24,
    unpack_24,
    unpack_int4,
    unpack_ternary_2bit,
    unpack_ternary_base3,
)

# INT4 --------------------------------------------------------------------------------


def test_int4_known_layout() -> None:
    packed = pack_int4(torch.tensor([1, 2, -1, 7]))
    assert packed.tolist() == [0x21, 0x7F]
    assert packed.dtype == torch.uint8


@pytest.mark.parametrize("signed", [True, False])
def test_int4_round_trip(signed: bool) -> None:
    lo, hi = (-8, 8) if signed else (0, 16)
    codes = torch.randint(lo, hi, (7, 64))
    packed = pack_int4(codes)
    assert packed.shape == (7, 32)
    assert torch.equal(unpack_int4(packed, signed=signed), codes.to(torch.int8))


def test_int4_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="even"):
        pack_int4(torch.zeros(3, dtype=torch.int8))
    with pytest.raises(ValueError, match=r"\[-8, 15\]"):
        pack_int4(torch.tensor([16, 0]))


# Ternary -----------------------------------------------------------------------------


def test_ternary_2bit_known_layout() -> None:
    # t + 1 = [2, 0, 1, 2] -> 0b10_01_00_10
    assert pack_ternary_2bit(torch.tensor([1, -1, 0, 1])).tolist() == [0b10010010]


def test_ternary_base3_known_layout() -> None:
    # (t + 1) = [2, 0, 1, 2, 1] -> 2 + 0*3 + 1*9 + 2*27 + 1*81 = 146
    assert pack_ternary_base3(torch.tensor([1, -1, 0, 1, 0])).tolist() == [146]


@pytest.mark.parametrize(
    ("pack", "unpack", "multiple", "bits"),
    [(pack_ternary_2bit, unpack_ternary_2bit, 4, 2.0), (pack_ternary_base3, unpack_ternary_base3, 5, 1.6)],
)
def test_ternary_round_trip_and_density(pack, unpack, multiple: int, bits: float) -> None:  # type: ignore[no-untyped-def]
    trits = torch.randint(-1, 2, (9, 20 * multiple))
    packed = pack(trits)
    assert torch.equal(unpack(packed), trits.to(torch.int8))
    assert 8 * packed.numel() / trits.numel() == pytest.approx(bits)


def test_ternary_extremes_fit_in_a_byte() -> None:
    assert pack_ternary_base3(torch.ones(5, dtype=torch.int8)).item() == 242


def test_ternary_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        pack_ternary_2bit(torch.zeros(6))
    with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
        pack_ternary_2bit(torch.tensor([2, 0, 0, 0]))
    with pytest.raises(ValueError, match="multiple of 5"):
        pack_ternary_base3(torch.zeros(4))


# 2:4 sparsity ------------------------------------------------------------------------


def test_mask_keeps_two_largest_per_group() -> None:
    w = torch.tensor([[0.1, -0.9, 0.5, 0.2, 3.0, 0.0, -4.0, 1.0]])
    assert mask_24(w).tolist() == [[False, True, True, False, True, False, True, False]]


def test_mask_tie_break_is_deterministic() -> None:
    w = torch.ones(1, 4)
    assert mask_24(w).tolist() == [[True, True, False, False]]


def test_every_group_has_exactly_two() -> None:
    m = mask_24(torch.randn(16, 64))
    assert torch.all(m.reshape(16, -1, 4).sum(-1) == 2)


def test_pack_24_round_trip_and_metadata() -> None:
    w = torch.tensor([[0.1, -0.9, 0.5, 0.2, 3.0, 0.0, -4.0, 1.0]])
    values, meta = pack_24(w)
    assert values[0].tolist() == pytest.approx([-0.9, 0.5, 3.0, -4.0])
    # group 0 keeps (1, 2) -> 1 | 2 << 2 = 0b1001; group 1 keeps (0, 2) -> 0b1000
    assert meta.tolist() == [[0b1000_1001]]
    assert torch.equal(unpack_24(values, meta, 8), prune_24(w))


def test_pack_24_random_round_trip() -> None:
    w = torch.randn(32, 128)
    values, meta = pack_24(w)
    assert values.shape == (32, 64)
    assert meta.shape == (32, 16)
    assert torch.equal(unpack_24(values, meta, 128), prune_24(w))


def test_pruning_keeps_most_energy() -> None:
    w = torch.randn(64, 256)
    kept = prune_24(w).pow(2).sum() / w.pow(2).sum()
    # for Gaussian weights the top 2 of 4 hold about 87% of the energy
    assert 0.8 < kept < 0.95


def test_sparse_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        mask_24(torch.randn(2, 6))
    with pytest.raises(ValueError, match="multiple of 8"):
        pack_24(torch.randn(2, 4))
