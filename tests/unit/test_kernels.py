"""Kernel helpers that run without a GPU."""

from __future__ import annotations


def test_int4_prefill_dispatch_rule() -> None:
    """Measured on an RTX 3090 (scripts/bench_int4_prefill.py): see use_dequant."""
    from bitsmith.kernels.int4_triton import use_dequant

    assert not use_dequant(1, 1536, 1536)  # decoding stays on the fused GEMV
    assert not use_dequant(4, 8960, 1536)
    assert use_dequant(16, 1536, 1536)  # square layer: the fused GEMM is slower at any size
    assert use_dequant(16, 8960, 1536)  # MLP down (K > N)
    assert not use_dequant(32, 1536, 8960)  # MLP up (N > K) keeps the fused GEMM below 64 rows
    assert use_dequant(64, 1536, 8960)
