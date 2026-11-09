"""Fused INT4 dequantize + matmul kernels in Triton (CUDA only).

The reference runtime unpacks the whole weight to a dense matrix and then calls a normal
matmul, which reads the packed bytes, writes a full precision copy and reads it again.
These kernels read the packed bytes once and dequantize in registers, which is what makes
low-bit weights faster in bandwidth bound decoding (ADR 0010).

Storage layout (same as ``Int4Codec``)::

    qweight  uint8   (N, K / 2)   element 2k in the low nibble, 2k + 1 in the high nibble
    scale    fp16    (N, K / G)
    zero     uint8   (N, K / G)   absent for symmetric weights

Inside a tile the two nibbles are split and re-interleaved with ``tl.join`` so the
activations can be read contiguously.

Two kernels:

* ``_gemv``  for a few rows of activations (token by token decoding). Each program owns a
  block of output features and a slice of K (split-K), reduces in registers, and adds its
  partial sum to an fp32 output with an atomic add. Large K tiles keep many bytes in
  flight per program, which is what a bandwidth bound kernel needs.
* ``_gemm``  for larger batches (prefill), with ``tl.dot`` on tensor cores.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:  # pragma: no cover - depends on the platform
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


def triton_available() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available()


if TRITON_AVAILABLE:  # pragma: no cover - exercised on GPU machines only

    @triton.jit
    def _dequant_tile(
        w_ptr,
        s_ptr,
        z_ptr,
        offs_n,
        mask_n,
        k0,
        K,
        stride_wn,
        stride_sn,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SYMMETRIC: tl.constexpr,
    ):
        """Load a (BLOCK_N, BLOCK_K) tile of packed weights and return it dequantized in fp32."""
        offs_h = tl.arange(0, BLOCK_K // 2)
        cols_h = k0 // 2 + offs_h
        packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + cols_h[None, :],
            mask=mask_n[:, None] & (cols_h < K // 2)[None, :],
            other=0,
        ).to(tl.int32)
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        q = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K)).to(tl.float32)  # interleave columns
        if SYMMETRIC:
            q = tl.where(q >= 8, q - 16, q)
        G_TILE: tl.constexpr = BLOCK_K // GROUP
        offs_g = k0 // GROUP + tl.arange(0, G_TILE)
        mask_g = mask_n[:, None] & (offs_g < K // GROUP)[None, :]
        s = tl.load(s_ptr + offs_n[:, None] * stride_sn + offs_g[None, :], mask=mask_g, other=0.0)
        q3 = tl.reshape(q, (BLOCK_N, G_TILE, GROUP))
        if not SYMMETRIC:
            z = tl.load(z_ptr + offs_n[:, None] * stride_sn + offs_g[None, :], mask=mask_g, other=0)
            q3 = q3 - z.to(tl.float32)[:, :, None]
        w = q3 * s.to(tl.float32)[:, :, None]
        return tl.reshape(w, (BLOCK_N, BLOCK_K))

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=w)
            for bn, bk, sk, w in (
                (4, 1024, 1, 4),
                (4, 1024, 4, 4),
                (8, 512, 4, 4),
                (8, 1024, 2, 4),
                (8, 1024, 4, 8),
                (16, 256, 4, 4),
                (16, 512, 2, 4),
                (16, 512, 4, 8),
                (16, 1024, 2, 8),
                (32, 256, 2, 4),
                (32, 512, 1, 8),
                (32, 512, 4, 8),
                (64, 256, 1, 8),
                (64, 256, 4, 8),
            )
        ],
        key=["N", "K", "GROUP"],
        reset_to_zero=["y_ptr"],
    )
    @triton.jit
    def _gemv(
        x_ptr,
        w_ptr,
        s_ptr,
        z_ptr,
        y_ptr,
        N,
        K,
        K_REAL,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_ym,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        SYMMETRIC: tl.constexpr,
    ):
        m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
            w = _dequant_tile(
                w_ptr, s_ptr, z_ptr, offs_n, mask_n, k0, K, stride_wn, stride_sn, GROUP, BLOCK_N, BLOCK_K, SYMMETRIC
            )
            kk = k0 + offs_k
            xv = tl.load(x_ptr + m * stride_xm + kk, mask=kk < K_REAL, other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.atomic_add(y_ptr + m * stride_ym + offs_n, acc, mask=mask_n)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=st)
            for bm, bn, w, st in (
                (16, 64, 4, 3),
                (32, 64, 4, 3),
                (64, 64, 4, 3),
                (64, 128, 4, 3),
                (128, 64, 4, 3),
                (128, 128, 8, 3),
                (128, 256, 8, 3),
            )
        ],
        key=["M", "N", "K", "GROUP"],
    )
    @triton.jit
    def _gemm(
        x_ptr,
        w_ptr,
        s_ptr,
        z_ptr,
        y_ptr,
        M,
        N,
        K,
        K_REAL,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_ym,
        GROUP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SYMMETRIC: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            w = _dequant_tile(
                w_ptr, s_ptr, z_ptr, offs_n, mask_n, k0, K, stride_wn, stride_sn, GROUP, BLOCK_N, BLOCK_K, SYMMETRIC
            )
            kk = k0 + offs_k
            xt = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                mask=mask_m[:, None] & (kk < K_REAL)[None, :],
                other=0.0,
            )
            acc += tl.dot(xt, tl.trans(w.to(xt.dtype)))
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


GEMV_MAX_ROWS = 4


def int4_linear(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    zero: Tensor | None,
    in_features: int,
    group_size: int,
    bias: Tensor | None = None,
) -> Tensor:
    """``x @ W^T + bias`` with ``W`` stored as packed INT4, computed by a fused kernel."""
    if not triton_available():
        raise RuntimeError("int4_linear needs Triton and a CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("int4_linear expects fp16 or bf16 activations")
    if group_size < 32 or group_size > 128 or group_size & (group_size - 1):
        raise ValueError("the triton kernel needs a power of two group_size between 32 and 128")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, in_features).contiguous()
    m, n = x2.shape[0], qweight.shape[0]
    k = qweight.shape[1] * 2  # padded width, a multiple of group_size
    symmetric = zero is None
    z = zero if zero is not None else scale  # dummy pointer, never read when symmetric
    if m <= GEMV_MAX_ROWS:
        y32 = torch.zeros(m, n, device=x.device, dtype=torch.float32)
        grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])  # noqa: E731
        _gemv[grid](
            x2,
            qweight,
            scale,
            z,
            y32,
            n,
            k,
            in_features,
            x2.stride(0),
            qweight.stride(0),
            scale.stride(0),
            y32.stride(0),
            GROUP=group_size,
            SYMMETRIC=symmetric,
        )
        y = y32.to(x.dtype)
    else:
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)
        grid_mm = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))  # noqa: E731
        _gemm[grid_mm](
            x2,
            qweight,
            scale,
            z,
            y,
            m,
            n,
            k,
            in_features,
            x2.stride(0),
            qweight.stride(0),
            scale.stride(0),
            y.stride(0),
            GROUP=group_size,
            BLOCK_K=group_size,
            SYMMETRIC=symmetric,
        )
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*lead, n)
