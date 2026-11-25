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

Three paths, chosen by the number of activation rows ``M``:

* ``_gemv``  for a few rows (token by token decoding, up to ``GEMV_MAX_ROWS``). Each program
  owns a block of output features and a slice of K (split-K), dequantizes that weight tile
  once for all rows, and reduces in registers. Large K tiles keep many bytes in flight per
  program, which is what a bandwidth bound kernel needs. Each K slice stores its partial
  sums in its own slot of an fp32 buffer (the first slice also zeroes the unused slots),
  and ``_sum_slots`` adds the slots in a fixed order and casts to the activation dtype.
  With fp32 atomics the order of the additions depended on which program finished first,
  and on a busy GPU the same input could round differently from one call to the next
  (ADR 0021).
* ``_gemm``  for medium batches, with ``tl.dot`` on tensor cores.
* ``_dequant`` + cuBLAS for many rows (prefill, ADR 0020). The GEMM is compute bound there,
  and every block of rows in ``_gemm`` dequantizes the same weights again. Unpacking the
  weight once to the activation dtype (about 2.5 bytes moved per weight) and calling cuBLAS
  costs a few percent over a 16-bit GEMM and reaches cuBLAS speed. The weights are the
  same values the fused kernels compute, so only the summation order changes.
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
    def _unpack_tile(
        w_ptr, offs_n, mask_n, k0, K, stride_wn, BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """Unsigned codes of a (BLOCK_N, BLOCK_K) tile, in column order, as int32.

        The packed bytes are read once per tile; each byte (INT4, INT2) or 3-byte word (INT3)
        yields several codes, which ``tl.join`` interleaves back to column order. Joining
        codes that are half a word apart, then a quarter, and so on, gives the natural order
        after the reshape.
        """
        row = w_ptr + offs_n[:, None] * stride_wn
        if BITS == 4:
            cols = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            b = tl.load(row + cols[None, :], mask=mask_n[:, None] & (cols < K // 2)[None, :], other=0).to(tl.int32)
            q = tl.join(b & 0xF, (b >> 4) & 0xF)
        elif BITS == 2:
            cols = k0 // 4 + tl.arange(0, BLOCK_K // 4)
            b = tl.load(row + cols[None, :], mask=mask_n[:, None] & (cols < K // 4)[None, :], other=0).to(tl.int32)
            q = tl.join(tl.join(b & 3, (b >> 4) & 3), tl.join((b >> 2) & 3, (b >> 6) & 3))
        else:  # BITS == 3: eight codes in a little-endian 24-bit word of three bytes
            words = k0 // 8 + tl.arange(0, BLOCK_K // 8)
            mask = mask_n[:, None] & (words < K // 8)[None, :]
            b0 = tl.load(row + 3 * words[None, :], mask=mask, other=0).to(tl.int32)
            b1 = tl.load(row + 3 * words[None, :] + 1, mask=mask, other=0).to(tl.int32)
            b2 = tl.load(row + 3 * words[None, :] + 2, mask=mask, other=0).to(tl.int32)
            w = b0 | (b1 << 8) | (b2 << 16)
            even = tl.join(tl.join(w & 7, (w >> 12) & 7), tl.join((w >> 6) & 7, (w >> 18) & 7))
            odd = tl.join(tl.join((w >> 3) & 7, (w >> 15) & 7), tl.join((w >> 9) & 7, (w >> 21) & 7))
            q = tl.join(even, odd)
        return tl.reshape(q, (BLOCK_N, BLOCK_K))

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
        BITS: tl.constexpr,
    ):
        """Load a (BLOCK_N, BLOCK_K) tile of packed weights and return it dequantized in fp32."""
        q = _unpack_tile(w_ptr, offs_n, mask_n, k0, K, stride_wn, BITS, BLOCK_N, BLOCK_K).to(tl.float32)
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
        key=["M", "N", "K", "GROUP", "BITS"],
    )
    @triton.jit
    def _gemv(
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
        stride_yk,
        stride_ym,
        BITS: tl.constexpr,  # positional, so the autotuner can key on it
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        SYMMETRIC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SLOTS: tl.constexpr,
    ):
        # One program per (feature block, K slice) serves every row: the weight tile is read
        # and dequantized once, then multiplied by up to BLOCK_M activation rows.
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
            w = _dequant_tile(
                w_ptr,
                s_ptr,
                z_ptr,
                offs_n,
                mask_n,
                k0,
                K,
                stride_wn,
                stride_sn,
                GROUP,
                BLOCK_N,
                BLOCK_K,
                SYMMETRIC,
                BITS,
            )
            kk = k0 + offs_k
            xv = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                mask=mask_m[:, None] & (kk < K_REAL)[None, :],
                other=0.0,
            ).to(tl.float32)
            if BLOCK_M == 1:
                acc += tl.sum(w * xv, axis=1)[None, :]
            else:
                acc += tl.sum(w[None, :, :] * xv[:, None, :], axis=2)
        # slot pid_k holds this slice's partial sums; slots SPLIT_K..SLOTS-1 are zeroed by the
        # first slice, so the buffer needs no memset and _sum_slots can add all of them
        y_ptrs = y_ptr + pid_k * stride_yk + offs_m[:, None] * stride_ym + offs_n[None, :]
        mask_y = mask_m[:, None] & mask_n[None, :]
        tl.store(y_ptrs, acc, mask=mask_y)
        if pid_k == 0:
            for s in tl.static_range(SPLIT_K, SLOTS):
                tl.store(y_ptrs + s * stride_yk, tl.zeros_like(acc), mask=mask_y)

    @triton.jit
    def _sum_slots(p_ptr, y_ptr, numel, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
        # y = ((p[0] + p[1]) + p[2]) + ..., always in this order, cast to y's dtype
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
        for s in tl.static_range(1, SLOTS):
            acc += tl.load(p_ptr + s * numel + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)

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
        key=["M", "N", "K", "GROUP", "BITS"],
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
        BITS: tl.constexpr,  # positional, so the autotuner can key on it
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
                w_ptr,
                s_ptr,
                z_ptr,
                offs_n,
                mask_n,
                k0,
                K,
                stride_wn,
                stride_sn,
                GROUP,
                BLOCK_N,
                BLOCK_K,
                SYMMETRIC,
                BITS,
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w)
            for bn, bk, w in (
                (16, 128, 4),
                (16, 256, 4),
                (32, 128, 4),
                (32, 256, 4),
                (64, 128, 4),
                (64, 256, 8),
                (128, 128, 8),
                (8, 512, 4),
            )
        ],
        key=["N", "K", "GROUP", "BITS"],
    )
    @triton.jit
    def _dequant(
        w_ptr,
        s_ptr,
        z_ptr,
        out_ptr,
        N,
        K,
        K_REAL,
        stride_wn,
        stride_sn,
        stride_on,
        BITS: tl.constexpr,  # positional, so the autotuner can key on it
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SYMMETRIC: tl.constexpr,
    ):
        offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        k0 = tl.program_id(1) * BLOCK_K
        mask_n = offs_n < N
        w = _dequant_tile(
            w_ptr, s_ptr, z_ptr, offs_n, mask_n, k0, K, stride_wn, stride_sn, GROUP, BLOCK_N, BLOCK_K, SYMMETRIC, BITS
        )
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = mask_n[:, None] & (kk < K_REAL)[None, :]
        tl.store(out_ptr + offs_n[:, None] * stride_on + kk[None, :], w.to(out_ptr.dtype.element_ty), mask=mask)


GEMV_MAX_ROWS = 4
SPLIT_K_MAX = 4  # the largest SPLIT_K among the GEMV configs
# From scripts/bench_int4_prefill.py on an RTX 3090: dequantize + cuBLAS beats the fused GEMM
# on every measured shape from 64 rows on, and at any row count when the layer is not wider
# than it is long (K >= N), where the fused GEMM's short K tiles leave it 1.2x to 2.9x slower.
DEQUANT_MIN_ROWS = 64


def use_dequant(rows: int, in_features: int, out_features: int) -> bool:
    """Whether ``int4_linear(path="auto")`` unpacks the weight and calls cuBLAS."""
    return rows > GEMV_MAX_ROWS and (rows >= DEQUANT_MIN_ROWS or in_features >= out_features)


def _check_bits(bits: int, symmetric: bool) -> None:
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 4, 3 or 2, not {bits}")
    if symmetric and bits != 4:
        raise ValueError("only INT4 weights can be symmetric (no zero point)")


def int4_dequantize(
    qweight: Tensor,
    scale: Tensor,
    zero: Tensor | None,
    in_features: int,
    group_size: int,
    dtype: torch.dtype = torch.float16,
    *,
    bits: int = 4,
) -> Tensor:
    """Unpack packed INT4, INT3 or INT2 weights to a dense ``(N, in_features)`` matrix in one kernel."""
    if not triton_available():
        raise RuntimeError("int4_dequantize needs Triton and a CUDA device")
    _check_bits(bits, zero is None)
    n, k = qweight.shape[0], qweight.shape[1] * 8 // bits
    out = torch.empty(n, in_features, device=qweight.device, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), triton.cdiv(k, meta["BLOCK_K"]))  # noqa: E731
    _dequant[grid](
        qweight,
        scale,
        zero if zero is not None else scale,
        out,
        n,
        k,
        in_features,
        qweight.stride(0),
        scale.stride(0),
        out.stride(0),
        bits,
        GROUP=group_size,
        SYMMETRIC=zero is None,
    )
    return out


def int4_linear(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    zero: Tensor | None,
    in_features: int,
    group_size: int,
    bias: Tensor | None = None,
    *,
    path: str = "auto",
    bits: int = 4,
) -> Tensor:
    """``x @ W^T + bias`` with ``W`` stored as packed INT4 (or INT3, INT2 with ``bits``).

    ``path="auto"``: up to ``GEMV_MAX_ROWS`` rows use the fused GEMV; above that, the
    shapes picked by :func:`use_dequant` unpack the weight once and call cuBLAS, and the
    rest use the fused tensor core GEMM. ``"fused"`` and ``"dequant"`` force one of the two
    (for benchmarks and tests).
    """
    if not triton_available():
        raise RuntimeError("int4_linear needs Triton and a CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("int4_linear expects fp16 or bf16 activations")
    if group_size < 32 or group_size > 128 or group_size & (group_size - 1):
        raise ValueError("the triton kernel needs a power of two group_size between 32 and 128")
    _check_bits(bits, zero is None)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, in_features).contiguous()
    m, n = x2.shape[0], qweight.shape[0]
    k = qweight.shape[1] * 8 // bits  # padded width, a multiple of group_size
    symmetric = zero is None
    z = zero if zero is not None else scale  # dummy pointer, never read when symmetric
    if path not in ("auto", "fused", "dequant"):
        raise ValueError(f"path must be auto, fused or dequant, not {path!r}")
    if path == "dequant" or (path == "auto" and use_dequant(m, in_features, n)):
        w = int4_dequantize(qweight, scale, zero, in_features, group_size, x.dtype, bits=bits)
        return torch.nn.functional.linear(x, w, bias.to(x.dtype) if bias is not None else None)
    if m <= GEMV_MAX_ROWS and path == "auto":
        # one slot per K slice, written (or zeroed) by the kernel itself
        partial = torch.empty(SPLIT_K_MAX, m, n, device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])  # noqa: E731
        _gemv[grid](
            x2,
            qweight,
            scale,
            z,
            partial,
            m,
            n,
            k,
            in_features,
            x2.stride(0),
            qweight.stride(0),
            scale.stride(0),
            partial.stride(0),
            partial.stride(1),
            bits,
            GROUP=group_size,
            SYMMETRIC=symmetric,
            BLOCK_M=triton.next_power_of_2(m),
            SLOTS=SPLIT_K_MAX,
        )
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)
        _sum_slots[(triton.cdiv(m * n, 1024),)](partial, y, m * n, SLOTS=SPLIT_K_MAX, BLOCK=1024)
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
            bits,
            GROUP=group_size,
            BLOCK_K=group_size,
            SYMMETRIC=symmetric,
        )
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*lead, n)
