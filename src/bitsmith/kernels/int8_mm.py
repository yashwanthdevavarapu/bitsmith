"""Real W8A8: INT8 activations times INT8 weights on tensor cores, INT32 accumulation.

The reference runtime simulates W8A8: it rounds activations to the INT8 grid but then
multiplies dequantized floats. Here the matmul itself is integer::

    s_x[t]  = max_k |x[t, k]| / 127                      per token, computed on the fly
    xq      = clamp(round(x / s_x), -127, 127)            int8, the same codes as the reference
    acc     = xq @ wq^T                                   int8 x int8 -> int32 (torch._int_mm)
    y       = acc * s_x * s_w^T + bias                    s_w: per output channel, fp16

``torch._int_mm`` calls cuBLASLt's INT8 GEMM. It needs more than 16 rows and both inner
dimensions divisible by 8; smaller inputs (single-token decode) are padded with zero rows,
which cost compute but not correctness.

A SmoothQuant or AWQ layer first multiplies its input by a per-channel ``input_scale``.
Passed to :func:`int8_linear`, that multiply happens inside the quantization pass instead
of as a separate pass over the activations, with the same rounding (the product is rounded
to the activation dtype, as ``x * input_scale`` would be).

On CUDA with Triton, the steps around the GEMM are fused: one kernel reads each row once
and writes its INT8 codes and scale, and one kernel turns the INT32 result into the output
dtype with both scales and the bias. In plain PyTorch those steps are about a dozen
passes over the activations and the output, which cost more than the GEMM on small layers.
The kernels round half to even and reproduce the PyTorch path bit for bit (tested on GPU).
"""

from __future__ import annotations

import torch
from torch import Tensor

from bitsmith.kernels.int4_triton import TRITON_AVAILABLE, triton_available
from bitsmith.quant.uniform import QuantSpec, compute_qparams, quantize_with

_ACT_SPEC = QuantSpec(bits=8, symmetric=True, granularity="channel")
MIN_ROWS = 17  # torch._int_mm requires mat1.size(0) > 16


def int8_mm_supported(in_features: int, out_features: int) -> bool:
    """Whether a layer of this shape can run on ``torch._int_mm``."""
    return in_features % 8 == 0 and out_features % 8 == 0


def quantize_rows_int8(x: Tensor, input_scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Per-row symmetric INT8 codes ``(rows, k)`` and fp32 scales ``(rows, 1)`` of
    ``x * input_scale`` (the product in ``x.dtype``)."""
    if input_scale is not None:
        x = x * input_scale.to(x.dtype)
    groups = x.reshape(-1, x.shape[-1]).to(torch.float32).unsqueeze(1)
    scale, zero = compute_qparams(groups, _ACT_SPEC)
    codes = quantize_with(groups, scale, zero, _ACT_SPEC).to(torch.int8)
    return codes.squeeze(1), scale.reshape(-1, 1)


if TRITON_AVAILABLE:  # pragma: no cover - exercised on GPU machines only
    import triton
    import triton.language as tl

    @triton.jit
    def _load_row(x_ptr, is_ptr, offs, mask, HAS_SCALE: tl.constexpr):
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if HAS_SCALE:  # round the product to the input dtype, exactly as x * input_scale does
            s = tl.load(is_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = (x.to(tl.float32) * s).to(x_ptr.dtype.element_ty)
        return x.to(tl.float32)

    @triton.jit
    def _quant_rows(
        x_ptr, is_ptr, q_ptr, s_ptr, K, stride_x, stride_q, HAS_SCALE: tl.constexpr, BLOCK: tl.constexpr
    ):  # fmt: skip
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        amax = tl.zeros([BLOCK], dtype=tl.float32)
        for k0 in range(0, K, BLOCK):
            x = _load_row(x_ptr + row * stride_x + k0, is_ptr + k0, offs, k0 + offs < K, HAS_SCALE)
            amax = tl.maximum(amax, tl.abs(x))
        # PyTorch divides by a scalar as a multiply by its fp32 reciprocal; do the same, bit for bit
        scale = tl.maximum(tl.max(amax, axis=0) * (1.0 / 127.0), 1e-12)
        tl.store(s_ptr + row, scale)
        for k0 in range(0, K, BLOCK):
            mask = k0 + offs < K
            x = _load_row(x_ptr + row * stride_x + k0, is_ptr + k0, offs, mask, HAS_SCALE)
            v = tl.math.div_rn(x, scale)
            f = tl.floor(v)
            d = v - f
            odd = (f - 2.0 * tl.floor(f * 0.5)) == 1.0
            r = tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(odd, f + 1.0, f)))  # half to even
            r = tl.minimum(tl.maximum(r, -127.0), 127.0)
            tl.store(q_ptr + row * stride_q + k0 + offs, r.to(tl.int8), mask=mask)

    @triton.jit
    def _rescale(
        acc_ptr, xs_ptr, ws_ptr, b_ptr, y_ptr, M, N, stride_a, stride_y,
        HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):  # fmt: skip
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        acc = tl.load(acc_ptr + offs_m[:, None] * stride_a + offs_n[None, :], mask=mask, other=0).to(tl.float32)
        xs = tl.load(xs_ptr + offs_m, mask=offs_m < M, other=0.0)
        ws = tl.load(ws_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        y = acc * xs[:, None] * ws[None, :]
        if HAS_BIAS:
            y += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_y + offs_n[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


def _quantize_rows_triton(
    x2: Tensor, input_scale: Tensor | None = None
) -> tuple[Tensor, Tensor]:  # pragma: no cover - GPU only
    rows, k = x2.shape
    q = torch.empty((max(rows, MIN_ROWS), k), dtype=torch.int8, device=x2.device)
    if rows < MIN_ROWS:
        q[rows:].zero_()
    s = torch.empty((rows, 1), dtype=torch.float32, device=x2.device)
    has_scale = input_scale is not None
    in_s = input_scale.to(x2.dtype).contiguous() if input_scale is not None else s
    _quant_rows[(rows,)](x2, in_s, q, s, k, x2.stride(0), q.stride(0), has_scale, BLOCK=1024)
    return q, s


def _rescale_triton(
    acc: Tensor, x_scale: Tensor, w_scale: Tensor, bias: Tensor | None, dtype: torch.dtype
) -> Tensor:  # pragma: no cover - GPU only
    m, n = acc.shape
    y = torch.empty((m, n), dtype=dtype, device=acc.device)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    b = bias if bias is not None else w_scale
    _rescale[grid](acc, x_scale, w_scale, b, y, m, n, acc.stride(0), y.stride(0), bias is not None, 64, 64)
    return y


def int8_linear(
    x: Tensor,
    qweight: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    *,
    input_scale: Tensor | None = None,
    fused: bool | None = None,
) -> Tensor:
    """``y = (x * input_scale) W^T + b`` with per-token INT8 activations and per-channel
    INT8 ``W`` (``(out, in)``). ``input_scale`` is optional, one value per input feature.

    ``fused`` picks the Triton kernels around the GEMM (default: on CUDA when available).
    """
    if qweight.dtype != torch.int8:
        raise ValueError("qweight must be int8")
    out_features, in_features = qweight.shape
    if x.shape[-1] != in_features:
        raise ValueError(f"expected {in_features} input features, got {x.shape[-1]}")
    if not int8_mm_supported(in_features, out_features):
        raise ValueError("torch._int_mm needs in_features and out_features divisible by 8")
    if input_scale is not None and input_scale.numel() != in_features:
        raise ValueError(f"input_scale needs {in_features} values, got {input_scale.numel()}")
    lead = x.shape[:-1]
    if fused is None:
        fused = x.is_cuda and triton_available()
    if fused:  # pragma: no cover - GPU only
        x2 = x.reshape(-1, in_features).contiguous()
        rows = x2.shape[0]
        xq, x_scale = _quantize_rows_triton(x2, input_scale)
        acc = torch._int_mm(xq, qweight.t())[:rows]
        y = _rescale_triton(acc, x_scale, w_scale.reshape(-1), bias, x.dtype)
        return y.reshape(*lead, out_features)
    xq, x_scale = quantize_rows_int8(x, input_scale)
    rows = xq.shape[0]
    if rows < MIN_ROWS:
        xq = torch.nn.functional.pad(xq, (0, 0, 0, MIN_ROWS - rows))
    acc = torch._int_mm(xq, qweight.t())[:rows]  # (rows, out) int32; W^T is a column-major view
    y = acc.to(torch.float32) * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y.to(x.dtype).reshape(*lead, out_features)
