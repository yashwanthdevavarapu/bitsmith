# 0010. Fused INT4 kernel in Triton as an optional backend

- Status: Accepted (amends 0004)
- Date: 2026-11-10

## Context

ADR 0004 kept the runtime readable by dequantizing the whole weight before every matmul.
That made every quantized model slower than its float version and left the roofline as the
only evidence that low-bit weights can be faster.

## Decision

Add `bitsmith.kernels.int4_linear`, written in Triton, and expose it as
`QLinear.set_backend("triton")` or `set_backend(model, "triton")`:

- **GEMV kernel** for up to 4 activation rows (decoding): each program owns a block of
  output features and a slice of K (split-K), dequantizes packed nibbles in registers,
  reduces, and atomically adds into an fp32 output. Nibbles are re-interleaved with
  `tl.join` so activations are read contiguously.
- **GEMM kernel** for larger batches: tiles dequantized to fp16 or bf16 and multiplied with
  `tl.dot` on tensor cores.
- The storage format is unchanged; the reference path stays the default and the source of
  truth. GPU tests check the kernel against it for fp16 and bf16, group sizes 32, 64 and
  128, symmetric and asymmetric codes, padded widths, and 1 to 300 rows.
- Benchmarks time kernels with CUDA graph replay, which removes Python launch overhead and
  measures GPU time only.

Two alternative GEMV formulations were measured and rejected because they were slower on an
RTX 3090: per-group scaling of integer dot products, and splitting activations with
`tl.split` instead of interleaving weights.

## Consequences

At batch size 1 on an RTX 3090 the kernel runs at 0.94x to 1.33x the speed of fp16 cuBLAS
(faster on 4 of 6 layer shapes) and 5x to 24x faster than the reference path, while reading 3.8x fewer weight bytes. It
reaches about 25% of peak DRAM bandwidth, so hand-written CUDA kernels such as Marlin are
still several times faster. For prefill (512 rows) it is about 2x slower than cuBLAS.
End to end decoding of small models remains dominated by per-layer launch overhead unless
the model is captured in a CUDA graph.
