# 0004. Pure PyTorch reference runtime, no custom kernels

- Status: Accepted, amended by [0010](0010-fused-int4-triton-kernel.md)
- Date: 2026-11-09

## Context

Fast low-bit inference needs fused kernels (dequantize inside the matmul) written in CUDA,
Triton or SIMD intrinsics. Those are hard to read, hard to test on every platform, and they
hide the numerics this project exists to show.

## Decision

`QLinear.forward` unpacks and dequantizes the full weight on every call, then runs
`F.linear`. Speed claims come from the analytic roofline (`bitsmith roofline`), which
assumes a fused kernel, and are never presented as measured speedups of this runtime.

## Consequences

- Every format runs anywhere PyTorch runs, including CPU-only CI.
- Memory at rest drops by the stated ratio, but measured latency goes **up**, because each
  forward writes and re-reads a dense copy of the weight. The README reports both numbers.
- A future fused backend can plug in behind the same codec interface without changing the
  storage format.
