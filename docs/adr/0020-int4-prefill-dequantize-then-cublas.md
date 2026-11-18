# 0020. INT4 with many rows: unpack the weight once, then cuBLAS

- Status: Accepted (amends ADR 0010)
- Date: 2026-11-18

## Context

The fused INT4 kernels of ADR 0010 dequantize weight tiles inside the matmul. That is the
right design for decoding, where a GEMV with one row is bound by how fast the weights can
be read. With many rows (prefill, batched requests) the matmul is bound by arithmetic
instead, and the fused `_gemm` falls behind: its K tile is one quantization group (32 to
128), so it gets little reuse from each tile, and every block of rows dequantizes the
same weights again. The prefill benchmark of ADR 0019 measured whole models with INT4
weights at 0.37x to 0.70x of bf16 speed.

## Decision

- **A third path in `int4_linear`**: for many rows, one Triton kernel (`int4_dequantize`,
  autotuned tile) unpacks the packed weight to the activation dtype, and `F.linear`
  multiplies with cuBLAS. It moves about 2.5 bytes per weight (read 0.5, write 2) before a
  GEMM that is compute bound, so its cost shrinks as the row count grows.
- **The same weights as the fused kernels.** The unpacking reuses `_dequant_tile`, so the
  values are bit for bit those the fused kernels compute; only the summation order of the
  matmul changes (tested against the codec and against the fused GEMM on GPU).
- **Dispatch from measurements** (`scripts/bench_int4_prefill.py`, six layer shapes from
  Qwen2.5-1.5B and a 7B model, 16 to 4096 rows): unpacking wins on every shape from 64 rows
  on, and at every row count when a layer is not wider than it is long (`K >= N`). Below
  64 rows the fused GEMM still wins on the wide MLP up projections (1.01x to 1.39x). The
  rule is `use_dequant(rows, K, N)`; the fused GEMV keeps decoding (4 rows or fewer).
- **`path="auto" | "fused" | "dequant"`** on `int4_linear`, so benchmarks and tests can
  force either path.

## Consequences

- Layers, CUDA graph replay on a shared RTX 3090 (`results/int4_prefill_rtx3090_shared.txt`),
  speed against fp16 cuBLAS:

  | Rows | Fused GEMM | Unpack + cuBLAS |
  |-----:|-----------:|----------------:|
  | 64 | 0.18x to 0.59x | 0.48x to 0.81x |
  | 512 | 0.29x to 0.50x | 0.82x to 0.88x |
  | 2048 | 0.34x to 0.48x | 0.87x to 0.95x |
  | 4096 | 0.43x to 0.51x | 0.93x to 1.09x |

  From 256 rows on, INT4 layers are 1.6x to 3.1x faster than before, and at 2048 rows
  they are within 5% to 13% of fp16.
- The unpacked weight is a temporary of the full 16-bit size, one layer at a time. Peak
  memory grows by the largest layer (27 MiB for Qwen2.5-1.5B's MLP, 86 MiB for a 7B MLP),
  not by the model.
- INT4 cannot beat fp16 this way: it does the fp16 GEMM plus the unpacking. Its prefill
  advantage is memory, while decoding keeps the fused GEMV's bandwidth gain. Beating fp16
  on prefill needs INT8 or FP8 tensor cores (ADR 0017, ADR 0019).
- A fused GEMM with larger K tiles and weight reuse across row blocks could still win in
  the 16 to 64 row range; that stays on the roadmap.
