# 0021. Fused kernels for INT3 and INT2, and one GEMV pass for all rows

- Status: Accepted (extends ADR 0010 and ADR 0020)
- Date: 2026-11-25

## Context

INT3 and INT2 (ADR 0014) store 3 and 2 bits per weight, so decoding, which is bound by
reading the weights, could be faster than with INT4. They only ran on the reference
runtime, which unpacks the whole matrix to 16 bits on every call and is slower than plain
fp16. Separately, the INT4 GEMV gave each activation row its own program: with 4 rows
(batch 4 decoding) it read and dequantized every weight tile four times, and ran at 0.4x
to 0.6x of fp16.

## Decision

- **One tile loader for 4, 3 and 2 bits** (`_unpack_tile`). INT4 splits each byte into two
  nibbles, INT2 into four 2-bit fields, and INT3 rebuilds each little-endian 24-bit word
  from three bytes and cuts it into eight 3-bit fields. `tl.join` puts the fields back in
  column order: joining fields half a word apart, then a quarter apart, and so on, gives the
  natural order after the reshape. Every other part of the kernels (groups, scales, zero
  points, the GEMV, the fused GEMM and the unpack + cuBLAS path of ADR 0020) is shared.
- **`bits=` on `int4_linear` and `int4_dequantize`**, and the `triton` backend accepts
  `int4`, `int4-sym`, `int3` and `int2` layers with group size 32, 64 or 128. INT3 and INT2
  are asymmetric only, as in the codec.
- **`BITS` is passed positionally**, so the autotuners key on it. Triton 3.1 only keys on
  positional arguments; with `BITS` as a keyword, a tile tuned for INT4 was reused for INT3
  on the same shape and needed more shared memory than the GPU has.
- **The GEMV serves all rows at once.** One program owns a block of output features and a
  slice of K, dequantizes that tile once and multiplies it by up to `GEMV_MAX_ROWS` (4)
  rows; the single-row case keeps the old reduction.
- **Split-K partial sums are added in a fixed order.** The GEMV used to add each K slice's
  partial sums into the output with fp32 atomics, so the order of the additions, and the
  rounding, depended on which program finished first. The decode benchmark caught it:
  on the shared GPU, 2 of about 15 CUDA graph runs of Qwen2.5-0.5B produced a different
  token from the eager run of the same steps (at token 29 of 128), while repeated runs on a
  quiet GPU agreed bit for bit. Each slice now stores to its own slot of a
  `(SPLIT_K_MAX, rows, N)` fp32 buffer, the first slice zeroes the slots its config does
  not use, and a small kernel (`_sum_slots`) adds the slots in order and casts to the
  activation dtype. A first version zeroed the buffer with `torch.zeros` and reduced with
  `sum(0).to(dtype)`, one kernel more per layer than the atomics, and decoding lost about
  10%; this one launches one kernel fewer than the atomics did (no memset).
- Tested on GPU: the Triton unpacking equals the codec bit for bit for every width, group
  size and a width that is not a multiple of the group, every path matches the reference
  runtime, and repeated GEMV calls are bitwise equal.

## Consequences

- Layers, CUDA graph replay on a shared RTX 3090, group 64
  (`results/lowbit_kernels_rtx3090_shared.txt`), speed against fp16 cuBLAS:

  | Rows | INT4 | INT3 | INT2 |
  |-----:|-----:|-----:|-----:|
  | 1 | 1.00x to 1.51x | 0.57x to 0.88x | **1.00x to 2.09x** |
  | 4 | 0.52x to 0.83x | 0.37x to 0.62x | 0.66x to 0.90x |

- **INT2 is the fastest decode format**: 1.00x to 1.38x faster than INT4 at one row and up
  to 2.09x faster than fp16 on 7B-sized layers. Its quality needs GPTQ (ADR 0014).
- Whole models from a CUDA graph (`results/decode_lowbit_rtx3090_shared.txt`): SmolLM2-135M
  decodes at 0.86x (INT4), 0.81x (INT3) and 0.91x (INT2) of bf16, Qwen2.5-0.5B at 0.93x,
  0.79x and **1.04x**. Graph and eager decoding give the same tokens in every run. The
  fixed-order reduction made INT4 decoding 8% to 10% faster than with atomics.
- **INT3 is slower than INT4** (0.43x to 0.88x of its speed) although it reads 25% fewer
  bytes. Three strided byte loads and an eight-way interleave per word make the kernel
  bound by unpacking work, not memory (65 to 92 GB/s on the five larger layers, against
  160 to 271 GB/s for INT4). A layout that splits each code into a 2-bit and a 1-bit
  plane, both aligned to bytes, would remove the straddling; it is on the roadmap.
- **4 rows got 1.1x to 2.3x faster** (INT4 1.14x to 1.55x, INT3 1.51x to 2.25x, INT2 1.03x
  to 1.17x, from two runs on the shared GPU), but all widths stay below fp16 there: the
  multiply now runs on CUDA cores for every row. Small batches need a tensor core GEMV.
