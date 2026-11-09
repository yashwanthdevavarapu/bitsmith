# 0002. Quantization conventions

- Status: Accepted
- Date: 2026-11-09

## Context

Integer quantizers differ in small ways across libraries: whether the symmetric range
includes the most negative code, whether the affine range must contain zero, what dtype
scales are stored in, and whether codes are computed before or after that cast.

## Decision

1. **Symmetric codes use a restricted range** `[-(2^(b-1) - 1), 2^(b-1) - 1]`. INT4 symmetric
   is `[-7, 7]`. The grid is symmetric, so `q(-x) = -q(x)` and there is no bias toward
   negative values. Cost: one of the 16 INT4 levels is unused.
2. **Affine ranges are widened to contain 0.** `x_min = min(min(x), 0)` and
   `x_max = max(max(x), 0)`. Zero is then exactly representable, which makes zero padding
   of partial groups harmless and keeps sparse or ReLU-like tensors exact at zero.
3. **Scales are stored in fp16 and rounded before computing codes.** Encoding with an fp32
   scale and decoding with its fp16 copy would add an error that is easy to miss.
   Exception: FP8 per-channel scales are fp32, because an E5M2 scale (`absmax / 57344`)
   falls below the fp16 normal range for typical weights.
4. **Rounding is round half to even** (`torch.round`). FP8 encoding also uses ties to even
   and is tested bit for bit against `torch.float8_e4m3fn` and `torch.float8_e5m2`.
5. **Partial groups are zero padded** internally and sliced away on decode, so any input
   width works with any group size.

## Consequences

`bits_per_weight` counts exactly what is stored (codes, scales, zero points, metadata).
Results are identical for fp32, fp16 and bf16 inputs holding the same values, because all
quantizer math runs in fp32.
