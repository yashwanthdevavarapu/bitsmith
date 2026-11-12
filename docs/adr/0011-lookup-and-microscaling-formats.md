# 0011. Lookup-table and microscaling formats: NF4, MXFP4, MXINT8

- Status: Accepted
- Date: 2026-11-11

## Context

Until v0.2 every 4-bit format was a uniform integer grid with a per-group fp16 scale.
Two other families are widely shipped and worth comparing on the same footing: NF4, which
uses non-uniform levels from a fixed table, and the OCP Microscaling (MX) formats, which
share a power-of-two scale across 32 elements so hardware can apply it with an exponent
add. Several choices are left to implementers in both cases.

## Decision

- **NF4 table is computed, not pasted.** It is built from the normal quantile function
  with the QLoRA offset 0.9677083 and tested against the published float32 values to
  1e-6. Ties between two levels go to the lower level.
- **NF4 block scale is fp16**, rounded before encoding, as for every other codec here
  (ADR 0002). QLoRA keeps fp32 absmax values, optionally quantized again ("double
  quantization"). fp16 keeps NF4 at the same storage as `int4-sym`, which makes the
  comparison direct. Block size follows `group_size`.
- **MX follows OCP MX v1.0 literally.** Block size is fixed at 32 and ignores
  `group_size`. The shared exponent is `floor(log2(absmax)) - emax_elem`, computed exactly
  with `frexp`, clamped to `[-127, 127]`; all-zero blocks get exponent -127. Elements
  saturate, so MXFP4 clips values between 6 and 8 times the scale, as the specification
  prescribes. We do not use the "round the scale up" variants that avoid clipping, because
  results must be comparable with hardware that follows the specification.
- **MXINT8 elements stay in `[-127, 127]`.** -128 is never produced, which keeps the grid
  symmetric.
- **Storage reuses existing layouts.** NF4 and FP4 codes are unsigned nibbles packed with
  `pack_int4`. E8M0 scales are stored as raw `uint8`.
- **No fused kernel yet.** Both formats run on the reference dequantize-then-matmul path.
  The Triton backend still accepts INT4 only.

## Consequences

- Every new scheme passes the shared codec tests (round trip, awkward widths, state dict,
  bits per weight) and is in `bitsmith sqnr`, the sweeps and the GPU test suite.
- Weight SQNR on Gaussian weights at G = 128: NF4 20.4 dB, MXFP4 18.9 dB, int4-sym
  18.6 dB, int4 19.9 dB.
- AWQ works with every codec as is. GPTQ originally produced uniform INT4 only; ADR 0012
  extends it to these formats.
- MX decode on real hardware is cheaper than our reference path suggests; speed numbers
  for these formats are not meaningful until a kernel exists.
