# 0012. GPTQ on non-uniform grids through a TableCodec interface

- Status: Accepted
- Date: 2026-11-12

## Context

ADR 0011 added NF4, MXFP4 and MXINT8. AWQ worked with them unchanged because it only
calls `encode` and `decode`. GPTQ did not: its loop computes integer codes with the
uniform formula `round(w / s) + z`, and its result is a `QTensor`, which describes uniform
grids only. The GPTQ update itself does not care about the grid: it needs the nearest
representable value for each column, given the scales.

## Decision

- **A `TableCodec` protocol** in `nn/codecs.py`, implemented by `NF4Codec` and `MXCodec`:
  `block_size`, `scale_buffer(weight)` (stored scales), `scale_values(scale)` (float32
  multipliers), `snap(x)` (nearest grid value) and `encode_with_scale(weight, scale)`.
  `encode(weight)` is now `encode_with_scale(weight, scale_buffer(weight))`, so plain
  round to nearest and GPTQ share a single encoding path.
- **A separate `gptq_quantize_table` loop** instead of generalizing `gptq_quantize`. The
  INT4 loop has a dynamic-group mode and produces `QTensor` codes. Keeping it untouched
  guarantees the published INT4 results cannot change. The new loop is about 30 lines
  and mirrors the old one.
- **Scales are always static**, computed from the original weights before any error
  feedback, as with act-order INT4 (ADR 0009). For MX this is the only choice that keeps
  the E8M0 exponent a pure function of the block, and it lets act-order visit columns in
  any order.
- **The result is codec buffers, not a `QTensor`.** `quantize_model(encoded=...)` and
  `QLinear.from_buffers` wrap them without re-quantizing. A test checks that decoding the
  buffers reproduces the GPTQ output bit for bit.
- Lookup tables are cached per device, because GPTQ calls `snap` once per column.

## Consequences

- `calibrate_and_quantize(method="gptq")` accepts `int4`, `int4-sym`, `nf4`, `mxfp4` and
  `mxint8`.
- On correlated synthetic inputs GPTQ lowers the expected output error about 4x for every
  grid format, in line with INT4 (3.8x).
- Two GPTQ loops must be kept in sync if the algorithm changes (for example a vectorized
  inner loop, roadmap v0.3). Both are covered by the same style of tests: equal to round
  to nearest when `H` is the identity, better than it on correlated inputs, and stable
  with dead inputs.
