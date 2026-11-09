# 0007. Explicit input scale instead of folding into norms

- Status: Accepted
- Date: 2026-11-09

## Context

SmoothQuant and AWQ rescale input channels: `y = (x / s) (W * s)^T`. Production toolkits
fold `1 / s` into the preceding LayerNorm or RMSNorm weight. That requires knowing the
model graph: which norm feeds which projections, and that all consumers share one `s`.

## Decision

`QLinear` stores an optional fp16 `input_scale` buffer and multiplies activations by it.
The buffer is counted in `weight_bytes`.

## Consequences

Works for any model built from `nn.Linear`, with no architecture specific code. Costs one
element-wise multiply per forward and 16 bits per input channel. Folding can be added later
as an optimization for known architectures without changing the math.
