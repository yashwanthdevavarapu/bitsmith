# 0005. One-shot calibration from full precision activations

- Status: Superseded by [0009](0009-block-sequential-calibration.md) (one-shot remains available with `sequential=False`)
- Date: 2026-11-09

## Context

Calibrated methods need input statistics for each linear layer. Two options:

- **Sequential**: quantize block 1, run calibration data through the quantized block,
  collect statistics for block 2, and so on. Later layers see the errors of earlier ones.
- **One-shot**: run the full precision model once and record every layer's inputs.

## Decision

Use one-shot capture (`collect_activation_stats`) with forward pre-hooks. GPTQ runs
without activation reordering (`act_order`), and `block_size` is rounded to a multiple of
`group_size` so a group never straddles two lazy-update blocks. Group parameters are
computed from the already corrected weights when a group starts.

## Consequences

One forward pass covers all layers and the code stays short. Sequential calibration usually
helps a little more on deep models and can be added as an option later. GPTQ memory grows
with `in_features^2` per layer for the Hessians (about 450 MB for a 135M parameter model).
