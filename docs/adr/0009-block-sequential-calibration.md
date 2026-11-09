# 0009. Block-sequential calibration and act-order GPTQ

- Status: Accepted (supersedes the default chosen in 0005)
- Date: 2026-11-10

## Context

With one-shot calibration (ADR 0005), GPTQ on SmolLM2-135M only reduced the INT4
perplexity gap from 28% to 24%, well short of what GPTQ achieves in published results.
Two things were missing:

1. Every layer was calibrated on full-precision inputs, so no layer could compensate for
   the errors of the quantized layers before it.
2. Columns were quantized in their storage order, so inputs with the most energy were often
   quantized late, when few columns were left to absorb their error.

## Decision

- `calibrate_and_quantize` groups layers by transformer block (the module path up to the
  first numeric index, for example `model.layers.3`) and processes blocks in order. Each
  block's statistics are collected with every earlier block already quantized. A forward
  hook on the block raises an internal exception to end each calibration pass as soon as
  that block has run, so block `i` costs about `i / n` of a full forward pass.
- GPTQ defaults to `act_order=True`: columns are processed by decreasing `diag(H)`. Group
  parameters are computed up front from the original weights ("static groups"), so the
  storage format stays contiguous groups and the INT4 kernels need no index map.
- `sequential=False` and `act_order=False` remain available.

## Consequences

On SmolLM2-135M (bf16, GPU) GPTQ now cuts the INT4 perplexity gap from 25.8% to about 11%.
Calibration costs one partial forward pass per block instead of one pass in total, and
only one block of Hessians is in memory at a time, which keeps GPTQ on a 1.5B model well
within a 24 GB GPU.
