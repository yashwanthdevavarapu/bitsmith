# 0008. Minimal runtime dependencies

- Status: Accepted
- Date: 2026-11-09

## Context

A library meant for learning and experiments should install fast and hold no surprises.

## Decision

Runtime dependencies are `torch` and `numpy` only. The CLI uses `argparse` and renders
tables and plots as plain text. `transformers` and `datasets` live in the `hf` extra,
`matplotlib` in the `plot` extra, and test and lint tools in the `dev` extra.

Triton is used only when it is already installed (it ships with CUDA builds of PyTorch on
Linux); it is never a declared dependency.

## Consequences

Installing the core package pulls nothing beyond PyTorch. Real model evaluation without the
`hf` extra fails with a clear message that names the extra to install.
