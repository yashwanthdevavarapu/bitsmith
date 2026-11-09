# 0006. Synthetic corpus and tiny model for CI

- Status: Accepted
- Date: 2026-11-09

## Context

End to end tests need a trained language model and an evaluation set. Downloading models
and datasets in CI is slow and flaky, and an untrained model gives meaningless perplexities.

## Decision

Generate text from a first order Markov chain with Dirichlet transition rows
(`MarkovCorpus`). Its entropy rate is computable, so the best achievable perplexity is known
exactly. Train a small transformer built from `nn.Linear` layers (`TinyLM`) on it for a few
hundred steps on CPU. Real model evaluation (SmolLM2 on WikiText-2) is opt in.

## Consequences

The whole pipeline (train, calibrate, quantize, evaluate, save, reload) runs in seconds in
CI and has a hard lower bound to check against. A tiny model is more robust to quantization
than a real LLM (it has no large activation outliers), so headline numbers come from the
real model run, not the synthetic one.
