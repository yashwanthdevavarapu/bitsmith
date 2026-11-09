# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-11-09

## Context

Quantization code is full of small choices (rounding mode, code range, where scales live,
which statistics calibration uses) that change results by a few tenths of a perplexity
point. Without a written reason, those choices get "fixed" back and forth.

## Decision

Record every decision that affects numerics, storage format, or the public API as a short
ADR in `docs/adr/`, numbered in order.

## Consequences

Reviewers can point at an ADR instead of re-arguing a choice. Changing a recorded decision
requires a new ADR that supersedes the old one.
