# 0003. Ternary storage: 2-bit default, base-3 optional

- Status: Accepted
- Date: 2026-11-09

## Context

A ternary weight carries log2(3) = 1.585 bits of information. Two practical encodings exist:

| Encoding | Trits per byte | Bits per weight | Decode |
|----------|---------------:|----------------:|--------|
| 2-bit fields | 4 | 2.0 | shift and mask |
| base 3 (3^5 = 243 <= 256) | 5 | 1.6 | repeated divide by 3 |

## Decision

`ternary` uses 2-bit fields (value + 1 in each field). `ternary-b3` offers the base-3 form
for anyone studying the density limit.

## Consequences

The 2-bit form maps directly to SIMD lanes and lookup tables, which is what inference
kernels want. The base-3 form saves 20% of storage but needs integer division per trit.
Both have round-trip tests and known-layout tests.
