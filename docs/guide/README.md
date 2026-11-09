# Technical guide

A walk through the math and storage formats behind every module, in the order data flows
through the library. Each section names the file that implements it.

1. [Uniform quantization](#1-uniform-quantization)
2. [FP8 emulation](#2-fp8-emulation)
3. [Bit packing](#3-bit-packing)
4. [2:4 structured sparsity](#4-24-structured-sparsity)
5. [QLinear and model conversion](#5-qlinear-and-model-conversion)
6. [Calibration: SmoothQuant, AWQ, GPTQ](#6-calibration)
7. [Evaluation](#7-evaluation)
8. [Roofline](#8-roofline)
9. [Fused INT4 kernel](#9-fused-int4-kernel)

---

## 1. Uniform quantization

`src/bitsmith/quant/uniform.py`

A b-bit uniform quantizer maps a real interval onto `2^b` evenly spaced levels.

**Affine (asymmetric)**, codes `q` in `[0, 2^b - 1]`:

```
x_lo = min(min(x), 0)          x_hi = max(max(x), 0)
s    = (x_hi - x_lo) / (2^b - 1)
z    = round(-x_lo / s)                          integer zero point
q    = clamp(round(x / s) + z, 0, 2^b - 1)
x'   = (q - z) * s
```

**Symmetric**, codes in `[-(2^(b-1) - 1), 2^(b-1) - 1]`, zero point fixed at 0:

```
s  = max|x| / (2^(b-1) - 1)
q  = clamp(round(x / s), -(2^(b-1) - 1), 2^(b-1) - 1)
x' = q * s
```

For any value inside the range the error is at most half a step: `|x - x'| <= s / 2`.
The test suite checks this bound for every bit width, mode and granularity.

**Granularity** decides how many values share one `(s, z)`:

| Granularity | Shares parameters across | Overhead (fp16 scale) |
|-------------|--------------------------|-----------------------|
| tensor | the whole tensor | negligible |
| channel | one output row | 16 bits per row |
| group of G | G consecutive inputs in one row | 16 / G bits per weight (+ 8 / G for a zero point) |

A single large weight stretches the step for everything that shares its scale. Smaller
groups contain that damage, which is why INT4 is almost always group-wise. With uniform
noise of variance `s^2 / 12`, each extra bit gives about 6.02 dB of SQNR; the test suite
checks the measured slope falls between 4.5 and 7 dB per bit.

## 2. FP8 emulation

`src/bitsmith/quant/fp8.py`

| Format | Exponent | Mantissa | Bias | Max | Special values |
|--------|---------:|---------:|-----:|----:|----------------|
| E4M3FN | 4 | 3 | 7 | 448 | no infinity, NaN = `S.1111.111` |
| E5M2 | 5 | 2 | 15 | 57344 | IEEE infinities and NaNs |

All 256 codes are decoded once into a lookup table. For non-negative finite values the bit
pattern is monotonic in the value, so encoding is a `searchsorted` over the table followed
by picking the nearer neighbour, with ties going to the even code (mantissa LSB 0).
Out-of-range values saturate to the largest finite magnitude. The encoder is compared bit
for bit with PyTorch's native `float8_e4m3fn` and `float8_e5m2` casts on 45,000 values.

Relative error is bounded by half an ulp: `2^-4` for E4M3 and `2^-3` for E5M2 in the
normal range. Unlike integers, FP8 spends its levels logarithmically, which suits
bell-shaped weight distributions with a few large values.

## 3. Bit packing

`src/bitsmith/packing/`

Memory is byte addressable, so sub-byte codes are packed into `uint8`.

**INT4**: two codes per byte, low nibble first.

```
byte = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
v0   = byte & 0x0F            v1 = byte >> 4
signed: v = (n ^ 8) - 8       maps 0..7 -> 0..7 and 8..15 -> -8..-1
```

Example: `[1, 2, -1, 7]` packs to `0x21 0x7F`.

**Ternary, 2-bit**: store `t + 1` in two bits, four per byte at bit offsets 0, 2, 4, 6.
`[1, -1, 0, 1]` packs to `0b10_01_00_10`.

**Ternary, base 3**: `byte = sum((t_i + 1) * 3^i)` for five trits. Maximum is 242, so it
fits a byte at 1.6 bits per weight. See ADR 0003 for the tradeoff.

## 4. 2:4 structured sparsity

`src/bitsmith/packing/sparse24.py`

In every group of four consecutive weights, keep the two with the largest magnitude.
Storage:

- `values`: the kept elements, half the original width
- `meta`: the two kept column indices per group (2 bits each), as a 4-bit nibble
  `idx0 | idx1 << 2`, two groups per byte

With fp16 values this is `(2 * 16 + 4) / 4 = 9` bits per weight. Sparse tensor cores read
exactly this layout and skip the zeros, doubling peak math throughput. For Gaussian weights
the kept half holds about 87% of the energy. Applied to a trained model without fine
tuning, the removed 13% hurts: see the results in the README.

## 5. QLinear and model conversion

`src/bitsmith/nn/`

A **codec** turns a dense `(out, in)` weight into a dict of buffers and back. `QLinear`
registers those buffers, so `state_dict()` holds the packed form and a checkpoint shrinks
on disk. Forward:

```
x -> x * input_scale        optional, SmoothQuant or AWQ (ADR 0007)
  -> fake quant per token   optional, act_bits
W  = codec.decode(buffers)  unpack + dequantize
y  = x @ W^T + bias
```

`quantize_model` walks `named_modules()`, swaps each `nn.Linear` for a `QLinear`, skips
names matching `exclude` (default `lm_head`), and returns a report with exact byte counts.

## 6. Calibration

`src/bitsmith/calib/`

Forward pre-hooks record per layer: the per-channel `max|x|`, the per-channel `mean|x|`,
optionally `X^T X`, and optionally a bounded sample of input rows. By default calibration
is **block-sequential** (ADR 0009): layers are grouped by transformer block, and block `i`
is calibrated on activations that already passed through quantized blocks `0..i-1`, so
later layers adapt to earlier errors. A hook ends each calibration pass right after the
current block.

**SmoothQuant** (`smoothquant.py`). LLM activations have a few channels up to 100x larger
than the rest, which ruins per-token INT8 activation quantization. For channel `j`:

```
s_j = max|X_j|^alpha / max|W_:,j|^(1 - alpha)
Y   = X W^T = (X diag(1/s)) (W diag(s))^T
```

The product is unchanged in exact arithmetic (tested to 1e-4), but activations become flat
and weights absorb a range they can afford. On synthetic data with three 40x outlier
channels, W8A8 output SQNR improves from 27.5 dB to 40.1 dB.

**AWQ** (`awq.py`). Weights in channels that see large activations matter more. Scale those
columns up by `s = mean|X|^alpha` (normalized) before quantizing, divide activations by `s`,
and grid search `alpha` in `[0, 1]` to minimize `||X W^T - (X / s) Q(W s)^T||^2`.
`alpha = 0` is plain round to nearest, so the search can never lose on its own data.

**GPTQ** (`gptq.py`). Minimize `||X W^T - X Q^T||^2` one column at a time. With
`H = 2 X^T X / n` and `U` the upper Cholesky factor of `H^-1`, after quantizing column `i`:

```
e        = (w_i - q_i) / U_ii
w_j     -= e * U_ij            for every later column j
```

This is the Optimal Brain Surgeon update: the rounding error of column `i` is pushed onto
the columns that are correlated with it, where later rounding can absorb it. With
`act_order=True` (the default) columns are visited by decreasing `diag(H)`, so the inputs
with the most energy are quantized first, while the most columns remain to absorb their
error. Group scales are then fixed up front from the original weights so the storage stays
in contiguous groups (ADR 0009). Updates are
applied lazily in blocks for speed. Dampening `H += 0.01 * mean(diag H) * I` keeps the
Cholesky factorization stable, and columns whose input is always zero are zeroed. When
`H` is the identity, there is nothing to compensate and GPTQ returns exactly the round to
nearest result; a unit test checks this.

## 7. Evaluation

`src/bitsmith/eval/`

**SQNR** = `10 log10(sum x^2 / sum (x - x')^2)` dB.

**Sliding-window perplexity**: windows of at most `window` tokens; each new window ends
`stride` tokens after the previous one and scores only those new tokens. Every token after
the first is scored exactly once. `stride` is capped at `window - 1` so non-overlapping
settings never drop a target. Tests check that a uniform model scores exactly `vocab` and
that a context-free bigram oracle gives identical perplexity under any window.

**Synthetic corpus**: a Markov chain with known entropy rate `H`, so `exp(H)` is the
perplexity floor that no model can beat (ADR 0006).

## 8. Roofline

`src/bitsmith/bench/roofline.py`

At batch size 1 each generated token reads every weight once and does one multiply-add with
it, so arithmetic intensity is `2 / (bits / 8) = 16 / bits` FLOP per byte: 1 for fp16,
4 for INT4. Every GPU's ridge point (peak FLOP/s divided by bandwidth) is far higher, so
decoding is bandwidth bound and

```
time per token >= weight_bytes / DRAM bandwidth
```

INT4 with group 128 (4.19 bits per weight) gives a 3.8x bound speedup over fp16. Reaching
it needs a fused kernel; this library's reference runtime does not (ADR 0004).

## 9. Fused INT4 kernel

`src/bitsmith/kernels/int4_triton.py`

The reference runtime computes `x @ dequant(W)^T` in two steps, so every call reads the
packed weight, writes a dense fp16 copy and reads it again: more traffic than plain fp16.
The Triton kernel does both steps in one pass:

```
for each tile of output features n and slice of input features k:
    load packed bytes (BLOCK_N, BLOCK_K / 2)            # 4 bits per weight from DRAM
    lo, hi = byte & 0xF, byte >> 4                       # unpack in registers
    q = interleave(lo, hi)                               # tl.join, back to column order
    w = (q - zero[group]) * scale[group]                 # dequantize in registers
    acc += sum(w * x[k], axis=1)                         # GEMV; tl.dot for GEMM
atomic add acc into y                                    # split-K partial sums
```

Measured on an RTX 3090 at batch 1, the kernel runs at 0.94x to 1.33x the speed of fp16
cuBLAS and 5x to 24x faster than the reference path (see the README). It reaches about 25% of
peak bandwidth; closing the rest of the gap needs techniques beyond this project's scope
(weight pre-shuffling for wide vector loads, asynchronous copies, and tuned tile shapes,
as in Marlin).
