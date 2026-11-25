# Technical guide

A walk through the math and storage formats behind every module, in the order data flows
through the library. Each section names the file that implements it.

1. [Uniform quantization](#1-uniform-quantization)
2. [FP8 emulation](#2-fp8-emulation)
   - [NF4](#nf4)
   - [MX microscaling](#mx-microscaling)
3. [Bit packing](#3-bit-packing)
4. [2:4 structured sparsity](#4-24-structured-sparsity)
5. [QLinear and model conversion](#5-qlinear-and-model-conversion)
6. [Calibration: SmoothQuant, AWQ, GPTQ](#6-calibration)
7. [Evaluation](#7-evaluation)
8. [Roofline](#8-roofline)
9. [Fused INT4 kernel](#9-fused-int4-kernel)
10. [KV-cache quantization](#10-kv-cache-quantization)
11. [CUDA graph decoding](#11-cuda-graph-decoding)

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

### NF4

`src/bitsmith/quant/nf4.py`

NF4 (NormalFloat4, from QLoRA) keeps 4-bit codes but spaces the 16 levels like a normal
distribution instead of evenly. Each block of G weights is divided by its absmax, so values
lie in `[-1, 1]`, and each value is replaced by the index of the nearest level:

```
pos = Phi^-1(linspace(0.9677083, 0.5, 9)[:8])      8 positive quantiles
neg = -Phi^-1(linspace(0.9677083, 0.5, 8)[:7])     7 negative quantiles
levels = sort(neg, 0, pos) / max|levels|            exactly -1, 0 and 1
```

`Phi^-1` is the normal quantile function. The table is built from this formula at import
time and tested against the published values to 1e-6. Storage: 4-bit codes packed with
the INT4 layout plus one fp16 absmax per block, so `4 + 16 / G` bits per weight, the same
as `int4-sym`. On a Gaussian weight NF4 beats `int4-sym` by about 1.8 dB at G = 128,
because more levels sit where most weights are.

### MX microscaling

`src/bitsmith/quant/mx.py`

The OCP Microscaling formats share one 8-bit power-of-two scale (E8M0, value
`2^(e - 127)`, `0xFF` is NaN) across a block of 32 elements:

```
shared = floor(log2(absmax)) - emax_elem      emax_elem = 2 for FP4, 0 for INT8
codes  = quantize(x / 2^shared)               saturating
```

| Format | Element | Element values | Bits per weight |
|--------|---------|----------------|----------------:|
| MXFP4 | FP4 E2M1 (bias 1) | 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives | 4.25 |
| MXINT8 | int8 times 2^-6 | -127/64 to 127/64 | 8.25 |

`floor(log2(absmax))` is read exactly from `frexp`, with no floating-point `log2`. FP4
elements are encoded with the same nearest-even table search as FP8. A power-of-two scale
cannot map the absmax onto the largest element exactly: after scaling, the FP4 absmax
lands in `[4, 8)`, so values between 6 and 8 saturate to 6. This is what the
specification prescribes, and it costs MXFP4 about 1.5 dB against NF4 on Gaussian
weights. In exchange, hardware can apply the scale as an exponent add instead of a
multiply.

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

**INT3**: eight unsigned codes in three bytes. The codes form one little-endian 24-bit
word, three bits each, stored low byte first (ADR 0014):

```
word   = v0 | v1 << 3 | v2 << 6 | ... | v7 << 21
byte_k = (word >> 8k) & 0xFF                      k = 0, 1, 2
v_i    = (word >> 3i) & 0b111
```

Example: `[1, 2, 3, 4, 5, 6, 7, 0]` gives `word = 0x1F58D1` and bytes `0xD1 0x58 0x1F`.
`v2` straddles the first two bytes, which is why the word is rebuilt before shifting.

**INT2**: four unsigned codes per byte at bit offsets 0, 2, 4, 6, the ternary layout below.
`[1, 2, 3, 0]` packs to `0x39`. INT3 and INT2 are asymmetric only, so all 8 or 4 levels
are used.

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

GPT-2 and a few other models use `transformers`' `Conv1D` instead of `nn.Linear`: the same
`y = x W + b`, with `W` stored as `(in, out)`. `linearize_conv1d` replaces each one with an
`nn.Linear` holding `W^T`; `quantize_model` and `calibrate_and_quantize` call it first, so
codecs, calibration hooks, AWQ and GPTQ only ever see `nn.Linear`. The swap is exact (a
test compares outputs), and `Conv1D` is recognised by name and shape, so bitsmith does
not import transformers for it.

**Backends.** `set_backend(model, name)` switches every layer that supports it:

| Backend | Layers | What runs |
|---------|--------|-----------|
| `reference` | all | dequantize, then `F.linear` (the source of truth, ADR 0004) |
| `triton` | INT4 weights | fused dequantize and matmul (section 9) |
| `int8` | INT8 weights with `act_bits=8` | integer W8A8 matmul (section 9) |

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

**GPTQ on grid formats** (`gptq_quantize_table`). Nothing in the update above depends on
the levels being evenly spaced: "quantize column `i`" only has to return the nearest
representable value. NF4 and the MX codecs therefore expose a small interface
(`TableCodec`): `scale_buffer` picks the stored block scales, `snap` rounds scaled values
onto the grid, and `encode_with_scale` stores weights with given scales. GPTQ fixes the
scales from the original weights, then for each column computes `q_i = snap(w_i / s) * s`
and runs the same update. The result is exactly what the codec stores (a test decodes the
buffers and compares bit for bit). On the correlated test inputs, GPTQ lowers the expected
output error about 4x for NF4, MXFP4 and MXINT8, the same as for INT4 (3.8x).

**Making the loop fast** (ADR 0013). The weight is stored transposed, so each input column
is a contiguous row. Updates are lazy on two levels: rank-1 updates stay inside a
32-column inner block, and one matmul pushes the block's errors to the columns after it.
On CUDA each inner block is captured once as a CUDA graph and replayed, turning about 300
kernel launches into one. The inverse Hessian factor uses `U = P L^-1 P` with
`P H P = L L^T` (`P` reverses the order), one Cholesky and one triangular inverse, in
float32 on CUDA with a float64 fallback. `scripts/bench_gptq.py` times each part.

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
store acc in slot k of a partial buffer                 # split-K
y = cast(((slot0 + slot1) + slot2) + slot3)              # fixed order, one kernel
```

The GEMV serves up to 4 rows: one program dequantizes its weight tile once and multiplies
it by every row. Each K slice writes its partial sums to its own slot and the slots are
added in order afterwards, so the result is the same bits on every call (fp32 atomics
added them in whatever order the programs finished; ADR 0021).

INT3 and INT2 use the same kernels with a different tile loader:

```
INT2   byte -> 4 fields (b >> 0, 2, 4, 6) & 3     join(join(c0, c2), join(c1, c3))
INT3   3 bytes -> 24-bit word -> 8 fields of 3    join(join(join(c0, c4), join(c2, c6)),
                                                       join(join(c1, c5), join(c3, c7)))
```

Joining fields half a word apart first, then a quarter apart, puts the codes back in
column order after the reshape. INT2 reads half the bytes of INT4 and is the fastest
decode format (up to 2.09x fp16 on 7B-sized layers); INT3 reads 25% fewer bytes than INT4
but is slower, because the codes straddle bytes and unpacking, not memory, is the limit.

Measured on an RTX 3090 at batch 1, the kernel runs at 0.94x to 1.33x the speed of fp16
cuBLAS and 5x to 24x faster than the reference path (see the README). It reaches about 25% of
peak bandwidth; closing the rest of the gap needs techniques beyond this project's scope
(weight pre-shuffling for wide vector loads, asynchronous copies, and tuned tile shapes,
as in Marlin).

With many rows the matmul is compute bound, and fusing stops paying: the fused GEMM's K
tile is one group, and every block of rows dequantizes the same weights again. There
`int4_linear` unpacks the weight once (`int4_dequantize`, the same `_dequant_tile` code, so
the same values) and calls cuBLAS:

```
rows <= 4                       fused GEMV                 decoding, bandwidth bound
rows >= 64  or  K >= N          unpack + cuBLAS            prefill, compute bound
otherwise                       fused tensor core GEMM     wide layers, few rows
```

The thresholds come from `scripts/bench_int4_prefill.py` (ADR 0020).

### Real INT8 W8A8

`src/bitsmith/kernels/int8_mm.py`

The reference runtime only simulates W8A8: activations are rounded to INT8 codes and then
multiplied as floats. The `int8` backend multiplies integers:

```
s_x = max|x[t]| / 127 per token     xq = clamp(round(x / s_x), -127, 127)   (int8)
acc = xq @ wq^T                      int8 x int8 -> int32 on tensor cores (torch._int_mm)
y   = acc * s_x * s_w^T + bias       s_w per output channel
```

The activation codes are computed by the same function as the simulated path, so both
backends agree to float rounding (tested). `torch._int_mm` needs more than 16 rows and
inner dimensions divisible by 8; shorter inputs are padded with zero rows.

A smoothed layer computes `(x * input_scale) W^T`. The quantization kernel multiplies each
element as it loads it and rounds the product to the activation dtype, exactly what the
separate multiply did, so the codes do not change and one pass over the activations is
saved (ADR 0019).

Where it pays off: INT8 tensor cores help GEMMs with many rows, so W8A8 speeds up prefill
(the prompt's forward pass), not single-token decoding. `scripts/bench_prefill.py` measures
whole models, eager and replayed from a CUDA graph.

## 10. KV-cache quantization

`src/bitsmith/kv/quant.py`, `src/bitsmith/kv/hf.py`

During decoding every layer keeps the keys and values of all earlier tokens. Per token
that is `2 x layers x kv_heads x head_dim` numbers, so the cache grows with context length
and batch size while the weights stay fixed. bitsmith stores it with an asymmetric grid
per group, an fp16 scale `s` and an fp16 minimum `m`:

```
s = (max - min) / (2^b - 1)     q = clamp(round((x - m) / s), 0, 2^b - 1)     x' = q s + m
bits per element = b + 32 / group
```

Two group shapes, chosen per tensor:

| Axis | One group is | When it is quantized |
|------|--------------|----------------------|
| per token | `group` consecutive channels of one token | as soon as the token is written |
| per channel | one channel over `group` consecutive tokens | once all `group` tokens exist |

Keys need per-channel groups. In trained models a few key channels are large for every
token, so a per-token grid stretches to fit them and rounds the other channels to a
handful of levels. Per channel, each channel gets its own range.
Values have no such channels and are combined across tokens by the attention weights, so
per token suits them. This is the KIVI layout, `k4cv4` in bitsmith's spec names
(`k<bits>[c]v<bits>[c][-g<group>]`, `c` = per channel).

A per-channel group cannot be quantized until it is complete, so the newest
`n mod group` tokens wait in full precision. `KVStream.append` returns the stored tokens
*before* quantizing a group that has just been completed, which is what token-by-token
decoding sees.

**Exact evaluation in chunks.** `kv_perplexity` feeds each window through a fresh cache in
chunks of `group` tokens. Queries in a chunk see earlier groups quantized and their own
group in full precision, exactly as when decoding one token at a time. A test checks that
the chunked logits match token-by-token decoding to float rounding. (A single forward pass
would be wrong both ways: Hugging Face's own quantized cache returns the prompt
unquantized, and quantizing the whole prompt first would let each query see a group
quantized with statistics of tokens after it.)

`quantized_cache(spec)` is a regular `transformers` cache, so it also works with
`model.generate(past_key_values=...)`. The cache holds the packed codes; the attention
still reads a dequantized copy built at each step, so this measures accuracy and memory,
not speed. A fused attention kernel that reads the packed cache is on the roadmap.

## 11. CUDA graph decoding

`src/bitsmith/decode.py`

At batch 1 one decode step of a 1.5B model runs a few hundred small kernels: norms,
projections, rotary embeddings, attention, MLPs, each one or a few launches. The GPU
finishes most of them faster than the CPU can launch the next one, so the step is bound
by launch overhead, not by memory bandwidth. This is why a faster INT4 GEMV alone did not
make `generate` faster than bf16.

`GraphDecoder` records one decode step as a CUDA graph and replays it for every token:

```
prefill the prompt eagerly into a StaticCache (fixed size, fixed addresses)
capture:  logits = model(ids_buf, cache_position=pos_buf, past_key_values=cache)
for each new token:
    ids_buf <- previous token        # device to device copy, no host sync
    pos_buf <- position
    graph.replay()                   # one launch for the whole step
    token <- argmax(logits)          # stays on the GPU
```

Requirements, all met by `transformers.StaticCache` and the bitsmith kernels: every
tensor the step touches keeps its address between replays (the cache is preallocated and
`reset()` zeroes it in place), nothing in the step synchronizes with the host, and every
kernel was compiled before capture (two warm-up steps on a side stream, which also runs
Triton's autotuning). The capture is done once; the warm-up steps write into the cache,
so the prompt is prefilled again before decoding.

Tests check that graph replay produces the same tokens as eager steps with bf16 weights,
the reference INT4 runtime and the fused kernel, and that the eager path matches
`generate` exactly.
