# Roadmap

Where bitsmith is going next. Every item below is (or will be) a GitHub issue you can pick
up; look for the milestone and labels named in each section.

Want to work on something? Comment on the issue first so nobody duplicates effort, then
follow [CONTRIBUTING.md](CONTRIBUTING.md). Ideas that are not listed here are welcome too:
open a [discussion](https://github.com/yashwanthdevavarapu/bitsmith/discussions) or a
feature request.

## Where we are: v0.2.0

- INT8, group-wise INT4, FP8 (E4M3, E5M2), ternary and 2:4 sparsity, with real bit packing
- SmoothQuant, AWQ, and block-sequential GPTQ with activation ordering
- A fused INT4 Triton kernel, up to 1.33x faster than fp16 cuBLAS for single-token decode
- Full WikiText-2 results for SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-1.5B on an RTX 3090

Since v0.2.0 (unreleased, see the [changelog](CHANGELOG.md)):

- NF4, MXFP4, MXINT8, INT3 and INT2, with GPTQ and AWQ on all of them
- A quantized KV cache (per-channel keys), alone and combined with quantized weights
- GPTQ 3.5x to 15x faster, CUDA graph decoding, a real INT8 W8A8 backend
- W8A8 prefill 1.19x to 1.39x faster than bf16 on Qwen2.5-1.5B, the first end to end speedup
- GPT-2 support (`Conv1D`) and a SmoothQuant that picks its strength per layer

Known gaps, from the [README](README.md#limitations): the fused INT4 kernel reaches about
25% of peak bandwidth, the quantized KV cache saves memory but not time, the INT8
backend only beats bf16 end to end on the 1.5B model, and INT4 prefill runs at about half
of bf16 speed.

## v0.3: Speed

Goal: make quantized models faster than bf16 end to end, not just per layer.

| Item | Labels |
|------|--------|
| ~~CUDA graph capture for the decode step~~ done, `GraphDecoder`, see ADR 0016 | `performance`, `area: kernels` |
| CUDA graph decoding with a quantized KV cache (a static, preallocated packed cache) | `performance`, `area: kernels`, `help wanted` |
| Faster INT4 GEMV: wide vector loads and weight pre-shuffling | `performance`, `area: kernels` |
| ~~Real INT8 W8A8 matmul on tensor cores with `torch._int_mm`~~ done, with fused Triton quantize and rescale, see ADR 0017 | `performance`, `area: kernels` |
| ~~End to end W8A8 speed~~ done, prefill 1.19x to 1.39x faster than bf16 on Qwen2.5-1.5B, see ADR 0019 | `performance`, `area: benchmarks` |
| Fuse per-token activation quantization into the previous layer's output (no extra pass) | `performance`, `area: kernels`, `help wanted` |
| Tune the INT4 prefill GEMM (currently about 2x slower than cuBLAS) | `performance`, `area: kernels` |
| ~~Faster GPTQ column loop~~ done, 3.5x to 15x per layer, see ADR 0013 | `performance`, `area: calibration` |

## v0.4: Formats

Goal: cover the formats people actually ship.

| Item | Labels |
|------|--------|
| ~~NF4 codec (lookup table, as in QLoRA)~~ done, see ADR 0011 | `area: formats` |
| ~~MXFP4 and MXINT8 microscaling formats~~ done, see ADR 0011 | `area: formats` |
| ~~GPTQ for NF4 and MX~~ done, see ADR 0012 | `area: calibration` |
| Fused Triton kernel for NF4 and MXFP4 (table lookup in registers) | `area: kernels`, `help wanted` |
| ~~INT3 and INT2 group codecs with packing~~ done, with GPTQ, see ADR 0014 | `area: formats` |
| Fused Triton GEMV for INT3 and INT2 | `area: kernels`, `help wanted` |
| ~~KV-cache quantization~~ done, per-channel keys and per-token values, see ADR 0015 | `area: formats` |
| Fused attention that reads the packed KV cache | `performance`, `area: kernels`, `help wanted` |
| ~~Results for weight and KV-cache quantization combined~~ done, `bitsmith kv --weights`, see the README | `area: benchmarks` |

## v0.5: Models and calibration

Goal: run on more model families and larger sizes.

| Item | Labels |
|------|--------|
| ~~Support GPT-2 style `Conv1D` layers in `quantize_model`~~ done, with GPT-2 results | `area: models` |
| ~~SmoothQuant that cannot make a layer worse~~ done, per-layer search, see ADR 0018 | `area: calibration` |
| Fold SmoothQuant and AWQ scales into norm layers | `area: calibration` |
| Results for a 7B model | `area: benchmarks`, `help wanted` |

## Always open

Small, well-scoped tasks for a first contribution, labelled `good first issue`:

- Add more GPUs to the roofline hardware list
- JSON output for `bitsmith sqnr` and `bitsmith roofline`
- An FP8 demo in `bitsmith pack`
- A Colab quickstart notebook
- Faster sampling in the synthetic Markov corpus
- Benchmark results from your own GPU

## How the roadmap changes

The maintainer reviews this file at each release. Items move between milestones based on
contributor interest and what the results show; decisions that change numerics, storage
formats or the public API are recorded as ADRs in [docs/adr](docs/adr/README.md).
