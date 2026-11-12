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

Known gaps, from the [README](README.md#limitations): end-to-end decoding is still slower
than bf16 because of launch overhead, the kernel reaches about 25% of peak bandwidth, and
only `nn.Linear` layers are converted.

## v0.3: Speed

Goal: make quantized models faster than bf16 end to end, not just per layer.

| Item | Labels |
|------|--------|
| CUDA graph capture for the decode step | `performance`, `area: kernels`, `help wanted` |
| Faster INT4 GEMV: wide vector loads and weight pre-shuffling | `performance`, `area: kernels` |
| Real INT8 W8A8 matmul on tensor cores with `torch._int_mm` | `performance`, `area: kernels`, `help wanted` |
| Tune the INT4 prefill GEMM (currently about 2x slower than cuBLAS) | `performance`, `area: kernels` |
| Vectorize the GPTQ column loop | `performance`, `area: calibration` |

## v0.4: Formats

Goal: cover the formats people actually ship.

| Item | Labels |
|------|--------|
| ~~NF4 codec (lookup table, as in QLoRA)~~ done, see ADR 0011 | `area: formats` |
| ~~MXFP4 and MXINT8 microscaling formats~~ done, see ADR 0011 | `area: formats` |
| ~~GPTQ for NF4 and MX~~ done, see ADR 0012 | `area: calibration` |
| Fused Triton kernel for NF4 and MXFP4 (table lookup in registers) | `area: kernels`, `help wanted` |
| INT3 and INT2 group codecs with packing | `area: formats` |
| KV-cache quantization | `area: formats`, `help wanted` |

## v0.5: Models and calibration

Goal: run on more model families and larger sizes.

| Item | Labels |
|------|--------|
| Support GPT-2 style `Conv1D` layers in `quantize_model` | `area: models`, `help wanted` |
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
