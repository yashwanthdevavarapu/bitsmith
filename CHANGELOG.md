# Changelog

All notable changes are listed here. The format follows Keep a Changelog and the project
uses Semantic Versioning.

## [0.2.0] - 2026-11-10

### Added

- Fused INT4 dequantize + matmul kernels in Triton (GEMV with split-K for decoding, GEMM
  with tensor cores for prefill), exposed as `QLinear.set_backend("triton")` and
  `set_backend(model, "triton")`.
- `scripts/bench_kernels.py`: CUDA graph timing of fp16 cuBLAS, the reference path and the
  fused kernel against the bandwidth roofline.
- GPU support across the library (calibration statistics stay on device) and a GPU test
  suite (85 tests) that checks CUDA results and the kernel against the reference path.
- `bitsmith hf` options: `--device`, `--dtype`, `--window`, `--stride`, `--n-calib`,
  `--calib-len`, and `--eval-tokens 0` for the full test split; decode speed with the
  fused kernel is reported alongside the reference path.
- GPU results for SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-1.5B on the full WikiText-2 test
  split. ADRs 0009 and 0010.

### Changed

- Calibration is block-sequential by default and GPTQ uses activation ordering with static
  groups. On SmolLM2-135M this cuts the INT4 GPTQ perplexity gap from about 24% to 11%.
- CPU-only results for SmolLM2-135M were replaced by the GPU runs.

## [0.1.0] - 2026-11-09

### Added

- Uniform quantization (affine and symmetric) at tensor, channel and group granularity,
  with zero padding for partial groups.
- Bit-exact OCP FP8 emulation (E4M3FN and E5M2), verified against PyTorch float8 casts.
- BitNet b1.58 style ternary quantization with 2-bit and base-3 packing.
- INT4 nibble packing and 2:4 structured sparsity with tensor core metadata layout.
- `QLinear` with eight weight schemes, optional input scaling and per-token activation
  quantization; `quantize_model` for in-place conversion.
- SmoothQuant, AWQ and GPTQ calibration with a one-call `calibrate_and_quantize` pipeline.
- Sliding-window perplexity, SQNR metrics, a Markov corpus with a known perplexity floor
  and a tiny transformer for fast end to end runs.
- Pareto sweep, decode roofline, Hugging Face WikiText-2 evaluation and the `bitsmith` CLI.
- 200+ tests, CI on Linux, Windows and macOS for Python 3.10 to 3.12, benchmark and
  release workflows, eight ADRs and a technical guide.
