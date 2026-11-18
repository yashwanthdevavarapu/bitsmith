# Changelog

All notable changes are listed here. The format follows Keep a Changelog and the project
uses Semantic Versioning.

## [Unreleased]

### Added

- NF4 codec (`nf4`): 16 normal-quantile levels with an fp16 absmax per block. The table
  is computed from the normal quantile function and tested against the published values.
- OCP Microscaling codecs `mxfp4` and `mxint8`: 32-element blocks with a shared E8M0
  scale, FP4 E2M1 or INT8 elements, following the MX v1.0 rules for the shared exponent
  and saturation (ADR 0011).
- AWQ works with the new formats; `bitsmith hf --configs` gains `nf4`, `nf4+awq`,
  `mxfp4`, `mxfp4+awq` and `mxint8`, and the default sweep includes `nf4-g64` and `mxfp4`.
- GPTQ for NF4, MXFP4 and MXINT8 (`gptq_quantize_table`, `--configs nf4+gptq mxfp4+gptq`):
  the codecs expose a `TableCodec` interface (block scales, grid rounding, encode with
  fixed scales), so GPTQ rounds each column onto the real grid (ADR 0012).
- `QLinear.from_buffers` and a `quantize_model(encoded=...)` argument for ready codec
  buffers.
- INT3 and INT2 group codecs (`int3`, `int2`) with real packing: eight 3-bit codes per
  three bytes, four 2-bit codes per byte (`pack_int3`, `pack_int2`), GPTQ support
  (`--configs int3 int3+gptq int2+gptq`) and `bitsmith pack --format int3|int2` (ADR 0014).
- KV-cache quantization (`bitsmith.kv`, ADR 0015): INT8, INT4, INT3 and INT2 keys and values
  with per-token or per-channel groups and packed storage, a Hugging Face cache
  (`quantized_cache`) that works with `generate`, an exact chunked perplexity evaluation,
  and `bitsmith kv --configs k8v8 k4cv4 ...` with results on three models.
- `bitsmith kv --weights nf4+gptq`: quantize the weights too and report total memory
  (model plus cache at `--context` tokens) against the unquantized model.
- `bitsmith.decode.GraphDecoder`: greedy decoding with a static cache and the one-token step
  replayed from a CUDA graph (ADR 0016), with `scripts/bench_decode.py`.
- Real W8A8: an `int8` backend that multiplies INT8 activations and weights with
  `torch._int_mm` (ADR 0017), selected with `set_backend(model, "int8")`, with fused
  Triton kernels for per-token quantization and the INT32 rescale, and
  `scripts/bench_int8.py`.
- GPT-2 results (`results/gpu_gpt2*.txt`).
- GPT-2 style `Conv1D` layers are converted like `nn.Linear` (`linearize_conv1d`), in
  `quantize_model`, `calibrate_and_quantize` and the sweeps.
- `scripts/kv_channel_stats.py`: how uneven key and value channels are in real models.
- `scripts/bench_prefill.py`: end to end prefill speed of bf16, W8A8 (real and simulated)
  and INT4, eager and CUDA graph, with perplexity from the timed backend (ADR 0019).
  W8A8 on INT8 tensor cores prefills Qwen2.5-1.5B 1.19x to 1.39x faster than bf16
  (`results/prefill_rtx3090_shared.*`).
- INT4 prefill path (ADR 0020): `int4_dequantize` unpacks a packed INT4 weight in one
  autotuned Triton kernel, and `int4_linear` uses it with cuBLAS for many rows (64 or more,
  or any layer with `K >= N`), with `path="auto" | "fused" | "dequant"` to force a path.
  INT4 layers with 256 or more rows are 1.6x to 3.1x faster; `scripts/bench_int4_prefill.py`
  measures the switch point (`results/int4_prefill_rtx3090_shared.*`).
- `int8_linear(input_scale=...)`: the SmoothQuant or AWQ input scale is applied inside the
  fused quantization kernel, bit for bit equal to multiplying first; the `int8` backend
  uses it.
- `pack_int4`, `pack_int3` and `pack_int2` take `validate=False` to skip the range check,
  which forces a GPU to host synchronization.
- `docs/assets/frontier_gpu_models.png`: the best perplexity at each bits per weight.
- `scripts/bench_gptq.py`: per-layer GPTQ timing for the Qwen2.5-1.5B shapes, with results
  for an RTX 3090 in `results/gptq_speed_rtx3090.*`.
- Technical guide sections on NF4, MX and GPTQ on grid formats; tests with known values.
- Apache 2.0 license, Code of Conduct, security policy, citation file and roadmap.
- Contributor guide with a first-contribution walkthrough and label reference.
- Issue templates for new formats and benchmark results, a contact link to Discussions,
  and an updated pull request checklist.
- `scripts/github_bootstrap.py`, which creates the project labels, milestones and starter
  issues on GitHub.
- Package metadata: license, author and project links.

### Changed

- The CPU test of half-precision layers is skipped on machines where PyTorch's fp16 or bf16
  CPU matmul crashes with an illegal instruction (seen on some GitHub Windows runners); a
  child process probes it once per session.
- `calibrate_and_quantize(method="smoothquant")` now picks the smoothing strength per layer
  on calibration data, including no smoothing (ADR 0018). A fixed `alpha = 0.5` made GPT-2
  73% worse; `alpha=` keeps the old behaviour. W8A8 + SmoothQuant results were re-measured.
- `set_backend` asks each layer which backends it supports (`QLinear.supports`).
- GPTQ is 3.5x to 15x faster per layer on a GPU (ADR 0013): one transposed loop with
  two-level lazy updates shared by INT4 and grid formats, CUDA graph replay of each
  32-column block, and a cheaper inverse Hessian factor computed in float32 on CUDA with
  a float64 fallback. Results are unchanged beyond float rounding. New arguments:
  `use_cuda_graph` and `inner_size` on both GPTQ functions, `dtype` on
  `damped_inverse_cholesky`.
- `fp8_encode` shares its nearest-even table search (`round_to_table`) with FP4.
- `Int4Codec` is now `IntCodec` with a `bits` field (4, 3 or 2); `Int4Codec` remains as an
  alias. INT4 group-size errors now say "multiple of 2".
- Sweeps keep the reference weights on the CPU, which saves one model's worth of GPU
  memory, and release cached GPU memory between configs.
- Result files are always written with LF line endings, also on Windows.

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
