# 0019. W8A8 measured end to end on prefill, with the input scale fused

- Status: Accepted (builds on ADR 0017 and ADR 0018)
- Date: 2026-11-17

## Context

ADR 0017 showed the `int8` backend beating fp16 on single layers (1.2x to 2.3x at 128 to
512 rows). A model is more than its linear layers: attention, norms, rotary embeddings
and the residual stream stay in bf16, and every W8A8 layer adds two small kernels
(quantize, rescale) around its GEMM. Whether the layer gains survive in a whole model
had not been measured.

Two details stood in the way:

- A SmoothQuant layer multiplied its input by `input_scale` before quantizing it: one
  more read and write of the activations per layer, in a separate kernel.
- On a 0.5B model, a plain PyTorch forward pass over 128 to 512 tokens takes about 20 ms
  whatever the prompt length: it is bound by Python and kernel launches. Timing that
  measures the launch count, not the arithmetic.

## Decision

- **The input scale is applied inside the quantization kernel.** `int8_linear` takes
  `input_scale`; the Triton kernel multiplies each loaded element and rounds the product
  to the activation dtype, which is what `x * input_scale` does, so the codes are bit for
  bit the same as before (tested on GPU for fp16 and bf16). `QLinear` passes its scale to
  the `int8` backend instead of multiplying first.
- **Prefill is the end to end benchmark for W8A8** (`scripts/bench_prefill.py`). One
  forward pass with `logits_to_keep=1` and no cache, the way generation starts, at batch
  x prompt shapes 1 x 128, 1 x 512, 1 x 2048 and 8 x 512. Single-token decoding is not
  where INT8 tensor cores help: those GEMMs have one row and are bound by weight bandwidth,
  where INT4 is the right tool (ADR 0010).
- **Two timings, reported side by side.** Eager (a plain call, what a `transformers` user
  sees) and CUDA graph replay (GPU work only, what a serving engine with graphs sees).
  The speed column compares graph times.
- **Interleaved rounds on the shared GPU.** All configurations of one shape are timed in
  turn, seven rounds of five calls, and the median is reported, so a busy period on the
  shared GPU slows them alike.
- **Perplexity from the backend that is timed**, on the first 32k WikiText-2 test tokens.
- **One process per model**, joined with `--merge`, so a large model gets the memory a
  shared GPU has free when it starts.

## Consequences

- Results on a shared RTX 3090 (`results/prefill_rtx3090_shared.txt`), graph replay,
  speed against bf16 at 1 x 128 / 1 x 512 / 1 x 2048 / 8 x 512:

  | Model | W8A8 int8 | INT4-g64 fused | W8A8 simulated | W8A8 perplexity |
  |-------|-----------|----------------|----------------|-----------------|
  | SmolLM2-135M | 0.65x / 0.99x / 0.97x / 0.88x | 0.43x to 0.70x | 0.22x to 0.36x | +1.1% |
  | Qwen2.5-0.5B | 0.89x / 1.05x / 1.05x / 1.02x | 0.37x to 0.53x | 0.23x to 0.34x | +1.0% |
  | Qwen2.5-1.5B | **1.19x / 1.26x / 1.32x / 1.39x** | 0.37x to 0.51x | 0.20x to 0.46x | +1.0% |

- W8A8 is the first configuration in bitsmith that makes a whole model faster than bf16:
  1.19x to 1.39x prefill on Qwen2.5-1.5B for +1.0% perplexity. The gain grows with the
  model and the token count, as the linear layers take a larger share of the time; on
  the two small models it is at or below parity.
- In eager mode W8A8 is 0.44x to 0.65x on short Qwen2.5-1.5B prompts, where launches
  dominate and W8A8 launches three kernels per layer instead of one, and 1.30x to 1.36x at
  2048 or more tokens.
- The real backend matches the simulated one on perplexity (10.200 against 10.216 on
  Qwen2.5-1.5B), so the accuracy results of the simulated path carry over.
- INT4 prefill runs at 0.37x to 0.70x of bf16: the fused INT4 GEMM is a decode kernel,
  already listed on the roadmap as slower than cuBLAS for many rows.
- The shared GPU adds noise that interleaving reduces but does not remove; the ratios are
  more reliable than the absolute times.
