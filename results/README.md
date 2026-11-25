# Results

Every file here is produced by a script; nothing is edited by hand.
`python scripts/reproduce.py` regenerates the CPU results and `--gpu` adds the GPU ones.

| File | Produced by |
|------|-------------|
| `gpu_smollm2_135m.*`, `gpu_qwen2.5_0.5b.*`, `gpu_qwen2.5_1.5b.*` | `bitsmith hf --model ... --device cuda --dtype bfloat16 --eval-tokens 0 --window 1024 --stride 512 --n-calib 128 --calib-len 512` |
| `kernels_rtx3090.*` | `python scripts/bench_kernels.py` |
| `int8_rtx3090_shared.*` | `python scripts/bench_int8.py` |
| `decode_rtx3090_shared.*` | `python scripts/bench_decode.py` |
| `prefill_rtx3090_shared.*` | `python scripts/bench_prefill.py`, one process per model, then `--merge` |
| `int4_prefill_rtx3090_shared.*` | `python scripts/bench_int4_prefill.py` |
| `lowbit_kernels_rtx3090_shared.*` | `python scripts/bench_lowbit.py` |
| `decode_lowbit_rtx3090_shared.*` | `python scripts/bench_decode.py --models HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B --schemes int4 int3 int2 --no-reference` |
| `tiny_sweep.*` | `bitsmith sweep` |
| `cli_sqnr.txt` | `bitsmith sqnr --shape 1024x1024 --outliers 8` |
| `cli_roofline.txt` | `bitsmith roofline --params 0.5e9` |
| `cli_pack_*.txt` | `bitsmith pack --format ...` |
| `example_0*.txt` | `examples/0*.py` |

## GPU runs

- NVIDIA GeForce RTX 3090 (24 GB), driver 535, CUDA 12.1, PyTorch 2.5.1, Triton 3.1,
  Transformers 4.57, Python 3.11
- Models in bf16; full WikiText-2 raw test split (about 300,000 tokens per tokenizer),
  window 1024, stride 512
- Calibration: 128 random sequences of 512 tokens from the WikiText-2 train split, seed 0
- Decode speed: median of 3 greedy runs of 32 new tokens after a 64 token prompt, after one
  warmup run, through Hugging Face `generate` without CUDA graphs
- Kernel timings: median of 200 CUDA graph replays after 20 warmup replays
- Files ending in `_shared` were measured while other jobs used the same GPU (up to 100%
  utilization); their ratios are more reliable than their absolute times
- Prefill: 64 calibration sequences of 512 tokens for SmoothQuant, seven interleaved rounds
  of five calls per configuration, perplexity on the first 32k test tokens in 1024-token
  windows

## CPU runs

- Windows 11, Python 3.11.4, PyTorch 2.6.0 CPU build, 8 GB RAM

Wall-clock columns (`ms/fwd`, `decode tok/s`, kernel times) depend on the machine and on
other load. Quality columns (`perplexity`, `SQNR`, `bits/w`) are deterministic for a given
PyTorch version.
