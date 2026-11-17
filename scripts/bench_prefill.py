"""End-to-end prefill speed: bf16 against W8A8 on INT8 tensor cores and fused INT4.

Prefill (reading the prompt) is one forward pass over many tokens, so its linear layers
are matrix multiplies with hundreds or thousands of rows: the regime where INT8 tensor
cores run at twice the fp16 rate. For each model this script builds:

* ``bf16``               the original model, cuBLAS
* ``w8a8 smoothquant``   INT8 weights and activations, SmoothQuant with the per-layer
                         search (ADR 0018), on the ``int8`` backend (``torch._int_mm``)
                         and on the reference runtime that only simulates it
* ``int4-g64``           round-to-nearest INT4 weights on the fused Triton kernel

and times one forward pass (``logits_to_keep=1``, no cache, as when generation starts)
over several prompt shapes, two ways:

* ``eager``: a plain PyTorch call, what a ``transformers`` user gets. On small models this
  is bound by Python and kernel launches, not by the GPU.
* ``graph``: the same pass replayed from a CUDA graph, so the time is GPU work alone. The
  speed column compares graph times; it is what a serving engine with graphs would see.

The GPU is shared with other jobs, so the configurations are timed in interleaved rounds
and the median per configuration is reported: a slow period hits all of them alike.
Perplexity on the first 32k WikiText-2 test tokens (1024-token windows) is measured with
the same backend that is timed.

One process per model keeps a shared GPU's memory free; ``--merge`` joins the runs::

    python scripts/bench_prefill.py --models Qwen/Qwen2.5-1.5B --json run_qwen1.5b.json
    python scripts/bench_prefill.py --merge run_*.json --json results/prefill.json > results/prefill.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from bitsmith.bench.hf import _require_hf, calibration_batches, wikitext2_tokens
from bitsmith.calib import calibrate_and_quantize
from bitsmith.eval.perplexity import sliding_window_perplexity
from bitsmith.kernels import triton_available
from bitsmith.nn import quantize_model, set_backend
from bitsmith.report import table

MODELS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
SHAPES = [(1, 128), (1, 512), (1, 2048), (8, 512)]  # (batch, prompt tokens)
ROUNDS, PER_ROUND = 7, 5
PPL_TOKENS, PPL_WINDOW = 32768, 1024


def time_ms(fn: Callable[[], object], reps: int) -> float:
    """Median of ``reps`` calls in milliseconds, CUDA events."""
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    """Record ``fn`` into a CUDA graph after warming it up on a side stream."""
    fn()  # Triton autotuning and lazy initialization happen outside the capture
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def perplexity(model: nn.Module, tokens: torch.Tensor) -> float:
    def logits(ids: torch.Tensor) -> torch.Tensor:
        return model(ids).logits  # type: ignore[no-any-return,operator]

    return sliding_window_perplexity(logits, tokens, window=PPL_WINDOW, stride=PPL_WINDOW).perplexity


@torch.no_grad()
def bench_model(model_id: str, n_calib: int, calib_len: int) -> list[dict[str, object]]:
    transformers, _ = _require_hf()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    base = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    test = wikitext2_tokens(tok, "test", PPL_TOKENS).cuda()
    calib = [c.cuda() for c in calibration_batches(wikitext2_tokens(tok, "train", 200_000), n_calib, calib_len)]

    w8a8 = copy.deepcopy(base)
    calibrate_and_quantize(w8a8, calib, method="smoothquant", scheme="int8", act_bits=8)
    int4 = copy.deepcopy(base)
    quantize_model(int4, scheme="int4", group_size=64)

    # (weights, kernel) -> (model, backend); the two W8A8 rows share one model
    configs: dict[tuple[str, str], tuple[nn.Module, str]] = {
        ("bf16", "cuBLAS"): (base, "reference"),
        ("w8a8 smoothquant", "int8 tensor cores"): (w8a8, "int8"),
        ("w8a8 smoothquant", "simulated"): (w8a8, "reference"),
    }
    if triton_available():
        configs[("int4-g64", "fused Triton")] = (int4, "triton")

    def select(key: tuple[str, str]) -> nn.Module:
        model, backend = configs[key]
        set_backend(model, backend)
        return model

    ppl = {key: perplexity(select(key), test) for key in configs}
    eager: dict[tuple[tuple[str, str], tuple[int, int]], list[float]] = {}
    graph: dict[tuple[tuple[str, str], tuple[int, int]], list[float]] = {}
    for batch, length in SHAPES:
        ids = test[1000 : 1000 + batch * length].reshape(batch, length)
        models = {key: configs[key][0] for key in configs}
        calls = {key: (lambda m=m, x=ids: m(x, logits_to_keep=1, use_cache=False)) for key, m in models.items()}
        graphs = {}
        for key, call in calls.items():  # a graph keeps the kernels of the backend set at capture
            select(key)
            graphs[key] = capture(call)
        for _ in range(ROUNDS):
            for key, call in calls.items():
                select(key)
                eager.setdefault((key, (batch, length)), []).append(time_ms(call, PER_ROUND))
                graph.setdefault((key, (batch, length)), []).append(time_ms(graphs[key].replay, PER_ROUND))
        del graphs
        torch.cuda.empty_cache()

    records: list[dict[str, object]] = []
    for weights, backend in configs:
        for batch, length in SHAPES:
            key = ((weights, backend), (batch, length))
            ms = statistics.median(graph[key])
            bf16 = statistics.median(graph[(("bf16", "cuBLAS"), (batch, length))])
            records.append(
                {
                    "model": model_id,
                    "weights": weights,
                    "backend": backend,
                    "batch": batch,
                    "prompt": length,
                    "eager_ms": statistics.median(eager[key]),
                    "graph_ms": ms,
                    "tokens_per_s": batch * length / ms * 1e3,
                    "speedup_vs_bf16": bf16 / ms,
                    "eager_speedup_vs_bf16": statistics.median(eager[(("bf16", "cuBLAS"), (batch, length))])
                    / statistics.median(eager[key]),
                    "perplexity": ppl[(weights, backend)],
                }
            )
    configs.clear()
    del base, w8a8, int4
    torch.cuda.empty_cache()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--calib-len", type=int, default=512)
    parser.add_argument("--merge", nargs="*", help="print and join the records of earlier --json files")
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    if args.merge:
        runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.merge]
        gpu, version = runs[0]["gpu"], runs[0]["torch"]
        records = [r for run in runs for r in run["records"]]
    else:
        torch.manual_seed(0)
        records = [r for m in args.models for r in bench_model(m, args.n_calib, args.calib_len)]
        gpu, version = torch.cuda.get_device_name(), torch.__version__
    print(
        f"Prefill on {gpu}: one forward pass, median ms over {ROUNDS} interleaved rounds of {PER_ROUND}; "
        f"graph = CUDA graph replay (GPU time); perplexity on the first {PPL_TOKENS // 1024}k WikiText-2 test tokens"
    )
    headers = ["model", "weights", "kernel", "batch x prompt", "eager ms", "eager vs bf16", "graph ms"]
    headers += ["graph tokens/s", "graph vs bf16", "perplexity"]
    rows = [
        [
            str(r["model"]).split("/")[1],
            r["weights"],
            r["backend"],
            f"{r['batch']} x {r['prompt']}",
            r["eager_ms"],
            f"{r['eager_speedup_vs_bf16']:.2f}x",
            r["graph_ms"],
            r["tokens_per_s"],
            f"{r['speedup_vs_bf16']:.2f}x",
            f"{r['perplexity']:.3f}",
        ]
        for r in records
    ]
    print(table(headers, rows))
    if args.json:
        payload = {"gpu": gpu, "torch": version, "shapes": SHAPES, "records": records}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
