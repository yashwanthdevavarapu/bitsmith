"""End-to-end greedy decode speed: Hugging Face ``generate`` against CUDA graph replay.

For each model: bf16 weights, and INT4 (group 64) weights on the reference runtime and on
the fused Triton kernel; ``--schemes int4 int3 int2`` adds the other low-bit formats. Each
is timed through ``model.generate`` (dynamic cache, one launch per kernel) and through
:class:`bitsmith.decode.GraphDecoder` (static cache, the decode step replayed as one CUDA
graph). Batch 1, 64-token prompt, 128 new tokens; the median of 5 runs after a warm-up.
The weights are round to nearest: calibration changes the codes, not the speed.

    python scripts/bench_decode.py --json results/decode_rtx3090.json > results/decode_rtx3090.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bitsmith.bench.hf import _require_hf, wikitext2_tokens
from bitsmith.decode import GraphDecoder, tokens_per_s
from bitsmith.kernels import triton_available
from bitsmith.nn import quantize_model, set_backend
from bitsmith.report import table

MODELS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
PROMPT, NEW, REPEATS = 64, 128, 5


def measure(model: torch.nn.Module, prompt: torch.Tensor) -> tuple[float, float, int, bool]:
    """(generate tok/s, graph tok/s, leading tokens equal to generate, graph equal to eager).

    Graph replay must give exactly the tokens of the same static-cache step run eagerly.
    Against ``generate`` (dynamic cache, other attention shapes) bf16 rounding differs, so
    a near tie in ``argmax`` can flip and the two continuations part after a while.
    """

    def hf() -> torch.Tensor:
        out = model.generate(  # type: ignore[operator]
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False
        )
        return out[:, prompt.shape[1] :]  # type: ignore[no-any-return]

    dec = GraphDecoder(model, PROMPT + NEW)
    graph_tokens = dec.generate(prompt, NEW)
    eager_tokens = GraphDecoder(model, PROMPT + NEW, use_graph=False).generate(prompt, NEW)
    differ = (hf() != graph_tokens)[0].nonzero()
    prefix = int(differ[0]) if differ.numel() else NEW
    return (
        tokens_per_s(hf, NEW, REPEATS),
        tokens_per_s(lambda: dec.generate(prompt, NEW), NEW, REPEATS),
        prefix,
        torch.equal(graph_tokens, eager_tokens),
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument(
        "--schemes", nargs="*", default=["int4"], help="weight formats on the triton backend (int4, int3, int2)"
    )
    parser.add_argument("--no-reference", action="store_true", help="skip the slow reference runtime rows")
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    transformers, _ = _require_hf()
    records = []
    for model_id in args.models:
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        prompt = wikitext2_tokens(tok, "test", 4096)[1000 : 1000 + PROMPT].unsqueeze(0).cuda()

        def load(model_id: str = model_id, eos: int = tok.eos_token_id) -> torch.nn.Module:
            m = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
            m.generation_config.pad_token_id = eos
            return m  # type: ignore[no-any-return]

        model = load()
        runs = [("bf16", "cuBLAS")]
        results = [measure(model, prompt)]
        for scheme in args.schemes:  # each scheme starts again from the bf16 weights
            del model
            torch.cuda.empty_cache()
            model = load()
            quantize_model(model, scheme=scheme, group_size=64)
            if not args.no_reference:
                runs.append((f"{scheme}-g64", "reference"))
                results.append(measure(model, prompt))
            if triton_available() and set_backend(model, "triton"):
                runs.append((f"{scheme}-g64", "fused Triton"))
                results.append(measure(model, prompt))
        for (weights, backend), (hf, graph, prefix, exact) in zip(runs, results, strict=True):
            records.append(
                {
                    "model": model_id,
                    "weights": weights,
                    "backend": backend,
                    "generate_tok_s": hf,
                    "graph_tok_s": graph,
                    "graph_speedup": graph / hf,
                    "prefix_equal_to_generate": prefix,
                    "graph_equals_eager": exact,
                }
            )
        del model
        torch.cuda.empty_cache()
    bf16 = {r["model"]: r["graph_tok_s"] for r in records if r["weights"] == "bf16"}
    rows = [
        [
            r["model"].split("/")[1],
            r["weights"],
            r["backend"],
            r["generate_tok_s"],
            r["graph_tok_s"],
            f"{r['graph_speedup']:.2f}x",
            f"{r['graph_tok_s'] / bf16[r['model']]:.2f}x",
            f"{r['prefix_equal_to_generate']}/{NEW}",
            "yes" if r["graph_equals_eager"] else "no",
        ]
        for r in records
    ]
    gpu = torch.cuda.get_device_name()
    print(f"Greedy decode on {gpu}, batch 1, {PROMPT}-token prompt, {NEW} new tokens, median of {REPEATS}")
    headers = ["model", "weights", "kernel", "generate tok/s", "graph tok/s", "graph gain", "vs bf16 graph"]
    headers += ["= generate", "graph = eager"]
    print(table(headers, rows))
    if args.json:
        payload = {"gpu": gpu, "torch": torch.__version__, "prompt": PROMPT, "new_tokens": NEW, "records": records}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
