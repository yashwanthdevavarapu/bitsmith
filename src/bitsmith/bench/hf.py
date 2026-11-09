"""Evaluate quantization configs on a real Hugging Face causal LM with WikiText-2.

Requires the optional ``hf`` extra (``transformers`` and ``datasets``).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from bitsmith.bench.pareto import SweepConfig, SweepPoint, run_sweep
from bitsmith.kernels import triton_available
from bitsmith.nn import set_backend

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _require_hf() -> tuple[Any, Any]:
    try:
        import datasets
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("install the hf extra: pip install 'bitsmith[hf]'") from exc
    return transformers, datasets


def wikitext2_tokens(tokenizer: Any, split: str = "test", max_tokens: int | None = None) -> Tensor:
    """Tokenize the WikiText-2 split joined with blank lines, as is standard."""
    _, datasets = _require_hf()
    ds = datasets.load_dataset(*WIKITEXT, split=split)
    text = "\n\n".join(ds["text"])
    if max_tokens:  # avoid tokenizing the whole train split when only a slice is needed
        text = text[: max_tokens * 8]
    ids = torch.tensor(tokenizer(text)["input_ids"], dtype=torch.long)
    return ids[:max_tokens] if max_tokens else ids


def calibration_batches(tokens: Tensor, n: int, seq_len: int, seed: int = 0) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, tokens.numel() - seq_len, (n,), generator=g)
    return [tokens[s : s + seq_len].unsqueeze(0) for s in starts.tolist()]


@torch.no_grad()
def decode_tokens_per_s(model: nn.Module, prompt: Tensor, new_tokens: int = 32, repeats: int = 3) -> float:
    """Greedy decode speed with the KV cache, median over ``repeats`` runs."""
    prompt = prompt.to(next(model.parameters()).device)
    rates = []
    for i in range(repeats + 1):  # the first run warms up kernels and is discarded
        if prompt.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(  # type: ignore[operator]
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
        if prompt.is_cuda:
            torch.cuda.synchronize()
        if i:
            rates.append(new_tokens / (time.perf_counter() - t0))
    return statistics.median(rates)


@dataclass
class HFResult:
    model_id: str
    n_params: int
    eval_tokens: int
    points: list[SweepPoint]
    decode_tps: dict[str, float]
    decode_tps_fused: dict[str, float]
    device: str
    dtype: str


def run_hf_sweep(
    model_id: str,
    configs: Sequence[SweepConfig],
    *,
    eval_tokens: int | None = 8192,
    n_calib: int = 16,
    calib_len: int = 256,
    window: int = 512,
    stride: int = 256,
    decode_new_tokens: int = 32,
    device: str = "cpu",
    dtype: str = "float32",
    progress: Callable[[str], None] | None = None,
) -> HFResult:
    """Sweep ``configs`` on a Hugging Face causal LM. ``eval_tokens=None`` uses the full test split."""
    transformers, _ = _require_hf()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    torch_dtype = getattr(torch, dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device).eval()
    test = wikitext2_tokens(tok, "test", eval_tokens)
    train = wikitext2_tokens(tok, "train", 200_000)
    calib = calibration_batches(train, n_calib, calib_len)
    prompt = test[:64].unsqueeze(0)

    decode: dict[str, float] = {}
    decode_fused: dict[str, float] = {}

    def measure(label: str, m: nn.Module) -> None:
        decode[label] = decode_tokens_per_s(m, prompt, decode_new_tokens)
        if device.startswith("cuda") and triton_available() and set_backend(m, "triton"):
            decode_fused[label] = decode_tokens_per_s(m, prompt, decode_new_tokens)
            set_backend(m, "reference")

    def logits(m: nn.Module, ids: Tensor) -> Tensor:
        return m(ids).logits  # type: ignore[no-any-return]

    points = run_sweep(
        model,
        test,
        calib,
        configs,
        logits_fn=logits,
        window=window,
        stride=stride,
        timing_input=test[:window].unsqueeze(0),
        progress=progress,
        on_model=measure,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return HFResult(model_id, n_params, int(test.numel()), points, decode, decode_fused, device, dtype)
