"""Sliding-window perplexity for causal language models.

The token stream is covered by windows of at most ``window`` tokens. Each new window ends
``stride`` tokens after the previous one and scores only those new tokens, so every token
after the first is counted exactly once and gets at least ``window - stride`` tokens of
left context (except near the start). ``stride`` is capped at ``window - 1`` so that even
non-overlapping settings keep one token of context and never skip a target.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

LogitsFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class PerplexityResult:
    perplexity: float
    nll: float  # mean negative log likelihood per token, in nats
    n_tokens: int
    n_windows: int


@torch.no_grad()
def sliding_window_perplexity(
    logits_fn: LogitsFn,
    tokens: Tensor,
    *,
    window: int = 512,
    stride: int = 256,
    device: torch.device | str | None = None,
) -> PerplexityResult:
    """Compute perplexity of ``tokens`` (1-D LongTensor) under ``logits_fn``.

    ``logits_fn`` takes ``(1, T)`` token ids and returns ``(1, T, vocab)`` logits.
    """
    if tokens.dim() != 1 or tokens.numel() < 2:
        raise ValueError("tokens must be a 1-D tensor with at least 2 elements")
    if window < 2 or not 0 < stride <= window:
        raise ValueError("require window >= 2 and 0 < stride <= window")
    tokens = tokens.to(device) if device is not None else tokens
    n = tokens.numel()
    step = min(stride, window - 1)  # leave at least one token of context for the first new target
    total_nll = 0.0
    total = 0
    windows = 0
    prev_end = 0
    while prev_end < n:
        end = min(window if prev_end == 0 else prev_end + step, n)
        begin = max(0, end - window)
        ids = tokens[begin:end].unsqueeze(0)
        logits = logits_fn(ids).to(torch.float32)
        # Position t predicts token t + 1, so targets are the new tokens in [prev_end, end).
        target_pos = torch.arange(max(prev_end, begin + 1), end, device=ids.device) - begin
        pred = logits[0, target_pos - 1]
        tgt = ids[0, target_pos]
        total_nll += float(F.cross_entropy(pred, tgt, reduction="sum"))
        total += int(target_pos.numel())
        windows += 1
        prev_end = end
    nll = total_nll / total
    return PerplexityResult(perplexity=math.exp(nll), nll=nll, n_tokens=total, n_windows=windows)
