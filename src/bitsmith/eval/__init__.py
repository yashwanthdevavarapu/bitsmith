"""Evaluation: tensor metrics, perplexity, synthetic corpus and a tiny reference model."""

from bitsmith.eval.corpus import MarkovCorpus
from bitsmith.eval.metrics import (
    ErrorReport,
    cosine_similarity,
    max_abs_error,
    mse,
    outlier_channels,
    sqnr_db,
)
from bitsmith.eval.perplexity import PerplexityResult, sliding_window_perplexity
from bitsmith.eval.tiny_lm import TinyLM, TinyLMConfig, train_tiny_lm

__all__ = [
    "ErrorReport",
    "MarkovCorpus",
    "PerplexityResult",
    "TinyLM",
    "TinyLMConfig",
    "cosine_similarity",
    "max_abs_error",
    "mse",
    "outlier_channels",
    "sliding_window_perplexity",
    "sqnr_db",
    "train_tiny_lm",
]
