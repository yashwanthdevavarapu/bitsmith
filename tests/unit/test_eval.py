from __future__ import annotations

import math

import pytest
import torch

from bitsmith.eval import (
    ErrorReport,
    MarkovCorpus,
    TinyLM,
    TinyLMConfig,
    cosine_similarity,
    mse,
    outlier_channels,
    sliding_window_perplexity,
    sqnr_db,
)

# metrics -------------------------------------------------------------------------------


def test_sqnr_known_values() -> None:
    x = torch.ones(1000)
    assert sqnr_db(x, x) == math.inf
    assert sqnr_db(x, x * 1.1) == pytest.approx(20.0)  # noise amplitude 10% -> 20 dB
    assert sqnr_db(x, x * 1.01) == pytest.approx(40.0)


def test_other_metrics() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 5.0])
    assert mse(a, b) == pytest.approx(4 / 3)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(torch.zeros(3), torch.zeros(3)) == 1.0
    rep = ErrorReport.compare(a, b)
    assert rep.max_abs_error == 2.0


def test_outlier_channel_detection() -> None:
    x = torch.randn(100, 64)
    x[:, [5, 40]] *= 30
    assert outlier_channels(x) == [5, 40]


# perplexity ----------------------------------------------------------------------------


def _uniform(vocab: int):  # type: ignore[no-untyped-def]
    return lambda ids: torch.zeros(1, ids.shape[1], vocab)


@pytest.mark.parametrize(("window", "stride"), [(16, 16), (16, 8), (32, 5), (1000, 1000)])
def test_uniform_model_has_perplexity_equal_to_vocab(window: int, stride: int) -> None:
    tokens = torch.randint(0, 50, (203,))
    res = sliding_window_perplexity(_uniform(50), tokens, window=window, stride=stride)
    assert res.perplexity == pytest.approx(50.0)
    assert res.n_tokens == 202  # every token after the first is scored exactly once


def test_context_free_model_is_window_invariant() -> None:
    corpus = MarkovCorpus(vocab_size=16, branching=3, seed=1)
    tokens = corpus.sample(500)
    log_t = corpus.transition.clamp_min(1e-12).log().to(torch.float32)

    def bigram(ids: torch.Tensor) -> torch.Tensor:
        return log_t[ids[0]].unsqueeze(0)

    a = sliding_window_perplexity(bigram, tokens, window=64, stride=32).perplexity
    b = sliding_window_perplexity(bigram, tokens, window=17, stride=3).perplexity
    assert a == pytest.approx(b, rel=1e-5)


def test_perplexity_validation() -> None:
    with pytest.raises(ValueError, match="1-D"):
        sliding_window_perplexity(_uniform(4), torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(ValueError, match="stride"):
        sliding_window_perplexity(_uniform(4), torch.zeros(10, dtype=torch.long), window=4, stride=8)


# corpus --------------------------------------------------------------------------------


def test_markov_corpus_structure() -> None:
    c = MarkovCorpus(vocab_size=32, branching=4, seed=3)
    torch.testing.assert_close(c.transition.sum(1), torch.ones(32, dtype=torch.float64))
    assert torch.all((c.transition > 0).sum(1) <= 4)
    assert 1.0 < c.optimal_perplexity() < 4.0
    assert torch.equal(c.sample(100), c.sample(100))
    assert not torch.equal(c.sample(100, seed=1), c.sample(100, seed=2))


def test_oracle_reaches_the_entropy_floor() -> None:
    c = MarkovCorpus(vocab_size=32, branching=4, seed=0)
    tokens = c.sample(30_000)
    log_t = c.transition.clamp_min(1e-12).log().to(torch.float32)
    res = sliding_window_perplexity(lambda ids: log_t[ids[0]].unsqueeze(0), tokens, window=512, stride=512)
    assert res.perplexity == pytest.approx(c.optimal_perplexity(), rel=0.03)


def test_corpus_validation() -> None:
    with pytest.raises(ValueError, match="branching"):
        MarkovCorpus(vocab_size=4, branching=5)


# tiny LM -------------------------------------------------------------------------------


def test_tiny_lm_shapes_and_length_guard() -> None:
    model = TinyLM(TinyLMConfig(vocab_size=20, dim=16, n_layers=1, n_heads=2, max_len=8))
    assert model(torch.zeros(2, 8, dtype=torch.long)).shape == (2, 8, 20)
    with pytest.raises(ValueError, match="max_len"):
        model(torch.zeros(1, 9, dtype=torch.long))
