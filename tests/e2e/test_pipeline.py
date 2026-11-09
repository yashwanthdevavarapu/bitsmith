"""Full pipeline on a trained model: train, calibrate, quantize, evaluate, save and reload."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from bitsmith.calib import calibrate_and_quantize
from bitsmith.eval import MarkovCorpus, TinyLMConfig, sliding_window_perplexity, train_tiny_lm
from bitsmith.nn import QLinear, quantize_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, float]:
    corpus = MarkovCorpus(vocab_size=64, branching=4, seed=0)
    train = corpus.sample(30_000, seed=1)
    test = corpus.sample(2_000, seed=2)
    cfg = TinyLMConfig(vocab_size=64, dim=64, n_layers=2, n_heads=4, max_len=64)
    model = train_tiny_lm(train, cfg, steps=250, batch_size=32, seq_len=64, seed=0)
    return model, train, test, corpus.optimal_perplexity()


def _ppl(model: torch.nn.Module, tokens: torch.Tensor) -> float:
    return sliding_window_perplexity(model, tokens, window=64, stride=32).perplexity


def test_model_learns_the_corpus(trained) -> None:  # type: ignore[no-untyped-def]
    model, _, test, floor = trained
    ppl = _ppl(model, test)
    assert ppl < 64 / 4  # far better than uniform
    assert ppl > floor * 0.97  # cannot beat the entropy floor


def test_quality_ordering_across_schemes(trained) -> None:  # type: ignore[no-untyped-def]
    model, train, test, _ = trained
    calib = [train[i * 64 : (i + 1) * 64].unsqueeze(0) for i in range(8)]
    base = _ppl(model, test)
    results = {}
    for label, method, scheme in [
        ("int8", "rtn", "int8"),
        ("int4", "rtn", "int4"),
        ("int4+gptq", "gptq", "int4"),
        ("int4+awq", "awq", "int4"),
        ("ternary", "rtn", "ternary"),
    ]:
        m = copy.deepcopy(model)
        calibrate_and_quantize(m, calib, method=method, scheme=scheme, group_size=64)  # type: ignore[arg-type]
        results[label] = _ppl(m, test)
    assert results["int8"] == pytest.approx(base, rel=0.005)
    assert results["int4"] < base * 1.10
    assert results["int4+gptq"] <= results["int4"] * 1.01
    assert results["int4+awq"] <= results["int4"] * 1.01
    assert results["ternary"] > results["int4"]


def test_quantized_checkpoint_round_trip(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model, _, test, _ = trained
    q1 = copy.deepcopy(model)
    quantize_model(q1, "int4", group_size=32)
    path = tmp_path / "int4.pt"
    torch.save(q1.state_dict(), path)

    q2 = copy.deepcopy(model)
    quantize_model(q2, "int4", group_size=32)  # same structure, then load packed buffers
    q2.load_state_dict(torch.load(path, weights_only=True))
    ids = test[:64].unsqueeze(0)
    assert torch.equal(q1(ids), q2(ids))
    packed = [m for m in q2.modules() if isinstance(m, QLinear)]
    assert packed and all(m.qweight.dtype == torch.uint8 for m in packed)
    # the packed checkpoint is much smaller than the float one
    float_path = tmp_path / "fp32.pt"
    torch.save(model.state_dict(), float_path)
    assert path.stat().st_size < 0.6 * float_path.stat().st_size
