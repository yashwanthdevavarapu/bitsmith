"""Real model smoke test. Opt in with BITSMITH_HF=1 (downloads about 270 MB)."""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(os.environ.get("BITSMITH_HF") != "1", reason="set BITSMITH_HF=1 to run"),
]


def test_smollm2_int8_is_near_lossless() -> None:
    pytest.importorskip("transformers")
    from bitsmith.bench import SweepConfig
    from bitsmith.bench.hf import run_hf_sweep

    res = run_hf_sweep(
        "HuggingFaceTB/SmolLM2-135M",
        [SweepConfig("fp32", "baseline"), SweepConfig("int8", "int8")],
        eval_tokens=1024,
        decode_new_tokens=4,
    )
    base, int8 = res.points
    assert base.perplexity < 60
    assert int8.delta_pct < 1.0
    assert int8.compression > 3.9
