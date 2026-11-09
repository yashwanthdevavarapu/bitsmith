from __future__ import annotations

import pytest
import torch
from torch import nn

from bitsmith.bench import (
    HARDWARE,
    SweepConfig,
    SweepPoint,
    decode_roofline,
    default_configs,
    pareto_front,
    run_sweep,
    sweep_rows,
    time_forward,
)
from bitsmith.report import scatter, table


def _point(label: str, bits: float, ppl: float) -> SweepPoint:
    return SweepPoint(label, "x", "rtn", bits, 0.0, 1.0, ppl, 0.0, 0.0, 0.0)


def test_pareto_front() -> None:
    pts = [
        _point("fp16", 16, 10.0),
        _point("int8", 8, 10.0),  # dominates fp16
        _point("int4", 4, 11.0),
        _point("bad4", 4.5, 12.0),  # dominated by int4
        _point("ternary", 2, 30.0),
    ]
    assert pareto_front(pts) == {"int8", "int4", "ternary"}


def test_roofline_numbers() -> None:
    hw = HARDWARE["rtx3050-laptop"]
    est = decode_roofline(0.5e9, 16.0, hw)
    assert est.weight_gb == pytest.approx(1.0)
    assert est.ms_per_token == pytest.approx(1e3 / 192, rel=1e-6)
    assert est.intensity == pytest.approx(1.0)
    assert est.bound == "memory"
    int4 = decode_roofline(0.5e9, 4.0, hw)
    assert int4.tokens_per_s / est.tokens_per_s == pytest.approx(4.0)
    assert hw.ridge_point > est.intensity


def test_roofline_compute_bound_and_validation() -> None:
    from bitsmith.bench import Hardware

    slow_chip = Hardware("toy", bandwidth_gbs=1e6, peak_tflops=0.001)
    assert decode_roofline(1e9, 4, slow_chip).bound == "compute"
    with pytest.raises(ValueError, match="positive"):
        decode_roofline(0, 4, slow_chip)


def test_default_configs_start_with_baseline() -> None:
    cfgs = default_configs(64)
    assert cfgs[0].scheme == "baseline"
    assert len({c.label for c in cfgs}) == len(cfgs)


class _Bigram(nn.Module):
    def __init__(self, vocab: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, 32)
        self.proj = nn.Linear(32, 32)
        self.lm_head = nn.Linear(32, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(torch.relu(self.proj(self.emb(ids))))


def test_run_sweep_small() -> None:
    model = _Bigram().eval()
    tokens = torch.randint(0, 16, (300,))
    calib = [tokens[:64].unsqueeze(0)]
    seen: list[str] = []
    points = run_sweep(
        model,
        tokens,
        calib,
        [SweepConfig("fp32", "baseline"), SweepConfig("int8", "int8"), SweepConfig("int4", "int4", group_size=32)],
        window=64,
        stride=32,
        progress=seen.append,
        on_model=lambda label, m: seen.append(f"model:{label}"),
    )
    assert [p.label for p in points] == ["fp32", "int8", "int4"]
    assert seen[:2] == ["fp32", "model:fp32"]
    assert points[0].bits_per_weight == 32.0
    assert points[0].delta_pct == 0.0
    assert points[1].compression == pytest.approx(32 / (8 + 16 / 32))
    assert any(p.on_frontier for p in points)
    rows = sweep_rows(points)
    assert len(rows) == 3 and len(rows[0]) == 9


def test_time_forward_positive() -> None:
    assert time_forward(nn.Linear(4, 4), torch.randn(1, 4), repeats=2) >= 0


def test_table_rendering() -> None:
    out = table(["name", "value"], [["a", 1.23456], ["bb", 12345.0], ["c", float("inf")], ["d", 7]])
    lines = out.splitlines()
    assert lines[0].startswith("| name")
    assert lines[1] == "|------|-------:|"
    assert "12,345" in out and "inf" in out and "1.235" in out


def test_scatter_rendering() -> None:
    out = scatter([(2, 30, "ternary"), (4, 11, "int4"), (16, 10, "fp16")], log_x=True, log_y=True, highlight={"int4"})
    assert "B = int4" in out
    assert "a = ternary" in out
    assert scatter([]) == "(no points)"
