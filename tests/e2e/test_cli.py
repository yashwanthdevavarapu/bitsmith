"""Drive the command line interface the way a user would."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bitsmith import __version__
from bitsmith.cli import main


def test_version_via_module_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "bitsmith", "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == f"bitsmith {__version__}"


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "int4" in out and "rtx3050-laptop" in out


@pytest.mark.parametrize("fmt", ["int4", "ternary", "ternary-b3", "sparse24"])
def test_pack(fmt: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pack", "--format", fmt, "--seed", "3"]) == 0
    assert "round trip" in capsys.readouterr().out


def test_sqnr(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sqnr", "--shape", "64x256", "--outliers", "2", "--group-size", "64"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n|") >= 12  # header, separator and 11 schemes
    assert "| nf4 " in out and "| mxfp4 " in out


def test_sqnr_rejects_bad_shape() -> None:
    with pytest.raises(SystemExit):
        main(["sqnr", "--shape", "64by64"])


def test_roofline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["roofline", "--params", "1e9", "--hardware", "t4"]) == 0
    out = capsys.readouterr().out
    assert "Tesla T4" in out and "memory" in out


@pytest.mark.slow
def test_sweep_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "sweep.json"
    args = [
        "sweep",
        "--vocab",
        "32",
        "--dim",
        "32",
        "--layers",
        "1",
        "--steps",
        "40",
        "--train-tokens",
        "4000",
        "--eval-tokens",
        "400",
        "--window",
        "64",
        "--group-size",
        "32",
        "--json",
        str(out_file),
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "Pareto frontier" in text
    data = json.loads(out_file.read_text())
    labels = [p["label"] for p in data["points"]]
    assert labels[0] == "fp32" and "int4-g32+gptq" in labels and "ternary" in labels
    assert data["optimal_perplexity"] > 1
    assert all(p["perplexity"] > 1 for p in data["points"])
