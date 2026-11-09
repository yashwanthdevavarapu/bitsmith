"""Render the README figures from the JSON results and captured terminal output.

Run after scripts/reproduce.py:  python scripts/make_figures.py
Writes PNG files to docs/assets/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bitsmith.eval import sqnr_db
from bitsmith.nn import SCHEMES, make_codec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"

# Light theme tokens (validated categorical slots 1 to 3 plus neutral ink).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # round-to-nearest
ORANGE = "#eb6834"  # calibrated (AWQ, GPTQ, SmoothQuant)
AQUA = "#1baf7a"  # baseline

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


def sqnr_chart(out_name: str) -> None:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(1024, 1024, generator=g) * 0.02
    rows = []
    for scheme in SCHEMES:
        codec = make_codec(scheme, 128)
        bufs = codec.encode(w)
        bits = 8 * sum(t.numel() * t.element_size() for t in bufs.values()) / w.numel()
        rows.append((scheme, sqnr_db(w, codec.decode(bufs, *w.shape)), bits))
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=BLUE, height=0.6)
    ax.invert_yaxis()
    for b, (_, s, bits) in zip(bars, rows, strict=True):
        ax.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"  {s:.1f} dB  ({bits:.2f} bits/w)",
            va="center",
            color=INK,
            fontsize=9.5,
        )
    ax.set_xlim(0, max(r[1] for r in rows) * 1.45)
    ax.set_axisbelow(True)
    ax.set_xlabel("weight SQNR in dB (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Weight fidelity per format, 1024x1024 Gaussian weight", loc="left")
    _save(fig, out_name)


def terminal_shot(text_file: Path, out_name: str, title: str, max_lines: int = 60) -> None:
    """Render captured CLI output as a terminal style image."""
    lines = text_file.read_text(encoding="utf-8").rstrip().splitlines()[:max_lines]
    lines = [f"$ {title}", *lines]
    width = max(len(line) for line in lines)
    h = 0.19 * len(lines) + 0.5
    fig = plt.figure(figsize=(max(6.0, 0.075 * width + 0.6), h))
    fig.patch.set_facecolor("#1a1a19")
    for i, line in enumerate(lines):
        fig.text(
            0.02,
            1 - (i + 1.2) / (len(lines) + 1.5),
            line,
            family="monospace",
            fontsize=9.5,
            color="#7fd1a8" if i == 0 else "#f0efec",
        )
    _save(fig, out_name)


GPU_MODELS = [
    ("gpu_smollm2_135m.json", "SmolLM2-135M"),
    ("gpu_qwen2.5_0.5b.json", "Qwen2.5-0.5B"),
    ("gpu_qwen2.5_1.5b.json", "Qwen2.5-1.5B"),
]
MODEL_COLORS = [BLUE, ORANGE, AQUA]
CHART_CONFIGS = [
    "int8",
    "fp8-e4m3",
    "w8a8",
    "w8a8+smoothquant",
    "int4-g64",
    "int4-g64+awq",
    "int4-g64+gptq",
]


def multi_model_chart(out_name: str) -> None:
    """Perplexity increase per config, one bar per model (grouped horizontal bars)."""
    runs = []
    for fname, label in GPU_MODELS:
        path = RESULTS / fname
        if path.exists():
            pts = {p["label"]: p for p in json.loads(path.read_text(encoding="utf-8"))["points"]}
            runs.append((label, pts))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 6.2))
    bar_h = 0.8 / len(runs)
    for j, ((label, pts), color) in enumerate(zip(runs, MODEL_COLORS, strict=False)):
        ys = [i + j * bar_h for i in range(len(CHART_CONFIGS))]
        vals = [pts[c]["delta_pct"] if c in pts else 0.0 for c in CHART_CONFIGS]
        bars = ax.barh(ys, vals, height=bar_h * 0.9, color=color, label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.1f}%", va="center", color=INK, fontsize=8.5)
    ax.set_yticks([i + bar_h * (len(runs) - 1) / 2 for i in range(len(CHART_CONFIGS))])
    ax.set_yticklabels(CHART_CONFIGS)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WikiText-2 perplexity increase over bf16, % (lower is better)")
    ax.legend(loc="lower right")
    ax.set_title("Cost of each format across three models (RTX 3090, full test set)", loc="left")
    _save(fig, out_name)


def kernel_chart(result_file: Path, out_name: str) -> None:
    data = json.loads(result_file.read_text(encoding="utf-8"))
    recs = [r for r in data["records"] if r["rows"] == 1]
    labels = [r["layer"] for r in recs]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    h = 0.38
    ys = range(len(recs))
    b1 = ax.barh(
        [y - h / 2 for y in ys], [r["fp16_us"] for r in recs], height=h * 0.92, color=BLUE, label="fp16 cuBLAS"
    )
    b2 = ax.barh(
        [y + h / 2 for y in ys],
        [r["int4_fused_us"] for r in recs],
        height=h * 0.92,
        color=ORANGE,
        label="int4 fused Triton kernel",
    )
    for bars, key in ((b1, "fp16_us"), (b2, "int4_fused_us")):
        for b, r in zip(bars, recs, strict=True):
            ax.text(
                b.get_width(), b.get_y() + b.get_height() / 2, f" {r[key]:.0f}", va="center", color=INK, fontsize=8.5
            )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("microseconds per call, batch 1 (CUDA graph replay, lower is better)")
    ax.set_xlim(0, max(r["fp16_us"] for r in recs) * 1.12)
    ax.legend(loc="upper right")
    ax.set_title(f"Decode GEMV on {data['gpu'].replace('NVIDIA GeForce ', '')}: fused INT4 vs fp16", loc="left")
    _save(fig, out_name)


def main() -> None:
    multi_model_chart("quality_gpu_models.png")
    if (RESULTS / "kernels_rtx3090.json").exists():
        kernel_chart(RESULTS / "kernels_rtx3090.json", "kernels_rtx3090.png")
    sqnr_chart("sqnr_formats.png")
    shots = [
        ("cli_pack_int4.txt", "terminal_pack.png", "bitsmith pack --format int4"),
        ("cli_sqnr.txt", "terminal_sqnr.png", "bitsmith sqnr --shape 1024x1024 --outliers 8"),
        ("tiny_sweep.txt", "terminal_sweep.png", "bitsmith sweep"),
        (
            "gpu_qwen2.5_1.5b.txt",
            "terminal_hf.png",
            "bitsmith hf --model Qwen/Qwen2.5-1.5B --device cuda --dtype bfloat16",
        ),
        ("kernels_rtx3090.txt", "terminal_kernels.png", "python scripts/bench_kernels.py"),
    ]
    for src, dst, cmd in shots:
        if (RESULTS / src).exists():
            terminal_shot(RESULTS / src, dst, cmd)


if __name__ == "__main__":
    main()
