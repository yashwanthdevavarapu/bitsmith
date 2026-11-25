"""Render the README figures from the JSON results and captured terminal output.

Run after scripts/reproduce.py:  python scripts/make_figures.py
Writes PNG files to docs/assets/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

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
    ("smollm2_135m", "SmolLM2-135M"),
    ("qwen2.5_0.5b", "Qwen2.5-0.5B"),
    ("qwen2.5_1.5b", "Qwen2.5-1.5B"),
]
# each model has a main sweep plus sweeps for the formats added later; all share one baseline
GPU_FILES = [
    "gpu_{}.json",
    "gpu_formats_{}.json",
    "gpu_formats_awq_{}.json",
    "gpu_formats_gptq_{}.json",
    "gpu_lowbit_{}.json",
    "gpu_smoothquant_{}.json",  # W8A8 + SmoothQuant with the per-layer search (ADR 0018) replaces older runs
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
    "nf4-g64",
    "nf4-g64+awq",
    "nf4-g64+gptq",
    "mxfp4",
    "mxfp4+awq",
    "mxfp4+gptq",
]


def _gpu_runs() -> list[tuple[str, dict[str, dict]]]:
    """Every GPU result point per model, merged across the sweep files."""
    runs = []
    for tag, label in GPU_MODELS:
        pts: dict[str, dict] = {}
        for pattern in GPU_FILES:
            path = RESULTS / pattern.format(tag)
            if path.exists():
                pts |= {p["label"]: p for p in json.loads(path.read_text(encoding="utf-8"))["points"]}
        if pts:
            runs.append((label, pts))
    return runs


def _frontier(points: list[dict]) -> list[dict]:
    """Points no other point beats on both bits per weight and perplexity, by bits."""
    best: list[dict] = []
    for p in sorted(points, key=lambda p: (p["bits_per_weight"], p["delta_pct"])):
        if not best or p["delta_pct"] < best[-1]["delta_pct"]:
            best.append(p)
    return best


NEEDS_TRAINING = ("ternary", "sparse24")  # they break any model not trained for them


def frontier_chart(out_name: str) -> None:
    """Perplexity increase against bits per weight: the best config at each size, per model."""
    runs = _gpu_runs()
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, pts), color in zip(runs, MODEL_COLORS, strict=False):
        cands = [p for p in pts.values() if p["label"] != "bf16" and not p["label"].startswith(NEEDS_TRAINING)]
        ax.scatter(
            [p["bits_per_weight"] for p in cands],
            [p["delta_pct"] for p in cands],
            s=14,
            color=color,
            alpha=0.3,
            linewidths=0,
        )
        front = _frontier(cands)
        xs = [p["bits_per_weight"] for p in front]
        ys = [p["delta_pct"] for p in front]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        if label == runs[-1][0]:  # name the configs once, on the largest model
            for p in front:
                ax.annotate(
                    p["label"],
                    (p["bits_per_weight"], p["delta_pct"]),
                    xytext=(6, -12),
                    textcoords="offset points",
                    color=INK_2,
                    fontsize=8.5,
                )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}%" if v >= 1 else f"{v:g}%"))
    ax.set_xlabel("bits per weight, all scales and metadata included")
    ax.set_ylabel("perplexity increase over bf16 (log)")
    ax.legend(loc="upper right")
    ax.set_title("Best quality at each size: the Pareto frontier per model", loc="left")
    _save(fig, out_name)


def multi_model_chart(out_name: str) -> None:
    """Perplexity increase per config, one bar per model (grouped horizontal bars)."""
    runs = _gpu_runs()
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 6.2))
    bar_h = 0.8 / len(runs)
    for j, ((label, pts), color) in enumerate(zip(runs, MODEL_COLORS, strict=False)):
        ys = [i + j * bar_h for i in range(len(CHART_CONFIGS))]
        have = [(y, pts[c]["delta_pct"]) for y, c in zip(ys, CHART_CONFIGS, strict=True) if c in pts]
        bars = ax.barh([y for y, _ in have], [v for _, v in have], height=bar_h * 0.9, color=color, label=label)
        for b, (_, v) in zip(bars, have, strict=True):
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


def _keys_per_channel(spec_name: str) -> bool:
    """``k4cv4`` -> True: a ``c`` right after the key bits means per-channel keys."""
    return "c" in spec_name.split("v")[0]


def kv_chart(out_name: str) -> None:
    """Perplexity cost of a quantized KV cache against its bits, one panel per model."""
    runs = []
    for tag, label in GPU_MODELS:
        path = RESULTS / f"kv_{tag}.json"
        if path.exists():
            runs.append((label, json.loads(path.read_text(encoding="utf-8"))["points"]))
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.8), sharey=True, squeeze=False)
    series = [("keys per token", BLUE, False), ("keys per channel", ORANGE, True)]
    for ax, (label, pts) in zip(axes[0], runs, strict=True):
        quant = [p for p in pts if not p["label"].endswith("cache")]
        for name, color, per_channel in series:
            sel = sorted(
                (p for p in quant if _keys_per_channel(p["label"]) == per_channel), key=lambda p: p["key_bits"]
            )
            ax.plot(
                [p["key_bits"] for p in sel],
                [max(p["delta_pct"], 0.01) for p in sel],
                color=color,
                lw=2,
                marker="o",
                ms=6,
                mec=SURFACE,
                mew=1.5,
                label=name,
            )
            for p in sel:
                ax.annotate(
                    p["label"],
                    (p["key_bits"], max(p["delta_pct"], 0.01)),
                    xytext=(5, 4),
                    textcoords="offset points",
                    color=INK_2,
                    fontsize=8,
                )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}%" if v >= 1 else f"{v:g}%")
        )
        ax.set_title(label, loc="left", fontsize=11.5)
        ax.set_xlabel("cache bits per element")
    axes[0][0].set_ylabel("perplexity increase over a bf16 cache")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("KV-cache quantization: keys grouped per token or per channel", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_name)


def _weights_short(point: dict) -> str:
    return "bf16" if point["weights"] == "bf16" else point["weights"].split("-")[0]


def memory_chart(out_name: str) -> None:
    """Model plus KV cache memory at 32k tokens, stacked, with the perplexity cost on top."""
    runs = []
    for tag, label in GPU_MODELS:
        path = RESULTS / f"kv_nf4_{tag}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append((label, data["context"], data["points"]))
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4.2), squeeze=False)
    for ax, (label, context, pts) in zip(axes[0], runs, strict=True):
        names = [_weights_short(p) + "\n" + p["label"].replace(" cache", "") for p in pts]
        model = [p["model_mib"] / 1024 for p in pts]
        cache = [(p["total_mib"] - p["model_mib"]) / 1024 for p in pts]
        xs = range(len(pts))
        ax.bar(xs, model, color=BLUE, width=0.7, label="model")
        ax.bar(xs, cache, bottom=model, color=ORANGE, width=0.7, label=f"KV cache, {context // 1024}k tokens")
        for x, m, c, p in zip(xs, model, cache, pts, strict=True):
            tag = "baseline" if x == 0 else f"+{p['delta_pct']:.1f}%"
            ax.text(x, m + c, tag, ha="center", va="bottom", color=INK, fontsize=8.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(label, loc="left", fontsize=11.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(m + c for m, c in zip(model, cache, strict=True)) * 1.15)
    axes[0][0].set_ylabel("GiB")
    axes[0][0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Weights and KV cache together: memory for one sequence, perplexity cost on top",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_name)


def prefill_chart(result_file: Path, out_name: str) -> None:
    """End-to-end prefill speed against bf16, per model and prompt shape."""
    if not result_file.exists():
        return
    data = json.loads(result_file.read_text(encoding="utf-8"))
    order = [label for _, label in GPU_MODELS]  # smallest model first
    names = {r["model"] for r in data["records"]}
    models = sorted(names, key=lambda m: order.index(m.split("/")[1]) if m.split("/")[1] in order else len(order))
    series = [
        (("w8a8 smoothquant", "int8 tensor cores"), "W8A8, INT8 tensor cores", ORANGE),
        (("int4-g64", "fused Triton"), "INT4, Triton backend", BLUE),
        (("w8a8 smoothquant", "simulated"), "W8A8, simulated", INK_2),
    ]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models, strict=True):
        recs = [r for r in data["records"] if r["model"] == model]
        shapes = list(dict.fromkeys((r["batch"], r["prompt"]) for r in recs))
        width = 0.8 / len(series)
        for i, (key, label, color) in enumerate(series):
            by_shape = {(r["batch"], r["prompt"]): r for r in recs if (r["weights"], r["backend"]) == key}
            xs = [j + (i - (len(series) - 1) / 2) * width for j, sh in enumerate(shapes) if sh in by_shape]
            ys = [by_shape[sh]["speedup_vs_bf16"] for sh in shapes if sh in by_shape]
            ax.bar(xs, ys, width=width, color=color, label=label)
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([f"{b} x {t}" for b, t in shapes], fontsize=8.5)
        ax.set_xlabel("batch x prompt tokens")
        ax.set_title(model.split("/")[1], loc="left", fontsize=11.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("GPU time speedup vs bf16")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        f"Prefill speed against bf16, CUDA graph replay on a shared {data['gpu'].replace('NVIDIA GeForce ', '')}",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_name)


def int4_prefill_chart(result_file: Path, out_name: str) -> None:
    """INT4 layers with many rows: fused GEMM against dequantize + cuBLAS, as a share of fp16 speed."""
    if not result_file.exists():
        return
    data = json.loads(result_file.read_text(encoding="utf-8"))
    layers = list(dict.fromkeys(r["layer"] for r in data["records"]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, layers, strict=False):
        recs = [r for r in data["records"] if r["layer"] == layer]
        rows = [r["rows"] for r in recs]
        for key, label, color in (
            ("dequant_cublas_us", "dequantize + cuBLAS", ORANGE),
            ("fused_gemm_us", "fused Triton GEMM", BLUE),
        ):
            ax.plot(rows, [r["fp16_us"] / r[key] for r in recs], marker="o", ms=4, color=color, label=label)
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rows)
        ax.set_xticklabels([str(r) for r in rows], fontsize=8)
        ax.set_title(layer.replace(" (", "\n("), loc="left", fontsize=10.5)
        ax.set_ylim(0, 1.15)
    for ax in axes[1]:
        ax.set_xlabel("activation rows (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed vs fp16 cuBLAS")
    axes[0][0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        f"INT4 layers with many rows on a shared {data['gpu'].replace('NVIDIA GeForce ', '')}: "
        "unpack once, then cuBLAS",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_name)


def lowbit_chart(result_file: Path, out_name: str) -> None:
    """Decode-shaped layers: INT4, INT3 and INT2 fused kernels as a share of fp16 speed."""
    if not result_file.exists():
        return
    data = json.loads(result_file.read_text(encoding="utf-8"))
    panels = [1, 4]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4.4), sharey=True, squeeze=False)
    series = (("int4", "INT4", BLUE), ("int3", "INT3", AQUA), ("int2", "INT2", ORANGE))
    for ax, rows in zip(axes[0], panels, strict=True):
        recs = [r for r in data["records"] if r["rows"] == rows]
        width = 0.8 / len(series)
        for i, (key, label, color) in enumerate(series):
            xs = [j + (i - 1) * width for j in range(len(recs))]
            ax.bar(xs, [r["fp16_us"] / r[f"{key}_us"] for r in recs], width=width, color=color, label=label)
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.set_xticks(range(len(recs)))
        short = [r["layer"].replace("qwen2.5-1.5b", "1.5B").replace("7b", "7B").replace(" (", "\n")[:-1] for r in recs]
        ax.set_xticklabels(short, fontsize=8)
        ax.set_title(f"{rows} activation row{'s' if rows > 1 else ''}", loc="left", fontsize=11.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("speed vs fp16 cuBLAS")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        f"Fused low-bit kernels for decoding, group {data['group_size']}, shared "
        f"{data['gpu'].replace('NVIDIA GeForce ', '')}",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
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
    frontier_chart("frontier_gpu_models.png")
    kv_chart("kv_gpu_models.png")
    memory_chart("memory_gpu_models.png")
    int4_prefill_chart(RESULTS / "int4_prefill_rtx3090_shared.json", "int4_prefill_rtx3090.png")
    lowbit_chart(RESULTS / "lowbit_kernels_rtx3090_shared.json", "lowbit_kernels_rtx3090.png")
    clean = RESULTS / "prefill_rtx3090.json"  # an idle GPU run replaces the shared one
    prefill_chart(clean if clean.exists() else RESULTS / "prefill_rtx3090_shared.json", "prefill_rtx3090.png")
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
