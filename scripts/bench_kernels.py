"""Measure INT4 linear layers on a GPU against fp16 cuBLAS and the bandwidth roofline.

For each layer shape and batch size this times three implementations:

* ``fp16``       ``F.linear`` with an fp16 weight (cuBLAS)
* ``int4-ref``   bitsmith reference runtime: unpack to fp16, then ``F.linear``
* ``int4-fused`` bitsmith Triton kernel that dequantizes inside the matmul

and reports the effective memory bandwidth for single-token decode (rows = 1), where the
weight read dominates. Requires CUDA and Triton.

Run: python scripts/bench_kernels.py --json results/kernels_rtx3090.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bitsmith.bench import HARDWARE
from bitsmith.nn import QLinear
from bitsmith.report import table

SHAPES = {
    "qwen2.5-1.5b attn (1536x1536)": (1536, 1536),
    "qwen2.5-1.5b mlp up (8960x1536)": (8960, 1536),
    "qwen2.5-1.5b mlp down (1536x8960)": (1536, 8960),
    "7b attn (4096x4096)": (4096, 4096),
    "7b mlp up (11008x4096)": (11008, 4096),
    "7b mlp down (4096x11008)": (4096, 11008),
}


def bench(fn, iters: int = 200, warmup: int = 20) -> float:  # type: ignore[no-untyped-def]
    """Median time in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def bench_graph(fn) -> float:  # type: ignore[no-untyped-def]
    """Time ``fn`` replayed from a CUDA graph: pure GPU time, no Python launch overhead."""
    fn()  # triggers Triton autotuning outside the capture
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return bench(graph.replay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="rtx3090", choices=list(HARDWARE))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rows", type=int, nargs="*", default=[1, 16, 512])
    parser.add_argument("--json")
    args = parser.parse_args()
    hw = HARDWARE[args.hardware]
    dev = torch.device("cuda")
    torch.manual_seed(0)

    records = []
    for label, (n, k) in SHAPES.items():
        lin = nn.Linear(k, n, bias=False).to(dev, torch.float16)
        q = QLinear.from_linear(lin, "int4", args.group_size)
        fp16_bytes = n * k * 2
        int4_bytes = q.weight_bytes()
        for m in args.rows:
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            q.set_backend("reference")
            t_ref = bench_graph(lambda x=x, q=q: q(x))
            t_fp16 = bench_graph(lambda x=x, lin=lin: F.linear(x, lin.weight))
            q.set_backend("triton")
            t_fused = bench_graph(lambda x=x, q=q: q(x))
            t_fused_eager = bench(lambda x=x, q=q: q(x))
            records.append(
                {
                    "layer": label,
                    "rows": m,
                    "fp16_us": t_fp16,
                    "int4_ref_us": t_ref,
                    "int4_fused_us": t_fused,
                    "int4_fused_eager_us": t_fused_eager,
                    "speedup_vs_fp16": t_fp16 / t_fused,
                    "fp16_gbs": fp16_bytes / t_fp16 / 1e3,
                    "int4_fused_gbs": int4_bytes / t_fused / 1e3,
                    "fp16_roofline_us": fp16_bytes / (hw.bandwidth_gbs * 1e3),
                    "int4_roofline_us": int4_bytes / (hw.bandwidth_gbs * 1e3),
                }
            )
            print(f"done {label} rows={m}", flush=True)

    name = torch.cuda.get_device_name()
    print(
        f"\n{name}, peak bandwidth {hw.bandwidth_gbs:.0f} GB/s, INT4 group {args.group_size}, torch {torch.__version__}"
    )
    for m in args.rows:
        rows = [
            [
                r["layer"],
                r["fp16_us"],
                r["int4_ref_us"],
                r["int4_fused_us"],
                r["speedup_vs_fp16"],
                r["fp16_gbs"],
                r["int4_fused_gbs"],
            ]
            for r in records
            if r["rows"] == m
        ]
        print(f"\nrows = {m}  (CUDA graph replay, microseconds; bandwidth = weight bytes / time)")
        print(table(["layer", "fp16", "int4 ref", "int4 fused", "fused speedup", "fp16 GB/s", "fused GB/s"], rows))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {"gpu": name, "torch": torch.__version__, "group_size": args.group_size, "records": records}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
