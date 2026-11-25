"""Decode-shaped speed of the fused low-bit kernels: fp16 against INT4, INT3 and INT2.

Decoding multiplies each weight matrix by one (or a few) activation rows, so its time is
set by how many weight bytes are read. Fewer bits per weight should mean less time, if
the kernel keeps the memory system busy. For each layer shape and row count this times:

* ``fp16``  ``F.linear`` with an fp16 weight (cuBLAS)
* ``int4``, ``int3``, ``int2``  the ``triton`` backend (fused GEMV for up to 4 rows,
  the same dispatch as in a model), group size 64, asymmetric

and reports each time and the achieved bandwidth (weight, scale and zero bytes / time).
All timings are CUDA graph replays.

    python scripts/bench_lowbit.py --json results/lowbit_kernels_rtx3090.json > results/lowbit_kernels_rtx3090.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_kernels import SHAPES, bench_graph
from torch import nn

from bitsmith.nn import QLinear
from bitsmith.report import table

ROWS = (1, 4, 16)
SCHEMES = ("int4", "int3", "int2")


def bench_shape(name: str, out_f: int, in_f: int, group: int) -> list[dict[str, object]]:
    lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)  # encode on the CPU, a shared GPU is short of memory
    layers = {s: QLinear.from_linear(lin, s, group).cuda() for s in SCHEMES}
    lin = lin.cuda()
    for q in layers.values():
        q.set_backend("triton")
    records: list[dict[str, object]] = []
    for rows in ROWS:
        x = torch.randn(rows, in_f, device="cuda", dtype=torch.float16)
        rec: dict[str, object] = {
            "layer": name,
            "rows": rows,
            "fp16_us": bench_graph(lambda x=x: F.linear(x, lin.weight)),
        }
        rec["fp16_bytes"] = out_f * in_f * 2
        for s, q in layers.items():
            rec[f"{s}_us"] = bench_graph(lambda x=x, q=q: q(x))
            rec[f"{s}_bytes"] = q.weight_bytes()
        records.append(rec)
        print(f"done {name} rows={rows}", flush=True)
    return records


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    torch.manual_seed(0)
    records = [r for name, (o, i) in SHAPES.items() for r in bench_shape(name, o, i, args.group_size)]
    gpu = torch.cuda.get_device_name()
    print(f"\nLow-bit decode kernels on {gpu}, group {args.group_size}: CUDA graph replay, microseconds (GB/s)")
    headers = ["layer", "rows", "fp16", "int4", "int3", "int2", "int3 vs int4", "int2 vs int4", "int2 vs fp16"]
    rows_out = []
    for r in records:

        def cell(key: str, r: dict[str, object] = r) -> str:
            us, nbytes = float(r[f"{key}_us"]), float(r[f"{key}_bytes"])  # type: ignore[arg-type]
            return f"{us:.1f} ({nbytes / us / 1e3:.0f})"

        t = {k: float(r[f"{k}_us"]) for k in ("fp16", *SCHEMES)}  # type: ignore[arg-type]
        rows_out.append(
            [
                r["layer"],
                r["rows"],
                cell("fp16"),
                cell("int4"),
                cell("int3"),
                cell("int2"),
                f"{t['int4'] / t['int3']:.2f}x",
                f"{t['int4'] / t['int2']:.2f}x",
                f"{t['fp16'] / t['int2']:.2f}x",
            ]
        )
    print(table(headers, rows_out))
    if args.json:
        payload = {"gpu": gpu, "torch": torch.__version__, "group_size": args.group_size, "records": records}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
