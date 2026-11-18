"""Where INT4 should stop using its fused GEMM: fused tensor core GEMM against unpacking the
weight once and calling cuBLAS, by layer shape and row count.

For each shape and number of activation rows:

* ``fp16``             ``F.linear`` with an fp16 weight (cuBLAS), the target
* ``fused gemm``       the Triton kernel that dequantizes every weight tile inside the GEMM
* ``dequant + cuBLAS`` one Triton kernel unpacks the weight to fp16, then ``F.linear``
* ``dequant only``     the unpacking kernel alone, the fixed cost of the second path

Every implementation is replayed from a CUDA graph. The smallest row count from which
dequantize + cuBLAS wins on every shape is what ``DEQUANT_MIN_ROWS`` is set from.

    python scripts/bench_int4_prefill.py --json results/int4_prefill_rtx3090.json > results/int4_prefill_rtx3090.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_kernels import SHAPES, bench_graph
from torch import nn

from bitsmith.kernels import int4_dequantize, int4_linear
from bitsmith.nn import QLinear
from bitsmith.report import table

ROWS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def bench_shape(name: str, out_f: int, in_f: int, g: int) -> list[dict[str, object]]:
    lin = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
    bufs = QLinear.from_linear(lin, "int4", g).packed_buffers()
    qw, sc, zp = bufs["qweight"], bufs["scale"], bufs["zero"]
    dequant = bench_graph(lambda: int4_dequantize(qw, sc, zp, in_f, g, torch.float16))
    records: list[dict[str, object]] = []
    for rows in ROWS:
        x = torch.randn(rows, in_f, device="cuda", dtype=torch.float16)
        fp16 = bench_graph(lambda x=x: F.linear(x, lin.weight))
        fused = bench_graph(lambda x=x: int4_linear(x, qw, sc, zp, in_f, g, path="fused"))
        dense = bench_graph(lambda x=x: int4_linear(x, qw, sc, zp, in_f, g, path="dequant"))
        records.append(
            {
                "layer": name,
                "rows": rows,
                "fp16_us": fp16,
                "fused_gemm_us": fused,
                "dequant_cublas_us": dense,
                "dequant_only_us": dequant,
            }
        )
        print(f"done {name} rows={rows}", flush=True)
    return records


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    torch.manual_seed(0)
    g = args.group_size
    records = [r for name, (out_f, in_f) in SHAPES.items() for r in bench_shape(name, out_f, in_f, g)]
    gpu = torch.cuda.get_device_name()
    print(f"\nINT4 group {g} on {gpu}: CUDA graph replay, median microseconds")
    headers = ["layer", "rows", "fp16", "fused gemm", "dequant + cuBLAS", "dequant only", "best vs fp16", "best"]
    rows_out = []
    for r in records:
        fused, dense = float(r["fused_gemm_us"]), float(r["dequant_cublas_us"])  # type: ignore[arg-type]
        best = min(fused, dense)
        winner = "dequant + cuBLAS" if dense < fused else "fused gemm"
        rows_out.append(
            [
                r["layer"],
                r["rows"],
                r["fp16_us"],
                r["fused_gemm_us"],
                r["dequant_cublas_us"],
                r["dequant_only_us"],
                f"{float(r['fp16_us']) / best:.2f}x",  # type: ignore[arg-type]
                winner,
            ]
        )
    print(table(headers, rows_out))
    if args.json:
        payload = {"gpu": gpu, "torch": torch.__version__, "group_size": g, "records": records}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
