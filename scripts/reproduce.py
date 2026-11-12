"""Regenerate every result in results/ and every figure in docs/assets/.

Usage:
    python scripts/reproduce.py            # CPU-only results, no downloads (about 2 minutes)
    python scripts/reproduce.py --gpu      # also the GPU model sweeps and kernel benchmark (CUDA + Triton)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GPU_MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", "smollm2_135m"),
    ("Qwen/Qwen2.5-0.5B", "qwen2.5_0.5b"),
    ("Qwen/Qwen2.5-1.5B", "qwen2.5_1.5b"),
]

HF_ARGS = [
    *("--device", "cuda", "--dtype", "bfloat16", "--eval-tokens", "0"),
    *("--window", "1024", "--stride", "512", "--n-calib", "128", "--calib-len", "512"),
]
# (output prefix, configs); every sweep also measures the bf16 baseline
GPU_SWEEPS = [
    ("gpu", ["int8", "fp8-e4m3", "w8a8", "w8a8+smoothquant", "int4", "int4+awq", "int4+gptq", "sparse24", "ternary"]),
    ("gpu_formats", ["nf4", "mxfp4", "mxint8"]),
    ("gpu_formats_awq", ["nf4+awq", "mxfp4+awq"]),
    ("gpu_formats_gptq", ["nf4+gptq", "mxfp4+gptq"]),
]


def run(args: list[str], out: str) -> None:
    print(f"-> {' '.join(args)}  > results/{out}", flush=True)
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=True)
    (RESULTS / out).write_text(proc.stdout, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="also run the GPU model sweeps and kernel benchmark")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    run(["-m", "bitsmith", "pack", "--format", "int4"], "cli_pack_int4.txt")
    run(["-m", "bitsmith", "pack", "--format", "ternary-b3"], "cli_pack_ternary_b3.txt")
    run(["-m", "bitsmith", "pack", "--format", "sparse24"], "cli_pack_sparse24.txt")
    run(["-m", "bitsmith", "sqnr", "--shape", "1024x1024", "--outliers", "8"], "cli_sqnr.txt")
    run(["-m", "bitsmith", "roofline", "--params", "0.5e9"], "cli_roofline.txt")
    run(["-m", "bitsmith", "sweep", "--json", "results/tiny_sweep.json"], "tiny_sweep.txt")
    run(["examples/01_quickstart.py"], "example_01_quickstart.txt")
    run(["examples/02_calibration.py"], "example_02_calibration.txt")
    run(["examples/03_roofline.py"], "example_03_roofline.txt")
    if args.gpu:
        run(["scripts/bench_kernels.py", "--json", "results/kernels_rtx3090.json"], "kernels_rtx3090.txt")
        for model, tag in GPU_MODELS:
            for prefix, configs in GPU_SWEEPS:
                run(
                    [
                        "-m",
                        "bitsmith",
                        "hf",
                        "--model",
                        model,
                        *HF_ARGS,
                        "--configs",
                        *configs,
                        "--json",
                        f"results/{prefix}_{tag}.json",
                    ],
                    f"{prefix}_{tag}.txt",
                )
    subprocess.run([sys.executable, "scripts/make_figures.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
