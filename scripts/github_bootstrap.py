"""Create the GitHub labels, milestones and starter issues for the project.

Uses the GitHub CLI (``gh``) with whatever account is logged in, so run
``gh auth status`` first. Safe to run more than once: existing labels are updated and
existing milestones and issues (matched by title) are skipped.

Usage:
    python scripts/github_bootstrap.py --dry-run     # show what would be created
    python scripts/github_bootstrap.py               # create it
    python scripts/github_bootstrap.py --repo owner/name
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

DEFAULT_REPO = "yashwanthdevavarapu/bitsmith"

# name -> (color, description)
LABELS: dict[str, tuple[str, str]] = {
    "good first issue": ("7057ff", "Small and well scoped, good for newcomers"),
    "help wanted": ("008672", "Larger work where outside help is welcome"),
    "bug": ("d73a4a", "Something gives a wrong result or crashes"),
    "enhancement": ("a2eeef", "New feature or improvement"),
    "performance": ("fbca04", "Faster or uses less memory"),
    "documentation": ("0075ca", "Docs, examples and notebooks"),
    "question": ("d876e3", "Further information is requested"),
    "area: kernels": ("1d76db", "GPU kernels and runtime backends"),
    "area: calibration": ("1d76db", "SmoothQuant, AWQ, GPTQ and statistics capture"),
    "area: formats": ("1d76db", "Quantizers, codecs and bit packing"),
    "area: models": ("1d76db", "Model conversion and architecture support"),
    "area: benchmarks": ("1d76db", "Evaluation, benchmark scripts and results"),
    "area: cli": ("1d76db", "The bitsmith command line interface"),
    "status: needs triage": ("ededed", "New, not reviewed by a maintainer yet"),
    "status: in progress": ("c2e0c6", "Someone is working on this"),
}

MILESTONES: dict[str, str] = {
    "v0.3: Speed": "Make quantized models faster than bf16 end to end. See ROADMAP.md.",
    "v0.4: Formats": "Cover the formats people actually ship: NF4, MX, INT3/INT2, KV cache.",
    "v0.5: Models": "More model families and larger sizes.",
}


@dataclass
class Issue:
    title: str
    labels: list[str]
    body: str
    milestone: str | None = None
    extra: list[str] = field(default_factory=list)


ISSUES: list[Issue] = [
    # ---------------------------------------------------------------- good first issues
    Issue(
        "Add more GPUs to the roofline hardware list",
        ["good first issue", "area: benchmarks"],
        """\
`bitsmith roofline` estimates decode speed from a GPU's memory bandwidth and peak FP16
throughput. The list currently has RTX 3050 Laptop, T4, RTX 3090, RTX 4090, A100, H100 and
M2 Pro.

**Task**: add common GPUs such as RTX 4060, RTX 4070, L4, A10G and L40S, using numbers from
the official datasheets.

**Files**
- `src/bitsmith/bench/roofline.py`: the `HARDWARE` dict
- `tests/unit/test_bench.py`: add a check for one of the new entries

**Done when** `bitsmith info` lists the new GPUs, `bitsmith roofline --hardware <new>` works,
and the PR description links the datasheet for each number.
""",
    ),
    Issue(
        "Add --json output to `bitsmith sqnr` and `bitsmith roofline`",
        ["good first issue", "area: cli"],
        """\
`bitsmith sweep` and `bitsmith hf` can write results to JSON with `--json path`, but
`bitsmith sqnr` and `bitsmith roofline` only print tables.

**Task**: add a `--json` option to both commands, using the existing `_write_json` helper.

**Files**
- `src/bitsmith/cli.py`: `cmd_sqnr`, `cmd_roofline` and their parsers in `build_parser`
- `tests/e2e/test_cli.py`: write to `tmp_path` and check the JSON fields

**Done when** both commands accept `--json`, the files contain one record per printed row,
and the tests cover it.
""",
    ),
    Issue(
        "Show FP8 bit layouts in `bitsmith pack`",
        ["good first issue", "area: cli", "area: formats"],
        """\
`bitsmith pack` explains INT4, ternary and 2:4 layouts, but not FP8.

**Task**: add `--format fp8-e4m3` and `--format fp8-e5m2`. Print a few values, their 8-bit
patterns split into sign, exponent and mantissa (for example `0 0111 000` for 1.0 in E4M3), the decoded
values, and the round trip check.

**Files**
- `src/bitsmith/cli.py`: `cmd_pack` and the `--format` choices
- `src/bitsmith/quant/fp8.py`: `fp8_encode` and `fp8_decode` already do the work
- `tests/e2e/test_cli.py`: extend `test_pack`

**Done when** both formats print and the round trip check passes.
""",
    ),
    Issue(
        "Add a Colab quickstart notebook",
        ["good first issue", "documentation"],
        """\
A notebook lets people try bitsmith without installing anything.

**Task**: add `examples/quickstart.ipynb` that installs bitsmith from GitHub, quantizes a
layer with every scheme, runs `calibrate_and_quantize` with GPTQ on
`HuggingFaceTB/SmolLM2-135M`, and compares perplexity on a short WikiText-2 slice. Add an
"Open in Colab" badge to the README.

**Done when** the notebook runs top to bottom on a free Colab GPU in under 10 minutes and is
saved with outputs cleared.
""",
    ),
    Issue(
        "Vectorize MarkovCorpus.sample",
        ["good first issue", "performance", "area: benchmarks"],
        """\
`MarkovCorpus.sample` draws one token at a time in a Python loop, which takes a few seconds
for the 60,000 tokens `bitsmith sweep` uses.

**Task**: make sampling faster, for example by drawing all uniform numbers up front and
doing the per-step lookup with precomputed cumulative tables, or by sampling in chunks.

**Files**
- `src/bitsmith/eval/corpus.py`
- `tests/unit/test_eval.py`: the existing tests (determinism, entropy floor) must still pass

**Note**: if the token sequence changes for a given seed, regenerate the committed results
with `make reproduce` in the same PR.

**Done when** sampling 100,000 tokens takes well under a second on a laptop CPU.
""",
    ),
    Issue(
        "Tokenize WikiText-2 without the 'sequence length is longer than maximum' warning",
        ["good first issue", "area: benchmarks"],
        """\
`bitsmith hf` tokenizes the whole test split in one call, which makes Transformers print
"Token indices sequence length is longer than the specified maximum sequence length".
It is harmless (we never feed the whole sequence to the model) but looks alarming.

**Task**: tokenize the text in chunks (for example per article or per 100,000 characters)
and concatenate the ids, or pass the tokenizer option that suppresses the length check.

**Files**
- `src/bitsmith/bench/hf.py`: `wikitext2_tokens`

**Done when** the warning no longer appears and the token count for the test split is
unchanged.
""",
    ),
    Issue(
        "Share benchmark results from your GPU",
        ["good first issue", "area: benchmarks", "help wanted"],
        """\
Kernel and model results currently come from one RTX 3090. Numbers from other GPUs show
where the fused INT4 kernel helps and where it does not.

**Task**: run

```
python scripts/bench_kernels.py --json kernels.json
bitsmith hf --model Qwen/Qwen2.5-0.5B --device cuda --dtype bfloat16 --eval-tokens 0 --json model.json
```

and open a pull request adding the files under `results/` (for example
`results/kernels_rtx4090.json`), or post them in a comment here.

**Done when** at least two more GPUs are in `results/` and listed in the README.
""",
    ),
    # ---------------------------------------------------------------- v0.3 speed
    Issue(
        "CUDA graph capture for the decode step",
        ["enhancement", "performance", "area: kernels", "help wanted"],
        """\
Per layer, the fused INT4 kernel is up to 1.33x faster than fp16 cuBLAS, but end-to-end decoding
through Hugging Face `generate` is still slower than bf16 (25.4 vs 42.4 tokens/s on
Qwen2.5-1.5B), because launching about 200 small kernels per token from Python dominates.

**Task**: capture one decode step in a CUDA graph and replay it. Options:
- a static KV cache plus `torch.compile(mode="reduce-overhead")`
- a manual `torch.cuda.CUDAGraph` around a forward with a static cache

**Files**
- `src/bitsmith/bench/hf.py`: `decode_tokens_per_s`
- `src/bitsmith/kernels/int4_triton.py`: kernels must be capture safe (they already are
  for the GEMV path in `scripts/bench_kernels.py`)

**Done when** `bitsmith hf` reports a graph-captured decode speed for bf16 and fused INT4,
and INT4 is faster than bf16 on Qwen2.5-1.5B.
""",
        milestone="v0.3: Speed",
    ),
    Issue(
        "Faster INT4 GEMV: wide vector loads and weight pre-shuffling",
        ["enhancement", "performance", "area: kernels"],
        """\
The fused INT4 GEMV reaches about 25% of the RTX 3090's 936 GB/s, against about 70% for
cuBLAS fp16 (see `results/kernels_rtx3090.txt`). Two variants that were tried and rejected
are recorded in ADR 0010.

**Ideas**
- Pre-shuffle packed weights at conversion time so each thread loads 16 bytes at once
- Dequantize with bit tricks instead of float conversions (as in Marlin)
- Tune split-K and tile sizes per shape

**Files**
- `src/bitsmith/kernels/int4_triton.py`
- `scripts/bench_kernels.py` to measure; `tests/gpu/test_cuda.py` must keep passing

**Done when** batch-1 bandwidth improves clearly on the 7B shapes, with numbers in the PR.
A storage layout change needs an ADR.
""",
        milestone="v0.3: Speed",
    ),
    Issue(
        "Real INT8 W8A8 matmul on tensor cores",
        ["enhancement", "performance", "area: kernels", "help wanted"],
        """\
`w8a8` and `w8a8+smoothquant` currently simulate activation quantization (fake quant) and
then run a float matmul, so they are slower than plain INT8 weights.

**Task**: add a backend for `int8` layers with `act_bits=8` that quantizes activations per
token to `int8` and calls `torch._int_mm` (int8 x int8 -> int32 on tensor cores), then
applies the activation and weight scales. Note `torch._int_mm` has shape requirements
(rows above 16, dimensions multiples of 8), so fall back to the reference path otherwise.

**Files**
- `src/bitsmith/nn/qlinear.py`: `set_backend` and `forward`
- `tests/gpu/test_cuda.py`: compare with the reference path

**Done when** W8A8 prefill is faster than fp16 on an Ampere GPU and matches the reference
within a tested error bound.
""",
        milestone="v0.3: Speed",
    ),
    Issue(
        "Tune the INT4 prefill GEMM",
        ["enhancement", "performance", "area: kernels"],
        """\
For 512 rows the fused INT4 GEMM is about 2x slower than cuBLAS fp16
(`results/kernels_rtx3090.txt`). Prefill is compute bound, so the goal is to get close to
cuBLAS, not to beat it.

**Ideas**: larger tiles and more pipeline stages in the autotune list, dequantizing a tile
once into shared memory and reusing it across row blocks, `BLOCK_K` larger than the group
size.

**Files**: `src/bitsmith/kernels/int4_triton.py` (`_gemm`), `scripts/bench_kernels.py`

**Done when** 512-row INT4 is within 1.3x of cuBLAS fp16 on the 7B shapes.
""",
        milestone="v0.3: Speed",
    ),
    Issue(
        "Reduce Python overhead in the GPTQ column loop",
        ["enhancement", "performance", "area: calibration"],
        """\
`gptq_quantize` loops over columns in Python. On a GPU this makes GPTQ CPU bound: during
the Qwen2.5 runs the GPU sat mostly idle while one CPU core ran at 100%.

**Ideas**: process each lazy-update block with `torch.compile`, or write the inner block
loop as a Triton kernel; the update between blocks is already a single matmul.

**Files**: `src/bitsmith/calib/gptq.py`, `tests/unit/test_calib.py` (results must not change
beyond float rounding)

**Done when** GPTQ on Qwen2.5-0.5B is at least 3x faster with the same perplexity.
""",
        milestone="v0.3: Speed",
    ),
    # ---------------------------------------------------------------- v0.4 formats
    Issue(
        "Add an NF4 codec",
        ["enhancement", "area: formats", "help wanted"],
        """\
NF4 (from QLoRA) stores 4-bit indices into a fixed table of 16 values placed at quantiles
of a normal distribution, with one absmax scale per block of 64 weights.

**Task**: add an `nf4` scheme.
1. Quantizer: normalize each block by its absmax, pick the nearest table entry.
2. Codec in `src/bitsmith/nn/codecs.py`, packed with `pack_int4` (unsigned).
3. Register in `SCHEMES` and `make_codec`; add rows to the tables in
   `tests/unit/test_nn.py` so all parametrized tests cover it.

**Done when** `nf4` passes the codec tests and appears in `bitsmith sqnr`, with its weight
SQNR reported next to `int4` and `int4-sym`.
""",
        milestone="v0.4: Formats",
    ),
    Issue(
        "Add MXFP4 and MXINT8 microscaling formats",
        ["enhancement", "area: formats"],
        """\
The OCP Microscaling (MX) formats share one 8-bit power-of-two scale (E8M0) per block of
32 elements. MXFP4 elements are FP4 E2M1; MXINT8 elements are int8.

**Task**: implement both as codecs, reusing the table-based approach from
`src/bitsmith/quant/fp8.py` for E2M1, and document the layout in the technical guide.

**Done when** both formats round trip with tested known values and appear in
`bitsmith sqnr` with correct bits per weight (4.25 and 8.25).
""",
        milestone="v0.4: Formats",
    ),
    Issue(
        "Add INT3 and INT2 group codecs with packing",
        ["enhancement", "area: formats"],
        """\
Only 4- and 8-bit integers are packed today. 3-bit and 2-bit weights (used with GPTQ in
some low-memory setups) need their own packing: for example 8 three-bit values in 3 bytes.

**Task**: add `pack_int3`/`unpack_int3` and `pack_int2`/`unpack_int2` in
`src/bitsmith/packing/`, codecs `int3` and `int2`, and make `gptq_quantize` usable with them.

**Done when** known-layout and round trip tests pass and `bitsmith sqnr` shows both.
""",
        milestone="v0.4: Formats",
    ),
    Issue(
        "KV-cache quantization",
        ["enhancement", "area: formats", "help wanted"],
        """\
For long contexts the KV cache can be larger than the weights. Storing keys and values in
INT8 or FP8 roughly halves that memory.

**Task**: design and implement KV-cache quantization for Hugging Face models (for example a
custom `Cache` class that stores quantized keys and values per head with per-token
scales), and measure perplexity and memory at 4k and 16k context.

**Done when** there is a short design note (ADR), an implementation with tests, and
results for one model.
""",
        milestone="v0.4: Formats",
    ),
    # ---------------------------------------------------------------- v0.5 models
    Issue(
        "Support GPT-2 style Conv1D layers in quantize_model",
        ["enhancement", "area: models", "help wanted"],
        """\
`quantize_model` only converts `nn.Linear`. GPT-2 and some other models use
`transformers.pytorch_utils.Conv1D`, whose weight is stored transposed `(in, out)`.

**Task**: convert `Conv1D` layers too (transpose the weight into a `QLinear`), and make
calibration hooks work for them.

**Files**: `src/bitsmith/nn/convert.py`, `src/bitsmith/calib/capture.py`, a test with a tiny
GPT-2 config (no download needed: build it from `GPT2Config`).

**Done when** `bitsmith hf --model gpt2` runs every config.
""",
        milestone="v0.5: Models",
    ),
    Issue(
        "Fold SmoothQuant and AWQ scales into norm layers",
        ["enhancement", "area: calibration"],
        """\
SmoothQuant and AWQ store a per-channel `input_scale` buffer in each `QLinear` and multiply
activations by it (ADR 0007). Production toolkits fold this scale into the preceding
LayerNorm or RMSNorm instead, which removes the extra multiply.

**Task**: for Llama and Qwen style blocks, find the norm that feeds q/k/v and gate/up,
search one shared scale for each group of consumers, and fold it into the norm weight.

**Done when** folded and unfolded models give the same outputs within a tested tolerance,
and an ADR records the design.
""",
        milestone="v0.5: Models",
    ),
    Issue(
        "Benchmark a 7B model",
        ["enhancement", "area: benchmarks", "help wanted"],
        """\
Published results stop at 1.5B parameters. INT4 behaves differently at 7B, and the fused
kernel benefits most there.

**Blocker to fix first**: `run_sweep` deep-copies the model for every config, which needs
twice the model memory. Add an option to reload the model from disk per config instead.

**Files**: `src/bitsmith/bench/pareto.py`, `src/bitsmith/bench/hf.py`

**Done when** results for a 7B model (for example Qwen2.5-7B) are in `results/` and the
README, run on a 24 GB GPU.
""",
        milestone="v0.5: Models",
    ),
]


def run(args: list[str], stdin: str | None = None) -> str:
    proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dry-run", action="store_true", help="print what would be created")
    args = parser.parse_args()

    if shutil.which("gh") is None:
        print("The GitHub CLI (gh) is not installed: https://cli.github.com", file=sys.stderr)
        return 1
    repo = args.repo
    print(f"Repository: {repo}{'  (dry run)' if args.dry_run else ''}")

    print("\nLabels")
    for name, (color, desc) in LABELS.items():
        print(f"  {name}")
        if not args.dry_run:
            run(["label", "create", name, "--repo", repo, "--color", color, "--description", desc, "--force"])

    print("\nMilestones")
    existing_ms: set[str] = set()
    if not args.dry_run:
        existing_ms = {m["title"] for m in json.loads(run(["api", f"repos/{repo}/milestones?state=all&per_page=100"]))}
    for title, desc in MILESTONES.items():
        if title in existing_ms:
            print(f"  {title}  (exists, skipped)")
            continue
        print(f"  {title}")
        if not args.dry_run:
            run(["api", f"repos/{repo}/milestones", "-f", f"title={title}", "-f", f"description={desc}"])

    print("\nIssues")
    existing: set[str] = set()
    if not args.dry_run:
        listed = run(["issue", "list", "--repo", repo, "--state", "all", "--limit", "1000", "--json", "title"])
        existing = {i["title"] for i in json.loads(listed)}
    for issue in ISSUES:
        tag = f" [{issue.milestone}]" if issue.milestone else ""
        if issue.title in existing:
            print(f"  {issue.title}  (exists, skipped)")
            continue
        print(f"  {issue.title}{tag}  labels: {', '.join(issue.labels)}")
        if args.dry_run:
            continue
        cmd = ["issue", "create", "--repo", repo, "--title", issue.title, "--body-file", "-"]
        for label in issue.labels:
            cmd += ["--label", label]
        if issue.milestone:
            cmd += ["--milestone", issue.milestone]
        print(f"    {run(cmd, stdin=issue.body).strip()}")

    print("\nDone." if not args.dry_run else "\nDry run only; nothing was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
