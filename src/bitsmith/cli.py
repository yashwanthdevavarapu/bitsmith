"""Command line interface: ``bitsmith <command> [options]``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from bitsmith import __version__
from bitsmith.bench import (
    HARDWARE,
    SWEEP_HEADERS,
    SweepConfig,
    decode_roofline,
    default_configs,
    run_sweep,
    sweep_rows,
)
from bitsmith.eval import ErrorReport, MarkovCorpus, TinyLMConfig, train_tiny_lm
from bitsmith.nn import SCHEMES, make_codec
from bitsmith.packing import (
    pack_24,
    pack_int4,
    pack_ternary_2bit,
    pack_ternary_base3,
    unpack_24,
    unpack_int4,
    unpack_ternary_2bit,
    unpack_ternary_base3,
)
from bitsmith.report import scatter, table


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _write_json(path: str | None, payload: object) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        _log(f"wrote {path}")


# info -------------------------------------------------------------------------------
def cmd_info(_: argparse.Namespace) -> int:
    print(f"bitsmith {__version__}  (torch {torch.__version__})")
    print("schemes:  " + ", ".join(SCHEMES))
    print("hardware: " + ", ".join(HARDWARE))
    return 0


# pack -------------------------------------------------------------------------------
def _bits(t: torch.Tensor) -> str:
    return " ".join(f"{int(b):08b}" for b in t.flatten())


def cmd_pack(args: argparse.Namespace) -> int:
    g = torch.Generator().manual_seed(args.seed)
    fmt = args.format
    if fmt == "int4":
        values = torch.randint(-8, 8, (8,), generator=g)
        packed = pack_int4(values)
        restored = unpack_int4(packed, signed=True)
        bpw = 4.0
    elif fmt in ("ternary", "ternary-b3"):
        n = 8 if fmt == "ternary" else 10
        values = torch.randint(-1, 2, (n,), generator=g)
        pack, unpack = (
            (pack_ternary_2bit, unpack_ternary_2bit) if fmt == "ternary" else (pack_ternary_base3, unpack_ternary_base3)
        )
        packed = pack(values)
        restored = unpack(packed)
        bpw = 2.0 if fmt == "ternary" else 1.6
    else:
        values = torch.randn(1, 8, generator=g).round(decimals=2)
        vals, meta = pack_24(values)
        restored = unpack_24(vals, meta, 8)
        values_kept = values * (restored != 0)
        print(f"dense    : {[round(v, 2) for v in values.flatten().tolist()]}")
        print(f"kept     : {[round(v, 2) for v in vals.flatten().tolist()]}")
        print(f"metadata : {_bits(meta)}  (two 4-bit groups: idx0 | idx1 << 2)")
        print(f"restored : {[round(v, 2) for v in restored.flatten().tolist()]}")
        ok = torch.equal(values_kept, restored)
        print(f"round trip of kept values exact: {ok}")
        print("storage  : 9.00 bits/weight with fp16 values (2 x 16 bits + 4 bits per group of 4)")
        return 0 if ok else 1
    print(f"values   : {values.tolist()}")
    print(f"packed   : {_bits(packed)}  ({packed.numel()} bytes)")
    print(f"restored : {restored.tolist()}")
    ok = torch.equal(values.to(torch.int8), restored)
    print(f"round trip exact: {ok}")
    print(f"storage  : {bpw:.2f} bits/weight, {16 / bpw:.1f}x smaller than fp16 (before scales)")
    return 0 if ok else 1


# sqnr -------------------------------------------------------------------------------
def _parse_shape(text: str) -> tuple[int, int]:
    try:
        rows, cols = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must look like 1024x1024") from exc
    return rows, cols


def cmd_sqnr(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    rows, cols = args.shape
    w = torch.randn(rows, cols) * 0.02
    if args.outliers:
        idx = torch.randperm(cols)[: args.outliers]
        w[:, idx] *= 20.0
    out = []
    for scheme in SCHEMES:
        codec = make_codec(scheme, args.group_size)
        bufs = codec.encode(w)
        approx = codec.decode(bufs, rows, cols)
        nbytes = sum(t.numel() * t.element_size() for t in bufs.values())
        r = ErrorReport.compare(w, approx)
        out.append([scheme, 8.0 * nbytes / w.numel(), r.sqnr_db, r.max_abs_error, r.cosine])
    print(f"weight {rows}x{cols}, N(0, 0.02^2), {args.outliers} outlier columns x20, group size {args.group_size}")
    print(table(["scheme", "bits/w", "SQNR dB", "max abs err", "cosine"], out))
    return 0


# sweep ------------------------------------------------------------------------------
def cmd_sweep(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    corpus = MarkovCorpus(vocab_size=args.vocab, branching=args.branching, seed=args.seed)
    train = corpus.sample(args.train_tokens, seed=args.seed + 1)
    test = corpus.sample(args.eval_tokens, seed=args.seed + 2)
    cfg = TinyLMConfig(vocab_size=args.vocab, dim=args.dim, n_layers=args.layers, n_heads=4, max_len=args.window)
    _log(f"training tiny LM ({args.steps} steps)")
    model = train_tiny_lm(train, cfg, steps=args.steps, seq_len=min(64, args.window), seed=args.seed)
    calib = [train[i * args.window : (i + 1) * args.window].unsqueeze(0) for i in range(8)]
    points = run_sweep(
        model,
        test,
        calib,
        default_configs(args.group_size),
        window=args.window,
        stride=args.window // 2,
        progress=lambda label: _log(f"  measuring {label}"),
    )
    floor = corpus.optimal_perplexity()
    print(f"tiny LM: dim {args.dim}, {args.layers} layers, vocab {args.vocab}; eval {args.eval_tokens} tokens")
    print(f"optimal perplexity of the corpus (entropy-rate floor): {floor:.3f}")
    print(table(SWEEP_HEADERS, sweep_rows(points)))
    print()
    print(
        scatter(
            [(p.bits_per_weight, p.perplexity, p.label) for p in points],
            x_label="bits per weight",
            y_label="perplexity",
            log_x=True,
            log_y=True,
            highlight={p.label for p in points if p.on_frontier},
        )
    )
    print("upper case marks = Pareto frontier")
    _write_json(
        args.json,
        {
            "kind": "tiny-sweep",
            "config": {k: v for k, v in vars(args).items() if k not in ("func", "json")},
            "optimal_perplexity": floor,
            "seconds": time.perf_counter() - t0,
            "points": [p.as_dict() for p in points],
        },
    )
    return 0


# roofline ---------------------------------------------------------------------------
# Bits per weight as stored by the codecs; per-row scales are negligible for large layers.
_ROOFLINE_BITS = {
    "fp16": 16.0,
    "int8": 8.0,
    "fp8-e4m3": 8.0,
    "int4-g128": 4 + 24 / 128,
    "int4-g64": 4 + 24 / 64,
    "ternary": 2.0,
}


def cmd_roofline(args: argparse.Namespace) -> int:
    hws = list(HARDWARE) if args.hardware == "all" else [args.hardware]
    rows = []
    for key in hws:
        hw = HARDWARE[key]
        base = decode_roofline(args.params, 16.0, hw)
        for label, bits in _ROOFLINE_BITS.items():
            est = decode_roofline(args.params, bits, hw)
            rows.append(
                [
                    hw.name,
                    label,
                    bits,
                    est.weight_gb,
                    est.intensity,
                    est.ms_per_token,
                    est.tokens_per_s,
                    est.tokens_per_s / base.tokens_per_s,
                    est.bound,
                ]
            )
    print(f"decode roofline, batch 1, {args.params / 1e9:.3g}B parameters (weights only, fused kernel assumed)")
    print(table(["hardware", "format", "bits/w", "GB", "FLOP/B", "ms/token", "tok/s", "speedup", "bound"], rows))
    return 0


# hf ---------------------------------------------------------------------------------
_DTYPE_LABEL = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}

_HF_CONFIGS = {
    "int8": SweepConfig("int8", "int8"),
    "fp8-e4m3": SweepConfig("fp8-e4m3", "fp8-e4m3"),
    "w8a8": SweepConfig("w8a8", "int8", act_bits=8),
    "w8a8+smoothquant": SweepConfig("w8a8+smoothquant", "int8", "smoothquant", act_bits=8),
    "int4": SweepConfig("int4-g64", "int4", group_size=64),
    "int4+awq": SweepConfig("int4-g64+awq", "int4", "awq", group_size=64),
    "int4+gptq": SweepConfig("int4-g64+gptq", "int4", "gptq", group_size=64),
    "nf4": SweepConfig("nf4-g64", "nf4", group_size=64),
    "nf4+awq": SweepConfig("nf4-g64+awq", "nf4", "awq", group_size=64),
    "nf4+gptq": SweepConfig("nf4-g64+gptq", "nf4", "gptq", group_size=64),
    "mxfp4": SweepConfig("mxfp4", "mxfp4"),
    "mxfp4+awq": SweepConfig("mxfp4+awq", "mxfp4", "awq"),
    "mxfp4+gptq": SweepConfig("mxfp4+gptq", "mxfp4", "gptq"),
    "mxint8": SweepConfig("mxint8", "mxint8"),
    "sparse24": SweepConfig("sparse24", "sparse24"),
    "ternary": SweepConfig("ternary", "ternary"),
}


def cmd_hf(args: argparse.Namespace) -> int:
    from bitsmith.bench.hf import run_hf_sweep

    names = [n for n in (args.configs or list(_HF_CONFIGS)) if n != "fp32"]
    baseline = SweepConfig(_DTYPE_LABEL[args.dtype], "baseline")
    configs = [baseline, *(_HF_CONFIGS[n] for n in names)]
    t0 = time.perf_counter()
    res = run_hf_sweep(
        args.model,
        configs,
        eval_tokens=args.eval_tokens or None,
        n_calib=args.n_calib,
        calib_len=args.calib_len,
        window=args.window,
        stride=args.stride,
        device=args.device,
        dtype=args.dtype,
        progress=lambda label: _log(f"  measuring {label}"),
    )
    rows = sweep_rows(res.points)
    headers = [*SWEEP_HEADERS, "decode tok/s"]
    for row, p in zip(rows, res.points, strict=True):
        row.append(res.decode_tps.get(p.label, float("nan")))
    if res.decode_tps_fused:
        headers.append("fused tok/s")
        for row, p in zip(rows, res.points, strict=True):
            row.append(res.decode_tps_fused.get(p.label, ""))
    where = f"{res.device}, {res.dtype}"
    print(f"{res.model_id}: {res.n_params / 1e6:.1f}M params, {where}, WikiText-2 test, {res.eval_tokens:,} tokens")
    print(table(headers, rows))
    _write_json(
        args.json,
        {
            "kind": "hf-sweep",
            "model": res.model_id,
            "n_params": res.n_params,
            "eval_tokens": res.eval_tokens,
            "seconds": time.perf_counter() - t0,
            "torch": torch.__version__,
            "device": res.device,
            "device_name": torch.cuda.get_device_name() if res.device.startswith("cuda") else "cpu",
            "dtype": res.dtype,
            "window": args.window,
            "stride": args.stride,
            "n_calib": args.n_calib,
            "calib_len": args.calib_len,
            "points": [
                p.as_dict()
                | {"decode_tok_s": res.decode_tps.get(p.label), "decode_tok_s_fused": res.decode_tps_fused.get(p.label)}
                for p in res.points
            ],
        },
    )
    return 0


# parser -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bitsmith", description="LLM weight quantization from first principles.")
    p.add_argument("--version", action="version", version=f"bitsmith {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show version, schemes and hardware profiles").set_defaults(func=cmd_info)

    sp = sub.add_parser("pack", help="show how a storage format lays out bits")
    sp.add_argument("--format", choices=["int4", "ternary", "ternary-b3", "sparse24"], default="int4")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_pack)

    sp = sub.add_parser("sqnr", help="weight-level error of every scheme on a random matrix")
    sp.add_argument("--shape", type=_parse_shape, default=(1024, 1024))
    sp.add_argument("--group-size", type=int, default=128)
    sp.add_argument("--outliers", type=int, default=0, help="number of columns scaled x20")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_sqnr)

    sp = sub.add_parser("sweep", help="train a tiny LM on a synthetic corpus and run the Pareto sweep")
    sp.add_argument("--vocab", type=int, default=128)
    sp.add_argument("--branching", type=int, default=4)
    sp.add_argument("--dim", type=int, default=128)
    sp.add_argument("--layers", type=int, default=2)
    sp.add_argument("--steps", type=int, default=400)
    sp.add_argument("--train-tokens", type=int, default=60_000)
    sp.add_argument("--eval-tokens", type=int, default=4_096)
    sp.add_argument("--window", type=int, default=128)
    sp.add_argument("--group-size", type=int, default=128)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--json", help="write results to this JSON file")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("roofline", help="bandwidth-bound decode speed estimate per format")
    sp.add_argument("--params", type=float, default=0.5e9)
    sp.add_argument("--hardware", choices=["all", *HARDWARE], default="all")
    sp.set_defaults(func=cmd_roofline)

    sp = sub.add_parser("hf", help="evaluate a Hugging Face causal LM on WikiText-2 (needs the hf extra)")
    sp.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    sp.add_argument("--configs", nargs="*", choices=list(_HF_CONFIGS))
    sp.add_argument("--eval-tokens", type=int, default=8192, help="0 = the full test split")
    sp.add_argument("--window", type=int, default=512)
    sp.add_argument("--stride", type=int, default=256)
    sp.add_argument("--n-calib", type=int, default=16, help="number of calibration sequences")
    sp.add_argument("--calib-len", type=int, default=256, help="tokens per calibration sequence")
    sp.add_argument("--device", default="cpu", help="cpu or cuda")
    sp.add_argument("--dtype", choices=list(_DTYPE_LABEL), default="float32")
    sp.add_argument("--json", help="write results to this JSON file")
    sp.set_defaults(func=cmd_hf)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
