"""Why low-bit weights speed up decoding: a bandwidth roofline for common model sizes.

Run: python examples/03_roofline.py
"""

from __future__ import annotations

from bitsmith.bench import HARDWARE, decode_roofline
from bitsmith.report import table

FORMATS = {"fp16": 16.0, "int8": 8 + 16 / 4096, "int4-g128": 4 + 24 / 128, "ternary": 2 + 16 / 4096}
MODELS = {"0.5B": 0.5e9, "7B": 7e9, "70B": 70e9}

for hw_key in ("rtx3050-laptop", "a100-40g"):
    hw = HARDWARE[hw_key]
    rows = []
    for model, n in MODELS.items():
        base = decode_roofline(n, 16.0, hw)
        for fmt, bits in FORMATS.items():
            est = decode_roofline(n, bits, hw)
            rows.append(
                [model, fmt, est.weight_gb, est.ms_per_token, est.tokens_per_s, est.tokens_per_s / base.tokens_per_s]
            )
    print(f"{hw.name}: {hw.bandwidth_gbs:.0f} GB/s, ridge point {hw.ridge_point:.0f} FLOP/byte")
    print(table(["model", "format", "weights GB", "ms/token", "tok/s bound", "speedup"], rows))
    print()

print("Decode at batch 1 runs at 16 / bits FLOP per byte (1 to 8), far below every ridge point,")
print("so it is bandwidth bound and time per token scales with bytes of weights read.")
