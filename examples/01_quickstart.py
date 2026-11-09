"""Quantize a single linear layer with every scheme, then convert a whole model.

Run: python examples/01_quickstart.py
"""

from __future__ import annotations

import torch
from torch import nn

from bitsmith import SCHEMES, QLinear, quantize_model
from bitsmith.eval import sqnr_db
from bitsmith.report import table

torch.manual_seed(0)

# 1. One layer, every scheme --------------------------------------------------------
linear = nn.Linear(1024, 1024)
nn.init.normal_(linear.weight, std=0.02)
x = torch.randn(8, 1024)
y_ref = linear(x)

rows = []
for scheme in SCHEMES:
    q = QLinear.from_linear(linear, scheme, group_size=128)
    rows.append([scheme, q.bits_per_weight(), q.weight_bytes() / 1024, sqnr_db(y_ref, q(x))])
print("Layer 1024x1024, group size 128 (output SQNR on random inputs)")
print(table(["scheme", "bits/w", "KiB", "output SQNR dB"], rows))


# 2. A whole model, keeping the output head in full precision -------------------------
class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))
        self.lm_head = nn.Linear(512, 1000)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.body(t))


model = MLPModel()
inp = torch.randn(4, 512)
before = model(inp)
report = quantize_model(model, "int4", group_size=64)
print()
print(report.summary())
print(f"output SQNR after conversion: {sqnr_db(before, model(inp)):.2f} dB")
print(model.body[0])
