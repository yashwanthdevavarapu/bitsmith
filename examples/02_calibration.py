"""SmoothQuant, AWQ and GPTQ on synthetic data with LLM-like activation statistics.

Run: python examples/02_calibration.py
"""

from __future__ import annotations

import torch

from bitsmith.calib import awq_search, gptq_quantize, output_error, smooth_weight, smoothing_factors
from bitsmith.eval import outlier_channels, sqnr_db
from bitsmith.nn import make_codec
from bitsmith.quant import QuantSpec, fake_quantize, quantize_activations_per_token

torch.manual_seed(0)
D_IN, D_OUT, N = 512, 512, 4096

# Activations with correlated channels and a handful of large outlier channels.
mix = torch.randn(D_IN, D_IN) / D_IN**0.5
x = torch.randn(N, D_IN) @ (torch.eye(D_IN) + 2 * mix)
x[:, [11, 200, 377]] *= 40
w = torch.randn(D_OUT, D_IN) * 0.02
y = x @ w.T
print(f"outlier channels detected: {outlier_channels(x)}")

# SmoothQuant ------------------------------------------------------------------------
s = smoothing_factors(x.abs().amax(0), w, alpha=0.5)
ws, inv = smooth_weight(w, s)
exact = (x * inv) @ ws.T
print("\nSmoothQuant (W8A8)")
print(f"  invariance check, max |difference| in fp32: {(exact - y).abs().max():.2e}")


def w8a8(xx: torch.Tensor, ww: torch.Tensor) -> torch.Tensor:
    return quantize_activations_per_token(xx, 8) @ fake_quantize(ww, QuantSpec(bits=8)).T


naive, smooth = sqnr_db(y, w8a8(x, w)), sqnr_db(y, w8a8(x * inv, ws))
print(f"  output SQNR naive W8A8:    {naive:6.2f} dB")
print(f"  output SQNR smoothed W8A8: {smooth:6.2f} dB  (+{smooth - naive:.2f} dB)")

# AWQ --------------------------------------------------------------------------------
res = awq_search(w, x.abs().mean(0), x[:1024], scheme="int4", group_size=128)
print("\nAWQ (INT4, group 128)")
print(
    f"  best alpha {res.alpha:.2f}; output MSE {res.baseline_loss:.3e} -> {res.loss:.3e} ({res.improvement:.2f}x lower)"
)

# GPTQ -------------------------------------------------------------------------------
h = 2 * x.T @ x / N
codec = make_codec("int4", 128)
rtn = codec.decode(codec.encode(w), D_OUT, D_IN)
gptq = gptq_quantize(w, h, group_size=128).qtensor.dequantize()
gptq_ao = gptq_quantize(w, h, group_size=128, act_order=True).qtensor.dequantize()
e_rtn, e_gptq, e_ao = output_error(w, rtn, h), output_error(w, gptq, h), output_error(w, gptq_ao, h)
print()
print("GPTQ (INT4, group 128), expected output error tr(dW H dW^T)")
print(f"  round to nearest:      {e_rtn:.3e}")
print(f"  GPTQ:                  {e_gptq:.3e} ({e_rtn / e_gptq:.2f}x lower)")
print(f"  GPTQ with act-order:   {e_ao:.3e} ({e_rtn / e_ao:.2f}x lower)")
print(
    f"  output SQNR: RTN {sqnr_db(y, x @ rtn.T):.2f} dB, GPTQ {sqnr_db(y, x @ gptq.T):.2f} dB, "
    f"act-order {sqnr_db(y, x @ gptq_ao.T):.2f} dB"
)
