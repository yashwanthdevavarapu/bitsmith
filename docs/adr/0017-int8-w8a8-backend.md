# 0017. Real INT8 W8A8 through torch._int_mm

- Status: Accepted
- Date: 2026-11-16

## Context

W8A8 with SmoothQuant gives the best 8-bit quality in the results (+0.8% to +1.1%), but
the reference runtime only simulates it: activations are rounded to INT8 codes, then
multiplied as floats. The point of quantizing activations is to use the INT8 tensor
cores, which on Ampere run at twice the fp16 rate.

## Decision

- **An `int8` backend** for `QLinear` layers with INT8 weights and `act_bits=8`. At run
  time it quantizes activations per token (symmetric, `[-127, 127]`), multiplies with
  `torch._int_mm` (cuBLASLt INT8 GEMM, INT32 accumulation) and rescales by the per-token
  and per-channel scales.
- **The same activation codes as the simulated path**: both call the same quantizer, so
  the backends differ only by float rounding in the matmul (tested on the CPU, where
  `torch._int_mm` also runs). Accuracy results stay valid for the real backend.
- **Padding**: `torch._int_mm` needs more than 16 rows; shorter inputs are padded with
  zero rows. Layers whose dimensions are not multiples of 8 keep the reference backend.
- `set_backend(model, name)` now asks each layer whether it supports the backend, so one
  call switches INT4 layers to `triton` or W8A8 layers to `int8` and leaves the rest.

## Consequences

- Measured results are in `results/int8_rtx3090.txt` and the README.
- In plain PyTorch the per-token quantization and the INT32 rescale were about a dozen
  passes over the activations and the output, more than the GEMM itself on small layers.
  Two Triton kernels now do them in one pass each, bit for bit equal to the PyTorch path,
  and the SmoothQuant input scale is applied inside the quantization pass (ADR 0019).
  Fusing the quantization into the previous layer's output would save the remaining pass.
