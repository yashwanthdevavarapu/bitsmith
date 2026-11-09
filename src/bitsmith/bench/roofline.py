"""Memory-bandwidth roofline for single-stream decoding.

At batch size 1 every generated token must stream all weights from DRAM once. Each
weight takes part in one multiply and one add, so arithmetic intensity is::

    I = 2 FLOPs / (bits_per_weight / 8 bytes) = 16 / bits_per_weight   FLOPs per byte

which is far below the ridge point of any modern accelerator. Decoding is therefore
bandwidth bound, and the lower bound on time per token is ``weight_bytes / bandwidth``.
This assumes a fused kernel that never materializes the dequantized weight (ADR 0004).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_gbs: float  # peak DRAM bandwidth, GB/s
    peak_tflops: float  # dense fp16 tensor throughput, TFLOP/s

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte above which a kernel becomes compute bound."""
        return self.peak_tflops * 1e12 / (self.bandwidth_gbs * 1e9)


# Public datasheet figures, rounded.
HARDWARE: dict[str, Hardware] = {
    "rtx3050-laptop": Hardware("RTX 3050 Laptop", 192.0, 17.0),
    "t4": Hardware("Tesla T4", 320.0, 65.0),
    "rtx3090": Hardware("RTX 3090", 936.0, 71.0),
    "rtx4090": Hardware("RTX 4090", 1008.0, 165.0),
    "a100-40g": Hardware("A100 40GB", 1555.0, 312.0),
    "h100-sxm": Hardware("H100 SXM", 3350.0, 989.0),
    "m2-pro": Hardware("Apple M2 Pro", 200.0, 13.6),
}


@dataclass(frozen=True)
class RooflineEstimate:
    hardware: str
    bits_per_weight: float
    weight_gb: float
    intensity: float
    ms_per_token: float
    tokens_per_s: float
    bound: str


def decode_roofline(n_params: float, bits_per_weight: float, hw: Hardware) -> RooflineEstimate:
    if n_params <= 0 or bits_per_weight <= 0:
        raise ValueError("n_params and bits_per_weight must be positive")
    weight_bytes = n_params * bits_per_weight / 8.0
    intensity = 16.0 / bits_per_weight
    seconds = weight_bytes / (hw.bandwidth_gbs * 1e9)
    compute_s = 2.0 * n_params / (hw.peak_tflops * 1e12)
    bound = "memory" if seconds >= compute_s else "compute"
    t = max(seconds, compute_s)
    return RooflineEstimate(
        hardware=hw.name,
        bits_per_weight=bits_per_weight,
        weight_gb=weight_bytes / 1e9,
        intensity=intensity,
        ms_per_token=t * 1e3,
        tokens_per_s=1.0 / t,
        bound=bound,
    )
