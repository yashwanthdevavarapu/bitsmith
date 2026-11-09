"""Optional fused GPU kernels. Everything here is skipped gracefully without CUDA and Triton."""

from bitsmith.kernels.int4_triton import int4_linear, triton_available

__all__ = ["int4_linear", "triton_available"]
