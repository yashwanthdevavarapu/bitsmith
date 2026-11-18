"""Shared fixtures."""

from __future__ import annotations

import functools
import subprocess
import sys

import pytest
import torch

_HALF_MATMUL = """
import torch
for dtype in (torch.float16, torch.bfloat16):
    torch.nn.functional.linear(torch.randn(2, 256, dtype=dtype), torch.randn(64, 256, dtype=dtype))
"""


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)


@pytest.fixture
def weight() -> torch.Tensor:
    """A realistic linear weight: Gaussian with std 0.02, shape (64, 256)."""
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 256, generator=g) * 0.02


@functools.cache
def _cpu_half_matmul_works() -> bool:
    """Whether fp16 and bf16 matmuls run on this CPU, probed in a child process.

    On some CI machines PyTorch picks a half-precision CPU kernel that uses an instruction
    the processor lacks, and the process dies with an illegal instruction (Windows
    0xC000001D) that no test can catch. Probing in a subprocess turns that into a skip.
    """
    probe = subprocess.run([sys.executable, "-c", _HALF_MATMUL], capture_output=True, timeout=120)
    return probe.returncode == 0


@pytest.fixture
def cpu_half_matmul() -> None:
    """Skip a test that multiplies fp16 or bf16 matrices on the CPU where that crashes."""
    if not _cpu_half_matmul_works():
        pytest.skip("fp16/bf16 matmul crashes on this CPU (illegal instruction in PyTorch)")
