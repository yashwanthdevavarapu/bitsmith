"""Shared fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)


@pytest.fixture
def weight() -> torch.Tensor:
    """A realistic linear weight: Gaussian with std 0.02, shape (64, 256)."""
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 256, generator=g) * 0.02
