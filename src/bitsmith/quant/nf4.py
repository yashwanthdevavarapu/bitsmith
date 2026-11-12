"""NormalFloat4 (NF4): 4-bit indices into a fixed table of normal quantiles.

Weights of trained networks are roughly normal, so instead of 16 evenly spaced levels NF4
places its levels at quantiles of N(0, 1), rescaled to ``[-1, 1]``. Each block of weights
is divided by its absmax and every value is snapped to the nearest table entry.

The table is asymmetric so that zero is exact: 8 positive quantiles, 7 negative ones and
0. The quantiles are taken between ``p = 0.5`` and ``p = OFFSET``, where ``OFFSET`` is
the midpoint between the two largest quantiles that 16 and 15 equal-mass bins would give
(the constant used by QLoRA). The construction is reproduced here and tested against the
published values.
"""

from __future__ import annotations

from functools import cache

import torch
from torch import Tensor

OFFSET = 0.9677083


@cache
def nf4_table() -> Tensor:
    """The 16 NF4 levels in ascending order, float32, exactly -1, 0 and 1 at the ends and middle."""
    pos = torch.special.ndtri(torch.linspace(OFFSET, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -torch.special.ndtri(torch.linspace(OFFSET, 0.5, 8, dtype=torch.float64)[:-1])
    levels = torch.cat([neg, torch.zeros(1, dtype=torch.float64), pos]).sort().values
    return (levels / levels.abs().max()).to(torch.float32)


@cache
def _table_on(device: torch.device) -> tuple[Tensor, Tensor]:
    """Levels and midpoints on ``device``, copied once (GPTQ calls this for every column)."""
    table = nf4_table().to(device)
    return table, (table[1:] + table[:-1]) / 2


def nf4_encode(x: Tensor) -> Tensor:
    """Map values in ``[-1, 1]`` (larger magnitudes saturate) to the index of the nearest level.

    Returns uint8 codes in ``[0, 15]``. A value exactly halfway between two levels goes to
    the lower one.
    """
    _, mids = _table_on(x.device)
    return torch.bucketize(x.to(torch.float32).contiguous(), mids).to(torch.uint8)


def nf4_decode(codes: Tensor) -> Tensor:
    """Look up the float32 level of each code."""
    return _table_on(codes.device)[0][codes.to(torch.long)]
