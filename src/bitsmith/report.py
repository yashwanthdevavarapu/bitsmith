"""Plain text rendering: aligned tables and an ASCII scatter plot. No extra dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v) or math.isnan(v):
            return str(v)
        if v != 0 and (abs(v) >= 1e7 or abs(v) < 1e-3):
            return f"{v:.3g}"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown-compatible table with right aligned numeric columns."""
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    numeric = [all(isinstance(row[i], int | float) for row in rows) and bool(rows) for i in range(len(headers))]

    def line(values: Sequence[str]) -> str:
        parts = [v.rjust(w) if num else v.ljust(w) for v, w, num in zip(values, widths, numeric, strict=True)]
        return "| " + " | ".join(parts) + " |"

    sep = "|" + "|".join(("-" * (w + 1)) + (":" if num else "-") for w, num in zip(widths, numeric, strict=True)) + "|"
    return "\n".join([line(list(headers)), sep, *(line(r) for r in cells)])


def scatter(
    points: Sequence[tuple[float, float, str]],
    *,
    width: int = 60,
    height: int = 16,
    x_label: str = "x",
    y_label: str = "y",
    log_x: bool = False,
    log_y: bool = False,
    highlight: set[str] | None = None,
) -> str:
    """ASCII scatter plot. Points are drawn as letters a, b, c, ... explained in a legend.

    Labels in ``highlight`` are drawn in upper case (used for the Pareto frontier).
    """
    if not points:
        return "(no points)"
    highlight = highlight or set()

    def ty(v: float) -> float:
        return math.log10(v) if log_y else v

    def tx(v: float) -> float:
        return math.log10(v) if log_x else v

    xs = [tx(p[0]) for p in points]
    ys = [ty(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    grid = [[" "] * width for _ in range(height)]
    legend = []
    for i, (x, y, label) in enumerate(points):
        col = round((tx(x) - x0) / (x1 - x0) * (width - 1))
        row = height - 1 - round((ty(y) - y0) / (y1 - y0) * (height - 1))
        mark = chr(ord("a") + i % 26)
        mark = mark.upper() if label in highlight else mark
        grid[row][col] = mark
        legend.append(f"  {mark} = {label}")
    top = f"{(10**y1 if log_y else y1):.4g}"
    bottom = f"{(10**y0 if log_y else y0):.4g}"
    pad = max(len(top), len(bottom))
    lines = [f"{y_label}{' (log)' if log_y else ''}"]
    for r, cells in enumerate(grid):
        tick = top if r == 0 else bottom if r == height - 1 else ""
        lines.append(f"{tick.rjust(pad)} |{''.join(cells)}")
    lines.append(" " * pad + " +" + "-" * width)
    left = f"{(10**x0 if log_x else x0):.4g}"
    right = f"{(10**x1 if log_x else x1):.4g}"
    lines.append(" " * pad + "  " + left + right.rjust(width - len(left)))
    lines.append(" " * pad + f"  {x_label}{' (log)' if log_x else ''}")
    return "\n".join(lines + legend)
