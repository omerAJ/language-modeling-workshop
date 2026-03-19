#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path


D_MODEL = 512
MAX_POS = 100
SLICE_POS = 36
SELECTED_DIMS = [0, 16, 64, 256]
COLORS = [
    "#b49dff",
    "#7b9aff",
    "#5ae88e",
    "#33e0f5",
]

WIDTH = 980
HEIGHT = 360
MARGIN_LEFT = 58
MARGIN_RIGHT = 24
MARGIN_TOP = 24
MARGIN_BOTTOM = 42
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM


def x_map(pos: float) -> float:
    return MARGIN_LEFT + (pos / MAX_POS) * PLOT_W


def y_map(value: float) -> float:
    return MARGIN_TOP + (1 - (value + 1) / 2) * PLOT_H


def signal(pos: float, dim: int) -> float:
    omega = 10000 ** (-(dim / D_MODEL))
    return math.sin(pos * omega)


def polyline_points(dim: int) -> str:
    points = []
    steps = MAX_POS * 4
    for step in range(steps + 1):
        pos = step / 4
        points.append(f"{x_map(pos):.2f},{y_map(signal(pos, dim)):.2f}")
    return " ".join(points)


def svg_text(x: float, y: float, text: str, size: int, fill: str, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
        f'font-family="Inter, Arial, sans-serif" font-weight="{weight}" text-anchor="{anchor}">{text}</text>'
    )


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "assets" / "generated" / "positional-signals.svg"
    output.parent.mkdir(parents=True, exist_ok=True)

    y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    x_ticks = [0, 20, 40, 60, 80, 100]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Sinusoidal positional signals across selected embedding dimensions">',
        '<rect width="100%" height="100%" rx="18" fill="#0f1524" />',
        f'<rect x="{MARGIN_LEFT:.2f}" y="{MARGIN_TOP:.2f}" width="{PLOT_W:.2f}" height="{PLOT_H:.2f}" rx="14" fill="#121a2c" stroke="#31415f" stroke-width="1.2" />',
    ]

    # Grid lines and axes.
    for y_tick in [-1.0, 0.0, 1.0]:
        y = y_map(y_tick)
        parts.append(
            f'<line x1="{MARGIN_LEFT:.2f}" y1="{y:.2f}" x2="{WIDTH - MARGIN_RIGHT:.2f}" y2="{y:.2f}" '
            'stroke="#2b3852" stroke-width="1" />'
        )
        parts.append(svg_text(MARGIN_LEFT - 10, y + 4, f"{int(y_tick) if y_tick in (-1.0, 0.0, 1.0) else y_tick:.0f}", 11, "#8ea1c8", "end"))

    for x_tick in x_ticks:
        x = x_map(x_tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{MARGIN_TOP:.2f}" x2="{x:.2f}" y2="{HEIGHT - MARGIN_BOTTOM:.2f}" '
            'stroke="#243149" stroke-width="1" stroke-dasharray="4 6" />'
        )
        parts.append(svg_text(x, HEIGHT - 12, str(x_tick), 11, "#8ea1c8", "middle"))

    parts.append(svg_text(WIDTH / 2, HEIGHT - 12, "position", 11, "#8ea1c8", "middle", "600"))

    # Vertical slice.
    slice_x = x_map(SLICE_POS)
    parts.append(
        f'<line x1="{slice_x:.2f}" y1="{MARGIN_TOP:.2f}" x2="{slice_x:.2f}" y2="{HEIGHT - MARGIN_BOTTOM:.2f}" '
        'stroke="#d7deea" stroke-width="1.5" stroke-dasharray="6 5" />'
    )
    parts.append(
        f'<rect x="{slice_x - 52:.2f}" y="{MARGIN_TOP + 10:.2f}" width="104" height="22" rx="6" '
        'fill="#1a2440" stroke="#6175a8" />'
    )
    parts.append(svg_text(slice_x, MARGIN_TOP + 25, f"pos = {SLICE_POS}", 11, "#dce6ff", "middle", "600"))

    # Plot lines.
    for dim, color in zip(SELECTED_DIMS, COLORS):
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{polyline_points(dim)}" />'
        )
        parts.append(
            f'<circle cx="{slice_x:.2f}" cy="{y_map(signal(SLICE_POS, dim)):.2f}" r="4.8" fill="{color}" stroke="#0f1524" stroke-width="1.6" />'
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
