#!/usr/bin/env python3
"""Plot cached decoding step times using only the standard library."""

import csv
import math
import sys
from pathlib import Path


def plot(source, destination):
    with open(source, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    points = [(int(row["position"]), float(row["milliseconds"])) for row in rows]
    if not points or any(not math.isfinite(y) or y < 0 for _, y in points):
        raise ValueError("CSV must contain finite, nonnegative step times")
    xmin = min(x for x, _ in points)
    xmax = max(x for x, _ in points)
    if xmin == xmax:
        xmax = xmin + 1
    ymax = max(y for _, y in points) * 1.1 or 1
    left, top, width, height = 90, 60, 760, 380

    def screen(x, y):
        return left + (x - xmin) / (xmax - xmin) * width, top + height - y / ymax * height

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
        '<rect width="900" height="520" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#222">',
        '<text x="450" y="30" text-anchor="middle" font-size="20">GPT-2 cached decoding latency</text>',
    ]
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        y = ymax * i / 5
        sx, _ = screen(x, 0)
        _, sy = screen(xmin, y)
        svg.extend([
            f'<path d="M {left} {sy:.2f} h {width}" stroke="#ddd"/>',
            f'<text x="{left - 12}" y="{sy + 5:.2f}" text-anchor="end">{y:.2f}</text>',
            f'<path d="M {sx:.2f} {top + height} v 6" stroke="#222"/>',
            f'<text x="{sx:.2f}" y="{top + height + 25}" text-anchor="middle">{x:.1f}</text>',
        ])
    coordinates = " ".join(f"{sx:.2f},{sy:.2f}" for sx, sy in (screen(x, y) for x, y in points))
    svg.extend([
        f'<path d="M {left} {top} v {height} h {width}" fill="none" stroke="#222"/>',
        f'<polyline points="{coordinates}" fill="none" stroke="#2563eb" stroke-width="2"/>',
        '<text x="470" y="500" text-anchor="middle">Position (token)</text>',
        '<text transform="translate(24 250) rotate(-90)" text-anchor="middle">Step latency (ms)</text>',
        '</g></svg>',
    ])
    Path(destination).write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 tools/plot_latency.py IN.csv OUT.svg")
    try:
        plot(sys.argv[1], sys.argv[2])
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"error: {error}")
