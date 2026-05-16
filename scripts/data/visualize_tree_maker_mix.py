#!/usr/bin/env python3
"""Visualize TreeMaker synthetic dataset composition from raw-manifest.jsonl."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "M": (255, 45, 45),
    "V": (0, 92, 255),
    "B": (15, 23, 42),
    "F": (160, 174, 192),
    "U": (160, 174, 192),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Synthetic dataset root containing raw-manifest.jsonl")
    parser.add_argument("--out", type=Path, help="Output PNG path. Defaults to <root>/qa/tree-maker-mix.png")
    parser.add_argument("--examples-per-topology", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(args.root / "raw-manifest.jsonl")
    output = args.out or args.root / "qa" / "tree-maker-mix.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_mix(rows, args.root, output, args.examples_per_topology)
    print(f"Wrote TreeMaker mix visualization to {output}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def draw_mix(rows: list[dict[str, Any]], root: Path, output: Path, examples_per_topology: int) -> None:
    width, height = 1800, 1700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_h = load_font(34, bold=True)
    font_b = load_font(24, bold=True)
    font = load_font(18)
    font_s = load_font(14)

    draw.text((50, 30), "TreeMaker Synthetic Topology Mix", fill=(20, 30, 48), font=font_h)
    draw.text((50, 74), f"{len(rows)} accepted samples from {root}", fill=(82, 96, 120), font=font)

    topology = Counter(tree(row).get("topology") for row in rows)
    symmetry = Counter(tree(row).get("symmetryClass") for row in rows)
    variant = Counter(tree(row).get("symmetryVariant") for row in rows)
    archetype = Counter(tree(row).get("archetype") for row in rows)

    draw_bar_panel(draw, (50, 130, 560, 440), "Topology", topology, font_b, font_s)
    draw_bar_panel(draw, (640, 130, 1070, 440), "Symmetry Class", symmetry, font_b, font_s)
    draw_bar_panel(draw, (1150, 130, 1750, 440), "Archetype", archetype, font_b, font_s)
    draw_bar_panel(draw, (50, 490, 560, 740), "Symmetry Variant", variant, font_b, font_s)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(tree(row).get("topology"))].append(row)

    draw.text((640, 490), "Example CPs By Topology", fill=(20, 30, 48), font=font_b)
    draw.text((640, 522), "M red, V blue, B black, F/U gray", fill=(82, 96, 120), font=font)
    x0, y0 = 640, 565
    thumb, gap = 220, 28
    ordered_topologies = ["radial-star", "hubbed-limbs", "spine-chain", "branched-hybrid"]
    for row_index, name in enumerate(ordered_topologies):
        y = y0 + row_index * (thumb + 54)
        draw.text((x0, y - 24), f"{name} ({len(grouped.get(name, []))})", fill=(20, 30, 48), font=font)
        for col, row in enumerate(grouped.get(name, [])[:examples_per_topology]):
            x = x0 + col * (thumb + gap)
            draw.rounded_rectangle((x, y, x + thumb, y + thumb), radius=10, outline=(205, 215, 230), width=2, fill=(252, 254, 255))
            fold = json.loads(resolve(root, row["foldPath"]).read_text(encoding="utf-8"))
            render_fold(draw, fold, (x + 12, y + 12, x + thumb - 12, y + thumb - 12))
            label = f"{tree(row).get('symmetryVariant')} / {row.get('edges')}e"
            draw.text((x + 8, y + thumb + 6), label, fill=(82, 96, 120), font=font_s)

    image.save(output)


def draw_bar_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, counts: Counter, font_b: ImageFont.ImageFont, font_s: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, outline=(205, 215, 230), width=2, fill=(252, 254, 255))
    draw.text((x0 + 18, y0 + 16), title, fill=(20, 30, 48), font=font_b)
    total = sum(value for key, value in counts.items() if key is not None)
    items = [(str(key), int(value)) for key, value in counts.most_common() if key is not None]
    max_value = max((value for _, value in items), default=1)
    bar_x = x0 + 160
    bar_w = x1 - bar_x - 38
    y = y0 + 70
    for label, value in items:
        pct = value / total if total else 0
        draw.text((x0 + 18, y + 4), label, fill=(51, 65, 85), font=font_s)
        draw.rounded_rectangle((bar_x, y + 3, bar_x + bar_w, y + 21), radius=7, fill=(232, 238, 247))
        draw.rounded_rectangle((bar_x, y + 3, bar_x + max(2, int(bar_w * value / max_value)), y + 21), radius=7, fill=(37, 99, 235))
        draw.text((bar_x + bar_w + 8, y + 1), f"{value} ({pct:.0%})", fill=(51, 65, 85), font=font_s)
        y += 34


def render_fold(draw: ImageDraw.ImageDraw, fold: dict[str, Any], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    for i in range(9):
        x = x0 + (x1 - x0) * i / 8
        y = y0 + (y1 - y0) * i / 8
        draw.line((x, y0, x, y1), fill=(226, 232, 240), width=1)
        draw.line((x0, y, x1, y), fill=(226, 232, 240), width=1)
    vertices = [tuple(map(float, point[:2])) for point in fold["vertices_coords"]]
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y) or 1
    scale = min((x1 - x0) / span, (y1 - y0) / span) * 0.94
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    panel_center = ((x0 + x1) / 2, (y0 + y1) / 2)

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        return (
            panel_center[0] + (point[0] - center[0]) * scale,
            panel_center[1] - (point[1] - center[1]) * scale,
        )

    assignments = fold.get("edges_assignment", ["U"] * len(fold["edges_vertices"]))
    for assignment in ["F", "U", "B", "M", "V"]:
        for index, (a, b) in enumerate(fold["edges_vertices"]):
            if assignments[index] != assignment:
                continue
            p1 = map_point(vertices[a])
            p2 = map_point(vertices[b])
            width = 3 if assignment in {"M", "V", "B"} else 1
            draw.line((*p1, *p2), fill=COLORS.get(assignment, COLORS["U"]), width=width)


def tree(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("treeMetadata") or row.get("tree_metadata") or {}


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
