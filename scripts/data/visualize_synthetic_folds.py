#!/usr/bin/env python3
"""Render a generic contact sheet for synthetic FOLD dataset releases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "M": (220, 38, 38),
    "V": (37, 99, 235),
    "B": (15, 23, 42),
    "F": (148, 163, 184),
    "U": (148, 163, 184),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Synthetic dataset root containing raw-manifest.jsonl")
    parser.add_argument("--out", type=Path, help="Output PNG path. Defaults to <root>/qa/synthetic-fold-contact-sheet.png")
    parser.add_argument("--limit", type=int, default=24, help="Maximum examples to render")
    parser.add_argument("--columns", type=int, default=4, help="Contact sheet columns")
    parser.add_argument("--family", help="Optional family filter")
    parser.add_argument(
        "--sort",
        choices=["manifest", "vertical-fraction-desc", "edges-desc"],
        default="manifest",
        help="Deterministic row ordering before taking --limit examples",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.root / "raw-manifest.jsonl")
    if args.family:
        rows = [row for row in rows if row.get("family") == args.family]
    rows = sort_rows(rows, args.sort)
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows matched the requested filters")

    output = args.out or args.root / "qa" / "synthetic-fold-contact-sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_contact_sheet(args.root, rows, output, max(1, args.columns))
    print(f"Wrote synthetic fold contact sheet to {output}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sort_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "vertical-fraction-desc":
        return sorted(rows, key=lambda row: float(tessellation(row).get("verticalCreaseLengthFraction") or -1), reverse=True)
    if mode == "edges-desc":
        return sorted(rows, key=lambda row: int(row.get("edges") or 0), reverse=True)
    return rows


def draw_contact_sheet(root: Path, rows: list[dict[str, Any]], output: Path, columns: int) -> None:
    tile = 260
    label_h = 82
    margin = 28
    gap = 18
    rows_count = math.ceil(len(rows) / columns)
    width = margin * 2 + columns * tile + (columns - 1) * gap
    header_h = 86
    height = margin + header_h + rows_count * (tile + label_h) + (rows_count - 1) * gap + margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_h = load_font(30, bold=True)
    font = load_font(16)
    font_s = load_font(13)

    draw.text((margin, margin), "Synthetic FOLD Contact Sheet", fill=(15, 23, 42), font=font_h)
    draw.text((margin, margin + 42), f"{len(rows)} examples from {root}", fill=(71, 85, 105), font=font)

    y0 = margin + header_h
    for index, row in enumerate(rows):
        col = index % columns
        row_index = index // columns
        x = margin + col * (tile + gap)
        y = y0 + row_index * (tile + label_h + gap)
        fold = json.loads(resolve(root, str(row["foldPath"])).read_text(encoding="utf-8"))
        draw.rectangle((x, y, x + tile, y + tile), outline=(203, 213, 225), width=2, fill=(248, 250, 252))
        render_fold(draw, fold, (x + 12, y + 12, x + tile - 12, y + tile - 12))
        labels = label_lines(row, fold)
        for line_index, line in enumerate(labels):
            draw.text((x + 2, y + tile + 7 + line_index * 18), line, fill=(51, 65, 85), font=font_s)

    image.save(output)


def label_lines(row: dict[str, Any], fold: dict[str, Any]) -> list[str]:
    meta = tessellation(row) or fold.get("tessellation_metadata") or {}
    sample_id = str(row.get("id") or Path(str(row.get("foldPath", ""))).stem)
    if len(sample_id) > 30:
        sample_id = sample_id[:27] + "..."
    family = str(row.get("family") or "?")
    bucket = str(row.get("bucket") or "?")
    edges = row.get("edges") or len(fold.get("edges_vertices", []))
    vertical = meta.get("verticalCreaseLengthFraction")
    spacing = meta.get("minRenderedSpacingPx1024")
    repeat_x = meta.get("repeatX")
    repeat_y = meta.get("repeatY")
    grid_x = meta.get("gridSizeX")
    grid_y = meta.get("gridSizeY")
    horizontal_interval = meta.get("horizontalPleatInterval")
    vertical_interval = meta.get("verticalPleatInterval")
    subfamily = str(meta.get("subfamily") or "")
    miura_skew = meta.get("miuraSkewFactor")
    miura_aspect = meta.get("miuraCellAspectRatio")

    lines = [sample_id, f"{family} / {bucket} / {edges} edges"]
    if subfamily == "miura-ori" and spacing is not None and repeat_x is not None and repeat_y is not None:
        diag = meta.get("diagonalCreaseLengthFraction")
        lines.append(f"miura {repeat_x}x{repeat_y}, diag={float(diag or 0):.2f}, min {float(spacing):.1f}px")
        if miura_skew is not None and miura_aspect is not None:
            lines.append(f"square sheet, skew {float(miura_skew):.2f}, cell aspect {float(miura_aspect):.2f}")
    elif vertical is not None and spacing is not None and repeat_x is not None and repeat_y is not None:
        lines.append(f"{repeat_x}x{repeat_y}, vertical={float(vertical):.2f}, min {float(spacing):.1f}px")
        if grid_x is not None and grid_y is not None and horizontal_interval is not None and vertical_interval is not None:
            lines.append(f"grid {grid_x}x{grid_y}, H every {horizontal_interval}, V every {vertical_interval}")
    else:
        lines.append("M red, V blue, B black")
    return lines


def render_fold(draw: ImageDraw.ImageDraw, fold: dict[str, Any], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
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
            width = 2 if assignment in {"M", "V", "B"} else 1
            draw.line((*p1, *p2), fill=COLORS.get(assignment, COLORS["U"]), width=width)


def tessellation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("tessellationMetadata") or row.get("tessellation_metadata") or {}


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
