#!/usr/bin/env python3
"""Render original CPs beside Rabbit Ear flat-folded previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Synthetic dataset root")
    parser.add_argument("--manifest", help="folded-manifest.jsonl path")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", help="Output contact sheet PNG")
    args = parser.parse_args(argv)

    root = Path(args.root)
    manifest_path = Path(args.manifest) if args.manifest else root / "qa" / "folded" / "folded-manifest.jsonl"
    output_path = Path(args.output) if args.output else root / "qa" / "folded" / "contact_sheet.png"
    rows = _load_jsonl(manifest_path)[: args.limit]
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"No folded preview rows found in {manifest_path}; nothing to render.")
        return

    image_dir = output_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        style = "mv_color_clean"
        original_image = image_dir / f"{row['id']}--cp.png"
        folded_image = image_dir / f"{row['id']}--flat-folded.png"
        render_fold(
            _load_json(root / row["originalFoldPath"]),
            original_image,
            image_size=512,
            padding=32,
            line_width=2,
            style=style,
            assignment_visibility="visible",
            seed=101 + index,
        )
        render_fold(
            _load_json(root / row["foldedFoldPath"]),
            folded_image,
            image_size=512,
            padding=32,
            line_width=2,
            style=style,
            assignment_visibility="visible",
            seed=1009 + index,
        )
        rendered.append({**row, "originalImage": original_image, "foldedImage": folded_image})

    _write_contact_sheet(rendered, output_path)
    print(f"Wrote folded preview contact sheet to {output_path}")


def _write_contact_sheet(rows: List[Dict[str, Any]], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    thumb = 180
    label_height = 44
    cols = 4
    sample_width = thumb * 2
    sample_height = thumb + label_height
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * sample_width, sheet_rows * sample_height), (244, 244, 244))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        col = index % cols
        sheet_row = index // cols
        x = col * sample_width
        y = sheet_row * sample_height
        for image_key, x_offset, title in [("originalImage", 0, "CP"), ("foldedImage", thumb, "folded")]:
            image = Image.open(row[image_key]).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + x_offset + (thumb - image.width) // 2, y))
            draw.text((x + x_offset + 4, y + thumb + 4), title, fill=(20, 20, 20))
        label = row["family"]
        label += f" faces={row['faces']}"
        draw.text((x + 4, y + thumb + 22), label[:48], fill=(20, 20, 20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def render_fold(
    fold: Dict[str, Any],
    image_path: Path,
    image_size: int,
    padding: int,
    line_width: int,
    style: str,
    assignment_visibility: str,
    seed: int,
) -> None:
    """Render a simple assignment-colored FOLD preview image."""

    from PIL import Image, ImageDraw, ImageFilter

    del style
    rng = random.Random(seed)
    scale = 3
    canvas_size = image_size * scale
    image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    vertices = _transform_vertices(fold["vertices_coords"], image_size, padding)
    vertices = [(x * scale, y * scale) for x, y in vertices]
    assignments = list(fold.get("edges_assignment", ["U"] * len(fold["edges_vertices"])))
    width = max(1, line_width * scale)

    for edge_index, (a, b) in enumerate(fold["edges_vertices"]):
        assignment = assignments[edge_index] if edge_index < len(assignments) else "U"
        if assignment_visibility == "active-only" and assignment not in {"M", "V", "B"}:
            continue
        draw.line(
            [_jitter(vertices[a], 0, rng), _jitter(vertices[b], 0, rng)],
            fill=_assignment_color(assignment),
            width=width,
        )

    image = image.filter(ImageFilter.GaussianBlur(radius=0.15 * scale))
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def _transform_vertices(vertices: List[List[float]], image_size: int, padding: int) -> List[tuple[float, float]]:
    usable = image_size - 2 * padding
    return [(padding + float(x) * usable, padding + float(y) * usable) for x, y in vertices]


def _assignment_color(assignment: str) -> tuple[int, int, int]:
    if assignment == "M":
        return (224, 48, 48)
    if assignment == "V":
        return (45, 91, 220)
    if assignment == "B":
        return (20, 20, 20)
    return (145, 145, 145)


def _jitter(point: tuple[float, float], amount: int, rng: random.Random) -> tuple[float, float]:
    if amount <= 0:
        return point
    return (point[0] + rng.uniform(-amount, amount), point[1] + rng.uniform(-amount, amount))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
