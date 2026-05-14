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

from src.data.synthetic.rendering import render_fold


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
        raise ValueError(f"No folded preview rows found in {manifest_path}")

    image_dir = output_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        style = "bp_color_clean" if row["family"] == "box-pleat" else "mv_color_clean"
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
        if row.get("bpSubfamily"):
            label = f"{label}/{row['bpSubfamily']}"
        label += f" faces={row['faces']}"
        draw.text((x + 4, y + thumb + 22), label[:48], fill=(20, 20, 20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
