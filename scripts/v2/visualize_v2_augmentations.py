#!/usr/bin/env python3
"""Generate visual QA contact sheets for V2 augmentation profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_dataset import (  # noqa: E402
    load_manifest_records,
    render_cpline_sample,
    resolve_fold_path,
)
from src.data.fold_parser import FOLDParser  # noqa: E402
from src.data.v2_augmentations import V2_AUGMENT_PROFILES, V2_LINE_STYLE_IDS  # noqa: E402

DEFAULT_PROFILES = (
    "v2-text",
    "v2-watermark",
    "v2-guide-grid",
    "v2-dashed",
    "v2-faint",
    "v2-ambiguous-mv",
    "v2-combined",
)
ASSIGNMENT_COLORS = {
    0: (230, 45, 45),
    1: (45, 100, 230),
    2: (20, 20, 20),
    3: (150, 150, 150),
}
STYLE_COLORS = {
    V2_LINE_STYLE_IDS["solid"]: (255, 220, 40),
    V2_LINE_STYLE_IDS["dashed"]: (30, 210, 210),
    V2_LINE_STYLE_IDS["faint"]: (255, 150, 45),
    V2_LINE_STYLE_IDS["monochrome"]: (165, 80, 230),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--examples-per-profile", type=int, default=6)
    parser.add_argument("--max-edges", type=int, default=220)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--cell-width", type=int, default=220)
    parser.add_argument(
        "--include-dark-mode",
        action="store_true",
        help="Also render matching v2-dark-* profiles.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated V2 profiles or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/v2_augmentations"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _parse_profiles(args.profiles)
    records = _select_records(
        manifest,
        split=args.split,
        count=args.examples_per_profile,
        max_edges=args.max_edges,
    )
    parser = FOLDParser()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = _render_rows(
        profiles=profiles,
        records=records,
        parser=parser,
        manifest=manifest,
        samples_dir=samples_dir,
        image_size=args.image_size,
        seed=args.seed,
    )
    _write_sheet_bundle(rows, output_dir=output_dir, cell_width=args.cell_width, prefix="")

    dark_rows: list[dict[str, Any]] = []
    if args.include_dark_mode:
        dark_profiles = [_dark_profile(profile) for profile in profiles if not profile.startswith("v2-dark-")]
        dark_rows = _render_rows(
            profiles=dark_profiles,
            records=records,
            parser=parser,
            manifest=manifest,
            samples_dir=samples_dir,
            image_size=args.image_size,
            seed=args.seed + 50000,
        )
        _write_sheet_bundle(dark_rows, output_dir=output_dir, cell_width=args.cell_width, prefix="dark_")

    _write_manifest(
        [*rows, *dark_rows],
        output_dir / "manifest.json",
        manifest=manifest,
        image_size=args.image_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "contact_sheet": str(output_dir / "contact_sheet.png"),
                "dark_contact_sheet": str(output_dir / "dark_contact_sheet.png") if dark_rows else None,
                "profiles": profiles,
                "examples": len(rows) + len(dark_rows),
            },
            indent=2,
        )
    )


def _parse_profiles(value: str) -> list[str]:
    profiles = list(V2_AUGMENT_PROFILES) if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    unsupported = [profile for profile in profiles if profile not in V2_AUGMENT_PROFILES]
    if unsupported:
        raise SystemExit(f"Unsupported V2 augmentation profiles: {', '.join(unsupported)}")
    return profiles


def _select_records(
    manifest: Path,
    *,
    split: str,
    count: int,
    max_edges: int | None,
) -> list[dict[str, Any]]:
    records = load_manifest_records(manifest)
    filtered = [
        record
        for record in records
        if record.get("split") == split and (max_edges is None or int(record["edges"]) <= max_edges)
    ]
    if len(filtered) < count:
        raise SystemExit(f"Need {count} records but found {len(filtered)} in {manifest} for split={split}")
    return filtered[:count]


def _render_rows(
    *,
    profiles: list[str],
    records: list[dict[str, Any]],
    parser: FOLDParser,
    manifest: Path,
    samples_dir: Path,
    image_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_idx, profile in enumerate(profiles):
        for sample_idx, record in enumerate(records):
            cp = parser.parse(resolve_fold_path(record, manifest))
            sample_seed = seed + profile_idx * 1009 + sample_idx * 37
            clean = render_cpline_sample(
                cp,
                image_size=image_size,
                padding=_padding(image_size),
                line_width=_line_width(image_size),
                augment_profile="clean",
            )
            augmented = render_cpline_sample(
                cp,
                image_size=image_size,
                padding=_padding(image_size),
                line_width=_line_width(image_size),
                augment_profile=profile,
                seed=sample_seed,
            )
            row = {
                "profile": profile,
                "record": record,
                "seed": sample_seed,
                "clean": clean,
                "augmented": augmented,
            }
            rows.append(row)
            prefix = f"{profile}_{sample_idx:02d}_{str(record['id'])[:48]}"
            Image.fromarray(augmented.image).save(samples_dir / f"{prefix}_augmented.png")
            Image.fromarray(_non_crease_overlay(augmented.image, augmented.v2_non_crease_mask)).save(
                samples_dir / f"{prefix}_non_crease.png"
            )
    return rows


def _write_sheet_bundle(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    cell_width: int,
    prefix: str,
) -> None:
    if not rows:
        return
    write_contact_sheet(rows, output_dir / f"{prefix}contact_sheet.png", cell_width=cell_width)
    profiles_dir = output_dir / f"{prefix}profiles"
    profiles_dir.mkdir(exist_ok=True)
    profiles = list(dict.fromkeys(str(row["profile"]) for row in rows))
    for profile in profiles:
        write_contact_sheet(
            [row for row in rows if row["profile"] == profile],
            profiles_dir / f"{profile}_contact_sheet.png",
            cell_width=cell_width,
        )


def _dark_profile(profile: str) -> str:
    return f"v2-dark-{profile.removeprefix('v2-')}"


def write_contact_sheet(rows: list[dict[str, Any]], path: Path, *, cell_width: int) -> None:
    if not rows:
        return
    columns = [
        ("clean", "Clean"),
        ("augmented", "Augmented"),
        ("non_crease", "Non-crease"),
        ("line_style", "Line style"),
        ("observed_assignment", "Observed M/V"),
        ("graph", "Target graph"),
    ]
    gap = 10
    label_height = 52
    header_height = 28
    cell_height = cell_width + label_height
    sheet_width = gap + len(columns) * (cell_width + gap)
    sheet_height = header_height + gap + len(rows) * (cell_height + gap)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col_idx, (_, label) in enumerate(columns):
        x = gap + col_idx * (cell_width + gap)
        draw.text((x + 4, 8), label, fill=(20, 20, 20), font=font)

    for row_idx, row in enumerate(rows):
        y = header_height + gap + row_idx * (cell_height + gap)
        augmented = row["augmented"]
        images = {
            "clean": row["clean"].image,
            "augmented": augmented.image,
            "non_crease": _non_crease_overlay(augmented.image, augmented.v2_non_crease_mask),
            "line_style": _line_style_overlay(augmented.image, augmented.v2_line_style),
            "observed_assignment": _assignment_overlay(augmented.image, augmented.v2_observed_assignment),
            "graph": _graph_overlay(
                augmented.image,
                augmented.pixel_vertices,
                augmented.edges,
                augmented.assignments,
            ),
        }
        row_label = f"{row['profile']}\n{row['record']['id']}\nseed={row['seed']}"
        for col_idx, (key, _) in enumerate(columns):
            x = gap + col_idx * (cell_width + gap)
            cell = _resize_cell(images[key], cell_width)
            sheet.paste(cell, (x, y))
            if col_idx == 0:
                draw.rectangle((x, y, x + cell_width, y + 36), fill=(255, 255, 255))
                draw.text((x + 4, y + 4), row_label, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_manifest(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    manifest: Path,
    image_size: int,
) -> None:
    payload = {
        "image_size": image_size,
        "source_manifest": str(manifest),
        "rows": [
            {
                "profile": row["profile"],
                "id": str(row["record"]["id"]),
                "fold_path": str(resolve_fold_path(row["record"], manifest)),
                "seed": row["seed"],
                "augmentation": row["augmented"].metadata,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _non_crease_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    active = mask > 0
    if np.any(active):
        color = np.array([255, 0, 180], dtype=np.float32)
        overlay[active] = (0.35 * overlay[active].astype(np.float32) + 0.65 * color).astype(np.uint8)
    return overlay


def _line_style_overlay(image: np.ndarray, line_style: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for style_id, color in STYLE_COLORS.items():
        active = line_style == style_id
        if np.any(active):
            overlay[active] = (
                0.35 * overlay[active].astype(np.float32) + 0.65 * np.array(color, dtype=np.float32)
            ).astype(np.uint8)
    return overlay


def _assignment_overlay(image: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for label, color in ASSIGNMENT_COLORS.items():
        active = assignment == label
        if np.any(active):
            overlay[active] = (
                0.35 * overlay[active].astype(np.float32) + 0.65 * np.array(color, dtype=np.float32)
            ).astype(np.uint8)
    return overlay


def _graph_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    for edge, assignment in zip(edges, assignments):
        cv2.line(
            overlay,
            _point(vertices[int(edge[0])]),
            _point(vertices[int(edge[1])]),
            ASSIGNMENT_COLORS.get(int(assignment), ASSIGNMENT_COLORS[3]),
            1,
            cv2.LINE_AA,
        )
    for vertex in vertices:
        cv2.circle(overlay, _point(vertex), 2, (255, 225, 25), -1, cv2.LINE_AA)
    return overlay


def _resize_cell(image: np.ndarray, cell_width: int) -> Image.Image:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return pil.resize((cell_width, cell_width), Image.Resampling.LANCZOS)


def _point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _padding(image_size: int) -> int:
    return max(8, int(32 * image_size / 1024))


def _line_width(image_size: int) -> int:
    return max(1, int(2 * image_size / 768))


if __name__ == "__main__":
    main()
