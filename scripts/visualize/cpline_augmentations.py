#!/usr/bin/env python3
"""Visualize CPLineNet augmentation profiles before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import (
    AUGMENT_PROFILES,
    BASE_AUGMENT_PROFILES,
    NON_IDENTITY_SQUARE_SYMMETRIES,
)
from src.data.cpline_dataset import load_manifest_records, render_cpline_sample, resolve_fold_path
from src.data.fold_parser import FOLDParser


ASSIGNMENT_COLORS = {
    0: (250, 80, 80),
    1: (40, 145, 255),
    2: (20, 20, 20),
    3: (150, 150, 150),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
        help="CPLine raw-manifest JSONL path. Rows must include foldPath, split, id, and edges.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--profiles",
        type=str,
        default="square-symmetry,line-style,dark-mode,print-light,print-medium,photo-light",
        help="Comma-separated profiles or 'all'.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-edges", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/phase3_augmentations"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _parse_profiles(args.profiles)
    records = _select_records(manifest, split=args.split, count=args.num_samples, max_edges=args.max_edges)
    parser = FOLDParser()
    cps = [(record, parser.parse(resolve_fold_path(record, manifest))) for record in records]

    for profile in profiles:
        profile_dir = output_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for sample_idx, (record, cp) in enumerate(cps):
            variants = _variants_for_profile(profile)
            for variant_idx, variant in enumerate(variants):
                seed = args.seed + sample_idx * 101 + variant_idx
                clean = render_cpline_sample(
                    cp,
                    image_size=args.image_size,
                    padding=_padding(args.image_size),
                    line_width=_line_width(args.image_size),
                    augment_profile="clean",
                )
                augmented = render_cpline_sample(
                    cp,
                    image_size=args.image_size,
                    padding=_padding(args.image_size),
                    line_width=_line_width(args.image_size),
                    augment_profile=profile,
                    seed=seed,
                    style_variant=variant if profile == "dark-mode" else None,
                    square_symmetry=variant if profile == "square-symmetry" else None,
                )
                rows.append(
                    {
                        "record": record,
                        "clean": clean,
                        "augmented": augmented,
                        "variant": variant,
                        "seed": seed,
                    }
                )
        sheet_path = profile_dir / f"contact_sheet_{args.image_size}.png"
        sidecar_path = profile_dir / f"contact_sheet_{args.image_size}.json"
        _write_contact_sheet(rows, profile=profile, output_path=sheet_path)
        _write_sidecar(rows, profile=profile, image_size=args.image_size, manifest=manifest, output_path=sidecar_path)
        print(f"Saved {profile}: {sheet_path}")
        print(f"Saved {profile}: {sidecar_path}")


def _parse_profiles(value: str) -> list[str]:
    if value == "all":
        return [profile for profile in BASE_AUGMENT_PROFILES if profile != "clean"]
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    unsupported = [profile for profile in profiles if profile not in AUGMENT_PROFILES]
    if unsupported:
        raise SystemExit(f"Unsupported profiles: {', '.join(unsupported)}")
    return profiles


def _select_records(manifest: Path, *, split: str, count: int, max_edges: int | None) -> list[dict[str, Any]]:
    records = load_manifest_records(manifest)
    filtered = [
        record
        for record in records
        if record.get("split") == split and (max_edges is None or int(record["edges"]) <= max_edges)
    ]
    if not filtered:
        raise SystemExit(f"No records found in {manifest} for split={split}")
    return filtered[:count]


def _variants_for_profile(profile: str) -> list[str | None]:
    if profile == "square-symmetry":
        return list(NON_IDENTITY_SQUARE_SYMMETRIES)
    if profile == "dark-mode":
        return ["dark-no-grid", "dark-grid", "dark-gray", "dark-bright"]
    return [None]


def _write_contact_sheet(rows: list[dict[str, Any]], *, profile: str, output_path: Path) -> None:
    columns = [
        "clean input",
        "augmented input",
        "line target",
        "junction target",
        "assignment target",
        "graph overlay",
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.2 * len(columns), max(2.8, 2.8 * len(rows))),
        squeeze=False,
    )
    for row_idx, row in enumerate(rows):
        clean = row["clean"]
        augmented = row["augmented"]
        images = [
            clean.image,
            augmented.image,
            _line_overlay(augmented.image, augmented.line_prob),
            _junction_overlay(augmented.image, augmented.junction_heatmap),
            _assignment_overlay(augmented.image, augmented.assignment),
            _graph_overlay(augmented.image, augmented.pixel_vertices, augmented.edges, augmented.assignments),
        ]
        row_label = str(row["record"]["id"])
        if row["variant"]:
            row_label = f"{row['variant']}\n{row_label}"
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(columns[col_idx], fontsize=10)
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    0.02,
                    0.98,
                    row_label,
                    transform=axes[row_idx, col_idx].transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
                )
    fig.suptitle(f"CPLine Augmentations: {profile}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_sidecar(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    image_size: int,
    manifest: Path,
    output_path: Path,
) -> None:
    payload = {
        "profile": profile,
        "image_size": image_size,
        "rows": [
            {
                "id": str(row["record"]["id"]),
                "fold_path": str(resolve_fold_path(row["record"], manifest)),
                "variant": row["variant"],
                "seed": row["seed"],
                "augmentation": row["augmented"].metadata,
            }
            for row in rows
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _line_overlay(image: np.ndarray, line_prob: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    mask = line_prob > 0.05
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([255, 220, 40])).astype(np.uint8)
    return overlay


def _junction_overlay(image: np.ndarray, junction_heatmap: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    heat = np.clip(junction_heatmap, 0.0, 1.0)
    color = np.zeros_like(overlay)
    color[..., 0] = (255 * heat).astype(np.uint8)
    color[..., 1] = (40 * heat).astype(np.uint8)
    color[..., 2] = (255 * heat).astype(np.uint8)
    mask = heat > 0.05
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * color[mask]).astype(np.uint8)
    return overlay


def _assignment_overlay(image: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for label, color in ASSIGNMENT_COLORS.items():
        mask = assignment == label
        if np.any(mask):
            overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array(color)).astype(np.uint8)
    return overlay


def _graph_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    for edge, assignment in zip(edges, assignments):
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        cv2.line(
            overlay,
            _point(p0),
            _point(p1),
            ASSIGNMENT_COLORS.get(int(assignment), ASSIGNMENT_COLORS[3]),
            1,
            lineType=cv2.LINE_AA,
        )
    for x, y in vertices:
        cv2.circle(overlay, (int(round(float(x))), int(round(float(y)))), 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)
    return overlay


def _point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _padding(image_size: int) -> int:
    return max(8, int(32 * image_size / 1024))


def _line_width(image_size: int) -> int:
    return max(1, int(2 * image_size / 768))


if __name__ == "__main__":
    main()
