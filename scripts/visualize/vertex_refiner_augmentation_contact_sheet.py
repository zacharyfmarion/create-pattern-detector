#!/usr/bin/env python3
"""Render one contact sheet of source-image augmentation profiles."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES  # noqa: E402
from src.data.cpline_dataset import load_manifest_records, render_cpline_sample, resolve_fold_path  # noqa: E402
from src.data.fold_parser import FOLDParser  # noqa: E402


DEFAULT_PROFILES = (
    "clean",
    "square-symmetry",
    "line-style-light",
    "print-light",
    "print-medium-lite",
    "faint-light",
    "vertex-light-rendered",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--variants-per-profile", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated augmentation profiles.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("visualizations/vertex_refiner_augmentations/contact_sheet.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    profiles = _parse_profiles(args.profiles)
    records = _select_records(
        manifest,
        split=args.split,
        count=args.samples,
        max_edges=args.max_edges,
    )
    parser = FOLDParser()
    cps = [(record, parser.parse(resolve_fold_path(record, manifest))) for record in records]
    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        variant_count = 1 if profile == "clean" else max(1, int(args.variants_per_profile))
        for variant_index in range(variant_count):
            images = []
            metadata = []
            for sample_index, (record, cp) in enumerate(cps):
                seed = args.seed + profile_index * 10_000 + variant_index * 1_000 + sample_index
                sample = render_cpline_sample(
                    cp,
                    image_size=args.image_size,
                    padding=_padding(args.image_size),
                    line_width=_line_width(args.image_size),
                    augment_profile=profile,
                    seed=seed,
                )
                images.append(sample.image)
                metadata.append(
                    {
                        "record_id": str(record["id"]),
                        "seed": seed,
                        "selected_profile": sample.metadata.get("selected_profile", profile),
                        "augmentation": sample.metadata,
                    }
                )
            rows.append(
                {
                    "profile": profile,
                    "variant_index": variant_index,
                    "images": images,
                    "metadata": metadata,
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_sheet(rows, records=records, out=out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "create-pattern-detector/vertex-refiner-augmentation-contact-sheet/v1",
                "manifest": manifest.as_posix(),
                "split": args.split,
                "max_edges": args.max_edges,
                "image_size": args.image_size,
                "profiles": profiles,
                "samples": [str(record["id"]) for record in records],
                "rows": [
                    {
                        "profile": row["profile"],
                        "variant_index": row["variant_index"],
                        "metadata": row["metadata"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sheet": out.as_posix(), "sidecar": sidecar.as_posix()}, indent=2))
    return 0


def _parse_profiles(value: str) -> list[str]:
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    unsupported = [profile for profile in profiles if profile not in AUGMENT_PROFILES]
    if unsupported:
        expected = ", ".join(AUGMENT_PROFILES)
        raise SystemExit(f"Unsupported profiles: {', '.join(unsupported)}. Expected one of: {expected}")
    return profiles


def _select_records(
    manifest: Path,
    *,
    split: str,
    count: int,
    max_edges: int | None,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in load_manifest_records(manifest)
        if record.get("split") == split and (max_edges is None or int(record["edges"]) <= max_edges)
    ]
    if not records:
        raise SystemExit(f"No records found in {manifest} for split={split}")
    return records[:count]


def _write_sheet(rows: list[dict[str, Any]], *, records: list[dict[str, Any]], out: Path) -> None:
    sample_count = len(records)
    fig, axes = plt.subplots(
        len(rows),
        sample_count,
        figsize=(3.2 * sample_count, max(2.4, 2.45 * len(rows))),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        selected_profiles = [str(meta["selected_profile"]) for meta in row["metadata"]]
        row_label = str(row["profile"])
        if len(set(selected_profiles)) == 1 and selected_profiles[0] != row["profile"]:
            row_label = f"{row_label}\n-> {selected_profiles[0]}"
        elif len(set(selected_profiles)) > 1:
            row_label = f"{row_label}\n-> mixed"
        if int(row["variant_index"]) > 0:
            row_label = f"{row_label}\nvariant {row['variant_index'] + 1}"
        for col_index, image in enumerate(row["images"]):
            ax = axes[row_index, col_index]
            ax.imshow(image)
            ax.axis("off")
            if row_index == 0:
                title = str(records[col_index]["id"])
                ax.set_title(title[:28], fontsize=8)
            if col_index == 0:
                ax.text(
                    -0.02,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
    fig.suptitle("VertexRefiner Source Augmentation Profiles", fontsize=13, y=0.998)
    fig.tight_layout(rect=(0.12, 0, 1, 0.992))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _padding(image_size: int) -> int:
    return max(8, int(32 * image_size / 1024))


def _line_width(image_size: int) -> int:
    return max(1, int(2 * image_size / 768))


if __name__ == "__main__":
    raise SystemExit(main())
