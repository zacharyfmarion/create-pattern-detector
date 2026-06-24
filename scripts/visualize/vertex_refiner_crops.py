#!/usr/bin/env python3
"""Generate visual QA contact sheets for VertexRefinerV1 crop targets."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import VertexRefinerCropDataset


BUCKETS = ("positives", "close_pairs", "boundary_contacts", "hard_negatives")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--proposals-per-sample", type=int, default=48)
    parser.add_argument("--per-bucket", type=int, default=16)
    parser.add_argument("--no-gt-anchors", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/vertex_refiner_crops"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VertexRefinerCropDataset(
        manifest,
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=not args.no_gt_anchors,
    )
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    for index in range(len(dataset)):
        item = dataset[index]
        target_meta = item["meta"]["target"]
        bucket = _bucket_for_target(target_meta)
        if len(buckets[bucket]) < args.per_bucket:
            buckets[bucket].append({"index": index, "item": item})
        if all(len(rows) >= args.per_bucket for rows in buckets.values()):
            break

    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for bucket, rows in buckets.items():
        if not rows:
            continue
        sheet = _make_contact_sheet([row["item"] for row in rows])
        path = output_dir / f"{bucket}.png"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        manifest_rows[bucket] = [
            {
                "dataset_index": row["index"],
                "record_id": row["item"]["meta"]["record_id"],
                "crop_origin_xy": list(row["item"]["meta"]["crop_origin_xy"]),
                "proposal": row["item"]["meta"]["proposal"],
                "target": row["item"]["meta"]["target"],
            }
            for row in rows
        ]
        print(f"Saved {bucket}: {path}")

    sidecar = {
        "schema": "create-pattern-detector/vertex-refiner-crop-contact-sheet/v1",
        "manifest": str(manifest),
        "split": args.split,
        "image_size": args.image_size,
        "seed": args.seed,
        "rows": manifest_rows,
    }
    sidecar_path = output_dir / "manifest.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Saved sidecar: {sidecar_path}")
    return 0


def _bucket_for_target(target_meta: dict[str, Any]) -> str:
    if int(target_meta.get("close_pair_count", 0)) > 0:
        return "close_pairs"
    if int(target_meta.get("kind_counts", {}).get("boundary_contact", 0)) > 0:
        return "boundary_contacts"
    if int(target_meta.get("vertex_count", 0)) > 0:
        return "positives"
    return "hard_negatives"


def _make_contact_sheet(items: list[dict[str, Any]], *, columns: int = 4) -> np.ndarray:
    tiles = [_preview_tile(item) for item in items]
    if not tiles:
        raise ValueError("contact sheet needs at least one item")
    tile_h, tile_w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return sheet


def _preview_tile(item: dict[str, Any]) -> np.ndarray:
    inputs = item["input"].detach().cpu().numpy()
    heatmap = item["vertex_heatmap"].detach().cpu().numpy()[0]
    gray = np.clip(inputs[0], 0.0, 1.0)
    ink = np.clip(inputs[1], 0.0, 1.0)
    base = np.repeat((gray * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    overlay = base.astype(np.float32)
    overlay[..., 0] = np.maximum(overlay[..., 0], heatmap * 255.0)
    overlay[..., 1] = np.maximum(overlay[..., 1], ink * 180.0)
    tile = np.clip(overlay, 0, 255).astype(np.uint8)
    target_meta = item["meta"]["target"]
    label = (
        f"v={target_meta.get('vertex_count', 0)} "
        f"cp={target_meta.get('close_pair_count', 0)} "
        f"{','.join(item['meta']['proposal']['provenance'][:2])}"
    )
    cv2.putText(tile, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(tile, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


if __name__ == "__main__":
    raise SystemExit(main())
