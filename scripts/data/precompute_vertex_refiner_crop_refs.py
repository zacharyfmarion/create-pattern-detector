#!/usr/bin/env python3
"""Precompute VertexRefiner crop proposal refs for a deterministic dataset slice."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import (  # noqa: E402
    VertexRefinerCropDataset,
    save_vertex_refiner_crop_refs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--line-width", type=int, default=None)
    parser.add_argument("--augment-profile", default="clean")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--proposals-per-sample", type=int, default=128)
    parser.add_argument(
        "--include-gt-training-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match the training/eval dataset setting that will consume this cache.",
    )
    parser.add_argument("--boundary-gt-anchor-repeats", type=int, default=0)
    parser.add_argument("--boundary-gt-anchor-jitter-px", type=float, default=6.0)
    parser.add_argument(
        "--auxiliary-mode",
        choices=["zero", "rendered-labels"],
        default="zero",
    )
    parser.add_argument("--input-version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument(
        "--crop-ref-progress-every",
        type=int,
        default=16,
        help="Print JSON progress every N selected records while refs are generated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = perf_counter()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    dataset = VertexRefinerCropDataset(
        manifest,
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        image_size=args.image_size,
        padding=args.padding,
        line_width=args.line_width,
        augment_profile=args.augment_profile,
        seed=args.seed,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=args.include_gt_training_anchors,
        boundary_gt_anchor_repeats=args.boundary_gt_anchor_repeats,
        boundary_gt_anchor_jitter_px=args.boundary_gt_anchor_jitter_px,
        auxiliary_mode=args.auxiliary_mode,
        input_version=args.input_version,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    save_vertex_refiner_crop_refs(output, dataset)
    summary = {
        "schema": "create-pattern-detector/vertex-refiner-crop-ref-precompute-summary/v1",
        "output": output.as_posix(),
        "records": len(dataset.records),
        "crop_refs": len(dataset.crop_refs),
        "config": dataset.crop_ref_config,
        "elapsed_seconds": perf_counter() - start,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
