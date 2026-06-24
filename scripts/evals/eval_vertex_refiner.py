#!/usr/bin/env python3
"""Evaluate a VertexRefinerV1 checkpoint on crop-level metrics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import VertexRefinerCropDataset, vertex_refiner_collate
from src.evaluation.vertex_refiner_eval import evaluate_vertex_refiner
from src.evaluation.vertex_refiner_global_merge import VertexMergeConfig
from src.evaluation.vertex_refiner_recall_diagnostics import (
    evaluate_full_pattern_vertex_recall,
    summarize_proposal_coverage,
)
from src.models import VertexRefinerV1, VertexRefinerV2
from src.models.losses import VertexRefinerLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--proposals-per-sample", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument(
        "--rendered-sample-cache-size",
        type=int,
        default=None,
        help="Optional max number of rendered full-pattern samples cached by the dataset.",
    )
    parser.add_argument("--model-version", choices=["v1", "v2"], default=None)
    parser.add_argument(
        "--crop-refs",
        type=Path,
        default=None,
        help="Optional precomputed crop-ref cache for this eval dataset selection.",
    )
    parser.add_argument(
        "--crop-ref-progress-every",
        type=int,
        default=0,
        help="Print dataset crop-ref construction progress every N selected records.",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--auxiliary-mode",
        choices=["zero", "rendered-labels"],
        default=None,
        help="Channels 4-6 source. Defaults to checkpoint config, then zero.",
    )
    parser.add_argument(
        "--include-gt-training-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include GT-centered proposals during eval. Keep false for product-style metrics.",
    )
    parser.add_argument("--heatmap-threshold", type=float, default=0.25)
    parser.add_argument("--match-tolerance-px", type=float, default=2.0)
    parser.add_argument(
        "--full-pattern-diagnostics",
        action="store_true",
        help="Also report proposal coverage and full-pattern decoded recall.",
    )
    parser.add_argument(
        "--global-merge",
        action="store_true",
        help="Merge overlapping crop predictions before full-pattern matching.",
    )
    parser.add_argument("--merge-radius-px", type=float, default=3.0)
    parser.add_argument("--merge-boundary-radius-px", type=float, default=None)
    parser.add_argument("--merge-min-score", type=float, default=0.0)
    parser.add_argument("--merge-min-member-score", type=float, default=0.0)
    parser.add_argument("--merge-min-support", type=int, default=1)
    parser.add_argument("--merge-min-support-fraction", type=float, default=0.0)
    parser.add_argument("--merge-ray-vote-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _select_device(args.device)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = args.base_channels or int(config.get("base_channels", 48))
    model_version = args.model_version or str(config.get("model_version", "v1"))
    auxiliary_mode = args.auxiliary_mode or str(config.get("auxiliary_mode", "zero"))
    model_cls = VertexRefinerV2 if model_version == "v2" else VertexRefinerV1
    model = model_cls(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    crop_refs = args.crop_refs if args.crop_refs is None or args.crop_refs.is_absolute() else REPO_ROOT / args.crop_refs
    dataset = VertexRefinerCropDataset(
        manifest,
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=args.include_gt_training_anchors,
        auxiliary_mode=auxiliary_mode,
        input_version=model_version,
        rendered_sample_cache_size=args.rendered_sample_cache_size,
        crop_refs_path=crop_refs,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vertex_refiner_collate,
    )
    metrics = evaluate_vertex_refiner(
        model,
        loader,
        device=device,
        criterion=VertexRefinerLoss(),
        heatmap_threshold=args.heatmap_threshold,
        match_tolerance_px=args.match_tolerance_px,
    )
    report = {
        "schema": "create-pattern-detector/vertex-refiner-crop-eval/v1",
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest.as_posix(),
        "split": args.split,
        "image_size": args.image_size,
        "base_channels": base_channels,
        "model_version": model_version,
        "auxiliary_mode": auxiliary_mode,
        "crop_refs": None if crop_refs is None else crop_refs.as_posix(),
        "crop_refs_source": dataset.crop_refs_source,
        "rendered_sample_cache_size": args.rendered_sample_cache_size,
        "include_gt_training_anchors": args.include_gt_training_anchors,
        "metrics": metrics,
    }
    if args.full_pattern_diagnostics:
        report["proposal_coverage"] = summarize_proposal_coverage(dataset)
        merge_config = (
            VertexMergeConfig(
                radius_px=args.merge_radius_px,
                boundary_merge_radius_px=args.merge_boundary_radius_px,
                min_score=args.merge_min_score,
                min_member_score=args.merge_min_member_score,
                min_support=args.merge_min_support,
                min_support_fraction=args.merge_min_support_fraction,
                ray_vote_fraction=args.merge_ray_vote_fraction,
            )
            if args.global_merge
            else None
        )
        report["full_pattern_metrics"] = evaluate_full_pattern_vertex_recall(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
            heatmap_threshold=args.heatmap_threshold,
            match_tolerance_px=args.match_tolerance_px,
            merge_config=merge_config,
        )
    print(json.dumps(report, indent=2), flush=True)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
