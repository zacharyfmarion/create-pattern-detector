#!/usr/bin/env python3
"""Sweep VertexRefiner proposal caps with one shared inference pass."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import CropRef, VertexRefinerCropDataset, vertex_refiner_collate
from src.data.vertex_refiner_proposals import (
    ProposalConfig,
    VertexProposal,
    generate_vertex_refiner_proposals,
    select_vertex_refiner_proposals,
)
from src.data.vertex_refiner_targets import SquareFrame, classify_vertex_kind
from src.evaluation.vertex_refiner_global_merge import VertexMergeConfig, merge_decoded_vertices
from src.evaluation.vertex_refiner_recall_diagnostics import _match_points, proposal_contains_vertex
from src.models import VertexRefinerV1, VertexRefinerV2, VertexRefinerV3
from src.models.vertex_refiner import DecodedVertex, decode_vertex_refiner_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--caps",
        default="128,160,192,256",
        help="Comma-separated per-record proposal caps to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--model-version", choices=["v1", "v2", "v3"], default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--auxiliary-mode", choices=["zero", "rendered-labels"], default=None)
    parser.add_argument("--rendered-sample-cache-size", type=int, default=4)
    parser.add_argument("--crop-ref-progress-every", type=int, default=0)
    parser.add_argument("--heatmap-threshold", type=float, default=0.25)
    parser.add_argument("--match-tolerance-px", type=float, default=2.0)
    parser.add_argument("--merge-radius-px", type=float, default=2.0)
    parser.add_argument("--merge-boundary-radius-px", type=float, default=None)
    parser.add_argument("--merge-min-score", type=float, default=0.25)
    parser.add_argument("--merge-min-member-score", type=float, default=0.0)
    parser.add_argument("--merge-min-support", type=int, default=1)
    parser.add_argument("--merge-min-support-fraction", type=float, default=0.45)
    parser.add_argument("--merge-ray-vote-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    caps = _parse_caps(args.caps)
    if not caps:
        raise ValueError("At least one cap is required")
    max_cap = max(caps)
    device = _select_device(args.device)
    checkpoint_path = _resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = args.base_channels or int(config.get("base_channels", 48))
    model_version = args.model_version or str(config.get("model_version", "v1"))
    auxiliary_mode = args.auxiliary_mode or str(config.get("auxiliary_mode", "zero"))
    model_cls = _model_class(model_version)
    model = model_cls(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest = _resolve(args.manifest)
    dataset = VertexRefinerCropDataset(
        manifest,
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        proposals_per_sample=max_cap,
        include_gt_training_anchors=False,
        auxiliary_mode=auxiliary_mode,
        input_version=model_version,
        rendered_sample_cache_size=args.rendered_sample_cache_size,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    selected_ref_indices_by_cap = _replace_with_union_crop_refs(dataset, caps)
    decoded_by_ref = _decode_all_crop_refs(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        heatmap_threshold=args.heatmap_threshold,
    )
    merge_config = VertexMergeConfig(
        radius_px=args.merge_radius_px,
        boundary_merge_radius_px=args.merge_boundary_radius_px,
        min_score=args.merge_min_score,
        min_member_score=args.merge_min_member_score,
        min_support=args.merge_min_support,
        min_support_fraction=args.merge_min_support_fraction,
        ray_vote_fraction=args.merge_ray_vote_fraction,
    )
    cap_reports = [
        _evaluate_cap(
            dataset,
            decoded_by_ref,
            cap=cap,
            selected_ref_indices_by_record=selected_ref_indices_by_cap[cap],
            merge_config=merge_config,
            match_tolerance_px=args.match_tolerance_px,
        )
        for cap in caps
    ]
    report = {
        "schema": "create-pattern-detector/vertex-refiner-proposal-cap-sweep/v1",
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest.as_posix(),
        "split": args.split,
        "limit": args.limit,
        "max_edges": args.max_edges,
        "image_size": args.image_size,
        "base_channels": base_channels,
        "model_version": model_version,
        "auxiliary_mode": auxiliary_mode,
        "seed": args.seed,
        "heatmap_threshold": args.heatmap_threshold,
        "match_tolerance_px": args.match_tolerance_px,
        "merge_config": {
            "radius_px": merge_config.radius_px,
            "boundary_merge_radius_px": merge_config.boundary_merge_radius_px,
            "min_score": merge_config.min_score,
            "min_member_score": merge_config.min_member_score,
            "min_support": merge_config.min_support,
            "min_support_fraction": merge_config.min_support_fraction,
            "ray_vote_fraction": merge_config.ray_vote_fraction,
        },
        "union_inference_crop_refs": len(dataset.crop_refs),
        "caps": cap_reports,
    }
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": out.as_posix(), "caps": cap_reports}, indent=2), flush=True)
    return 0


@torch.no_grad()
def _decode_all_crop_refs(
    model: torch.nn.Module,
    dataset: VertexRefinerCropDataset,
    *,
    device: torch.device,
    batch_size: int,
    heatmap_threshold: float,
) -> list[list[DecodedVertex]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vertex_refiner_collate,
    )
    decoded_by_ref: list[list[DecodedVertex]] = [[] for _ in dataset.crop_refs]
    cursor = 0
    for batch in loader:
        outputs = model(batch["input"].to(device))
        decoded = decode_vertex_refiner_batch(
            outputs,
            crop_origins_xy=[tuple(meta["crop_origin_xy"]) for meta in batch["meta"]],
            square_frames=[_square_frame_from_meta(meta) for meta in batch["meta"]],
            heatmap_threshold=heatmap_threshold,
        )
        for vertices in decoded:
            decoded_by_ref[cursor] = vertices
            cursor += 1
    if cursor != len(decoded_by_ref):
        raise RuntimeError(f"Decoded {cursor} crop refs, expected {len(decoded_by_ref)}")
    return decoded_by_ref


def _evaluate_cap(
    dataset: VertexRefinerCropDataset,
    decoded_by_ref: list[list[DecodedVertex]],
    *,
    cap: int,
    selected_ref_indices_by_record: dict[int, list[int]],
    merge_config: VertexMergeConfig,
    match_tolerance_px: float,
) -> dict[str, Any]:
    overall = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    false_negative_split = Counter()
    coverage_counts = Counter()
    decoded_total = 0
    merged_total = 0
    proposal_counts: list[int] = []
    record_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(dataset.records):
        sample = dataset._render_record(record_index)
        selected_ref_indices = selected_ref_indices_by_record.get(record_index, [])
        proposal_counts.append(len(selected_ref_indices))
        proposals = [dataset.crop_refs[index].proposal for index in selected_ref_indices]
        raw_predictions = [
            vertex
            for index in selected_ref_indices
            for vertex in decoded_by_ref[index]
        ]
        decoded_total += len(raw_predictions)
        predictions = merge_decoded_vertices(
            raw_predictions,
            proposals=proposals,
            config=merge_config,
        )
        merged_total += len(predictions)
        gt_xy = np.asarray(sample.pixel_vertices, dtype=np.float32).reshape(-1, 2)
        pred_xy = np.asarray([[vertex.x, vertex.y] for vertex in predictions], dtype=np.float32).reshape(
            -1, 2
        )
        matched_gt, matched_pred, errors = _match_points(
            gt_xy,
            pred_xy,
            tolerance_px=match_tolerance_px,
        )
        record_counts = Counter(
            gt=len(gt_xy),
            pred=len(predictions),
            matched=len(matched_gt),
            false_positive=len(predictions) - len(matched_pred),
            false_negative=len(gt_xy) - len(matched_gt),
        )
        overall.update(record_counts)
        for vertex_index, vertex in enumerate(gt_xy):
            kind = classify_vertex_kind(
                vertex_index,
                sample.pixel_vertices,
                sample.edges,
                sample.assignments,
                sample.square_frame,
                image_size=int(sample.metadata["image_size"]),
            )
            covered = any(proposal_contains_vertex(vertex, proposal) for proposal in proposals)
            matched = vertex_index in matched_gt
            by_kind[kind]["total"] += 1
            if covered:
                coverage_counts["covered_gt"] += 1
                by_kind[kind]["covered"] += 1
            if matched:
                coverage_counts["matched_gt"] += 1
                by_kind[kind]["matched"] += 1
                if covered:
                    coverage_counts["covered_matched_gt"] += 1
                else:
                    coverage_counts["uncovered_matched_gt"] += 1
            else:
                if covered:
                    false_negative_split["covered_but_not_matched"] += 1
                else:
                    false_negative_split["no_crop_coverage"] += 1
        record_rows.append(
            {
                "record_index": record_index,
                "record_id": str(record["id"]),
                "proposal_count": len(selected_ref_indices),
                "gt_vertices": int(record_counts["gt"]),
                "predictions": int(record_counts["pred"]),
                "matched": int(record_counts["matched"]),
                "false_positive": int(record_counts["false_positive"]),
                "false_negative": int(record_counts["false_negative"]),
                "mean_error_px": float(np.mean(errors)) if errors else None,
            }
        )
    precision = _ratio(overall["matched"], overall["pred"])
    recall = _ratio(overall["matched"], overall["gt"])
    false_negative_count = overall["false_negative"]
    total_gt = sum(counts["total"] for counts in by_kind.values())
    return {
        "cap": cap,
        "proposal_counts": _count_summary(proposal_counts),
        "decoded_predictions_total_before_global_nms": int(decoded_total),
        "decoded_predictions_total_after_global_merge": int(merged_total),
        "metrics": {
            "true_positive": int(overall["matched"]),
            "false_positive": int(overall["false_positive"]),
            "false_negative": int(false_negative_count),
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2.0 * precision * recall, precision + recall),
        },
        "proposal_coverage": _ratio(coverage_counts["covered_gt"], total_gt),
        "covered_conditional_recall": _ratio(
            coverage_counts["covered_matched_gt"],
            coverage_counts["covered_gt"],
        ),
        "false_negative_split": {
            "due_to_no_crop_coverage": int(false_negative_split["no_crop_coverage"]),
            "due_to_covered_but_not_matched": int(false_negative_split["covered_but_not_matched"]),
            "no_crop_fraction": _ratio(false_negative_split["no_crop_coverage"], false_negative_count),
            "covered_miss_fraction": _ratio(
                false_negative_split["covered_but_not_matched"],
                false_negative_count,
            ),
        },
        "by_kind": {
            kind: {
                "total": int(counts["total"]),
                "covered": int(counts["covered"]),
                "matched": int(counts["matched"]),
                "proposal_coverage": _ratio(counts["covered"], counts["total"]),
                "recall": _ratio(counts["matched"], counts["total"]),
                "covered_conditional_recall": _ratio(counts["matched"], counts["covered"]),
            }
            for kind, counts in sorted(by_kind.items(), key=lambda item: item[0])
        },
        "worst_records": sorted(
            record_rows,
            key=lambda row: (row["false_negative"] + row["false_positive"], row["false_negative"]),
            reverse=True,
        )[:8],
    }


def _replace_with_union_crop_refs(
    dataset: VertexRefinerCropDataset,
    caps: list[int],
) -> dict[int, dict[int, list[int]]]:
    """Replace dataset crop refs with the union of exact per-cap selections."""
    selected_ref_indices_by_cap: dict[int, dict[int, list[int]]] = {
        cap: defaultdict(list) for cap in caps
    }
    union_refs: list[CropRef] = []
    union_index_by_key: dict[tuple[Any, ...], int] = {}
    for record_index in range(len(dataset.records)):
        sample = dataset._render_record(record_index)
        proposals = generate_vertex_refiner_proposals(
            source_ink_probability=sample.source_ink_probability,
            junction_probability=sample.cpline_junction_probability,
            junction_offset=sample.cpline_junction_offset,
            square_frame=sample.square_frame,
            gt_vertices=sample.pixel_vertices,
            include_gt_training_anchors=False,
            config=ProposalConfig(crop_size=96),
        )
        for cap in caps:
            selected = select_vertex_refiner_proposals(
                proposals,
                max_count=cap,
                crop_size=96,
                image_shape=sample.source_ink_probability.shape,
            )
            for proposal in selected:
                key = _proposal_union_key(record_index, proposal)
                union_index = union_index_by_key.get(key)
                if union_index is None:
                    union_index = len(union_refs)
                    union_index_by_key[key] = union_index
                    union_refs.append(CropRef(record_index=record_index, proposal=proposal))
                selected_ref_indices_by_cap[cap][record_index].append(union_index)
    dataset.crop_refs = union_refs
    dataset.crop_refs_source = "exact_cap_selection_union"
    return {
        cap: {record_index: list(indices) for record_index, indices in by_record.items()}
        for cap, by_record in selected_ref_indices_by_cap.items()
    }


def _proposal_union_key(record_index: int, proposal: VertexProposal) -> tuple[Any, ...]:
    return (
        int(record_index),
        round(float(proposal.x), 6),
        round(float(proposal.y), 6),
        round(float(proposal.score), 6),
        tuple(proposal.provenance),
    )


def _square_frame_from_meta(meta: dict[str, Any]) -> SquareFrame | None:
    frame = meta.get("square_frame") if isinstance(meta, dict) else None
    if not isinstance(frame, dict):
        return None
    return SquareFrame(
        x_min=float(frame["x_min"]),
        y_min=float(frame["y_min"]),
        x_max=float(frame["x_max"]),
        y_max=float(frame["y_max"]),
    )


def _count_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(min(values)),
        "mean": float(np.mean(values)),
        "max": int(max(values)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _parse_caps(value: str) -> list[int]:
    caps = [int(part.strip()) for part in value.split(",") if part.strip()]
    return sorted(set(caps))


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _model_class(model_version: str) -> type[torch.nn.Module]:
    if model_version == "v1":
        return VertexRefinerV1
    if model_version == "v2":
        return VertexRefinerV2
    if model_version == "v3":
        return VertexRefinerV3
    raise ValueError(f"Unsupported model_version: {model_version}")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
