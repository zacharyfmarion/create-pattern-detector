"""Full-pattern recall diagnostics for VertexRefinerV1 proposal crops."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.vertex_refiner_dataset import VertexRefinerCropDataset, vertex_refiner_collate
from src.data.vertex_refiner_proposals import VertexProposal, crop_origin_for_center
from src.data.vertex_refiner_targets import classify_vertex_kind
from src.data.vertex_refiner_targets import SquareFrame
from src.evaluation.vertex_refiner_eval import _finalize_match_stats, _new_match_stats
from src.evaluation.vertex_refiner_global_merge import (
    VertexMergeConfig,
    merge_decoded_vertices,
    summarize_merge,
)
from src.models.vertex_refiner import DecodedVertex, decode_vertex_refiner_batch
from src.models.vertex_refiner_contract import CROP_SIZE_PX


def proposal_contains_vertex(
    vertex_xy: np.ndarray,
    proposal: VertexProposal,
    *,
    crop_size: int = CROP_SIZE_PX,
    margin_px: float = 0.0,
) -> bool:
    origin_x, origin_y = crop_origin_for_center((proposal.x, proposal.y), crop_size=crop_size)
    x = float(vertex_xy[0])
    y = float(vertex_xy[1])
    margin = float(margin_px)
    return (
        origin_x + margin <= x < origin_x + crop_size - margin
        and origin_y + margin <= y < origin_y + crop_size - margin
    )


def summarize_proposal_coverage(
    dataset: VertexRefinerCropDataset,
    *,
    margin_px: float = 0.0,
) -> dict[str, Any]:
    """Measure how many full-pattern GT vertices are inside selected crop proposals."""
    refs_by_record = _crop_refs_by_record(dataset)
    overall = Counter(total=0, covered=0)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    proposal_counts: list[int] = []
    for record_index in range(len(dataset.records)):
        sample = dataset._render_record(record_index)
        proposals = [ref.proposal for ref in refs_by_record.get(record_index, [])]
        proposal_counts.append(len(proposals))
        for vertex_index, vertex in enumerate(sample.pixel_vertices):
            kind = classify_vertex_kind(
                vertex_index,
                sample.pixel_vertices,
                sample.edges,
                sample.assignments,
                sample.square_frame,
                image_size=int(sample.metadata["image_size"]),
            )
            covered = any(
                proposal_contains_vertex(vertex, proposal, margin_px=margin_px)
                for proposal in proposals
            )
            overall["total"] += 1
            by_kind[kind]["total"] += 1
            if covered:
                overall["covered"] += 1
                by_kind[kind]["covered"] += 1
    return {
        "total_gt_vertices": int(overall["total"]),
        "covered_by_crop": int(overall["covered"]),
        "uncovered_by_crop": int(overall["total"] - overall["covered"]),
        "coverage": _ratio(overall["covered"], overall["total"]),
        "margin_px": float(margin_px),
        "proposal_counts": _proposal_count_summary(proposal_counts),
        "by_kind": {
            kind: {
                "total": int(counts["total"]),
                "covered": int(counts["covered"]),
                "coverage": _ratio(counts["covered"], counts["total"]),
            }
            for kind, counts in sorted(by_kind.items(), key=lambda item: item[0])
        },
    }


@torch.no_grad()
def evaluate_full_pattern_vertex_recall(
    model: torch.nn.Module,
    dataset: VertexRefinerCropDataset,
    *,
    device: torch.device,
    batch_size: int,
    heatmap_threshold: float = 0.25,
    match_tolerance_px: float = 2.0,
    merge_config: VertexMergeConfig | None = None,
) -> dict[str, Any]:
    """Decode crop predictions globally and split misses by crop coverage."""
    was_training = model.training
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vertex_refiner_collate,
    )
    predictions_by_record: dict[str, list[DecodedVertex]] = defaultdict(list)
    for batch in loader:
        inputs = batch["input"].to(device)
        outputs = model(inputs)
        crop_origins = [tuple(meta["crop_origin_xy"]) for meta in batch["meta"]]
        decoded = decode_vertex_refiner_batch(
            outputs,
            crop_origins_xy=crop_origins,
            square_frames=[_square_frame_from_meta(meta) for meta in batch["meta"]],
            heatmap_threshold=heatmap_threshold,
        )
        for pred_vertices, meta in zip(decoded, batch["meta"]):
            predictions_by_record[str(meta["record_id"])].extend(pred_vertices)
    if was_training:
        model.train()

    refs_by_record = _crop_refs_by_record(dataset)
    overall = _new_match_stats()
    split = Counter(
        total_gt=0,
        covered_gt=0,
        matched_gt=0,
        covered_matched_gt=0,
        uncovered_matched_gt=0,
    )
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    decoded_total = 0
    merged_total = 0
    merge_summaries: list[dict[str, Any]] = []
    for record_index in range(len(dataset.records)):
        sample = dataset._render_record(record_index)
        record_id = str(dataset.records[record_index]["id"])
        proposals = [ref.proposal for ref in refs_by_record.get(record_index, [])]
        raw_predictions = predictions_by_record.get(record_id, [])
        decoded_total += len(raw_predictions)
        if merge_config is None:
            predictions = raw_predictions
        else:
            predictions = merge_decoded_vertices(
                raw_predictions,
                proposals=proposals,
                config=merge_config,
            )
            merge_summaries.append(summarize_merge(raw_predictions, predictions))
        merged_total += len(predictions)
        pred_xy = np.asarray([[vertex.x, vertex.y] for vertex in predictions], dtype=np.float32).reshape(
            -1, 2
        )
        matched_gt, matched_pred, errors = _match_points(
            sample.pixel_vertices,
            pred_xy,
            tolerance_px=match_tolerance_px,
        )
        _update_global_match_stats(
            overall,
            gt_count=len(sample.pixel_vertices),
            pred_count=len(predictions),
            matched_pred_count=len(matched_pred),
            errors=errors,
        )
        for vertex_index, vertex in enumerate(sample.pixel_vertices):
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
            split["total_gt"] += 1
            by_kind[kind]["total"] += 1
            if covered:
                split["covered_gt"] += 1
                by_kind[kind]["covered"] += 1
            if matched:
                split["matched_gt"] += 1
                by_kind[kind]["matched"] += 1
                if covered:
                    split["covered_matched_gt"] += 1
                    by_kind[kind]["covered_matched"] += 1
                else:
                    split["uncovered_matched_gt"] += 1
    false_negatives = int(split["total_gt"] - split["matched_gt"])
    no_crop_misses = int(
        (split["total_gt"] - split["covered_gt"]) - split["uncovered_matched_gt"]
    )
    covered_misses = int(split["covered_gt"] - split["covered_matched_gt"])
    return {
        **_finalize_match_stats(overall),
        "decoded_predictions_total_before_global_nms": int(decoded_total),
        "decoded_predictions_total_after_global_merge": int(merged_total),
        "proposal_coverage": _ratio(split["covered_gt"], split["total_gt"]),
        "covered_conditional_recall": _ratio(
            split["covered_matched_gt"],
            split["covered_gt"],
        ),
        "false_negative_split": {
            "due_to_no_crop_coverage": no_crop_misses,
            "due_to_covered_but_not_matched": covered_misses,
            "no_crop_fraction": _ratio(no_crop_misses, false_negatives),
            "covered_miss_fraction": _ratio(covered_misses, false_negatives),
        },
        "by_kind": {
            kind: {
                "total": int(counts["total"]),
                "covered": int(counts["covered"]),
                "proposal_coverage": _ratio(counts["covered"], counts["total"]),
                "matched": int(counts["matched"]),
                "recall": _ratio(counts["matched"], counts["total"]),
                "covered_conditional_recall": _ratio(
                    counts["covered_matched"],
                    counts["covered"],
                ),
            }
            for kind, counts in sorted(by_kind.items(), key=lambda item: item[0])
        },
        "global_merge": None
        if merge_config is None
        else {
            "radius_px": float(merge_config.radius_px),
            "boundary_merge_radius_px": (
                None
                if merge_config.boundary_merge_radius_px is None
                else float(merge_config.boundary_merge_radius_px)
            ),
            "min_score": float(merge_config.min_score),
            "min_member_score": float(merge_config.min_member_score),
            "min_support": int(merge_config.min_support),
            "min_support_fraction": float(merge_config.min_support_fraction),
            "ray_vote_fraction": float(merge_config.ray_vote_fraction),
            "summary": _combine_merge_summaries(merge_summaries),
        },
    }


def _crop_refs_by_record(dataset: VertexRefinerCropDataset) -> dict[int, list[Any]]:
    refs_by_record: dict[int, list[Any]] = defaultdict(list)
    for ref in dataset.crop_refs:
        refs_by_record[int(ref.record_index)].append(ref)
    return refs_by_record


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


def _match_points(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    *,
    tolerance_px: float,
) -> tuple[set[int], set[int], list[float]]:
    gt = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    pred = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    if len(gt) == 0 or len(pred) == 0:
        return set(), set(), []
    distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    candidates = [
        (float(distances[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(len(gt))
        for pred_index in range(len(pred))
        if float(distances[gt_index, pred_index]) <= float(tolerance_px)
    ]
    candidates.sort(key=lambda item: item[0])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    errors: list[float] = []
    for distance, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        errors.append(distance)
    return matched_gt, matched_pred, errors


def _update_global_match_stats(
    stats: dict[str, Any],
    *,
    gt_count: int,
    pred_count: int,
    matched_pred_count: int,
    errors: list[float],
) -> None:
    stats["samples"] += 1
    stats["true_positive"] += int(matched_pred_count)
    stats["false_positive"] += int(pred_count - matched_pred_count)
    stats["false_negative"] += int(gt_count - matched_pred_count)
    stats["errors"].extend(float(error) for error in errors)


def _proposal_count_summary(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(min(counts)),
        "mean": float(np.mean(counts)),
        "max": int(max(counts)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _combine_merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    raw = sum(int(summary["raw_predictions"]) for summary in summaries)
    merged = sum(int(summary["merged_predictions"]) for summary in summaries)
    support_values: list[int] = []
    support_fraction_values: list[float] = []
    for summary in summaries:
        support = summary.get("support_count", {})
        support_fraction = summary.get("support_fraction", {})
        # Per-record summaries only expose aggregate support, so keep global
        # counts conservative here. Detailed support histograms belong in a
        # visualization/report, not the lightweight eval JSON.
        if int(support.get("max", 0)) > 0:
            support_values.append(int(support["max"]))
        if float(support_fraction.get("max", 0.0)) > 0.0:
            support_fraction_values.append(float(support_fraction["mean"]))
    return {
        "raw_predictions": raw,
        "merged_predictions": merged,
        "suppressed_predictions": max(raw - merged, 0),
        "max_record_support": _proposal_count_summary(support_values),
        "mean_record_support_fraction": _float_count_summary(support_fraction_values),
    }


def _float_count_summary(counts: list[float]) -> dict[str, Any]:
    if not counts:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(counts)),
        "mean": float(np.mean(counts)),
        "max": float(max(counts)),
    }
