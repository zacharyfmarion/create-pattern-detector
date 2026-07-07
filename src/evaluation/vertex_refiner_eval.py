"""Crop-level evaluation helpers for VertexRefinerV1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.losses.vertex_refiner_loss import VertexRefinerLoss
from src.models.vertex_refiner import DecodedVertex, decode_vertex_refiner_batch
from src.data.vertex_refiner_targets import SquareFrame

VERTEX_REFINER_TARGET_KEYS = (
    "vertex_heatmap",
    "boundary_contact_heatmap",
    "vertex_offset",
    "vertex_offset_mask",
    "vertex_kind",
    "vertex_kind_mask",
    "boundary_side",
    "boundary_side_mask",
    "degree",
    "degree_mask",
    "incident_rays",
    "incident_ray_mask",
)


def vertex_refiner_targets_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(device) for key in VERTEX_REFINER_TARGET_KEYS}


@torch.no_grad()
def evaluate_vertex_refiner(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: VertexRefinerLoss | None = None,
    heatmap_threshold: float = 0.25,
    match_tolerance_px: float = 2.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    loss_batches = 0
    overall = _new_match_stats()
    slices: dict[str, dict[str, Any]] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device)
        outputs = model(inputs)
        if criterion is not None:
            losses = criterion(outputs, vertex_refiner_targets_to_device(batch, device))
            total_loss += float(losses["total"].detach().cpu())
            loss_batches += 1
        decoded = decode_vertex_refiner_batch(
            outputs,
            crop_origins_xy=[(0.0, 0.0)] * inputs.shape[0],
            square_frames=[
                _local_square_frame_from_meta(meta)
                for meta in batch.get("meta", [{}] * inputs.shape[0])
            ],
            heatmap_threshold=heatmap_threshold,
        )
        for pred_vertices, gt_vertices, meta in zip(
            decoded,
            batch["local_vertices"],
            batch.get("meta", [{}] * len(decoded)),
        ):
            tp, fp, fn, matched_errors = match_decoded_vertices(
                pred_vertices,
                gt_vertices.detach().cpu().numpy(),
                tolerance_px=match_tolerance_px,
            )
            _update_match_stats(overall, tp, fp, fn, matched_errors)
            for slice_name in vertex_refiner_slice_names(meta):
                stats = slices.setdefault(slice_name, _new_match_stats())
                _update_match_stats(stats, tp, fp, fn, matched_errors)
    if was_training:
        model.train()
    result = {
        **_finalize_match_stats(overall),
        "loss": None if loss_batches == 0 else total_loss / loss_batches,
        "slices": {
            name: _finalize_match_stats(stats)
            for name, stats in sorted(slices.items(), key=lambda item: item[0])
        },
    }
    return result


def match_decoded_vertices(
    pred_vertices: list[DecodedVertex],
    gt_vertices: np.ndarray,
    *,
    tolerance_px: float,
) -> tuple[int, int, int, list[float]]:
    pred = np.asarray([[vertex.x, vertex.y] for vertex in pred_vertices], dtype=np.float32).reshape(
        -1, 2
    )
    gt = np.asarray(gt_vertices, dtype=np.float32).reshape(-1, 2)
    if len(gt) == 0:
        return 0, len(pred), 0, []
    if len(pred) == 0:
        return 0, 0, len(gt), []
    distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    candidates = [
        (float(distances[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(len(gt))
        for pred_index in range(len(pred))
        if float(distances[gt_index, pred_index]) <= tolerance_px
    ]
    candidates.sort(key=lambda item: item[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    errors: list[float] = []
    for distance, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        errors.append(distance)
    tp = len(errors)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, errors


def vertex_refiner_slice_names(meta: dict[str, Any]) -> list[str]:
    """Return deterministic hard-slice labels for one crop's target metadata."""
    target = meta.get("target", {}) if isinstance(meta, dict) else {}
    kind_counts = target.get("kind_counts", {}) if isinstance(target, dict) else {}
    names: list[str] = []
    vertex_count = int(target.get("vertex_count", 0)) if isinstance(target, dict) else 0
    names.append("positive" if vertex_count > 0 else "empty")
    if int(target.get("close_pair_count", 0)) > 0:
        names.append("close_pair")
    if int(kind_counts.get("boundary_contact", 0)) > 0:
        names.append("boundary_contact")
    if int(kind_counts.get("corner", 0)) > 0:
        names.append("corner")
    if int(kind_counts.get("endpoint_or_dangling", 0)) > 0:
        names.append("endpoint_or_dangling")
    proposal = meta.get("proposal", {}) if isinstance(meta, dict) else {}
    provenance = proposal.get("provenance", []) if isinstance(proposal, dict) else []
    if "gt_training_anchor" in provenance:
        names.append("gt_training_anchor")
    return names


def _local_square_frame_from_meta(meta: dict[str, Any]) -> SquareFrame | None:
    frame = meta.get("square_frame") if isinstance(meta, dict) else None
    origin = meta.get("crop_origin_xy") if isinstance(meta, dict) else None
    if not isinstance(frame, dict) or origin is None:
        return None
    origin_x, origin_y = origin
    return SquareFrame(
        x_min=float(frame["x_min"]) - float(origin_x),
        y_min=float(frame["y_min"]) - float(origin_y),
        x_max=float(frame["x_max"]) - float(origin_x),
        y_max=float(frame["y_max"]) - float(origin_y),
    )


def _new_match_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "errors": [],
    }


def _update_match_stats(
    stats: dict[str, Any],
    tp: int,
    fp: int,
    fn: int,
    errors: list[float],
) -> None:
    stats["samples"] += 1
    stats["true_positive"] += int(tp)
    stats["false_positive"] += int(fp)
    stats["false_negative"] += int(fn)
    stats["errors"].extend(float(error) for error in errors)


def _finalize_match_stats(stats: dict[str, Any]) -> dict[str, Any]:
    tp = int(stats["true_positive"])
    fp = int(stats["false_positive"])
    fn = int(stats["false_negative"])
    errors = list(stats["errors"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "samples": int(stats["samples"]),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error_px": None if not errors else float(np.mean(errors)),
        "max_error_px": None if not errors else float(np.max(errors)),
    }
