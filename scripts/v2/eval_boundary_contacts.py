#!/usr/bin/env python3
"""Evaluate CPLineNet-V2 boundary-contact localization."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES
from src.data.cpline_dataset import CplineFoldDataset, cpline_collate
from src.data.v2_boundary_targets import V2_VERTEX_TYPE_IDS
from src.models import CPLineNet
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode


@dataclass(frozen=True)
class Point:
    side: int
    coord: float
    row: float
    col: float
    score: float = 1.0


@dataclass
class CountStats:
    pred: int = 0
    target: int = 0
    matched: int = 0
    corner_fp: int = 0
    match_coord_errors: list[float] | None = None

    def __post_init__(self) -> None:
        if self.match_coord_errors is None:
            self.match_coord_errors = []

    def add(self, other: CountStats) -> None:
        self.pred += other.pred
        self.target += other.target
        self.matched += other.matched
        self.corner_fp += other.corner_fp
        assert self.match_coord_errors is not None
        assert other.match_coord_errors is not None
        self.match_coord_errors.extend(other.match_coord_errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--augment-profile", choices=AUGMENT_PROFILES, default="clean")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=120)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batchnorm-mode", choices=BATCHNORM_MODES, default="eval")
    parser.add_argument("--thresholds", type=str, default="0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--match-tolerance", type=float, default=0.035)
    parser.add_argument("--side-band-px", type=float, default=None)
    parser.add_argument("--nms-radius-px", type=float, default=None)
    parser.add_argument("--max-per-side", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = CPLineNet(
        backbone=str(config.get("backbone", "tiny")),
        hidden_channels=int(config.get("hidden_channels", 128)),
        v2_heads=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dataset = CplineFoldDataset(
        manifest,
        split=args.split,
        limit=args.count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=args.augment_profile,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cpline_collate,
    )
    side_band_px = args.side_band_px
    if side_band_px is None:
        side_band_px = max(4.0, 0.04 * float(args.image_size))
    nms_radius_px = args.nms_radius_px
    if nms_radius_px is None:
        nms_radius_px = max(5.0, 0.025 * float(args.image_size))

    aggregate: dict[float, CountStats] = {threshold: CountStats() for threshold in thresholds}
    by_profile: dict[float, dict[str, CountStats]] = {
        threshold: defaultdict(CountStats) for threshold in thresholds
    }
    rows: list[dict[str, Any]] = []
    separation = {
        "target_contacts": 0,
        "target_contacts_pred_contact": 0,
        "target_contacts_pred_corner": 0,
        "target_contacts_side_correct": 0,
        "target_contact_coord_abs_error": [],
        "target_corners": 0,
        "target_corners_pred_corner": 0,
        "target_corners_pred_contact": 0,
    }

    with model_eval_with_batchnorm_mode(model, batchnorm_mode=args.batchnorm_mode):
        for batch in tqdm(loader, desc="V2 boundary-contact eval"):
            images = batch["image"].to(device)
            outputs = model(images)
            contact_probs = torch.sigmoid(outputs["boundary_contact_logits"]).cpu().numpy()[:, 0]
            vertex_preds = outputs["vertex_type_logits"].argmax(dim=1).cpu().numpy()
            side_preds = outputs["boundary_side_logits"].argmax(dim=1).cpu().numpy()
            coord_preds = outputs.get("boundary_coord")
            coord_values = None if coord_preds is None else coord_preds.cpu().numpy()[:, 0]

            for item_idx, meta in enumerate(batch["meta"]):
                frame = _frame_from_meta(meta, args.image_size)
                target_points = _target_points(batch, item_idx)
                corner_points = _corner_points(batch["v2_vertex_type"][item_idx].numpy())
                _update_separation(
                    separation,
                    target_points=target_points,
                    corner_points=corner_points,
                    vertex_pred=vertex_preds[item_idx],
                    side_pred=side_preds[item_idx],
                    coord_pred=None if coord_values is None else coord_values[item_idx],
                )
                profile = str(meta["augmentation"].get("selected_profile", args.augment_profile))
                row_thresholds: dict[str, Any] = {}
                for threshold in thresholds:
                    pred_points = _predict_points(
                        contact_probs[item_idx],
                        frame=frame,
                        threshold=threshold,
                        side_band_px=side_band_px,
                        nms_radius_px=nms_radius_px,
                        max_per_side=args.max_per_side,
                    )
                    stats = _match_points(
                        pred_points,
                        target_points,
                        corner_points=corner_points,
                        match_tolerance=args.match_tolerance,
                        corner_tolerance_px=nms_radius_px,
                    )
                    aggregate[threshold].add(stats)
                    by_profile[threshold][profile].add(stats)
                    row_thresholds[f"{threshold:.3f}"] = _stats_payload(stats)
                rows.append(
                    {
                        "id": str(meta["id"]),
                        "profile": profile,
                        "target_contacts": len(target_points),
                        "thresholds": row_thresholds,
                    }
                )

    threshold_payload = {f"{threshold:.3f}": _stats_payload(stats) for threshold, stats in aggregate.items()}
    best_threshold = max(thresholds, key=lambda value: threshold_payload[f"{value:.3f}"]["f1"])
    profile_payload = {
        profile: _stats_payload(stats)
        for profile, stats in sorted(by_profile[best_threshold].items())
    }
    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_config": config,
        "manifest": manifest.as_posix(),
        "split": args.split,
        "augment_profile": args.augment_profile,
        "image_size": args.image_size,
        "count": len(dataset),
        "max_edges": args.max_edges,
        "side_band_px": side_band_px,
        "nms_radius_px": nms_radius_px,
        "match_tolerance": args.match_tolerance,
        "best_threshold": best_threshold,
        "thresholds": threshold_payload,
        "by_profile_at_best_threshold": profile_payload,
        "corner_contact_separation": _separation_payload(separation),
        "rows": rows,
    }
    summary_path = output_dir / "boundary_contact_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_threshold_chart(summary, output_dir / "boundary_contact_thresholds.png")
    _write_profile_chart(summary, output_dir / "boundary_contact_profiles.png")
    print(f"Saved V2 boundary-contact metrics: {summary_path}")
    print(f"Saved threshold chart: {output_dir / 'boundary_contact_thresholds.png'}")
    print(f"Saved profile chart: {output_dir / 'boundary_contact_profiles.png'}")
    print(json.dumps(_brief_summary(summary), indent=2))


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise SystemExit("At least one threshold is required")
    return thresholds


def _frame_from_meta(meta: dict[str, Any], image_size: int) -> dict[str, float]:
    frame = meta.get("augmentation", {}).get("v2_boundary", {}).get("frame", {})
    return {
        "x_min": float(frame.get("x_min", 0.0)),
        "y_min": float(frame.get("y_min", 0.0)),
        "x_max": float(frame.get("x_max", image_size - 1)),
        "y_max": float(frame.get("y_max", image_size - 1)),
    }


def _target_points(batch: dict[str, Any], item_idx: int) -> list[Point]:
    mask = batch["v2_boundary_mask"][item_idx].numpy()
    side = batch["v2_boundary_side"][item_idx].numpy()
    coord = batch["v2_boundary_coord"][item_idx, 0].numpy()
    offset = batch["v2_boundary_offset"][item_idx].permute(1, 2, 0).numpy()
    points: list[Point] = []
    for row, col in np.argwhere(mask):
        points.append(
            Point(
                side=int(side[row, col]),
                coord=float(coord[row, col]),
                row=float(row) + float(offset[row, col, 1]),
                col=float(col) + float(offset[row, col, 0]),
            )
        )
    return points


def _corner_points(vertex_type: np.ndarray) -> list[Point]:
    corner_mask = (vertex_type == V2_VERTEX_TYPE_IDS["corner"]).astype(np.uint8)
    count, labels, _, centroids = cv2.connectedComponentsWithStats(corner_mask, connectivity=8)
    points: list[Point] = []
    for label_idx in range(1, count):
        rows, cols = np.nonzero(labels == label_idx)
        if len(rows) == 0:
            continue
        centroid_col, centroid_row = centroids[label_idx]
        points.append(Point(side=-1, coord=0.0, row=float(centroid_row), col=float(centroid_col)))
    return points


def _predict_points(
    heatmap: np.ndarray,
    *,
    frame: dict[str, float],
    threshold: float,
    side_band_px: float,
    nms_radius_px: float,
    max_per_side: int,
) -> list[Point]:
    local_max = _local_maxima(heatmap, radius=max(1, int(round(nms_radius_px / 2.0))))
    points: list[Point] = []
    for side in range(4):
        side_mask = _side_mask(heatmap.shape, frame=frame, side=side, side_band_px=side_band_px)
        coords = np.argwhere(side_mask & local_max & (heatmap >= threshold))
        candidates = [
            Point(
                side=side,
                coord=_side_coord(row=int(row), col=int(col), frame=frame, side=side),
                row=float(row),
                col=float(col),
                score=float(heatmap[int(row), int(col)]),
            )
            for row, col in coords
        ]
        candidates.sort(key=lambda point: point.score, reverse=True)
        kept: list[Point] = []
        for point in candidates:
            if len(kept) >= max_per_side:
                break
            if all(_point_distance(point, other) > nms_radius_px for other in kept):
                kept.append(point)
        points.extend(kept)
    return points


def _local_maxima(heatmap: np.ndarray, *, radius: int) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    dilated = cv2.dilate(heatmap.astype(np.float32), kernel)
    return heatmap >= dilated - 1e-6


def _side_mask(
    shape: tuple[int, int],
    *,
    frame: dict[str, float],
    side: int,
    side_band_px: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    x_min = frame["x_min"]
    y_min = frame["y_min"]
    x_max = frame["x_max"]
    y_max = frame["y_max"]
    if side == 0:
        return (np.abs(yy - y_min) <= side_band_px) & (xx >= x_min - side_band_px) & (xx <= x_max + side_band_px)
    if side == 1:
        return (np.abs(xx - x_max) <= side_band_px) & (yy >= y_min - side_band_px) & (yy <= y_max + side_band_px)
    if side == 2:
        return (np.abs(yy - y_max) <= side_band_px) & (xx >= x_min - side_band_px) & (xx <= x_max + side_band_px)
    return (np.abs(xx - x_min) <= side_band_px) & (yy >= y_min - side_band_px) & (yy <= y_max + side_band_px)


def _side_coord(*, row: int, col: int, frame: dict[str, float], side: int) -> float:
    if side in (0, 2):
        width = max(1.0, frame["x_max"] - frame["x_min"])
        return float(np.clip((float(col) - frame["x_min"]) / width, 0.0, 1.0))
    height = max(1.0, frame["y_max"] - frame["y_min"])
    return float(np.clip((float(row) - frame["y_min"]) / height, 0.0, 1.0))


def _match_points(
    pred_points: list[Point],
    target_points: list[Point],
    *,
    corner_points: list[Point],
    match_tolerance: float,
    corner_tolerance_px: float,
) -> CountStats:
    unmatched = set(range(len(target_points)))
    matched = 0
    errors: list[float] = []
    for pred in sorted(pred_points, key=lambda point: point.score, reverse=True):
        candidates = [
            (abs(pred.coord - target_points[idx].coord), idx)
            for idx in unmatched
            if target_points[idx].side == pred.side and abs(pred.coord - target_points[idx].coord) <= match_tolerance
        ]
        if not candidates:
            continue
        error, target_idx = min(candidates)
        unmatched.remove(target_idx)
        matched += 1
        errors.append(float(error))
    corner_fp = sum(
        1
        for pred in pred_points
        if any(_point_distance(pred, corner) <= corner_tolerance_px for corner in corner_points)
    )
    return CountStats(
        pred=len(pred_points),
        target=len(target_points),
        matched=matched,
        corner_fp=corner_fp,
        match_coord_errors=errors,
    )


def _point_distance(first: Point, second: Point) -> float:
    return float(np.hypot(first.row - second.row, first.col - second.col))


def _update_separation(
    separation: dict[str, Any],
    *,
    target_points: list[Point],
    corner_points: list[Point],
    vertex_pred: np.ndarray,
    side_pred: np.ndarray,
    coord_pred: np.ndarray | None,
) -> None:
    for point in target_points:
        row = int(round(point.row))
        col = int(round(point.col))
        if not _in_bounds(vertex_pred, row, col):
            continue
        predicted_vertex = int(vertex_pred[row, col])
        separation["target_contacts"] += 1
        if predicted_vertex == V2_VERTEX_TYPE_IDS["boundary_contact"]:
            separation["target_contacts_pred_contact"] += 1
        if predicted_vertex == V2_VERTEX_TYPE_IDS["corner"]:
            separation["target_contacts_pred_corner"] += 1
        if int(side_pred[row, col]) == point.side:
            separation["target_contacts_side_correct"] += 1
        if coord_pred is not None:
            separation["target_contact_coord_abs_error"].append(abs(float(coord_pred[row, col]) - point.coord))

    for corner in corner_points:
        row = int(round(corner.row))
        col = int(round(corner.col))
        if not _in_bounds(vertex_pred, row, col):
            continue
        predicted_vertex = int(vertex_pred[row, col])
        separation["target_corners"] += 1
        if predicted_vertex == V2_VERTEX_TYPE_IDS["corner"]:
            separation["target_corners_pred_corner"] += 1
        if predicted_vertex == V2_VERTEX_TYPE_IDS["boundary_contact"]:
            separation["target_corners_pred_contact"] += 1


def _in_bounds(array: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < array.shape[0] and 0 <= col < array.shape[1]


def _stats_payload(stats: CountStats) -> dict[str, Any]:
    precision = stats.matched / stats.pred if stats.pred else 0.0
    recall = stats.matched / stats.target if stats.target else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = stats.match_coord_errors or []
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": stats.matched,
        "pred": stats.pred,
        "target": stats.target,
        "corner_fp": stats.corner_fp,
        "corner_fp_rate": stats.corner_fp / stats.pred if stats.pred else 0.0,
        "match_coord_mae": float(np.mean(errors)) if errors else None,
    }


def _separation_payload(separation: dict[str, Any]) -> dict[str, Any]:
    contact_count = int(separation["target_contacts"])
    corner_count = int(separation["target_corners"])
    coord_errors = separation["target_contact_coord_abs_error"]
    return {
        "target_contacts": contact_count,
        "target_contact_vertex_accuracy": _ratio(separation["target_contacts_pred_contact"], contact_count),
        "target_contact_as_corner_rate": _ratio(separation["target_contacts_pred_corner"], contact_count),
        "target_contact_side_accuracy": _ratio(separation["target_contacts_side_correct"], contact_count),
        "target_contact_coord_mae": float(np.mean(coord_errors)) if coord_errors else None,
        "target_corners": corner_count,
        "target_corner_vertex_accuracy": _ratio(separation["target_corners_pred_corner"], corner_count),
        "target_corner_as_contact_rate": _ratio(separation["target_corners_pred_contact"], corner_count),
    }


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _write_threshold_chart(summary: dict[str, Any], path: Path) -> None:
    thresholds = [float(key) for key in summary["thresholds"]]
    payloads = [summary["thresholds"][f"{threshold:.3f}"] for threshold in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, [item["precision"] for item in payloads], marker="o", label="precision")
    ax.plot(thresholds, [item["recall"] for item in payloads], marker="o", label="recall")
    ax.plot(thresholds, [item["f1"] for item in payloads], marker="o", label="F1")
    ax.set_xlabel("contact threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Boundary-contact threshold sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_profile_chart(summary: dict[str, Any], path: Path) -> None:
    profiles = list(summary["by_profile_at_best_threshold"])
    if not profiles:
        return
    f1_values = [summary["by_profile_at_best_threshold"][profile]["f1"] for profile in profiles]
    recall_values = [summary["by_profile_at_best_threshold"][profile]["recall"] for profile in profiles]
    x = np.arange(len(profiles))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(profiles)), 4.5))
    ax.bar(x - width / 2, f1_values, width, label="F1")
    ax.bar(x + width / 2, recall_values, width, label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(f"Boundary contacts by profile @ threshold {summary['best_threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _brief_summary(summary: dict[str, Any]) -> dict[str, Any]:
    best_key = f"{summary['best_threshold']:.3f}"
    return {
        "best_threshold": summary["best_threshold"],
        "aggregate": summary["thresholds"][best_key],
        "corner_contact_separation": summary["corner_contact_separation"],
    }


if __name__ == "__main__":
    main()
