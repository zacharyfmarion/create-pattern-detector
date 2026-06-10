#!/usr/bin/env python3
"""Close-pair junction resolution eval.

Measures whether a CPLineNet checkpoint can resolve GT vertex pairs closer
than ~8px into two distinct vertices — the failure mode documented in
docs/v2-close-pair-junction-recovery.md (the junction heatmap fuses such
pairs into a single blob and legacy sub-pixel offsets cannot split them).

Two decoders are reported side by side:
- peaks: legacy local-maxima + anchor sub-pixel offset refinement;
- offset-cluster: all pixels above threshold vote at (pixel + offset * radius)
  and votes are mean-shift-clustered, which can recover two vertices from one
  fused blob when the model was trained with radius offsets.

Usage:
    .venv/bin/python scripts/evals/eval_close_pairs.py \
        --checkpoint checkpoints/<run>/latest.pt \
        --offset-radius-px 3.0 --val-count 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from src.data.cpline_dataset import CplineFoldDataset
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode
from src.models.cpline_net import CPLineNet

PAIR_DISTANCE_PX = 8.0
MATCH_TOLERANCE_PX = 1.5
JUNCTION_THRESHOLD = 0.5
CLUSTER_BANDWIDTH_PX = 1.5
MIN_CLUSTER_SUPPORT = 2.0
NMS_RADIUS_PX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="hrnet_w18")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument(
        "--offset-radius-px",
        type=float,
        default=0.0,
        help="Offset normalization radius the checkpoint was trained with (0 = legacy sub-pixel).",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--batchnorm-mode",
        choices=BATCHNORM_MODES,
        default="batch-stats",
        help="BatchNorm behavior at inference; these checkpoints require batch-stats.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_peaks(probs: np.ndarray, offsets: np.ndarray, *, offset_scale: float) -> np.ndarray:
    """Legacy decode: local maxima + per-anchor offset refinement (vectorized)."""
    size = probs.shape[0]
    window_max = probs.copy()
    for dy in range(-NMS_RADIUS_PX, NMS_RADIUS_PX + 1):
        for dx in range(-NMS_RADIUS_PX, NMS_RADIUS_PX + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.full_like(probs, -np.inf)
            ys = slice(max(0, dy), size + min(0, dy))
            xs = slice(max(0, dx), size + min(0, dx))
            ys_src = slice(max(0, -dy), size + min(0, -dy))
            xs_src = slice(max(0, -dx), size + min(0, -dx))
            shifted[ys, xs] = probs[ys_src, xs_src]
            window_max = np.maximum(window_max, shifted)
    peak_mask = (probs >= JUNCTION_THRESHOLD) & (probs >= window_max)
    peak_mask[:NMS_RADIUS_PX, :] = False
    peak_mask[-NMS_RADIUS_PX:, :] = False
    peak_mask[:, :NMS_RADIUS_PX] = False
    peak_mask[:, -NMS_RADIUS_PX:] = False
    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    px = xs + offsets[0, ys, xs] * offset_scale
    py = ys + offsets[1, ys, xs] * offset_scale
    return np.stack([px, py], axis=1).astype(np.float32)


def decode_offset_clusters(
    probs: np.ndarray,
    offsets: np.ndarray,
    *,
    offset_scale: float,
) -> np.ndarray:
    """Offset-vote decode: threshold pixels vote at pixel + offset; mean-shift."""
    ys, xs = np.where(probs >= JUNCTION_THRESHOLD)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    votes_x = xs + offsets[0, ys, xs] * offset_scale
    votes_y = ys + offsets[1, ys, xs] * offset_scale
    weights = probs[ys, xs]
    order = np.argsort(-weights)
    votes = np.stack([votes_x, votes_y], axis=1)[order]
    weights = weights[order]

    centers: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    mass: list[float] = []
    for vote, weight in zip(votes, weights):
        assigned = False
        for index, center in enumerate(centers):
            if np.hypot(*(vote - center)) <= CLUSTER_BANDWIDTH_PX:
                sums[index] += vote * weight
                mass[index] += weight
                centers[index] = sums[index] / mass[index]
                assigned = True
                break
        if not assigned:
            centers.append(vote.copy())
            sums.append(vote * weight)
            mass.append(float(weight))
    kept = [center for center, support in zip(centers, mass) if support >= MIN_CLUSTER_SUPPORT]
    return np.array(kept, dtype=np.float32).reshape(-1, 2)


def close_pairs(vertices: np.ndarray, degrees: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(vertices)):
        if degrees[i] < 1:
            continue
        for j in range(i + 1, len(vertices)):
            if degrees[j] < 1:
                continue
            if np.hypot(*(vertices[i] - vertices[j])) < PAIR_DISTANCE_PX:
                pairs.append((i, j))
    return pairs


def pair_resolved(pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    if len(pred) == 0:
        return False
    dist_a = np.hypot(pred[:, 0] - a[0], pred[:, 1] - a[1])
    dist_b = np.hypot(pred[:, 0] - b[0], pred[:, 1] - b[1])
    best_a = int(np.argmin(dist_a))
    best_b = int(np.argmin(dist_b))
    return (
        dist_a[best_a] <= MATCH_TOLERANCE_PX
        and dist_b[best_b] <= MATCH_TOLERANCE_PX
        and best_a != best_b
    )


def junction_pr(pred: np.ndarray, gt: np.ndarray, tolerance: float = 3.0) -> tuple[int, int, int]:
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    used = np.zeros(len(pred), dtype=bool)
    tp = 0
    for vertex in gt:
        dist = np.hypot(pred[:, 0] - vertex[0], pred[:, 1] - vertex[1])
        dist[used] = np.inf
        best = int(np.argmin(dist))
        if dist[best] <= tolerance:
            used[best] = True
            tp += 1
    return tp, int((~used).sum()), len(gt) - tp


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    offset_clamp = 1.0 if args.offset_radius_px > 0 else 0.5
    model = CPLineNet(
        backbone=args.backbone,
        pretrained=False,
        hidden_channels=args.hidden_channels,
        v2_heads=True,
        junction_offset_clamp=offset_clamp,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dataset = CplineFoldDataset(
        args.manifest,
        split="val",
        limit=args.val_count,
        image_size=args.image_size,
        augment_profile="clean",
        seed=args.seed,
    )
    offset_scale = args.offset_radius_px if args.offset_radius_px > 0 else 1.0

    totals = {
        "samples": 0,
        "pairs": 0,
        "pairs_resolved_peaks": 0,
        "pairs_resolved_clusters": 0,
        "tp_peaks": 0,
        "fp_peaks": 0,
        "fn_peaks": 0,
        "tp_clusters": 0,
        "fp_clusters": 0,
        "fn_clusters": 0,
    }
    with torch.no_grad(), model_eval_with_batchnorm_mode(model, batchnorm_mode=args.batchnorm_mode):
        for index in range(len(dataset)):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            outputs = model(image)
            probs = torch.sigmoid(outputs["junction_logits"])[0, 0].cpu().numpy()
            offsets = outputs["junction_offset"][0].cpu().numpy()

            vertices = item["graph"]["vertices"].numpy()
            edges = item["graph"]["edges"].numpy()
            degrees = np.zeros(len(vertices), dtype=np.int32)
            for v1, v2 in edges:
                degrees[int(v1)] += 1
                degrees[int(v2)] += 1
            gt_active = vertices[degrees >= 1]

            pred_peaks = decode_peaks(probs, offsets, offset_scale=offset_scale)
            pred_clusters = decode_offset_clusters(probs, offsets, offset_scale=offset_scale)

            totals["samples"] += 1
            for i, j in close_pairs(vertices, degrees):
                totals["pairs"] += 1
                totals["pairs_resolved_peaks"] += int(
                    pair_resolved(pred_peaks, vertices[i], vertices[j])
                )
                totals["pairs_resolved_clusters"] += int(
                    pair_resolved(pred_clusters, vertices[i], vertices[j])
                )
            for name, pred in (("peaks", pred_peaks), ("clusters", pred_clusters)):
                tp, fp, fn = junction_pr(pred, gt_active)
                totals[f"tp_{name}"] += tp
                totals[f"fp_{name}"] += fp
                totals[f"fn_{name}"] += fn

    def rate(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    report = {
        "checkpoint": args.checkpoint.as_posix(),
        "offset_radius_px": args.offset_radius_px,
        "batchnorm_mode": args.batchnorm_mode,
        "samples": totals["samples"],
        "close_pairs": totals["pairs"],
        "pair_resolution_peaks": rate(totals["pairs_resolved_peaks"], totals["pairs"]),
        "pair_resolution_clusters": rate(totals["pairs_resolved_clusters"], totals["pairs"]),
        "junction_precision_peaks": rate(
            totals["tp_peaks"], totals["tp_peaks"] + totals["fp_peaks"]
        ),
        "junction_recall_peaks": rate(totals["tp_peaks"], totals["tp_peaks"] + totals["fn_peaks"]),
        "junction_precision_clusters": rate(
            totals["tp_clusters"], totals["tp_clusters"] + totals["fp_clusters"]
        ),
        "junction_recall_clusters": rate(
            totals["tp_clusters"], totals["tp_clusters"] + totals["fn_clusters"]
        ),
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
