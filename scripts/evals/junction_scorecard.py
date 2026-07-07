#!/usr/bin/env python3
"""Junction scorecard: the checkpoint-selection metric for junction recall/precision.

Runs a CPLineNet checkpoint over a val split and scores junction detection with
the PRODUCT decode — peak-gated offset-vote clustering, mirroring
`offset_cluster_primitives` in ori-studio `crates/oristudio-cp-detect/src/
evidence_extract.rs` (threshold floor 0.40, vote floor = 0.6 * threshold,
bandwidth 1.2px, merge 1.0px, keep a cluster iff its strongest vote clears the
threshold). Keep these constants in sync with the Rust; the 2026-07-02
mass-cull incident happened precisely because eval and product decode drifted.

Reported per family and overall:
- junction recall: interior crease-degree>=3 GT junctions matched within 2px;
- extras: predicted peaks >3px from every degree>=1 GT vertex;
- per-CP clean rate: samples with 0 misses AND 0 extras (topology is
  all-or-nothing downstream, so this is the number that tracks solve_recovered);
- miss taxonomy: absent (< vote floor near GT), sub-threshold, localization
  (fired >= threshold nearby but no peak within 2px), split by close-pair vs
  isolated (nearest labeled vertex < 12px).

Usage:
    .venv/bin/python scripts/evals/junction_scorecard.py \
        --checkpoint checkpoints/<run>/full/latest.pt \
        --manifest data/generated/synthetic/cp_training_mix_v4_search225/raw-manifest.jsonl \
        --val-count 512 --out reports/junction-scorecard.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from src.data.cpline_dataset import CplineFoldDataset
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode
from src.models.cpline_net import CPLineNet

# Product decode constants — mirror evidence_extract.rs.
JUNCTION_PEAK_THRESHOLD = 0.40
VOTE_FLOOR_FACTOR = 0.6
CLUSTER_BANDWIDTH_PX = 1.2
CLUSTER_MERGE_PX = 1.0

MATCH_TOLERANCE_PX = 2.0
EXTRA_TOLERANCE_PX = 3.0
CLOSE_PAIR_NND_PX = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v4_search225/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=2500)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="hrnet_w18")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--offset-radius-px", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=JUNCTION_PEAK_THRESHOLD)
    parser.add_argument(
        "--profile",
        default="clean",
        help="Render profile for the eval images (clean = geometry recall).",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--batchnorm-mode", choices=BATCHNORM_MODES, default="batch-stats")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_peak_gated_clusters(
    probs: np.ndarray,
    offsets: np.ndarray,
    *,
    radius_px: float,
    threshold: float,
) -> np.ndarray:
    """Product decode: offset votes, seed-anchored clusters, peak gate."""
    vote_floor = threshold * VOTE_FLOOR_FACTOR
    ys, xs = np.nonzero(probs >= vote_floor)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    w = probs[ys, xs]
    vx = xs + np.clip(offsets[0, ys, xs], -1.0, 1.0) * radius_px
    vy = ys + np.clip(offsets[1, ys, xs], -1.0, 1.0) * radius_px
    order = np.argsort(-w)
    vx, vy, w = vx[order], vy[order], w[order]

    # Seed-anchored assignment on a spatial grid (identical semantics to the
    # Rust loop; the grid only accelerates the nearest-seed search).
    seeds: list[tuple[float, float]] = []
    sums: list[list[float]] = []
    mass: list[float] = []
    peak: list[float] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell = CLUSTER_BANDWIDTH_PX
    for x, y, wt in zip(vx, vy, w):
        gx, gy = int(x / cell), int(y / cell)
        assigned = False
        for cx in range(gx - 1, gx + 2):
            for cy in range(gy - 1, gy + 2):
                for idx in grid.get((cx, cy), ()):
                    dx = x - seeds[idx][0]
                    dy = y - seeds[idx][1]
                    if dx * dx + dy * dy <= CLUSTER_BANDWIDTH_PX * CLUSTER_BANDWIDTH_PX:
                        sums[idx][0] += x * wt
                        sums[idx][1] += y * wt
                        mass[idx] += wt
                        peak[idx] = max(peak[idx], wt)
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break
        if not assigned:
            idx = len(seeds)
            seeds.append((x, y))
            sums.append([x * wt, y * wt])
            mass.append(wt)
            peak.append(wt)
            grid.setdefault((gx, gy), []).append(idx)

    clusters = [
        ((sums[i][0] / mass[i], sums[i][1] / mass[i]), peak[i], mass[i])
        for i in range(len(seeds))
    ]
    clusters.sort(key=lambda c: -c[2])
    merged: list[list] = []
    for center, pk, ms in clusters:
        absorbed = False
        for m in merged:
            if math.hypot(center[0] - m[0][0], center[1] - m[0][1]) <= CLUSTER_MERGE_PX:
                tot = m[2] + ms
                m[0] = (
                    (m[0][0] * m[2] + center[0] * ms) / tot,
                    (m[0][1] * m[2] + center[1] * ms) / tot,
                )
                m[1] = max(m[1], pk)
                m[2] = tot
                absorbed = True
                break
        if not absorbed:
            merged.append([center, pk, ms])
    kept = [m[0] for m in merged if m[1] >= threshold]
    return np.array(kept, dtype=np.float32).reshape(-1, 2)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CPLineNet(
        backbone=args.backbone,
        pretrained=False,
        hidden_channels=args.hidden_channels,
        v2_heads=True,
        junction_offset_clamp=1.0 if args.offset_radius_px > 0 else 0.5,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dataset = CplineFoldDataset(
        args.manifest,
        split=args.split,
        limit=args.val_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=args.profile,
        seed=args.seed,
    )
    vote_floor = args.threshold * VOTE_FLOOR_FACTOR

    per_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    taxonomy = defaultdict(int)
    with torch.no_grad(), model_eval_with_batchnorm_mode(model, batchnorm_mode=args.batchnorm_mode):
        for index in range(len(dataset)):
            item = dataset[index]
            family = item["meta"]["family"] or "unknown"
            image = item["image"].unsqueeze(0).to(device)
            outputs = model(image)
            probs = torch.sigmoid(outputs["junction_logits"])[0, 0].cpu().numpy()
            offsets = outputs["junction_offset"][0].cpu().numpy()
            peaks = decode_peak_gated_clusters(
                probs, offsets, radius_px=args.offset_radius_px, threshold=args.threshold
            )

            vertices = item["graph"]["vertices"].numpy()
            edges = item["graph"]["edges"].numpy()
            assignments = item["graph"]["assignments"].numpy()
            crease_deg = np.zeros(len(vertices), dtype=np.int32)
            border_deg = np.zeros(len(vertices), dtype=np.int32)
            any_deg = np.zeros(len(vertices), dtype=np.int32)
            for (v1, v2), assignment in zip(edges, assignments):
                any_deg[v1] += 1
                any_deg[v2] += 1
                if assignment in (0, 1):
                    crease_deg[v1] += 1
                    crease_deg[v2] += 1
                elif assignment == 2:
                    border_deg[v1] += 1
                    border_deg[v2] += 1
            targets = np.nonzero((crease_deg >= 3) & (border_deg == 0))[0]
            labeled = vertices[any_deg >= 1]

            misses = 0
            for vi in targets:
                p = vertices[vi]
                dmin = (
                    float(np.hypot(peaks[:, 0] - p[0], peaks[:, 1] - p[1]).min())
                    if len(peaks)
                    else np.inf
                )
                if dmin <= MATCH_TOLERANCE_PX:
                    continue
                misses += 1
                nnd = np.sort(np.hypot(labeled[:, 0] - p[0], labeled[:, 1] - p[1]))
                nnd = float(nnd[1]) if len(nnd) > 1 else np.inf
                xi, yi = int(round(p[0])), int(round(p[1]))
                window = probs[max(0, yi - 3) : yi + 4, max(0, xi - 3) : xi + 4]
                act = float(window.max()) if window.size else 0.0
                kind = (
                    "absent"
                    if act < vote_floor
                    else ("sub_threshold" if act < args.threshold else "localization")
                )
                locality = "close_pair" if nnd < CLOSE_PAIR_NND_PX else "isolated"
                taxonomy[f"{kind}.{locality}"] += 1

            extras = 0
            for pk in peaks:
                if (
                    len(labeled) == 0
                    or np.hypot(labeled[:, 0] - pk[0], labeled[:, 1] - pk[1]).min()
                    > EXTRA_TOLERANCE_PX
                ):
                    extras += 1

            stats = per_family[family]
            stats["samples"] += 1
            stats["junctions"] += len(targets)
            stats["misses"] += misses
            stats["extras"] += extras
            stats["clean"] += int(misses == 0 and extras == 0)

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "profile": args.profile,
        "threshold": args.threshold,
        "offset_radius_px": args.offset_radius_px,
        "decode": "peak-gated-offset-cluster (evidence_extract.rs parity)",
        "families": {},
        "miss_taxonomy": dict(sorted(taxonomy.items())),
    }
    totals = defaultdict(float)
    for family, stats in sorted(per_family.items()):
        for key, value in stats.items():
            totals[key] += value
        report["families"][family] = _summarize(stats)
    report["overall"] = _summarize(totals)

    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _summarize(stats: dict[str, float]) -> dict[str, float]:
    junctions = max(stats["junctions"], 1.0)
    samples = max(stats["samples"], 1.0)
    return {
        "samples": int(stats["samples"]),
        "junctions": int(stats["junctions"]),
        "recall": round(1.0 - stats["misses"] / junctions, 5),
        "misses": int(stats["misses"]),
        "extras": int(stats["extras"]),
        "extras_per_sample": round(stats["extras"] / samples, 3),
        "clean_rate": round(stats["clean"] / samples, 4),
    }


if __name__ == "__main__":
    main()
