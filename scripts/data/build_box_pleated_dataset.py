#!/usr/bin/env python3
"""Build a standard dataset root from Stage B box-pleat generator output.

The Stage B generator (tools/synthetic-generator, `bun run box-pleated-generate`)
writes folds/ + metadata/ + manifest.jsonl. This ingest adds what the training
mix builder and CplineFoldDataset expect (family field, raw-manifest.jsonl,
recipe/qa), applies quality tiering, and optionally masks assignment noise:
vertices that fail Maekawa get their incident non-border creases relabeled to U
(honest "unassigned" instead of confident-wrong M/V — the junction/line heads
never consume M/V, so geometry supervision is unaffected).

Usage:
  python scripts/data/build_box_pleated_dataset.py \
      --stage-b ~/Documents/datasets/create-pattern-detector/synthetic/.staging/box_pleated_v1_stageb \
      --out ~/Documents/datasets/create-pattern-detector/synthetic/box_pleated_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

FAMILY = "box-pleated"
DATASET_VERSION = "box-pleated/v1"

BUCKET_THRESHOLDS = [
    ("tiny", 0, 60),
    ("small", 60, 180),
    ("medium", 180, 500),
    ("dense", 500, 1200),
    ("superdense", 1200, 10**9),
]


def bucket_for_edges(num_edges: int) -> str:
    for name, lo, hi in BUCKET_THRESHOLDS:
        if lo <= num_edges < hi:
            return name
    return "superdense"


def maekawa_mask_to_unknown(fold: dict) -> int:
    """Relabel non-border creases at Maekawa-violating interior vertices to U."""
    vertices = np.asarray(fold["vertices_coords"], dtype=np.float64)
    edges = fold["edges_vertices"]
    assignment = list(fold["edges_assignment"])
    incident: dict[int, list[int]] = {}
    border_touch: set[int] = set()
    for edge_index, (a, b) in enumerate(edges):
        if assignment[edge_index] == "B":
            border_touch.add(a)
            border_touch.add(b)
            continue
        incident.setdefault(a, []).append(edge_index)
        incident.setdefault(b, []).append(edge_index)
    relabeled = 0
    for vertex, edge_ids in incident.items():
        if vertex in border_touch:
            continue
        labels = [assignment[e] for e in edge_ids]
        m = labels.count("M")
        v = labels.count("V")
        if "U" in labels:
            continue  # already masked / undetermined at this vertex
        if abs(m - v) == 2:
            continue  # Maekawa holds
        for e in edge_ids:
            if assignment[e] in ("M", "V"):
                assignment[e] = "U"
                relabeled += 1
    fold["edges_assignment"] = assignment
    return relabeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage-b", type=Path, required=True, help="Stage B output dir")
    parser.add_argument("--out", type=Path, required=True, help="Dataset root to create")
    parser.add_argument("--dataset-name", default="box_pleated_v1")
    parser.add_argument(
        "--max-conflicts",
        type=int,
        default=None,
        help="Drop CPs with more Maekawa conflicts than this (default: keep all)",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=None,
        help="Keep only CPs with at least this many edges (dense-supplement builds)",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Skip (seed, scale) pairs already present in this raw-manifest.jsonl "
        "(prevents duplicating CPs across box-pleated roots)",
    )
    parser.add_argument(
        "--no-maekawa-mask",
        action="store_true",
        help="Keep generator M/V labels at Maekawa-violating vertices instead of relabeling to U",
    )
    args = parser.parse_args()

    out = args.out.expanduser()
    folds_dir = out / "folds"
    metadata_dir = out / "metadata"
    qa_dir = out / "qa"
    for d in (folds_dir, metadata_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)

    stage_rows = [
        json.loads(line)
        for line in (args.stage_b.expanduser() / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    excluded_pairs: set[tuple[int, int]] = set()
    if args.exclude_manifest is not None:
        for line in args.exclude_manifest.expanduser().read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                excluded_pairs.add((int(row["seed"]), int(row["scale"])))

    rows = []
    dropped = 0
    masked_edges_total = 0
    for stage_row in sorted(stage_rows, key=lambda r: (int(r["seed"]), int(r["scale"]))):
        if args.max_conflicts is not None and stage_row["quality"]["conflicts"] > args.max_conflicts:
            dropped += 1
            continue
        if args.min_edges is not None and int(stage_row["edges"]) < args.min_edges:
            dropped += 1
            continue
        if (int(stage_row["seed"]), int(stage_row["scale"])) in excluded_pairs:
            dropped += 1
            continue
        sample_id = f"{args.dataset_name}-{stage_row['id']}"
        fold = json.loads((args.stage_b.expanduser() / stage_row["foldPath"]).read_text())
        masked = 0
        if not args.no_maekawa_mask:
            masked = maekawa_mask_to_unknown(fold)
            masked_edges_total += masked
        assignments = {k: int(v) for k, v in sorted(Counter(fold["edges_assignment"]).items())}
        fold_path = folds_dir / f"{sample_id}.fold"
        fold_path.write_text(json.dumps(fold, separators=(",", ":")), encoding="utf-8")
        metadata = {
            "id": sample_id,
            "config": {
                "id": sample_id,
                "family": FAMILY,
                "source": "box-pleated-generator",
                "seed": stage_row["seed"],
                "scale": stage_row["scale"],
                "leafCount": stage_row["leafCount"],
                "grid": stage_row["grid"],
            },
            "quality": stage_row["quality"],
            "maekawaMaskedEdges": masked,
            "labelPolicy": {
                "assignmentSource": "box-pleated-generator",
                "geometrySource": "box-pleated-generator-exact",
                "trainingEligible": True,
                "notes": [
                    "Geometry is exact by construction (integer grid packing).",
                    "M/V labels at Maekawa-violating vertices are relabeled to U at "
                    "ingest unless --no-maekawa-mask; junction/line supervision is "
                    "unaffected by assignment noise.",
                ],
            },
        }
        (metadata_dir / f"{sample_id}.json").write_text(
            json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
        )
        rows.append(
            {
                "id": sample_id,
                "family": FAMILY,
                "bucket": bucket_for_edges(int(stage_row["edges"])),
                "split": stage_row["split"],
                "foldPath": f"folds/{fold_path.name}",
                "metadataPath": f"metadata/{sample_id}.json",
                "vertices": int(stage_row["vertices"]),
                "edges": int(stage_row["edges"]),
                "assignments": assignments,
                "seed": int(stage_row["seed"]),
                "scale": int(stage_row["scale"]),
                "grid": int(stage_row["grid"]),
                "quality": stage_row["quality"],
                "maekawaMaskedEdges": masked,
            }
        )

    with (out / "raw-manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    qa = {
        "accepted": len(rows),
        "dropped": dropped,
        "datasetVersion": DATASET_VERSION,
        "familyCounts": {FAMILY: len(rows)},
        "bucketCounts": dict(sorted(Counter(r["bucket"] for r in rows).items())),
        "splitCounts": dict(sorted(Counter(r["split"] for r in rows).items())),
        "scaleCounts": dict(sorted(Counter(str(r["scale"]) for r in rows).items())),
        "maekawaMaskedEdges": masked_edges_total,
        "assignmentTotals": dict(
            sorted(
                Counter(
                    label for r in rows for label, n in r["assignments"].items() for _ in range(n)
                ).items()
            )
        ),
    }
    payload = json.dumps(qa, indent=2, sort_keys=True) + "\n"
    (out / "qa.json").write_text(payload, encoding="utf-8")
    (qa_dir / "build-summary.json").write_text(payload, encoding="utf-8")
    recipe = {
        "name": args.dataset_name,
        "source": "box-pleated-generator (tools/synthetic-generator Stage A/B)",
        "adapterVersion": DATASET_VERSION,
        "stageB": str(args.stage_b.expanduser()),
        "maxConflicts": args.max_conflicts,
        "minEdges": args.min_edges,
        "excludeManifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "maekawaMask": not args.no_maekawa_mask,
        "family": FAMILY,
        "buckets": [
            {"name": n, "minEdges": lo, "maxEdges": hi} for n, lo, hi in BUCKET_THRESHOLDS
        ],
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
