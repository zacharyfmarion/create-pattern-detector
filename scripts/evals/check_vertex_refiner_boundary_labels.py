#!/usr/bin/env python3
"""Fail if vertex-refiner samples lose native boundary/corner labels."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_dataset import load_manifest_records, resolve_fold_path, select_records
from src.data.fold_parser import FOLDParser
from src.data.vertex_refiner_dataset import render_vertex_refiner_sample
from src.data.vertex_refiner_targets import classify_vertex_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-boundary", type=int, default=1)
    parser.add_argument("--min-corner", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    records = select_records(
        load_manifest_records(manifest),
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        seed=args.seed,
    )
    parser = FOLDParser()
    counts: Counter[str] = Counter()
    assignment_counts: Counter[int] = Counter()
    for record_index, record in enumerate(records):
        cp = parser.parse(resolve_fold_path(record, manifest))
        sample = render_vertex_refiner_sample(
            cp,
            image_size=args.image_size,
            padding=max(8, int(32 * args.image_size / 1024)),
            line_width=max(1, int(2 * args.image_size / 768)),
            seed=args.seed + record_index,
            auxiliary_mode="zero",
        )
        assignment_counts.update(int(value) for value in sample.assignments.tolist())
        for vertex_index in range(len(sample.pixel_vertices)):
            counts[
                classify_vertex_kind(
                    vertex_index,
                    sample.pixel_vertices,
                    sample.edges,
                    sample.assignments,
                    sample.square_frame,
                    image_size=args.image_size,
                )
            ] += 1
    report = {
        "manifest": manifest.as_posix(),
        "records": len(records),
        "kind_counts": dict(sorted(counts.items())),
        "assignment_counts": {str(key): value for key, value in sorted(assignment_counts.items())},
    }
    print(json.dumps(report, indent=2), flush=True)
    if counts["boundary_contact"] < int(args.min_boundary):
        print("fatal: boundary_contact labels collapsed or are absent", file=sys.stderr)
        return 2
    if counts["corner"] < int(args.min_corner):
        print("fatal: corner labels collapsed or are absent", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
