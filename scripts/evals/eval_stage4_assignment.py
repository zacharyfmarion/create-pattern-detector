#!/usr/bin/env python3
"""Stage 4 assignment/report smoke evaluation on fixed synthetic fixtures."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_dataset import render_cpline_sample
from src.data.fold_parser import CreasePattern
from src.vectorization import (
    AttributedPlanarGraph,
    PlanarGraphResult,
    RepairConfig,
    assign_edges_from_logits,
    build_quality_report,
    conservative_repair,
    graph_to_fold_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/stage4_assignment"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--profiles",
        type=str,
        default="clean,line-style,dark-mode,print-light",
        help="Comma-separated render profiles to evaluate with oracle geometry.",
    )
    parser.add_argument("--infer-assignments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for profile in [part.strip() for part in args.profiles.split(",") if part.strip()]:
        sample = render_cpline_sample(
            simple_mv_cp(),
            image_size=args.image_size,
            padding=max(8, int(16 * args.image_size / 128)),
            line_width=max(1, int(2 * args.image_size / 128)),
            augment_profile=profile,
            seed=0,
            square_symmetry="identity",
        )
        result = oracle_result_from_sample(sample, image_size=args.image_size)
        assignment = assign_edges_from_logits(
            result,
            assignment_probabilities_from_target(sample.assignment),
            line_prob=sample.line_prob,
        )
        graph = AttributedPlanarGraph.from_planar_result(result, assignment)
        repair = conservative_repair(
            graph,
            line_prob=sample.line_prob,
            config=RepairConfig(
                image_size=args.image_size,
                weak_edge_support_threshold=0.05,
                reconstruct_square_border_chain=False,
                infer_assignments=args.infer_assignments,
            ),
        )
        report = build_quality_report(repair.graph, repair_actions=repair.actions)
        accuracy = assignment_accuracy(repair.graph.edges_assignment, sample.assignments)
        row = {
            "profile": profile,
            "selected_profile": sample.metadata["selected_profile"],
            "palette_kind": sample.metadata["params"].get("palette_kind", "assignment"),
            "assignment_target_mode": sample.metadata["params"].get(
                "assignment_target_mode", "original"
            ),
            "status": report.status,
            "assignment_accuracy": accuracy,
            "assignment_counts": report.summary["assignment_counts"],
            "sources": {
                "observed": int(sum(source == "observed" for source in repair.graph.assignment_source)),
                "unknown": int(sum(source == "unknown" for source in repair.graph.assignment_source)),
                "inferred": int(sum(source == "inferred" for source in repair.graph.assignment_source)),
            },
            "warning_codes": [warning.code for warning in report.warnings],
        }
        rows.append(row)
        (output_dir / f"{profile.replace('-', '_')}.fold").write_text(
            json.dumps(graph_to_fold_dict(repair.graph, report=report), indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

    summary = {
        "image_size": args.image_size,
        "profiles": rows,
        "mean_assignment_accuracy": float(np.mean([row["assignment_accuracy"] for row in rows])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2), flush=True)


def simple_mv_cp() -> CreasePattern:
    return CreasePattern(
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        edges=np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
            dtype=np.int64,
        ),
        assignments=np.array([2, 2, 2, 2, 0, 1, 1, 1], dtype=np.int8),
    )


def oracle_result_from_sample(sample: Any, *, image_size: int) -> PlanarGraphResult:
    return PlanarGraphResult(
        vertices_coords=(sample.pixel_vertices / float(image_size - 1)).clip(0.0, 1.0),
        edges_vertices=sample.edges.astype(np.int64),
        edges_assignment=np.full(len(sample.edges), 3, dtype=np.int8),
        edge_support=np.ones(len(sample.edges), dtype=np.float32),
        vertex_support=np.ones(len(sample.pixel_vertices), dtype=np.float32),
        pixel_vertices=sample.pixel_vertices.astype(np.float32),
    )


def assignment_probabilities_from_target(target: np.ndarray) -> np.ndarray:
    h, w = target.shape
    probabilities = np.zeros((4, h, w), dtype=np.float32)
    probabilities[3, :, :] = 1.0
    for class_idx in range(4):
        mask = target == class_idx
        probabilities[:, mask] = 0.0
        probabilities[class_idx, mask] = 1.0
    return probabilities


def assignment_accuracy(pred: np.ndarray, expected: np.ndarray) -> float:
    if len(expected) == 0:
        return 0.0
    return float(np.mean(np.asarray(pred, dtype=np.int8) == np.asarray(expected, dtype=np.int8)))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
