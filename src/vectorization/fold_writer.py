"""FOLD export for Stage 4 attributed crease-pattern graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.fold_parser import FOLDParser
from src.vectorization.constraint_repair import RepairAction
from src.vectorization.edge_assignment import AttributedPlanarGraph
from src.vectorization.quality_report import QualityReport


def graph_to_fold_dict(
    graph: AttributedPlanarGraph,
    *,
    report: QualityReport | None = None,
    repair_actions: list[RepairAction] | None = None,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Export an attributed graph as a minimal FOLD dictionary."""
    payload: dict[str, Any] = {
        "file_spec": 1.1,
        "file_creator": "cp-detector stage4",
        "file_classes": ["singleModel"],
        "frame_classes": ["creasePattern"],
        "vertices_coords": graph.vertices_coords.astype(float).tolist(),
        "edges_vertices": graph.edges_vertices.astype(int).tolist(),
        "edges_assignment": [
            FOLDParser.ASSIGNMENT_LABELS[int(value)] for value in graph.edges_assignment
        ],
    }
    if include_metadata:
        payload["cp_detector"] = _metadata_payload(
            graph,
            report=report,
            repair_actions=repair_actions,
        )
    return payload


def save_fold(
    graph: AttributedPlanarGraph,
    output_path: str | Path,
    *,
    report: QualityReport | None = None,
    repair_actions: list[RepairAction] | None = None,
    include_metadata: bool = True,
) -> None:
    """Write a Stage 4 attributed graph to a `.fold` file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            graph_to_fold_dict(
                graph,
                report=report,
                repair_actions=repair_actions,
                include_metadata=include_metadata,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _metadata_payload(
    graph: AttributedPlanarGraph,
    *,
    report: QualityReport | None,
    repair_actions: list[RepairAction] | None,
) -> dict[str, Any]:
    actions = repair_actions if repair_actions is not None else []
    if report is not None and not actions:
        actions = report.repair_actions
    metadata: dict[str, Any] = {
        "schema": "cp-detector/stage4/v1",
        "status": report.status if report is not None else None,
        "edge_support": graph.edge_support.astype(float).tolist(),
        "assignment_confidence": graph.assignment_confidence.astype(float).tolist(),
        "assignment_margin": graph.assignment_margin.astype(float).tolist(),
        "assignment_source": list(graph.assignment_source),
        "repair_actions": [action.to_dict() for action in actions],
    }
    if graph.assignment_probabilities is not None:
        metadata["assignment_probabilities"] = graph.assignment_probabilities.astype(float).tolist()
    if report is not None:
        report_payload = report.to_dict()
        metadata["warnings"] = report_payload["warnings"]
        metadata["summary"] = report_payload["summary"]
        metadata["structural_validity"] = report_payload["structural_validity"]
    else:
        metadata["warnings"] = []
    return metadata
