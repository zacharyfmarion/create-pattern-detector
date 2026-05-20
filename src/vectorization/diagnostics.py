"""UI-friendly diagnostics for Stage 4 graph inspection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.fold_parser import FOLDParser
from src.vectorization.metrics import match_vertices

ASSIGNMENT_NAMES = {0: "M", 1: "V", 2: "B", 3: "U"}

FAILED_CODES = {
    "empty_graph",
    "unparseable_fold",
    "duplicate_edges",
    "zero_length_edges",
    "illegal_crossings",
}
OUTSIDE_CODES = {
    "incomplete_border",
    "very_short_edges",
    "crowded_junctions",
    "weak_edges",
    "dense_geometry",
    "dense_input_evidence",
}
ASSIGNMENT_AMBIGUITY_CODES = {"low_confidence_assignments", "unknown_assignments"}
ORIGAMI_DIAGNOSTIC_CODES = {
    "even_degree_failures",
    "kawasaki_residuals",
    "maekawa_failures",
}
AMBIGUOUS_CODES = ASSIGNMENT_AMBIGUITY_CODES | ORIGAMI_DIAGNOSTIC_CODES


def build_stage4_diagnostic_payload(
    *,
    row: Mapping[str, Any],
    image_url: str,
    image_size: int,
    gt_vertices: Any,
    gt_edges: Any,
    gt_assignments: Any,
    pred_vertices: Any,
    pred_edges: Any,
    pred_assignments: Any,
    pred_edge_support: Any,
    pred_assignment_confidence: Any,
    pred_assignment_margin: Any,
    pred_assignment_source: list[str],
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    vertex_tolerance_px: float,
) -> dict[str, Any]:
    """Build the JSON payload consumed by the Stage Inspector UI."""
    gt_vertices_array = np.asarray(gt_vertices, dtype=np.float32)
    gt_edges_array = np.asarray(gt_edges, dtype=np.int64)
    gt_assignments_array = np.asarray(gt_assignments, dtype=np.int8)
    pred_vertices_array = np.asarray(pred_vertices, dtype=np.float32)
    pred_edges_array = np.asarray(pred_edges, dtype=np.int64)
    pred_assignments_array = np.asarray(pred_assignments, dtype=np.int8)
    pred_support_array = np.asarray(pred_edge_support, dtype=np.float32)
    pred_confidence_array = np.asarray(pred_assignment_confidence, dtype=np.float32)
    pred_margin_array = np.asarray(pred_assignment_margin, dtype=np.float32)

    gt_to_pred, pred_to_gt, vertex_errors = match_vertices(
        gt_vertices_array,
        pred_vertices_array,
        tolerance=vertex_tolerance_px,
    )
    edge_matches = _match_edges_with_state(
        gt_edges_array=gt_edges_array,
        pred_edges_array=pred_edges_array,
        gt_assignments=gt_assignments_array,
        pred_assignments=pred_assignments_array,
        pred_to_gt=pred_to_gt,
    )
    warning_entries = [dict(item) for item in report.get("warnings", [])]
    repair_entries = [dict(item) for item in report.get("repair_actions", [])]

    pred_vertex_issues, pred_edge_issues = _issue_maps(
        vertex_count=len(pred_vertices_array),
        edge_count=len(pred_edges_array),
        entries=warning_entries,
    )
    pred_vertex_repairs, pred_edge_repairs = _issue_maps(
        vertex_count=len(pred_vertices_array),
        edge_count=len(pred_edges_array),
        entries=repair_entries,
    )
    gt_incident = _incident_edges(len(gt_vertices_array), gt_edges_array)
    pred_incident = _incident_edges(len(pred_vertices_array), pred_edges_array)
    gt_vertices_payload = [
        {
            "id": int(idx),
            "x": _float(point[0]),
            "y": _float(point[1]),
            "matchedPredVertex": gt_to_pred.get(int(idx)),
            "degree": len(gt_incident[idx]),
            "incidentEdges": gt_incident[idx],
        }
        for idx, point in enumerate(gt_vertices_array)
    ]
    pred_vertices_payload = [
        {
            "id": int(idx),
            "x": _float(point[0]),
            "y": _float(point[1]),
            "matchedGtVertex": pred_to_gt.get(int(idx)),
            "matchErrorPx": _match_error_for_pred(idx, pred_to_gt, gt_vertices_array, pred_vertices_array),
            "degree": len(pred_incident[idx]),
            "incidentEdges": pred_incident[idx],
            "issues": sorted(pred_vertex_issues[idx]),
            "repairs": sorted(pred_vertex_repairs[idx]),
            "kawasakiResidual": _kawasaki_residual_for_vertex(idx, warning_entries),
        }
        for idx, point in enumerate(pred_vertices_array)
    ]
    pred_edges_payload = []
    for edge_idx, edge in enumerate(pred_edges_array):
        match = edge_matches["pred"][edge_idx]
        issues = set(pred_edge_issues[edge_idx])
        if match["state"] == "extra":
            issues.add("extra_predicted_edge")
        assignment = int(pred_assignments_array[edge_idx])
        pred_edges_payload.append(
            {
                "id": int(edge_idx),
                "vertices": [int(edge[0]), int(edge[1])],
                "assignment": ASSIGNMENT_NAMES.get(assignment, "U"),
                "assignmentIndex": assignment,
                "support": _float(pred_support_array[edge_idx]),
                "confidence": _float(pred_confidence_array[edge_idx]),
                "margin": _float(pred_margin_array[edge_idx]),
                "source": str(pred_assignment_source[edge_idx]),
                "issues": sorted(issues),
                "repairs": sorted(pred_edge_repairs[edge_idx]),
                "match": match,
            }
        )

    gt_edges_payload = []
    for edge_idx, edge in enumerate(gt_edges_array):
        match = edge_matches["gt"][edge_idx]
        issues = ["missing_gt_edge"] if match["state"] == "missing" else []
        assignment = int(gt_assignments_array[edge_idx])
        gt_edges_payload.append(
            {
                "id": int(edge_idx),
                "vertices": [int(edge[0]), int(edge[1])],
                "assignment": ASSIGNMENT_NAMES.get(assignment, "U"),
                "assignmentIndex": assignment,
                "issues": issues,
                "match": match,
            }
        )

    warning_counts = Counter(str(item.get("code", "unknown")) for item in warning_entries)
    repair_counts = Counter(str(item.get("code", "unknown")) for item in repair_entries)
    row_dict = dict(row)
    structural = dict(report.get("structural_validity") or metrics.get("structural_validity") or {})
    return {
        "key": _example_key(row_dict),
        "stage": "stage4",
        "imageUrl": image_url,
        "imageSize": int(image_size),
        "row": _json_safe(row_dict),
        "metrics": _json_safe(dict(metrics)),
        "status": str(report.get("status", row_dict.get("status", "failed"))),
        "structuralValidity": _json_safe(structural),
        "warnings": _json_safe(warning_entries),
        "warningCounts": dict(sorted(warning_counts.items())),
        "repairs": _json_safe(repair_entries),
        "repairCounts": dict(sorted(repair_counts.items())),
        "whatIfStatuses": compute_what_if_statuses(
            warning_entries,
            repair_entries,
            structural_validity=structural,
            edge_count=len(pred_edges_array),
        ),
        "graph": {
            "groundTruth": {
                "vertices": gt_vertices_payload,
                "edges": gt_edges_payload,
            },
            "prediction": {
                "vertices": pred_vertices_payload,
                "edges": pred_edges_payload,
            },
            "matches": {
                "vertexTolerancePx": _float(vertex_tolerance_px),
                "matchedVertices": len(gt_to_pred),
                "meanVertexErrorPx": _float(np.mean(vertex_errors)) if vertex_errors else 0.0,
                "matchedPredEdges": edge_matches["matchedPredEdges"],
                "matchedGtEdges": edge_matches["matchedGtEdges"],
                "missingGtEdges": edge_matches["missingGtEdges"],
                "extraPredEdges": edge_matches["extraPredEdges"],
            },
        },
    }


def compute_what_if_statuses(
    warnings: list[Mapping[str, Any]],
    repairs: list[Mapping[str, Any]],
    *,
    structural_validity: Mapping[str, Any] | None = None,
    edge_count: int = 1,
) -> dict[str, str]:
    """Recompute status with selected warning groups treated as informational."""
    structural = structural_validity or {}
    codes = {str(warning.get("code", "")) for warning in warnings}
    return {
        "current": _status_from_codes(
            codes,
            repairs,
            structural_validity=structural,
            edge_count=edge_count,
            ignore=set(),
        ),
        "ignoreOrigamiDiagnostics": _status_from_codes(
            codes,
            repairs,
            structural_validity=structural,
            edge_count=edge_count,
            ignore=ORIGAMI_DIAGNOSTIC_CODES,
        ),
        "ignoreAssignmentUncertainty": _status_from_codes(
            codes,
            repairs,
            structural_validity=structural,
            edge_count=edge_count,
            ignore=ASSIGNMENT_AMBIGUITY_CODES,
        ),
        "ignoreEnvelopeWarnings": _status_from_codes(
            codes,
            repairs,
            structural_validity=structural,
            edge_count=edge_count,
            ignore=OUTSIDE_CODES,
        ),
        "structuralOnly": _status_from_codes(
            codes,
            repairs,
            structural_validity=structural,
            edge_count=edge_count,
            ignore=OUTSIDE_CODES | AMBIGUOUS_CODES,
        ),
    }


def _status_from_codes(
    codes: set[str],
    repairs: list[Mapping[str, Any]],
    *,
    structural_validity: Mapping[str, Any],
    edge_count: int,
    ignore: set[str],
) -> str:
    active = codes - ignore
    if edge_count == 0 or active & FAILED_CODES or structural_validity.get("parseable_fold") is False:
        return "failed"
    if active & OUTSIDE_CODES:
        return "outside_v1_envelope"
    if active & AMBIGUOUS_CODES:
        return "ambiguous"
    if repairs:
        return "repaired"
    return "valid"


def _match_edges_with_state(
    *,
    gt_edges_array: np.ndarray,
    pred_edges_array: np.ndarray,
    gt_assignments: np.ndarray,
    pred_assignments: np.ndarray,
    pred_to_gt: dict[int, int],
) -> dict[str, Any]:
    gt_lookup = {
        (int(min(v1, v2)), int(max(v1, v2))): int(idx)
        for idx, (v1, v2) in enumerate(gt_edges_array)
    }
    pred_matches: list[dict[str, Any]] = []
    gt_matches: list[dict[str, Any]] = [
        {"state": "missing", "predEdge": None, "assignmentCorrect": False}
        for _ in range(len(gt_edges_array))
    ]
    matched_gt: set[int] = set()
    matched_pred: list[int] = []

    for pred_idx, (pred_v1, pred_v2) in enumerate(pred_edges_array):
        gt_v1 = pred_to_gt.get(int(pred_v1))
        gt_v2 = pred_to_gt.get(int(pred_v2))
        gt_idx = None
        if gt_v1 is not None and gt_v2 is not None:
            gt_idx = gt_lookup.get((min(gt_v1, gt_v2), max(gt_v1, gt_v2)))
        if gt_idx is None or gt_idx in matched_gt:
            pred_matches.append(
                {
                    "state": "extra",
                    "gtEdge": None,
                    "assignmentCorrect": False,
                }
            )
            continue
        matched_gt.add(gt_idx)
        matched_pred.append(int(pred_idx))
        assignment_correct = int(gt_assignments[gt_idx]) == int(pred_assignments[pred_idx])
        pred_matches.append(
            {
                "state": "matched",
                "gtEdge": int(gt_idx),
                "assignmentCorrect": bool(assignment_correct),
            }
        )
        gt_matches[gt_idx] = {
            "state": "matched",
            "predEdge": int(pred_idx),
            "assignmentCorrect": bool(assignment_correct),
        }

    missing_gt = [idx for idx, match in enumerate(gt_matches) if match["state"] == "missing"]
    extra_pred = [idx for idx, match in enumerate(pred_matches) if match["state"] == "extra"]
    return {
        "pred": pred_matches,
        "gt": gt_matches,
        "matchedPredEdges": matched_pred,
        "matchedGtEdges": sorted(matched_gt),
        "missingGtEdges": missing_gt,
        "extraPredEdges": extra_pred,
    }


def _issue_maps(
    *,
    vertex_count: int,
    edge_count: int,
    entries: list[Mapping[str, Any]],
) -> tuple[list[set[str]], list[set[str]]]:
    vertex_issues = [set() for _ in range(vertex_count)]
    edge_issues = [set() for _ in range(edge_count)]
    for entry in entries:
        code = str(entry.get("code", "unknown"))
        for vertex_idx in entry.get("vertex_indices", []) or []:
            if 0 <= int(vertex_idx) < vertex_count:
                vertex_issues[int(vertex_idx)].add(code)
        for edge_idx in entry.get("edge_indices", []) or []:
            if 0 <= int(edge_idx) < edge_count:
                edge_issues[int(edge_idx)].add(code)
    return vertex_issues, edge_issues


def _incident_edges(vertex_count: int, edges: np.ndarray) -> list[list[int]]:
    incident = [[] for _ in range(vertex_count)]
    for edge_idx, (v1, v2) in enumerate(edges):
        if 0 <= int(v1) < vertex_count:
            incident[int(v1)].append(int(edge_idx))
        if 0 <= int(v2) < vertex_count:
            incident[int(v2)].append(int(edge_idx))
    return incident


def _kawasaki_residual_for_vertex(
    vertex_idx: int,
    warnings: list[Mapping[str, Any]],
) -> float | None:
    for warning in warnings:
        if warning.get("code") != "kawasaki_residuals":
            continue
        residuals = (warning.get("details") or {}).get("residuals") or {}
        value = residuals.get(str(vertex_idx))
        if value is not None:
            return _float(value)
    return None


def _match_error_for_pred(
    pred_idx: int,
    pred_to_gt: dict[int, int],
    gt_vertices: np.ndarray,
    pred_vertices: np.ndarray,
) -> float | None:
    gt_idx = pred_to_gt.get(int(pred_idx))
    if gt_idx is None:
        return None
    return _float(np.linalg.norm(pred_vertices[pred_idx] - gt_vertices[gt_idx]))


def _example_key(row: Mapping[str, Any]) -> str:
    raw_id = str(row.get("id", "example"))
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_id)[:90]
    return f"{row.get('profile', 'profile')}__{int(row.get('sample_index', 0)):03d}__{safe}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _float(value: Any) -> float:
    return float(np.asarray(value).item())


def assignment_name(value: int) -> str:
    return FOLDParser.ASSIGNMENT_LABELS[int(value)]
