"""Graph matching and structural checks for deterministic vectorization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.fold_parser import CreasePattern, FOLDParser
from src.vectorization.planar_graph_builder import PlanarGraphResult


@dataclass
class StructuralValidity:
    parseable_fold: bool
    no_duplicate_edges: bool
    no_zero_length_edges: bool
    no_illegal_crossings: bool
    complete_border_when_present: bool
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return (
            self.parseable_fold
            and self.no_duplicate_edges
            and self.no_zero_length_edges
            and self.no_illegal_crossings
            and self.complete_border_when_present
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


@dataclass
class GraphMetrics:
    vertex_precision: float
    vertex_recall: float
    edge_precision: float
    edge_recall: float
    assignment_accuracy: float
    mean_vertex_error_px: float
    matched_vertices: int
    matched_edges: int
    gt_vertices: int
    pred_vertices: int
    gt_edges: int
    pred_edges: int
    assignment_total: int
    assignment_correct: int
    assignment_by_class: dict[str, dict[str, float | int]]
    structural_validity: StructuralValidity
    downstream_base_computation: str = "skipped: Rabbit Ear/FOLD CLI validator not restored"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["structural_validity"] = self.structural_validity.to_dict()
        return payload


def evaluate_graph(
    result: PlanarGraphResult,
    gt_vertices: np.ndarray,
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    vertex_tolerance_px: float = 5.0,
) -> GraphMetrics:
    gt_to_pred, pred_to_gt, errors = match_vertices(
        gt_vertices,
        result.pixel_vertices,
        tolerance=vertex_tolerance_px,
    )
    matched_gt_edges, matched_pred_edges, assignment_correct = match_edges(
        gt_edges,
        gt_assignments,
        result.edges_vertices,
        result.edges_assignment,
        pred_to_gt,
    )

    assignment_by_class = _assignment_by_class(
        matched_gt_edges,
        gt_assignments,
        assignment_correct,
    )

    structural = validate_structure(result)
    matched_vertices = len(gt_to_pred)
    matched_edges = len(matched_gt_edges)
    assignment_total = len(assignment_correct)
    assignment_correct_count = int(sum(assignment_correct))

    return GraphMetrics(
        vertex_precision=_ratio(matched_vertices, len(result.pixel_vertices)),
        vertex_recall=_ratio(matched_vertices, len(gt_vertices)),
        edge_precision=_ratio(matched_edges, len(result.edges_vertices)),
        edge_recall=_ratio(matched_edges, len(gt_edges)),
        assignment_accuracy=_ratio(assignment_correct_count, assignment_total),
        mean_vertex_error_px=float(np.mean(errors)) if errors else 0.0,
        matched_vertices=matched_vertices,
        matched_edges=matched_edges,
        gt_vertices=int(len(gt_vertices)),
        pred_vertices=int(len(result.pixel_vertices)),
        gt_edges=int(len(gt_edges)),
        pred_edges=int(len(result.edges_vertices)),
        assignment_total=assignment_total,
        assignment_correct=assignment_correct_count,
        assignment_by_class=assignment_by_class,
        structural_validity=structural,
    )


def match_vertices(
    gt_vertices: np.ndarray,
    pred_vertices: np.ndarray,
    tolerance: float,
) -> tuple[dict[int, int], dict[int, int], list[float]]:
    if len(gt_vertices) == 0 or len(pred_vertices) == 0:
        return {}, {}, []
    distances = np.linalg.norm(gt_vertices[:, None, :] - pred_vertices[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    gt_to_pred: dict[int, int] = {}
    pred_to_gt: dict[int, int] = {}
    errors: list[float] = []
    for gt_idx, pred_idx in zip(rows, cols):
        distance = float(distances[gt_idx, pred_idx])
        if distance <= tolerance:
            gt_to_pred[int(gt_idx)] = int(pred_idx)
            pred_to_gt[int(pred_idx)] = int(gt_idx)
            errors.append(distance)
    return gt_to_pred, pred_to_gt, errors


def match_edges(
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    pred_edges: np.ndarray,
    pred_assignments: np.ndarray,
    pred_to_gt_vertex: dict[int, int],
) -> tuple[list[int], list[int], list[bool]]:
    gt_lookup = {
        (int(min(v1, v2)), int(max(v1, v2))): idx for idx, (v1, v2) in enumerate(gt_edges)
    }
    matched_gt_edges: list[int] = []
    matched_pred_edges: list[int] = []
    assignment_correct: list[bool] = []
    seen_gt: set[int] = set()
    for pred_idx, (pred_v1, pred_v2) in enumerate(pred_edges):
        if int(pred_v1) not in pred_to_gt_vertex or int(pred_v2) not in pred_to_gt_vertex:
            continue
        gt_v1 = pred_to_gt_vertex[int(pred_v1)]
        gt_v2 = pred_to_gt_vertex[int(pred_v2)]
        key = (min(gt_v1, gt_v2), max(gt_v1, gt_v2))
        gt_idx = gt_lookup.get(key)
        if gt_idx is None or gt_idx in seen_gt:
            continue
        seen_gt.add(gt_idx)
        matched_gt_edges.append(gt_idx)
        matched_pred_edges.append(pred_idx)
        assignment_correct.append(int(gt_assignments[gt_idx]) == int(pred_assignments[pred_idx]))
    return matched_gt_edges, matched_pred_edges, assignment_correct


def validate_structure(result: PlanarGraphResult) -> StructuralValidity:
    errors: list[str] = []

    parseable_fold = True
    try:
        FOLDParser().parse_dict(
            {
                "vertices_coords": result.vertices_coords.tolist(),
                "edges_vertices": result.edges_vertices.tolist(),
                "edges_assignment": [
                    FOLDParser.ASSIGNMENT_LABELS[int(value)] for value in result.edges_assignment
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001 - structural diagnostic
        parseable_fold = False
        errors.append(f"parseable_fold: {exc}")

    edge_keys = [tuple(sorted(map(int, edge))) for edge in result.edges_vertices]
    no_duplicate_edges = len(edge_keys) == len(set(edge_keys))
    if not no_duplicate_edges:
        errors.append("duplicate edges found")

    no_zero_length_edges = True
    for edge in result.edges_vertices:
        if np.linalg.norm(result.pixel_vertices[edge[0]] - result.pixel_vertices[edge[1]]) < 1e-6:
            no_zero_length_edges = False
            errors.append("zero-length edge found")
            break

    no_illegal_crossings = _no_illegal_crossings(result)
    if not no_illegal_crossings:
        errors.append("illegal crossing found")

    has_border = bool(np.any(result.edges_assignment == 2))
    complete_border = True
    if has_border:
        complete_border = int(np.sum(result.edges_assignment == 2)) >= 4
        if not complete_border:
            errors.append("fewer than four border edges")

    return StructuralValidity(
        parseable_fold=parseable_fold,
        no_duplicate_edges=no_duplicate_edges,
        no_zero_length_edges=no_zero_length_edges,
        no_illegal_crossings=no_illegal_crossings,
        complete_border_when_present=complete_border,
        errors=errors,
    )


def metrics_from_results(metrics: list[GraphMetrics]) -> dict:
    if not metrics:
        return {}
    totals = {
        "files": len(metrics),
        "gt_vertices": sum(item.gt_vertices for item in metrics),
        "pred_vertices": sum(item.pred_vertices for item in metrics),
        "matched_vertices": sum(item.matched_vertices for item in metrics),
        "gt_edges": sum(item.gt_edges for item in metrics),
        "pred_edges": sum(item.pred_edges for item in metrics),
        "matched_edges": sum(item.matched_edges for item in metrics),
        "assignment_total": sum(item.assignment_total for item in metrics),
        "assignment_correct": sum(item.assignment_correct for item in metrics),
        "structurally_valid_files": sum(item.structural_validity.valid for item in metrics),
    }
    return {
        **totals,
        "vertex_precision": _ratio(totals["matched_vertices"], totals["pred_vertices"]),
        "vertex_recall": _ratio(totals["matched_vertices"], totals["gt_vertices"]),
        "edge_precision": _ratio(totals["matched_edges"], totals["pred_edges"]),
        "edge_recall": _ratio(totals["matched_edges"], totals["gt_edges"]),
        "assignment_accuracy": _ratio(totals["assignment_correct"], totals["assignment_total"]),
        "structural_validity_rate": _ratio(totals["structurally_valid_files"], totals["files"]),
        "mean_vertex_error_px": float(np.mean([item.mean_vertex_error_px for item in metrics])),
    }


def _assignment_by_class(
    matched_gt_edges: list[int],
    gt_assignments: np.ndarray,
    assignment_correct: list[bool],
) -> dict[str, dict[str, float | int]]:
    class_names = {0: "M", 1: "V", 2: "B", 3: "U"}
    totals = {name: 0 for name in class_names.values()}
    correct = {name: 0 for name in class_names.values()}
    for gt_idx, is_correct in zip(matched_gt_edges, assignment_correct):
        name = class_names.get(int(gt_assignments[gt_idx]), "U")
        totals[name] += 1
        correct[name] += int(is_correct)
    return {
        name: {
            "total": totals[name],
            "correct": correct[name],
            "accuracy": _ratio(correct[name], totals[name]),
        }
        for name in ["M", "V", "B", "U"]
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _no_illegal_crossings(result: PlanarGraphResult) -> bool:
    vertices = result.pixel_vertices
    edges = result.edges_vertices
    for i, edge_a in enumerate(edges):
        a0, a1 = int(edge_a[0]), int(edge_a[1])
        for edge_b in edges[i + 1 :]:
            b0, b1 = int(edge_b[0]), int(edge_b[1])
            if len({a0, a1, b0, b1}) < 4:
                continue
            if _segments_intersect(vertices[a0], vertices[a1], vertices[b0], vertices[b1]):
                return False
    return True


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-6

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
