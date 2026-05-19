"""Stage 4 graph quality reports and origami-aware diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.vectorization.constraint_repair import RepairAction
from src.vectorization.edge_assignment import AttributedPlanarGraph
from src.vectorization.metrics import StructuralValidity, validate_structure

STATUS_VALUES = ("valid", "repaired", "ambiguous", "outside_v1_envelope", "failed")


@dataclass(frozen=True)
class QualityReportConfig:
    image_size: int | None = None
    weak_edge_support_threshold: float = 0.40
    short_edge_warning_px: float = 8.0
    crowded_junction_px: float = 8.0
    low_assignment_confidence: float = 0.60
    low_assignment_margin: float = 0.12
    kawasaki_tolerance_radians: float = 0.12


@dataclass
class QualityWarning:
    code: str
    message: str
    severity: str = "warning"
    edge_indices: list[int] = field(default_factory=list)
    vertex_indices: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    status: str
    warnings: list[QualityWarning]
    structural_validity: StructuralValidity
    repair_actions: list[RepairAction] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "structural_validity": self.structural_validity.to_dict(),
            "repair_actions": [action.to_dict() for action in self.repair_actions],
            "summary": self.summary,
        }


def build_quality_report(
    graph: AttributedPlanarGraph,
    *,
    repair_actions: Iterable[RepairAction] = (),
    config: QualityReportConfig | None = None,
) -> QualityReport:
    """Build the Stage 4 status/warning report for a graph."""
    cfg = config or QualityReportConfig()
    actions = list(repair_actions)
    structural = validate_structure(graph.to_planar_result())
    warnings: list[QualityWarning] = []

    warnings.extend(_structural_warnings(graph, structural))
    warnings.extend(_support_and_envelope_warnings(graph, cfg))
    warnings.extend(_assignment_warnings(graph, cfg))
    warnings.extend(_origami_constraint_warnings(graph, cfg))

    status = _status_from_warnings(warnings, structural, actions, graph)
    summary = {
        "vertices": graph.num_vertices,
        "edges": graph.num_edges,
        "assignment_counts": _assignment_counts(graph.edges_assignment),
        "observed_edges": int(sum(source == "observed" for source in graph.assignment_source)),
        "unknown_edges": int(sum(source == "unknown" for source in graph.assignment_source)),
        "inferred_edges": int(sum(source == "inferred" for source in graph.assignment_source)),
        "mean_edge_support": float(np.mean(graph.edge_support)) if graph.num_edges else 0.0,
        "mean_assignment_confidence": (
            float(np.mean(graph.assignment_confidence)) if graph.num_edges else 0.0
        ),
    }
    return QualityReport(
        status=status,
        warnings=warnings,
        structural_validity=structural,
        repair_actions=actions,
        summary=summary,
    )


def _structural_warnings(
    graph: AttributedPlanarGraph,
    structural: StructuralValidity,
) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []
    if graph.num_edges == 0:
        warnings.append(
            QualityWarning(
                code="empty_graph",
                message="No edges were detected in the graph.",
                severity="error",
            )
        )
    if not structural.parseable_fold:
        warnings.append(
            QualityWarning(
                code="unparseable_fold",
                message="The predicted graph cannot be parsed as a FOLD graph.",
                severity="error",
                details={"errors": list(structural.errors)},
            )
        )
    if not structural.no_duplicate_edges:
        warnings.append(
            QualityWarning(
                code="duplicate_edges",
                message="Duplicate edges remain in the graph.",
                severity="error",
            )
        )
    if not structural.no_zero_length_edges:
        warnings.append(
            QualityWarning(
                code="zero_length_edges",
                message="Zero-length edges remain in the graph.",
                severity="error",
            )
        )
    if not structural.no_illegal_crossings:
        warnings.append(
            QualityWarning(
                code="illegal_crossings",
                message="Edges cross away from graph vertices.",
                severity="error",
            )
        )
    border_count = int(np.sum(graph.edges_assignment == 2))
    if border_count < 4:
        warnings.append(
            QualityWarning(
                code="incomplete_border",
                message="Fewer than four square border edges were recovered.",
                severity="warning",
                details={"border_edge_count": border_count},
            )
        )
    return warnings


def _support_and_envelope_warnings(
    graph: AttributedPlanarGraph,
    cfg: QualityReportConfig,
) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []
    weak = np.where(np.asarray(graph.edge_support) < cfg.weak_edge_support_threshold)[0]
    if len(weak):
        warnings.append(
            QualityWarning(
                code="weak_edges",
                message="Some edges have weak line-evidence support.",
                edge_indices=[int(idx) for idx in weak],
                details={"threshold": cfg.weak_edge_support_threshold},
            )
        )

    lengths = _edge_lengths(graph)
    short = np.where(lengths < cfg.short_edge_warning_px)[0]
    if len(short):
        warnings.append(
            QualityWarning(
                code="very_short_edges",
                message="Some predicted edges are shorter than the Phase 3 V1 readable-geometry envelope.",
                edge_indices=[int(idx) for idx in short],
                details={
                    "threshold_px": cfg.short_edge_warning_px,
                    "min_length_px": float(np.min(lengths[short])),
                },
            )
        )

    crowded = _crowded_vertices(graph, cfg.crowded_junction_px)
    if crowded:
        warnings.append(
            QualityWarning(
                code="crowded_junctions",
                message="Some junctions are closer than the Phase 3 V1 readable-geometry envelope.",
                vertex_indices=sorted(crowded),
                details={"threshold_px": cfg.crowded_junction_px},
            )
        )
    return warnings


def _assignment_warnings(
    graph: AttributedPlanarGraph,
    cfg: QualityReportConfig,
) -> list[QualityWarning]:
    low: list[int] = []
    unknown: list[int] = []
    for idx in range(graph.num_edges):
        if graph.assignment_source[idx] == "unknown":
            unknown.append(idx)
        if (
            float(graph.assignment_confidence[idx]) < cfg.low_assignment_confidence
            or float(graph.assignment_margin[idx]) < cfg.low_assignment_margin
        ):
            low.append(idx)
    warnings: list[QualityWarning] = []
    if low:
        warnings.append(
            QualityWarning(
                code="low_confidence_assignments",
                message="Some edge assignments have low confidence or a small class margin.",
                edge_indices=low,
                details={
                    "confidence_threshold": cfg.low_assignment_confidence,
                    "margin_threshold": cfg.low_assignment_margin,
                },
            )
        )
    if unknown:
        warnings.append(
            QualityWarning(
                code="unknown_assignments",
                message="Some edge assignments are visually ambiguous and remain unassigned.",
                edge_indices=unknown,
            )
        )
    return warnings


def _origami_constraint_warnings(
    graph: AttributedPlanarGraph,
    cfg: QualityReportConfig,
) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []
    adjacency = _edge_adjacency(graph)
    interior = _interior_vertices(graph, adjacency)
    odd_vertices: list[int] = []
    kawasaki_violations: list[int] = []
    maekawa_violations: list[int] = []
    kawasaki_residuals: dict[str, float] = {}

    for vertex_idx in interior:
        incident = [idx for idx in adjacency[vertex_idx] if int(graph.edges_assignment[idx]) != 2]
        if not incident:
            continue
        if len(incident) % 2 != 0:
            odd_vertices.append(vertex_idx)
        if len(incident) >= 4 and len(incident) % 2 == 0:
            residual = _kawasaki_residual(graph, vertex_idx, incident)
            if residual > cfg.kawasaki_tolerance_radians:
                kawasaki_violations.append(vertex_idx)
                kawasaki_residuals[str(vertex_idx)] = residual
        if incident and all(int(graph.edges_assignment[idx]) in (0, 1) for idx in incident):
            m_count = sum(int(graph.edges_assignment[idx]) == 0 for idx in incident)
            v_count = sum(int(graph.edges_assignment[idx]) == 1 for idx in incident)
            if abs(m_count - v_count) != 2:
                maekawa_violations.append(vertex_idx)

    if odd_vertices:
        warnings.append(
            QualityWarning(
                code="even_degree_failures",
                message="Interior vertices with crease evidence should have even degree.",
                vertex_indices=odd_vertices,
            )
        )
    if kawasaki_violations:
        warnings.append(
            QualityWarning(
                code="kawasaki_residuals",
                message="Interior vertex sector angles violate the Kawasaki tolerance.",
                vertex_indices=kawasaki_violations,
                details={
                    "tolerance_radians": cfg.kawasaki_tolerance_radians,
                    "residuals": kawasaki_residuals,
                },
            )
        )
    if maekawa_violations:
        warnings.append(
            QualityWarning(
                code="maekawa_failures",
                message="Fully observed M/V assignments violate Maekawa's theorem.",
                vertex_indices=maekawa_violations,
            )
        )
    return warnings


def _status_from_warnings(
    warnings: list[QualityWarning],
    structural: StructuralValidity,
    repair_actions: list[RepairAction],
    graph: AttributedPlanarGraph,
) -> str:
    codes = {warning.code for warning in warnings}
    failed_codes = {
        "empty_graph",
        "unparseable_fold",
        "duplicate_edges",
        "zero_length_edges",
        "illegal_crossings",
    }
    outside_codes = {"incomplete_border", "very_short_edges", "crowded_junctions", "weak_edges"}
    ambiguous_codes = {
        "low_confidence_assignments",
        "unknown_assignments",
        "even_degree_failures",
        "kawasaki_residuals",
        "maekawa_failures",
    }
    if codes & failed_codes or not structural.parseable_fold or graph.num_edges == 0:
        return "failed"
    if codes & outside_codes:
        return "outside_v1_envelope"
    if codes & ambiguous_codes:
        return "ambiguous"
    if repair_actions:
        return "repaired"
    return "valid"


def _assignment_counts(assignments: np.ndarray) -> dict[str, int]:
    names = {0: "M", 1: "V", 2: "B", 3: "U"}
    return {name: int(np.sum(assignments == idx)) for idx, name in names.items()}


def _edge_lengths(graph: AttributedPlanarGraph) -> np.ndarray:
    if graph.num_edges == 0:
        return np.empty(0, dtype=np.float32)
    p0 = graph.pixel_vertices[graph.edges_vertices[:, 0]]
    p1 = graph.pixel_vertices[graph.edges_vertices[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1).astype(np.float32)


def _crowded_vertices(graph: AttributedPlanarGraph, threshold_px: float) -> set[int]:
    crowded: set[int] = set()
    vertices = graph.pixel_vertices
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if np.linalg.norm(vertices[i] - vertices[j]) < threshold_px:
                crowded.add(i)
                crowded.add(j)
    return crowded


def _edge_adjacency(graph: AttributedPlanarGraph) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(graph.num_vertices)]
    for edge_idx, (v1, v2) in enumerate(graph.edges_vertices):
        adjacency[int(v1)].append(edge_idx)
        adjacency[int(v2)].append(edge_idx)
    return adjacency


def _interior_vertices(graph: AttributedPlanarGraph, adjacency: list[list[int]]) -> list[int]:
    interior = []
    for vertex_idx, incident in enumerate(adjacency):
        if incident and not any(int(graph.edges_assignment[idx]) == 2 for idx in incident):
            interior.append(vertex_idx)
    return interior


def _kawasaki_residual(
    graph: AttributedPlanarGraph,
    vertex_idx: int,
    incident_edges: list[int],
) -> float:
    center = graph.pixel_vertices[vertex_idx]
    angles = []
    for edge_idx in incident_edges:
        v1, v2 = graph.edges_vertices[edge_idx]
        other = int(v2) if int(v1) == vertex_idx else int(v1)
        vector = graph.pixel_vertices[other] - center
        if np.linalg.norm(vector) <= 1e-6:
            continue
        angles.append(float(np.arctan2(vector[1], vector[0]) % (2.0 * np.pi)))
    if len(angles) < 4 or len(angles) % 2 != 0:
        return 0.0
    ordered = np.sort(np.asarray(angles, dtype=np.float64))
    sectors = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * np.pi]))
    residual = abs(float(np.sum(sectors[0::2]) - np.sum(sectors[1::2])))
    return min(residual, abs(2.0 * np.pi - residual))
