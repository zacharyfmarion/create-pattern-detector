"""Deterministic vectorization utilities for crease-pattern graphs."""

from .constraint_repair import (
    RepairAction,
    RepairConfig,
    RepairResult,
    conservative_repair,
)
from .cpline_adapter import cpline_outputs_to_evidence
from .diagnostics import build_stage4_diagnostic_payload, compute_what_if_statuses
from .edge_assignment import (
    AttributedPlanarGraph,
    EdgeAssignmentConfig,
    EdgeAssignmentResult,
    assign_edges_from_logits,
    attribute_graph_from_logits,
)
from .evidence import (
    RenderedVectorizerEvidence,
    render_vectorizer_evidence,
    render_vectorizer_evidence_from_pixels,
)
from .fold_writer import graph_to_fold_dict, save_fold
from .metrics import GraphMetrics, StructuralValidity, evaluate_graph
from .planar_graph_builder import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    PlanarGraphResult,
    VectorizerEvidence,
)
from .quality_report import (
    QualityReport,
    QualityReportConfig,
    QualityWarning,
    build_quality_report,
)
from .square_topology_decoder import SquareTopologyDecoder, SquareTopologyDecoderConfig

__all__ = [
    "AttributedPlanarGraph",
    "EdgeAssignmentConfig",
    "EdgeAssignmentResult",
    "GraphMetrics",
    "PlanarGraphBuilder",
    "PlanarGraphBuilderConfig",
    "PlanarGraphResult",
    "QualityReport",
    "QualityReportConfig",
    "QualityWarning",
    "RepairAction",
    "RepairConfig",
    "RepairResult",
    "RenderedVectorizerEvidence",
    "SquareTopologyDecoder",
    "SquareTopologyDecoderConfig",
    "StructuralValidity",
    "VectorizerEvidence",
    "assign_edges_from_logits",
    "attribute_graph_from_logits",
    "build_quality_report",
    "build_stage4_diagnostic_payload",
    "compute_what_if_statuses",
    "cpline_outputs_to_evidence",
    "conservative_repair",
    "evaluate_graph",
    "graph_to_fold_dict",
    "render_vectorizer_evidence",
    "render_vectorizer_evidence_from_pixels",
    "save_fold",
]
