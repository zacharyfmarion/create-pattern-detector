"""Deterministic vectorization utilities for crease-pattern graphs."""

from .cpline_adapter import cpline_outputs_to_evidence
from .evidence import (
    RenderedVectorizerEvidence,
    render_vectorizer_evidence,
    render_vectorizer_evidence_from_pixels,
)
from .metrics import GraphMetrics, StructuralValidity, evaluate_graph
from .planar_graph_builder import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    PlanarGraphResult,
    VectorizerEvidence,
)

__all__ = [
    "GraphMetrics",
    "PlanarGraphBuilder",
    "PlanarGraphBuilderConfig",
    "PlanarGraphResult",
    "RenderedVectorizerEvidence",
    "StructuralValidity",
    "VectorizerEvidence",
    "cpline_outputs_to_evidence",
    "evaluate_graph",
    "render_vectorizer_evidence",
    "render_vectorizer_evidence_from_pixels",
]
