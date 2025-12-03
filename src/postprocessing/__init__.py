"""
Post-processing module for converting pixel predictions to graph structure.
"""

from .graph_extraction import GraphExtractor, ExtractedGraph, GraphExtractorConfig
from .skeletonize import skeletonize_segmentation, find_skeleton_branch_points
from .junctions import detect_junctions, add_boundary_vertices
from .edge_tracing import trace_edges, TracedEdge
from .cleanup import assign_edge_labels, remove_short_edges, merge_collinear_edges

__all__ = [
    # Main classes
    "GraphExtractor",
    "ExtractedGraph",
    "GraphExtractorConfig",
    # Skeletonization
    "skeletonize_segmentation",
    "find_skeleton_branch_points",
    # Junction detection
    "detect_junctions",
    "add_boundary_vertices",
    # Edge tracing
    "trace_edges",
    "TracedEdge",
    # Cleanup
    "assign_edge_labels",
    "remove_short_edges",
    "merge_collinear_edges",
]
