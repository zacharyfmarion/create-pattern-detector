"""
Main GraphExtractor class for converting pixel predictions to graph structure.

This implements an OVER-COMPLETE candidate graph approach:
- Prioritize recall over precision at extraction time
- Include extra vertices and edges that might be spurious
- Let the downstream Graph Head learn to filter/refine

Key insight: If a real crease never shows up as a candidate edge,
the graph head can't resurrect it. But if we include extra edges,
the graph head can learn to drop them.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .skeletonize import skeletonize_segmentation, find_skeleton_endpoints
from .junctions import (
    detect_junctions_overcomplete,
    add_boundary_vertices_overcomplete,
)
from .edge_tracing import trace_edges_overcomplete, edges_to_arrays
from .edge_detection import (
    detect_edges_direct,
    detect_edges_hybrid,
    filter_overlapping_edges,
    edges_to_arrays as direct_edges_to_arrays,
)
from .cleanup import (
    assign_edge_labels,
    remove_short_edges,
    merge_collinear_edges,
    snap_to_boundary,
    remove_isolated_vertices,
)


@dataclass
class ExtractedGraph:
    """
    Extracted graph from pixel predictions.

    Coordinates are in image pixel space (x, y).
    """
    vertices: np.ndarray  # (N, 2) float32 - vertex coordinates (x, y)
    edges: np.ndarray  # (E, 2) int64 - vertex index pairs
    assignments: np.ndarray  # (E,) int8 - M=0, V=1, B=2, U=3
    confidence: np.ndarray  # (E,) float32 - edge confidence scores
    is_boundary: np.ndarray  # (N,) bool - whether vertex is on boundary
    edge_paths: List[np.ndarray] = field(default_factory=list)  # Edge pixel paths

    def to_fold_format(self, image_size: int = 512) -> Dict[str, Any]:
        """
        Convert to FOLD format dictionary.

        Args:
            image_size: Image size for normalizing coordinates to [0, 1]

        Returns:
            FOLD format dictionary with vertices_coords, edges_vertices, etc.
        """
        # Normalize coordinates to [0, 1]
        vertices_coords = (self.vertices / (image_size - 1)).tolist()

        # Convert edges to list of pairs
        edges_vertices = self.edges.tolist()

        # Convert assignments to FOLD format strings
        assignment_map = {0: 'M', 1: 'V', 2: 'B', 3: 'U'}
        edges_assignment = [assignment_map[a] for a in self.assignments]

        return {
            "file_spec": 1,
            "file_creator": "cp-custom-detector",
            "file_classes": ["singleModel"],
            "frame_classes": ["creasePattern"],
            "vertices_coords": vertices_coords,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment,
        }

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_edges(self) -> int:
        return len(self.edges)


@dataclass
class GraphExtractorConfig:
    """
    Configuration for over-complete graph extraction.

    Parameters are biased toward HIGH RECALL - we'd rather include
    spurious edges/vertices than miss real ones. The Graph Head
    will learn to filter out the noise.
    """
    # Edge detection method: "hybrid", "direct", or "skeleton" (default)
    edge_detection_method: str = "skeleton"

    # Junction detection - LOW thresholds for high recall
    junction_threshold: float = 0.55  # Just above 0.5 baseline - catch more peaks
    junction_min_distance: int = 3  # Small NMS distance - allow close vertices
    use_skeleton_junctions: bool = True  # Include topology-based junctions
    use_skeleton_endpoints: bool = True  # Include skeleton endpoints as vertices
    junction_merge_distance: float = 3.0  # Soft merge only very close vertices

    # Direct/Hybrid edge detection settings
    direct_min_coverage: float = 0.5  # Min fraction of line covered by crease pixels
    direct_line_width: int = 3  # Width of line to sample
    direct_max_edge_length: float = None  # Max edge length (None = auto, for "direct" only)
    hybrid_vertex_radius: int = 8  # Radius to search for skeleton near vertex (for "hybrid")

    # Skeleton-based edge tracing (when edge_detection_method="skeleton")
    junction_radius: int = 5  # Radius for finding skeleton near junction
    min_edge_length: float = 3.0  # Very short min - include spurs
    bridge_gap_pixels: int = 2  # Bridge small skeleton gaps
    include_endpoint_edges: bool = True  # Include edges to skeleton endpoints

    # Cleanup - MINIMAL cleanup for over-complete graph
    merge_collinear: bool = False  # Don't merge - let graph head decide
    collinear_angle_threshold: float = 10.0
    snap_boundary: bool = True  # Still snap boundary vertices
    boundary_snap_distance: float = 5.0
    remove_isolated: bool = False  # Don't remove - might be needed
    include_boundary_corners: bool = True  # Include image corners as vertices


class GraphExtractor:
    """
    Extract OVER-COMPLETE graph structure from pixel head predictions.

    This extractor prioritizes RECALL over precision:
    - Include all plausible vertices (heatmap peaks, skeleton junctions, endpoints)
    - Include all plausible edges (even short spurs and over-segmented creases)
    - Minimal cleanup - let the Graph Head learn to filter

    Pipeline:
    1. Skeletonize segmentation mask
    2. Detect over-complete vertices (heatmap + skeleton junctions + endpoints + corners)
    3. Trace over-complete edges (bridge gaps, include short spurs)
    4. Assign labels and compute confidence
    5. Minimal cleanup (snap boundary only)
    """

    def __init__(self, config: Optional[GraphExtractorConfig] = None):
        """
        Initialize graph extractor.

        Args:
            config: Extraction configuration (uses defaults if None)
        """
        self.config = config or GraphExtractorConfig()

    def extract(
        self,
        segmentation: np.ndarray,
        junction_heatmap: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> ExtractedGraph:
        """
        Extract over-complete graph from pixel predictions.

        The resulting graph will have:
        - More vertices than the true graph (some duplicates, some spurious)
        - More edges than the true graph (some redundant, some short spurs)

        This is intentional - the Graph Head will learn to prune.

        Args:
            segmentation: (H, W) segmentation mask with class labels
                         BG=0, M=1, V=2, B=3, U=4
            junction_heatmap: (H, W) junction probability heatmap (0-1)
            orientation: (H, W, 2) optional orientation field (cos θ, sin θ)

        Returns:
            ExtractedGraph with vertices, edges, assignments, etc.
        """
        h, w = segmentation.shape
        image_size = max(h, w)

        # Step 1: Skeletonize (needed for vertex detection even with direct edge method)
        skeleton, skeleton_labels = skeletonize_segmentation(
            segmentation,
            preserve_labels=True,
        )

        # Step 2: Detect over-complete vertices
        # Include: heatmap peaks + skeleton junctions + skeleton endpoints + boundary corners
        vertices, vertex_sources = detect_junctions_overcomplete(
            junction_heatmap,
            skeleton=skeleton,
            threshold=self.config.junction_threshold,
            min_distance=self.config.junction_min_distance,
            use_skeleton_junctions=self.config.use_skeleton_junctions,
            use_skeleton_endpoints=self.config.use_skeleton_endpoints,
            merge_distance=self.config.junction_merge_distance,
        )

        # Add boundary vertices (crease-border intersections + paper corners)
        vertices, is_boundary = add_boundary_vertices_overcomplete(
            vertices,
            skeleton,
            boundary_distance=self.config.boundary_snap_distance,
            segmentation=segmentation,
            include_corners=self.config.include_boundary_corners,
        )

        # Step 3: Detect edges using configured method
        if self.config.edge_detection_method == "hybrid":
            # Hybrid: use skeleton for connectivity, validate with segmentation
            detected_edges = detect_edges_hybrid(
                vertices,
                segmentation,
                skeleton,
                orientation=orientation,
                vertex_radius=self.config.hybrid_vertex_radius,
                min_coverage=self.config.direct_min_coverage,
                line_width=self.config.direct_line_width,
                boundary_vertices=is_boundary,
            )

            # Filter overlapping edges
            detected_edges = filter_overlapping_edges(detected_edges, vertices)

            if len(detected_edges) == 0:
                return ExtractedGraph(
                    vertices=vertices,
                    edges=np.empty((0, 2), dtype=np.int64),
                    assignments=np.empty(0, dtype=np.int8),
                    confidence=np.empty(0, dtype=np.float32),
                    is_boundary=is_boundary,
                    edge_paths=[],
                )

            # Convert to arrays
            edges, assignments, confidence = direct_edges_to_arrays(detected_edges)
            edge_paths = []

        elif self.config.edge_detection_method == "direct":
            # Direct vertex-to-vertex edge detection (slow for dense graphs)
            detected_edges = detect_edges_direct(
                vertices,
                segmentation,
                orientation=orientation,
                max_edge_length=self.config.direct_max_edge_length,
                min_coverage=self.config.direct_min_coverage,
                line_width=self.config.direct_line_width,
                boundary_vertices=is_boundary,
            )

            # Filter overlapping edges
            detected_edges = filter_overlapping_edges(detected_edges, vertices)

            if len(detected_edges) == 0:
                return ExtractedGraph(
                    vertices=vertices,
                    edges=np.empty((0, 2), dtype=np.int64),
                    assignments=np.empty(0, dtype=np.int8),
                    confidence=np.empty(0, dtype=np.float32),
                    is_boundary=is_boundary,
                    edge_paths=[],
                )

            # Convert to arrays
            edges, assignments, confidence = direct_edges_to_arrays(detected_edges)
            edge_paths = []

        else:
            # Skeleton-based edge tracing (original method)
            traced_edges = trace_edges_overcomplete(
                skeleton,
                vertices,
                orientation=orientation,
                junction_radius=self.config.junction_radius,
                min_edge_length=self.config.min_edge_length,
                bridge_gap_pixels=self.config.bridge_gap_pixels,
                include_endpoint_edges=self.config.include_endpoint_edges,
            )

            if len(traced_edges) == 0:
                return ExtractedGraph(
                    vertices=vertices,
                    edges=np.empty((0, 2), dtype=np.int64),
                    assignments=np.empty(0, dtype=np.int8),
                    confidence=np.empty(0, dtype=np.float32),
                    is_boundary=is_boundary,
                    edge_paths=[],
                )

            # Convert edges to arrays
            edges, edge_paths = edges_to_arrays(traced_edges, len(vertices))

            # Assign labels
            assignments, confidence = assign_edge_labels(
                traced_edges,
                segmentation,
                skeleton_labels,
            )

            # Remove very short edges
            edges, edge_paths, assignments, confidence = remove_short_edges(
                edges, edge_paths, assignments, confidence, vertices,
                min_length=self.config.min_edge_length,
            )

        # Step 4: Cleanup (common for both methods)

        # Merge collinear edges if enabled
        if self.config.merge_collinear and len(edges) > 0:
            edges, assignments, vertices = merge_collinear_edges(
                edges, assignments, vertices,
                angle_threshold=self.config.collinear_angle_threshold,
            )
            edge_paths = []

        # Snap boundary vertices to actual boundary
        if self.config.snap_boundary:
            vertices = snap_to_boundary(vertices, is_boundary, image_size, segmentation)

        # Remove isolated vertices if enabled
        if self.config.remove_isolated and len(edges) > 0:
            vertices, edges, is_boundary = remove_isolated_vertices(
                vertices, edges, is_boundary
            )

        # Ensure confidence array matches edge count
        if len(edges) != len(confidence):
            confidence = np.ones(len(edges), dtype=np.float32)

        return ExtractedGraph(
            vertices=vertices,
            edges=edges,
            assignments=assignments,
            confidence=confidence,
            is_boundary=is_boundary if is_boundary is not None else np.zeros(len(vertices), dtype=bool),
            edge_paths=edge_paths,
        )

    def extract_from_model_output(
        self,
        output: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> ExtractedGraph:
        """
        Extract graph from model output dictionary.

        Convenience method that handles tensor conversion and thresholding.

        Args:
            output: Model output dictionary with keys:
                   - 'segmentation': (H, W) or (1, C, H, W) logits
                   - 'junction': (H, W) or (1, 1, H, W) heatmap
                   - 'orientation': (H, W, 2) or (1, 2, H, W) field
            threshold: Junction heatmap threshold

        Returns:
            ExtractedGraph
        """
        # Handle segmentation
        seg = output['segmentation']
        if seg.ndim == 4:  # (B, C, H, W)
            seg = seg[0].argmax(axis=0)  # (H, W)
        elif seg.ndim == 3:  # (C, H, W)
            seg = seg.argmax(axis=0)

        # Handle junction heatmap
        junction = output['junction']
        if junction.ndim == 4:  # (B, 1, H, W)
            junction = junction[0, 0]  # (H, W)
        elif junction.ndim == 3:  # (1, H, W)
            junction = junction[0]

        # Handle orientation
        orientation = output.get('orientation')
        if orientation is not None:
            if orientation.ndim == 4:  # (B, 2, H, W)
                orientation = orientation[0].transpose(1, 2, 0)  # (H, W, 2)
            elif orientation.ndim == 3:  # (2, H, W)
                orientation = orientation.transpose(1, 2, 0)

        # Update threshold
        old_threshold = self.config.junction_threshold
        self.config.junction_threshold = threshold

        result = self.extract(seg, junction, orientation)

        self.config.junction_threshold = old_threshold

        return result
