"""
Main GraphExtractor class for converting pixel predictions to graph structure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .skeletonize import skeletonize_segmentation
from .junctions import detect_junctions, add_boundary_vertices
from .edge_tracing import trace_edges, edges_to_arrays
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
    """Configuration for graph extraction."""
    # Junction detection
    junction_threshold: float = 0.70  # Optimal threshold for focal loss model (93% precision, 73% recall)
    junction_min_distance: int = 10  # More spacing between junctions
    use_skeleton_junctions: bool = False  # Disable by default - adds too many spurious junctions
    junction_merge_distance: float = 5.0

    # Edge tracing
    junction_radius: int = 5  # Larger radius for finding skeleton near junction
    min_edge_length: float = 10.0

    # Cleanup
    merge_collinear: bool = True
    collinear_angle_threshold: float = 10.0
    snap_boundary: bool = True
    boundary_snap_distance: float = 5.0
    remove_isolated: bool = True


class GraphExtractor:
    """
    Extract graph structure from pixel head predictions.

    Pipeline:
    1. Skeletonize segmentation mask
    2. Detect junctions from heatmap + skeleton
    3. Trace edges between junctions
    4. Assign labels and compute confidence
    5. Cleanup (merge collinear, snap boundary, etc.)
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
        Extract graph from pixel predictions.

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

        # Step 1: Skeletonize
        skeleton, skeleton_labels = skeletonize_segmentation(
            segmentation,
            preserve_labels=True,
        )

        # Step 2: Detect junctions
        # use_skeleton_junctions controls both fallback and refinement behavior
        junctions = detect_junctions(
            junction_heatmap,
            skeleton=skeleton if self.config.use_skeleton_junctions else None,
            threshold=self.config.junction_threshold,
            min_distance=self.config.junction_min_distance,
            use_skeleton_fallback=self.config.use_skeleton_junctions,
            use_skeleton_refinement=self.config.use_skeleton_junctions,
            merge_distance=self.config.junction_merge_distance,
        )

        # Add boundary vertices (pass segmentation to find where creases meet border)
        vertices, is_boundary = add_boundary_vertices(
            junctions,
            skeleton,
            boundary_distance=self.config.boundary_snap_distance,
            segmentation=segmentation,
        )

        # Step 3: Trace edges
        traced_edges = trace_edges(
            skeleton,
            vertices,
            orientation=orientation,
            junction_radius=self.config.junction_radius,
            min_edge_length=5,  # Use a lower threshold during tracing
        )

        if len(traced_edges) == 0:
            # No edges found
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

        # Step 4: Assign labels
        assignments, confidence = assign_edge_labels(
            traced_edges,
            segmentation,
            skeleton_labels,
        )

        # Step 5: Cleanup

        # Remove short edges
        edges, edge_paths, assignments, confidence = remove_short_edges(
            edges, edge_paths, assignments, confidence, vertices,
            min_length=self.config.min_edge_length,
        )

        # Merge collinear edges
        if self.config.merge_collinear and len(edges) > 0:
            edges, assignments, vertices = merge_collinear_edges(
                edges, assignments, vertices,
                angle_threshold=self.config.collinear_angle_threshold,
            )
            # Note: edge_paths are invalidated after merging
            # We'd need to re-trace or concatenate paths
            edge_paths = []  # Clear paths after merge

        # Snap boundary vertices (use segmentation to find paper boundary)
        if self.config.snap_boundary:
            vertices = snap_to_boundary(vertices, is_boundary, image_size, segmentation)

        # Remove isolated vertices
        if self.config.remove_isolated and len(edges) > 0:
            vertices, edges, is_boundary = remove_isolated_vertices(
                vertices, edges, is_boundary
            )

        # Recompute confidence array to match edge count
        if len(edges) != len(confidence):
            # Confidence was invalidated by merge, set to 1.0
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
