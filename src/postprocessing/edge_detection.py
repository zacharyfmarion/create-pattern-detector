"""
Direct vertex-to-vertex edge detection.

Two approaches are provided:
1. detect_edges_direct: Checks all nearby vertex pairs (slow for dense graphs)
2. detect_edges_hybrid: Uses skeleton for connectivity, then validates with segmentation (faster)

The hybrid approach is recommended for production use.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import label
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class DetectedEdge:
    """An edge detected between two vertices."""
    start_idx: int
    end_idx: int
    confidence: float  # How well the edge matches the segmentation
    assignment: int  # Majority class along the edge (0=M, 1=V, 2=B, 3=U)
    orientation_score: float  # How well edge aligns with orientation field


def detect_edges_direct(
    vertices: np.ndarray,
    segmentation: np.ndarray,
    orientation: Optional[np.ndarray] = None,
    max_edge_length: float = None,
    min_coverage: float = 0.7,
    line_width: int = 3,
    boundary_vertices: Optional[np.ndarray] = None,
) -> List[DetectedEdge]:
    """
    Detect edges by directly checking vertex-to-vertex connections.

    For each pair of vertices within max_edge_length, we:
    1. Draw a line between them
    2. Sample the segmentation mask along the line
    3. Create an edge if enough crease pixels exist

    Uses KDTree for efficient neighbor lookup to avoid O(n²) complexity.

    Args:
        vertices: (N, 2) array of (x, y) vertex coordinates
        segmentation: (H, W) segmentation mask (BG=0, M=1, V=2, B=3, U=4)
        orientation: (H, W, 2) optional orientation field (cos θ, sin θ)
        max_edge_length: Maximum edge length to consider (default: image diagonal / 4)
        min_coverage: Minimum fraction of line that must be crease pixels
        line_width: Width of line to sample (pixels)
        boundary_vertices: (N,) bool array indicating which vertices are on boundary

    Returns:
        List of DetectedEdge objects
    """
    if len(vertices) < 2:
        return []

    h, w = segmentation.shape

    if max_edge_length is None:
        # Use a reasonable default based on image size
        max_edge_length = np.sqrt(h**2 + w**2) / 4

    # Build KDTree for efficient neighbor lookup
    tree = KDTree(vertices)

    # Find all pairs within max_edge_length using KDTree
    # This is O(n log n) instead of O(n²)
    pairs = tree.query_pairs(r=max_edge_length, output_type='ndarray')

    edges = []

    for i, j in pairs:
        dist = np.linalg.norm(vertices[i] - vertices[j])

        if dist < 1.0:  # Skip nearly coincident vertices
            continue

        # Check if there's a valid edge between these vertices
        edge = _check_edge(
            vertices[i], vertices[j],
            i, j,
            segmentation,
            orientation,
            min_coverage,
            line_width,
            boundary_vertices[i] if boundary_vertices is not None else False,
            boundary_vertices[j] if boundary_vertices is not None else False,
        )

        if edge is not None:
            edges.append(edge)

    return edges


def _check_edge(
    v1: np.ndarray,
    v2: np.ndarray,
    idx1: int,
    idx2: int,
    segmentation: np.ndarray,
    orientation: Optional[np.ndarray],
    min_coverage: float,
    line_width: int,
    is_boundary_1: bool,
    is_boundary_2: bool,
) -> Optional[DetectedEdge]:
    """
    Check if there's a valid edge between two vertices.

    Samples the segmentation along the line and checks for crease coverage.
    """
    h, w = segmentation.shape

    # Generate points along the line
    line_length = np.linalg.norm(v2 - v1)
    num_samples = max(int(line_length * 2), 10)  # Sample at sub-pixel resolution

    t = np.linspace(0, 1, num_samples)
    line_points = v1[None, :] + t[:, None] * (v2 - v1)[None, :]

    # Sample segmentation along the line (with width)
    labels = []
    valid_samples = 0

    # Edge direction for perpendicular sampling
    edge_dir = (v2 - v1) / line_length
    perp_dir = np.array([-edge_dir[1], edge_dir[0]])

    for point in line_points:
        # Sample across line width
        for offset in range(-(line_width // 2), line_width // 2 + 1):
            sample_point = point + offset * perp_dir
            x, y = int(round(sample_point[0])), int(round(sample_point[1]))

            if 0 <= x < w and 0 <= y < h:
                label = segmentation[y, x]
                if label > 0:  # Non-background
                    labels.append(label)
                valid_samples += 1

    if valid_samples == 0:
        return None

    # Calculate coverage (fraction of samples that are crease pixels)
    coverage = len(labels) / valid_samples

    # For boundary edges, we expect border class (3) along with creases
    # For interior edges, we expect mostly M/V/U
    if is_boundary_1 and is_boundary_2:
        # Both endpoints on boundary - this is a border edge
        expected_class = 3  # Border
        min_coverage_adjusted = min_coverage
    elif is_boundary_1 or is_boundary_2:
        # One endpoint on boundary - edge meets boundary
        min_coverage_adjusted = min_coverage * 0.8  # More lenient
    else:
        # Interior edge
        min_coverage_adjusted = min_coverage

    if coverage < min_coverage_adjusted:
        return None

    # Determine edge assignment by majority vote
    if len(labels) == 0:
        return None

    labels_arr = np.array(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_label = unique[majority_idx]
    assignment_confidence = counts[majority_idx] / len(labels)

    # Convert from segmentation labels (1-4) to assignment (0-3)
    assignment = int(majority_label - 1)

    # Check orientation consistency if available
    orientation_score = 1.0
    if orientation is not None:
        orientation_score = _check_orientation_consistency(
            line_points, edge_dir, orientation
        )

        # Reject edges with poor orientation alignment (unless it's a border)
        if orientation_score < 0.3 and assignment != 2:  # Not a border
            return None

    # Combined confidence
    confidence = coverage * assignment_confidence * orientation_score

    return DetectedEdge(
        start_idx=idx1,
        end_idx=idx2,
        confidence=confidence,
        assignment=assignment,
        orientation_score=orientation_score,
    )


def _check_orientation_consistency(
    line_points: np.ndarray,
    edge_dir: np.ndarray,
    orientation: np.ndarray,
) -> float:
    """
    Check if the orientation field along the line is consistent with edge direction.

    The orientation field is bidirectional (a line has no preferred direction),
    so we check |cos(angle)| between edge direction and orientation.
    """
    h, w = orientation.shape[:2]

    scores = []
    for point in line_points[::3]:  # Sample every 3rd point
        x, y = int(round(point[0])), int(round(point[1]))

        if 0 <= x < w and 0 <= y < h:
            orient = orientation[y, x]
            orient_norm = np.linalg.norm(orient)

            if orient_norm > 0.1:  # Only use significant orientations
                orient = orient / orient_norm
                # Bidirectional, so take absolute value
                score = abs(np.dot(edge_dir, orient))
                scores.append(score)

    if len(scores) == 0:
        return 0.5  # No orientation info, neutral score

    return np.mean(scores)


def filter_overlapping_edges(
    edges: List[DetectedEdge],
    vertices: np.ndarray,
    angle_threshold: float = 15.0,
) -> List[DetectedEdge]:
    """
    Filter out overlapping/redundant edges.

    When multiple edges go through similar paths, keep the one with highest confidence.
    """
    if len(edges) <= 1:
        return edges

    angle_threshold_rad = np.radians(angle_threshold)

    # Sort by confidence (highest first)
    edges_sorted = sorted(edges, key=lambda e: e.confidence, reverse=True)

    kept = []
    removed = set()

    for i, edge1 in enumerate(edges_sorted):
        if i in removed:
            continue

        kept.append(edge1)

        v1_start = vertices[edge1.start_idx]
        v1_end = vertices[edge1.end_idx]
        v1_mid = (v1_start + v1_end) / 2
        v1_dir = v1_end - v1_start
        v1_len = np.linalg.norm(v1_dir)
        if v1_len > 0:
            v1_dir = v1_dir / v1_len

        # Check remaining edges for overlap
        for j, edge2 in enumerate(edges_sorted[i+1:], start=i+1):
            if j in removed:
                continue

            v2_start = vertices[edge2.start_idx]
            v2_end = vertices[edge2.end_idx]
            v2_mid = (v2_start + v2_end) / 2
            v2_dir = v2_end - v2_start
            v2_len = np.linalg.norm(v2_dir)
            if v2_len > 0:
                v2_dir = v2_dir / v2_len

            # Check if edges are nearly parallel and close
            cos_angle = abs(np.dot(v1_dir, v2_dir))
            if cos_angle < np.cos(angle_threshold_rad):
                continue  # Not parallel enough

            # Check if midpoints are close
            mid_dist = np.linalg.norm(v1_mid - v2_mid)
            max_len = max(v1_len, v2_len)

            if mid_dist < max_len * 0.3:  # Midpoints within 30% of length
                removed.add(j)

    return kept


def edges_to_arrays(
    edges: List[DetectedEdge],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DetectedEdge list to array format.

    Returns:
        edge_indices: (E, 2) array of vertex index pairs
        assignments: (E,) array of edge assignments
        confidences: (E,) array of confidence scores
    """
    if len(edges) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float32),
        )

    edge_indices = np.array(
        [[e.start_idx, e.end_idx] for e in edges],
        dtype=np.int64
    )
    assignments = np.array([e.assignment for e in edges], dtype=np.int8)
    confidences = np.array([e.confidence for e in edges], dtype=np.float32)

    return edge_indices, assignments, confidences


def detect_edges_hybrid(
    vertices: np.ndarray,
    segmentation: np.ndarray,
    skeleton: np.ndarray,
    orientation: Optional[np.ndarray] = None,
    vertex_radius: int = 8,
    min_coverage: float = 0.5,
    line_width: int = 3,
    boundary_vertices: Optional[np.ndarray] = None,
    max_edge_length: float = None,
) -> List[DetectedEdge]:
    """
    Hybrid edge detection: use skeleton for connectivity, validate with segmentation.

    This is faster than pure direct detection because we only check vertex pairs
    that have skeleton connectivity between them AND are within max_edge_length.

    Algorithm:
    1. For each vertex, find skeleton pixels within vertex_radius
    2. Use connected components on skeleton to find which vertices share connectivity
    3. Filter by max_edge_length using KDTree for efficiency
    4. For each valid candidate pair, validate by checking segmentation coverage

    Args:
        vertices: (N, 2) array of (x, y) vertex coordinates
        segmentation: (H, W) segmentation mask (BG=0, M=1, V=2, B=3, U=4)
        skeleton: (H, W) binary skeleton mask
        orientation: (H, W, 2) optional orientation field
        vertex_radius: Radius around vertex to look for skeleton pixels
        min_coverage: Minimum segmentation coverage to validate edge
        line_width: Width of line to sample for validation
        boundary_vertices: (N,) bool array indicating boundary vertices
        max_edge_length: Maximum edge length to consider (default: image diagonal / 3)

    Returns:
        List of DetectedEdge objects
    """
    if len(vertices) < 2:
        return []

    h, w = skeleton.shape
    n_vertices = len(vertices)

    # Set default max_edge_length
    if max_edge_length is None:
        max_edge_length = np.sqrt(h**2 + w**2) / 3

    # Step 1: Build KDTree for distance filtering
    tree = KDTree(vertices)
    nearby_pairs = set(tuple(sorted(p)) for p in tree.query_pairs(r=max_edge_length, output_type='ndarray'))

    # Step 2: Label skeleton connected components
    labeled_skeleton, num_components = label(skeleton)

    # Step 3: For each vertex, find which skeleton components it connects to
    vertex_components = {}  # vertex_idx -> set of component labels

    for i, (vx, vy) in enumerate(vertices):
        vx_int, vy_int = int(round(vx)), int(round(vy))
        components = set()

        for dy in range(-vertex_radius, vertex_radius + 1):
            for dx in range(-vertex_radius, vertex_radius + 1):
                ny, nx = vy_int + dy, vx_int + dx
                if 0 <= ny < h and 0 <= nx < w:
                    comp = labeled_skeleton[ny, nx]
                    if comp > 0:  # Non-zero = skeleton pixel
                        components.add(comp)

        vertex_components[i] = components

    # Step 4: Find vertex pairs that share a skeleton component
    skeleton_pairs = set()

    # Build component -> vertices mapping
    component_to_vertices = {}
    for v_idx, comps in vertex_components.items():
        for comp in comps:
            if comp not in component_to_vertices:
                component_to_vertices[comp] = []
            component_to_vertices[comp].append(v_idx)

    # For each component, add all pairs of vertices connected to it
    for comp, v_list in component_to_vertices.items():
        for i, v1 in enumerate(v_list):
            for v2 in v_list[i+1:]:
                if v1 != v2:
                    skeleton_pairs.add((min(v1, v2), max(v1, v2)))

    # Step 5: Intersect skeleton connectivity with distance filter
    candidate_pairs = skeleton_pairs & nearby_pairs

    # Step 6: If too many candidates, use simplified validation
    # For very complex graphs, skip per-pixel validation and rely on skeleton connectivity
    MAX_CANDIDATES_FOR_FULL_CHECK = 5000
    use_fast_mode = len(candidate_pairs) > MAX_CANDIDATES_FOR_FULL_CHECK

    # Step 7: Validate each candidate pair with segmentation check
    edges = []

    for v1_idx, v2_idx in candidate_pairs:
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]

        dist = np.linalg.norm(v2 - v1)
        if dist < 1.0:  # Skip nearly coincident vertices
            continue

        if use_fast_mode:
            # Fast mode: trust skeleton connectivity, just do quick label sampling
            edge = _check_edge_fast(
                v1, v2,
                v1_idx, v2_idx,
                segmentation,
            )
        else:
            # Full mode: detailed per-pixel validation
            edge = _check_edge(
                v1, v2,
                v1_idx, v2_idx,
                segmentation,
                orientation,
                min_coverage,
                line_width,
                boundary_vertices[v1_idx] if boundary_vertices is not None else False,
                boundary_vertices[v2_idx] if boundary_vertices is not None else False,
            )

        if edge is not None:
            edges.append(edge)

    return edges


def _check_edge_fast(
    v1: np.ndarray,
    v2: np.ndarray,
    idx1: int,
    idx2: int,
    segmentation: np.ndarray,
) -> Optional[DetectedEdge]:
    """
    Fast edge check - samples a few points along the edge to determine assignment.

    Used when there are too many candidate pairs for full validation.
    """
    h, w = segmentation.shape

    # Sample 5 points along the line
    num_samples = 5
    t = np.linspace(0.1, 0.9, num_samples)  # Avoid endpoints
    line_points = v1[None, :] + t[:, None] * (v2 - v1)[None, :]

    labels = []
    for point in line_points:
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < w and 0 <= y < h:
            label = segmentation[y, x]
            if label > 0:  # Non-background
                labels.append(label)

    if len(labels) < 2:
        return None

    # Determine assignment by majority vote
    labels_arr = np.array(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    assignment = int(majority_label - 1)  # Convert from seg labels (1-4) to assignment (0-3)

    confidence = len(labels) / num_samples

    return DetectedEdge(
        start_idx=idx1,
        end_idx=idx2,
        confidence=confidence,
        assignment=assignment,
        orientation_score=1.0,
    )
