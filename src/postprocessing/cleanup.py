"""
Graph cleanup and label assignment utilities.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .edge_tracing import TracedEdge


def assign_edge_labels(
    edges: List[TracedEdge],
    segmentation: np.ndarray,
    skeleton_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign M/V/B/U labels to each edge based on segmentation.

    Uses majority voting along the edge path.

    Args:
        edges: List of TracedEdge objects with paths
        segmentation: (H, W) segmentation mask with class labels
                     BG=0, M=1, V=2, B=3, U=4
        skeleton_labels: (H, W) optional pre-computed skeleton labels

    Returns:
        assignments: (E,) array of class labels (0=M, 1=V, 2=B, 3=U)
        confidences: (E,) array of confidence scores (fraction of pixels matching)
    """
    if len(edges) == 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)

    h, w = segmentation.shape
    assignments = []
    confidences = []

    for edge in edges:
        # Sample segmentation along the edge path
        labels = []
        for x, y in edge.path:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < h and 0 <= ix < w:
                label = segmentation[iy, ix]
                if label > 0:  # Ignore background
                    labels.append(label)

        if len(labels) == 0:
            # No valid labels found, default to Unassigned
            assignments.append(3)  # U
            confidences.append(0.0)
            continue

        # Majority vote
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label = unique[majority_idx]
        confidence = counts[majority_idx] / len(labels)

        # Convert from segmentation labels (1-4) to assignment labels (0-3)
        # Seg: BG=0, M=1, V=2, B=3, U=4
        # Assignment: M=0, V=1, B=2, U=3
        assignment = majority_label - 1

        assignments.append(assignment)
        confidences.append(confidence)

    return np.array(assignments, dtype=np.int8), np.array(confidences, dtype=np.float32)


def remove_short_edges(
    edges: np.ndarray,
    edge_paths: List[np.ndarray],
    assignments: np.ndarray,
    confidences: np.ndarray,
    vertices: np.ndarray,
    min_length: float = 10.0,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Remove edges shorter than minimum length.

    Args:
        edges: (E, 2) edge vertex indices
        edge_paths: List of (L, 2) path arrays
        assignments: (E,) edge assignments
        confidences: (E,) edge confidences
        vertices: (V, 2) vertex coordinates
        min_length: Minimum edge length to keep

    Returns:
        Filtered edges, paths, assignments, confidences
    """
    if len(edges) == 0:
        return edges, edge_paths, assignments, confidences

    # Compute edge lengths
    lengths = np.sqrt(np.sum((vertices[edges[:, 0]] - vertices[edges[:, 1]]) ** 2, axis=1))

    # Keep edges that are long enough
    keep = lengths >= min_length

    return (
        edges[keep],
        [p for p, k in zip(edge_paths, keep) if k],
        assignments[keep],
        confidences[keep],
    )


def merge_collinear_edges(
    edges: np.ndarray,
    assignments: np.ndarray,
    vertices: np.ndarray,
    angle_threshold: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge nearly collinear edges at degree-2 vertices.

    If a vertex has exactly 2 edges that are nearly collinear and have
    the same assignment, merge them into a single edge and remove the vertex.

    Args:
        edges: (E, 2) edge vertex indices
        assignments: (E,) edge assignments
        vertices: (V, 2) vertex coordinates
        angle_threshold: Maximum angle (degrees) between edges to merge

    Returns:
        new_edges: (E', 2) merged edge indices
        new_assignments: (E',) merged assignments
        new_vertices: (V', 2) vertices with degree-2 removed
    """
    if len(edges) == 0:
        return edges, assignments, vertices

    angle_threshold_rad = np.radians(angle_threshold)

    # Compute vertex degrees
    num_vertices = len(vertices)
    degree = np.zeros(num_vertices, dtype=np.int32)
    for v1, v2 in edges:
        degree[v1] += 1
        degree[v2] += 1

    # Find degree-2 vertices
    degree_2_vertices = np.where(degree == 2)[0]

    # Build adjacency for quick edge lookup
    vertex_edges = {i: [] for i in range(num_vertices)}
    for edge_idx, (v1, v2) in enumerate(edges):
        vertex_edges[v1].append((edge_idx, v2))
        vertex_edges[v2].append((edge_idx, v1))

    # Track which edges to merge
    edges_to_remove = set()
    new_edges_list = []
    new_assignments_list = []

    for v in degree_2_vertices:
        if len(vertex_edges[v]) != 2:
            continue

        (edge1_idx, neighbor1), (edge2_idx, neighbor2) = vertex_edges[v]

        if edge1_idx in edges_to_remove or edge2_idx in edges_to_remove:
            continue

        # Check if same assignment
        if assignments[edge1_idx] != assignments[edge2_idx]:
            continue

        # Check angle between edges
        dir1 = vertices[neighbor1] - vertices[v]
        dir2 = vertices[neighbor2] - vertices[v]

        norm1 = np.linalg.norm(dir1)
        norm2 = np.linalg.norm(dir2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        dir1 /= norm1
        dir2 /= norm2

        # Edges are collinear if they point in opposite directions
        # (since they both point away from the shared vertex)
        cos_angle = np.dot(dir1, dir2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # For collinear edges pointing away from vertex, angle should be ~π
        if abs(angle - np.pi) < angle_threshold_rad:
            # Merge edges
            edges_to_remove.add(edge1_idx)
            edges_to_remove.add(edge2_idx)
            new_edges_list.append([neighbor1, neighbor2])
            new_assignments_list.append(assignments[edge1_idx])

    # Build new edge list
    final_edges = []
    final_assignments = []

    for i, (edge, assignment) in enumerate(zip(edges, assignments)):
        if i not in edges_to_remove:
            final_edges.append(edge)
            final_assignments.append(assignment)

    final_edges.extend(new_edges_list)
    final_assignments.extend(new_assignments_list)

    if len(final_edges) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int8), vertices

    return (
        np.array(final_edges, dtype=np.int64),
        np.array(final_assignments, dtype=np.int8),
        vertices,  # Keep all vertices for now (could prune unused ones)
    )


def snap_to_boundary(
    vertices: np.ndarray,
    is_boundary: np.ndarray,
    image_size: int,
    segmentation: np.ndarray = None,
) -> np.ndarray:
    """
    Snap boundary vertices to the paper boundary (if available) or image edge.

    When segmentation is provided, snaps to the actual paper boundary (border class).
    Otherwise falls back to snapping to image edges.

    Args:
        vertices: (V, 2) vertex coordinates (x, y)
        is_boundary: (V,) boolean array indicating boundary vertices
        image_size: Size of the image (assumed square)
        segmentation: Optional (H, W) segmentation with border class = 3

    Returns:
        vertices: (V, 2) with boundary vertices snapped to paper/image edges
    """
    vertices = vertices.copy()

    if segmentation is not None:
        # Find paper boundary from border class (3)
        border_mask = segmentation == 3
        if border_mask.any():
            border_coords = np.argwhere(border_mask)  # (N, 2) as (y, x)
            y_coords = border_coords[:, 0]
            x_coords = border_coords[:, 1]

            paper_left = x_coords.min()
            paper_right = x_coords.max()
            paper_top = y_coords.min()
            paper_bottom = y_coords.max()

            for i in np.where(is_boundary)[0]:
                x, y = vertices[i]

                # Find nearest paper edge
                dist_left = abs(x - paper_left)
                dist_right = abs(x - paper_right)
                dist_top = abs(y - paper_top)
                dist_bottom = abs(y - paper_bottom)

                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                # Only snap if we're reasonably close to a paper edge (within 10px)
                if min_dist <= 10:
                    if min_dist == dist_left:
                        vertices[i, 0] = paper_left
                    elif min_dist == dist_right:
                        vertices[i, 0] = paper_right
                    elif min_dist == dist_top:
                        vertices[i, 1] = paper_top
                    else:
                        vertices[i, 1] = paper_bottom

            return vertices

    # Fallback: snap to image edges (old behavior)
    for i in np.where(is_boundary)[0]:
        x, y = vertices[i]

        dist_left = x
        dist_right = image_size - 1 - x
        dist_top = y
        dist_bottom = image_size - 1 - y

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            vertices[i, 0] = 0
        elif min_dist == dist_right:
            vertices[i, 0] = image_size - 1
        elif min_dist == dist_top:
            vertices[i, 1] = 0
        else:
            vertices[i, 1] = image_size - 1

    return vertices


def remove_isolated_vertices(
    vertices: np.ndarray,
    edges: np.ndarray,
    is_boundary: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Remove vertices with no connected edges.

    Args:
        vertices: (V, 2) vertex coordinates
        edges: (E, 2) edge vertex indices
        is_boundary: (V,) optional boundary flags

    Returns:
        new_vertices: (V', 2) vertices with isolated removed
        new_edges: (E, 2) edges with updated indices
        new_is_boundary: (V',) updated boundary flags (if provided)
    """
    if len(edges) == 0:
        return vertices, edges, is_boundary

    # Find vertices that appear in edges
    used_vertices = set(edges.flatten())

    # Create mapping from old to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(vertices)):
        if old_idx in used_vertices:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Filter vertices
    keep_mask = np.array([i in used_vertices for i in range(len(vertices))])
    new_vertices = vertices[keep_mask]

    # Remap edge indices
    new_edges = np.array([
        [old_to_new[v1], old_to_new[v2]]
        for v1, v2 in edges
    ], dtype=np.int64)

    # Remap boundary flags
    new_is_boundary = None
    if is_boundary is not None:
        new_is_boundary = is_boundary[keep_mask]

    return new_vertices, new_edges, new_is_boundary
