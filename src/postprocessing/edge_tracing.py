"""
Edge tracing algorithm for connecting junctions along skeleton paths.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from .skeletonize import get_skeleton_neighbors


@dataclass
class TracedEdge:
    """A traced edge between two junctions."""
    start_idx: int  # Index of start junction
    end_idx: int    # Index of end junction
    path: np.ndarray  # (L, 2) array of (x, y) pixel coordinates along edge
    length: float   # Euclidean length of edge


def trace_edges(
    skeleton: np.ndarray,
    junctions: np.ndarray,
    orientation: Optional[np.ndarray] = None,
    junction_radius: int = 3,
    min_edge_length: int = 5,
) -> List[TracedEdge]:
    """
    Trace edges between junctions along the skeleton.

    Args:
        skeleton: (H, W) binary skeleton mask
        junctions: (N, 2) array of (x, y) junction coordinates
        orientation: (H, W, 2) orientation field (cos θ, sin θ) - used for ambiguity
        junction_radius: Radius around junction to start/stop tracing
        min_edge_length: Minimum edge length in pixels

    Returns:
        edges: List of TracedEdge objects
    """
    if len(junctions) == 0:
        return []

    h, w = skeleton.shape

    # Build KDTree for fast junction lookup
    junction_tree = KDTree(junctions)

    # Create junction mask for quick lookup
    junction_mask = np.zeros((h, w), dtype=np.int32)
    junction_mask.fill(-1)  # -1 means no junction

    for i, (x, y) in enumerate(junctions):
        ix, iy = int(round(x)), int(round(y))
        # Mark junction and nearby pixels
        for dy in range(-junction_radius, junction_radius + 1):
            for dx in range(-junction_radius, junction_radius + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < h and 0 <= nx < w:
                    junction_mask[ny, nx] = i

    # Track which edges we've already found (to avoid duplicates)
    found_edges: Set[Tuple[int, int]] = set()
    edges: List[TracedEdge] = []

    # For each junction, trace outward along skeleton branches
    for start_idx, (jx, jy) in enumerate(junctions):
        jx_int, jy_int = int(round(jx)), int(round(jy))

        # Find skeleton pixels near this junction
        start_pixels = _find_skeleton_near_junction(
            skeleton, jy_int, jx_int, junction_radius
        )

        # Trace from each starting pixel
        for sy, sx in start_pixels:
            # Skip if we've already traced this edge
            edge = _trace_single_edge(
                skeleton,
                junction_mask,
                start_idx,
                sy,
                sx,
                jy_int,
                jx_int,
                orientation,
            )

            if edge is None:
                continue

            # Create canonical edge key (smaller index first)
            edge_key = (min(edge.start_idx, edge.end_idx),
                        max(edge.start_idx, edge.end_idx))

            if edge_key in found_edges:
                continue

            if edge.length < min_edge_length:
                continue

            found_edges.add(edge_key)
            edges.append(edge)

    return edges


def _find_skeleton_near_junction(
    skeleton: np.ndarray,
    jy: int,
    jx: int,
    radius: int,
) -> List[Tuple[int, int]]:
    """
    Find skeleton pixels near a junction that could start edge traces.

    Returns pixels that are:
    1. On the skeleton
    2. Within radius of the junction
    3. On the "edge" of the junction area (not surrounded by junction)
    """
    h, w = skeleton.shape
    candidates = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = jy + dy, jx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not skeleton[ny, nx]:
                continue

            # Check if this pixel has neighbors outside the junction radius
            has_outside_neighbor = False
            for ndy in [-1, 0, 1]:
                for ndx in [-1, 0, 1]:
                    if ndy == 0 and ndx == 0:
                        continue
                    nny, nnx = ny + ndy, nx + ndx
                    if not (0 <= nny < h and 0 <= nnx < w):
                        continue
                    if skeleton[nny, nnx]:
                        dist = np.sqrt((nny - jy) ** 2 + (nnx - jx) ** 2)
                        if dist > radius:
                            has_outside_neighbor = True
                            break
                if has_outside_neighbor:
                    break

            if has_outside_neighbor:
                candidates.append((ny, nx))

    return candidates


def _trace_single_edge(
    skeleton: np.ndarray,
    junction_mask: np.ndarray,
    start_idx: int,
    start_y: int,
    start_x: int,
    junction_y: int,
    junction_x: int,
    orientation: Optional[np.ndarray] = None,
    max_steps: int = 10000,
) -> Optional[TracedEdge]:
    """
    Trace a single edge from a starting pixel until hitting another junction.

    Args:
        skeleton: Binary skeleton mask
        junction_mask: (H, W) array where value is junction index or -1
        start_idx: Index of the starting junction
        start_y, start_x: Starting pixel coordinates
        junction_y, junction_x: Junction coordinates (to avoid going back)
        orientation: Optional orientation field for resolving ambiguities
        max_steps: Maximum trace steps (prevent infinite loops)

    Returns:
        TracedEdge if we reach another junction, None otherwise
    """
    h, w = skeleton.shape
    path = [(start_x, start_y)]  # Store as (x, y)

    visited = set()
    visited.add((start_y, start_x))
    visited.add((junction_y, junction_x))  # Don't go back to start junction

    current_y, current_x = start_y, start_x

    for _ in range(max_steps):
        # Get skeleton neighbors
        neighbors = get_skeleton_neighbors(skeleton, current_y, current_x)

        if len(neighbors) == 0:
            # Dead end - no edge found
            return None

        # Filter out visited neighbors
        unvisited = []
        for ny, nx in neighbors:
            if (ny, nx) not in visited:
                unvisited.append((ny, nx))

        if len(unvisited) == 0:
            # All neighbors visited - no edge found
            return None

        # Choose next pixel
        if len(unvisited) == 1:
            next_y, next_x = unvisited[0]
        else:
            # Multiple choices - use orientation if available
            next_y, next_x = _choose_next_pixel(
                unvisited, current_y, current_x, path, orientation
            )

        # Check if we've reached another junction
        end_junction = junction_mask[next_y, next_x]
        if end_junction >= 0 and end_junction != start_idx:
            # Found another junction!
            path.append((next_x, next_y))

            # Compute edge length
            path_array = np.array(path, dtype=np.float32)
            diffs = np.diff(path_array, axis=0)
            length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

            return TracedEdge(
                start_idx=start_idx,
                end_idx=end_junction,
                path=path_array,
                length=length,
            )

        # Continue tracing
        visited.add((next_y, next_x))
        path.append((next_x, next_y))
        current_y, current_x = next_y, next_x

    # Max steps reached without finding junction
    return None


def _choose_next_pixel(
    candidates: List[Tuple[int, int]],
    current_y: int,
    current_x: int,
    path: List[Tuple[int, int]],
    orientation: Optional[np.ndarray],
) -> Tuple[int, int]:
    """
    Choose the next pixel when there are multiple candidates.

    Uses orientation field if available, otherwise chooses based on
    continuity with the current direction.
    """
    if len(path) < 2:
        # Not enough history, just pick first
        return candidates[0]

    # Compute current direction from recent path
    prev_x, prev_y = path[-2]
    curr_x, curr_y = path[-1]
    current_dir = np.array([curr_x - prev_x, curr_y - prev_y], dtype=np.float32)
    current_dir_norm = np.linalg.norm(current_dir)

    if current_dir_norm < 1e-6:
        return candidates[0]

    current_dir /= current_dir_norm

    # Score each candidate by how well it continues the current direction
    best_score = -np.inf
    best_candidate = candidates[0]

    for ny, nx in candidates:
        cand_dir = np.array([nx - current_x, ny - current_y], dtype=np.float32)
        cand_dir_norm = np.linalg.norm(cand_dir)

        if cand_dir_norm < 1e-6:
            continue

        cand_dir /= cand_dir_norm

        # Score is dot product (higher = more aligned)
        score = np.dot(current_dir, cand_dir)

        # If orientation field is available, also consider it
        if orientation is not None:
            orient = orientation[ny, nx]
            orient_norm = np.linalg.norm(orient)
            if orient_norm > 0.1:  # Only use if orientation is significant
                orient = orient / orient_norm
                # Orientation is bidirectional, so take abs of dot product
                orient_score = abs(np.dot(cand_dir, orient))
                score = 0.5 * score + 0.5 * orient_score

        if score > best_score:
            best_score = score
            best_candidate = (ny, nx)

    return best_candidate


def edges_to_arrays(
    edges: List[TracedEdge],
    num_vertices: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Convert list of TracedEdge to edge array format.

    Args:
        edges: List of TracedEdge objects
        num_vertices: Total number of vertices

    Returns:
        edge_indices: (E, 2) array of vertex index pairs
        edge_paths: List of (L, 2) path arrays for each edge
    """
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64), []

    edge_indices = np.array(
        [[e.start_idx, e.end_idx] for e in edges],
        dtype=np.int64
    )
    edge_paths = [e.path for e in edges]

    return edge_indices, edge_paths


# =============================================================================
# OVER-COMPLETE EDGE TRACING (for Graph Head pipeline)
# =============================================================================


def trace_edges_overcomplete(
    skeleton: np.ndarray,
    vertices: np.ndarray,
    orientation: Optional[np.ndarray] = None,
    junction_radius: int = 5,
    min_edge_length: float = 3.0,
    bridge_gap_pixels: int = 2,
    include_endpoint_edges: bool = True,
) -> List[TracedEdge]:
    """
    Trace OVER-COMPLETE edges between vertices along the skeleton.

    This intentionally generates MORE edges than needed - the Graph Head
    will learn to filter spurious ones. We prioritize RECALL over precision.

    Key behaviors:
    1. Bridge small gaps in the skeleton (up to bridge_gap_pixels)
    2. Include short spurs (down to min_edge_length)
    3. Keep over-segmented edges (multiple edges for one true crease)
    4. Include edges to skeleton endpoints

    Args:
        skeleton: (H, W) binary skeleton mask
        vertices: (N, 2) array of (x, y) vertex coordinates
        orientation: (H, W, 2) orientation field (cos θ, sin θ)
        junction_radius: Radius around junction to start/stop tracing
        min_edge_length: Very short minimum (include spurs)
        bridge_gap_pixels: Bridge skeleton gaps up to this size
        include_endpoint_edges: Include edges to skeleton endpoints

    Returns:
        edges: List of TracedEdge objects
    """
    if len(vertices) == 0:
        return []

    h, w = skeleton.shape

    # Optionally bridge small gaps in skeleton
    if bridge_gap_pixels > 0:
        skeleton = _bridge_skeleton_gaps(skeleton, max_gap=bridge_gap_pixels)

    # Build KDTree for fast vertex lookup
    vertex_tree = KDTree(vertices)

    # Create vertex mask for quick lookup
    vertex_mask = np.zeros((h, w), dtype=np.int32)
    vertex_mask.fill(-1)  # -1 means no vertex

    for i, (x, y) in enumerate(vertices):
        ix, iy = int(round(x)), int(round(y))
        # Mark vertex and nearby pixels
        for dy in range(-junction_radius, junction_radius + 1):
            for dx in range(-junction_radius, junction_radius + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if vertex_mask[ny, nx] == -1:  # Don't overwrite
                        vertex_mask[ny, nx] = i

    # Track which edges we've found
    found_edges: Set[Tuple[int, int]] = set()
    edges: List[TracedEdge] = []

    # For each vertex, trace outward along skeleton branches
    for start_idx, (vx, vy) in enumerate(vertices):
        vx_int, vy_int = int(round(vx)), int(round(vy))

        # Find skeleton pixels near this vertex
        start_pixels = _find_skeleton_near_junction(
            skeleton, vy_int, vx_int, junction_radius
        )

        # If no skeleton pixels found near vertex, try extending the search
        if len(start_pixels) == 0 and bridge_gap_pixels > 0:
            start_pixels = _find_skeleton_near_junction(
                skeleton, vy_int, vx_int, junction_radius + bridge_gap_pixels
            )

        # Trace from each starting pixel
        for sy, sx in start_pixels:
            edge = _trace_single_edge_overcomplete(
                skeleton,
                vertex_mask,
                start_idx,
                sy,
                sx,
                vy_int,
                vx_int,
                orientation,
                include_endpoint_edges,
                vertices,
                vertex_tree,
            )

            if edge is None:
                continue

            # Create canonical edge key
            edge_key = (
                min(edge.start_idx, edge.end_idx),
                max(edge.start_idx, edge.end_idx),
            )

            if edge_key in found_edges:
                continue

            # Very lenient minimum length (just to filter noise)
            if edge.length < min_edge_length:
                continue

            # Allow self-loops? No, that doesn't make sense for creases
            if edge.start_idx == edge.end_idx:
                continue

            found_edges.add(edge_key)
            edges.append(edge)

    return edges


def _bridge_skeleton_gaps(skeleton: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """
    Bridge small gaps in the skeleton using morphological closing.

    Args:
        skeleton: Binary skeleton mask
        max_gap: Maximum gap size to bridge

    Returns:
        skeleton_bridged: Skeleton with gaps filled
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.morphology import skeletonize as skimage_skeletonize

    # Dilate, then erode to close small gaps
    # Use a small structuring element
    dilated = binary_dilation(skeleton, iterations=max_gap)
    closed = binary_erosion(dilated, iterations=max_gap)

    # Re-skeletonize to maintain 1-pixel width
    bridged = skimage_skeletonize(closed)

    return bridged


def _trace_single_edge_overcomplete(
    skeleton: np.ndarray,
    vertex_mask: np.ndarray,
    start_idx: int,
    start_y: int,
    start_x: int,
    vertex_y: int,
    vertex_x: int,
    orientation: Optional[np.ndarray],
    include_endpoint_edges: bool,
    vertices: np.ndarray,
    vertex_tree: KDTree,
    max_steps: int = 10000,
) -> Optional[TracedEdge]:
    """
    Trace a single edge, with over-complete behavior.

    This is more permissive than the standard version:
    - Will create edges to skeleton endpoints (not just vertices)
    - More tolerant of tracing issues
    """
    h, w = skeleton.shape
    path = [(start_x, start_y)]  # Store as (x, y)

    visited = set()
    visited.add((start_y, start_x))
    visited.add((vertex_y, vertex_x))  # Don't go back to start vertex

    current_y, current_x = start_y, start_x

    for _ in range(max_steps):
        # Get skeleton neighbors
        neighbors = get_skeleton_neighbors(skeleton, current_y, current_x)

        if len(neighbors) == 0:
            # Dead end - check if we should create endpoint edge
            if include_endpoint_edges and len(path) >= 3:
                # Create edge to this endpoint
                # Find nearest vertex to endpoint
                endpoint = np.array([[current_x, current_y]], dtype=np.float32)
                dist, idx = vertex_tree.query(endpoint, k=1)

                if dist[0] < 10:  # Within reasonable distance
                    end_idx = idx[0]
                    if end_idx != start_idx:
                        path_array = np.array(path, dtype=np.float32)
                        diffs = np.diff(path_array, axis=0)
                        length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

                        return TracedEdge(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            path=path_array,
                            length=length,
                        )
            return None

        # Filter out visited neighbors
        unvisited = [(ny, nx) for ny, nx in neighbors if (ny, nx) not in visited]

        if len(unvisited) == 0:
            # All neighbors visited - no edge found
            return None

        # Choose next pixel
        if len(unvisited) == 1:
            next_y, next_x = unvisited[0]
        else:
            next_y, next_x = _choose_next_pixel(
                unvisited, current_y, current_x, path, orientation
            )

        # Check if we've reached another vertex
        end_vertex = vertex_mask[next_y, next_x]
        if end_vertex >= 0 and end_vertex != start_idx:
            # Found another vertex
            path.append((next_x, next_y))
            path_array = np.array(path, dtype=np.float32)
            diffs = np.diff(path_array, axis=0)
            length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

            return TracedEdge(
                start_idx=start_idx,
                end_idx=end_vertex,
                path=path_array,
                length=length,
            )

        # Continue tracing
        visited.add((next_y, next_x))
        path.append((next_x, next_y))
        current_y, current_x = next_y, next_x

    # Max steps reached
    return None
