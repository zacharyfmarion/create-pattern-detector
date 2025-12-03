"""
Junction detection from heatmap and skeleton topology.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max
from typing import Optional, Tuple

from .skeletonize import find_skeleton_branch_points


def detect_junctions_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Detect junction positions from a junction heatmap.

    Args:
        heatmap: (H, W) junction probability heatmap (0-1)
        threshold: Minimum heatmap value for a valid junction
        min_distance: Minimum distance between detected junctions (NMS)
        sigma: Gaussian blur sigma for smoothing before peak detection

    Returns:
        junctions: (N, 2) array of (x, y) junction coordinates
    """
    # Smooth heatmap to reduce noise
    if sigma > 0:
        smoothed = gaussian_filter(heatmap, sigma=sigma)
    else:
        smoothed = heatmap

    # Find local maxima
    # peak_local_max returns (y, x) coordinates
    peaks = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )

    # Convert from (y, x) to (x, y) for consistency with image coordinates
    if len(peaks) > 0:
        junctions = peaks[:, ::-1].astype(np.float32)  # (N, 2) as (x, y)
    else:
        junctions = np.empty((0, 2), dtype=np.float32)

    return junctions


def detect_junctions_from_skeleton(
    skeleton: np.ndarray,
    cluster_distance: int = 3,
) -> np.ndarray:
    """
    Detect junction positions from skeleton branch points.

    Branch points are skeleton pixels with >2 neighbors.
    Nearby branch points are clustered to get single junction positions.

    Args:
        skeleton: (H, W) binary skeleton mask
        cluster_distance: Merge branch points within this distance

    Returns:
        junctions: (N, 2) array of (x, y) junction coordinates
    """
    # Find branch points (pixels with >2 neighbors)
    branch_points = find_skeleton_branch_points(skeleton)  # (N, 2) as (y, x)

    if len(branch_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Cluster nearby branch points
    junctions = _cluster_points(branch_points, cluster_distance)

    # Convert from (y, x) to (x, y)
    junctions = junctions[:, ::-1].astype(np.float32)

    return junctions


def _cluster_points(points: np.ndarray, max_distance: float) -> np.ndarray:
    """
    Cluster nearby points and return cluster centroids.

    Simple greedy clustering: for each point, merge with nearby points.

    Args:
        points: (N, 2) array of point coordinates
        max_distance: Maximum distance to merge points

    Returns:
        centroids: (M, 2) array of cluster centroid coordinates
    """
    if len(points) == 0:
        return points

    from scipy.spatial.distance import cdist

    # Compute pairwise distances
    distances = cdist(points, points)

    # Greedy clustering
    used = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if used[i]:
            continue

        # Find all points within max_distance
        nearby = np.where((distances[i] <= max_distance) & ~used)[0]
        used[nearby] = True

        # Compute centroid
        cluster_points = points[nearby]
        centroid = cluster_points.mean(axis=0)
        clusters.append(centroid)

    return np.array(clusters)


def merge_junction_detections(
    heatmap_junctions: np.ndarray,
    skeleton_junctions: np.ndarray,
    merge_distance: float = 5.0,
) -> np.ndarray:
    """
    Merge junctions from heatmap and skeleton detection.

    Prefers heatmap positions when both methods detect nearby junctions.
    Adds skeleton-only junctions that weren't detected by heatmap.

    Args:
        heatmap_junctions: (N, 2) array of (x, y) from heatmap detection
        skeleton_junctions: (M, 2) array of (x, y) from skeleton detection
        merge_distance: Distance threshold for merging detections

    Returns:
        junctions: (K, 2) array of merged (x, y) junction coordinates
    """
    if len(heatmap_junctions) == 0:
        return skeleton_junctions.copy()

    if len(skeleton_junctions) == 0:
        return heatmap_junctions.copy()

    from scipy.spatial.distance import cdist

    # Start with all heatmap junctions (they're more accurate)
    merged = list(heatmap_junctions)

    # Check each skeleton junction
    distances = cdist(skeleton_junctions, heatmap_junctions)

    for i, skel_junction in enumerate(skeleton_junctions):
        # If no heatmap junction is nearby, add the skeleton junction
        if distances[i].min() > merge_distance:
            merged.append(skel_junction)

    return np.array(merged, dtype=np.float32)


def detect_junctions(
    heatmap: np.ndarray,
    skeleton: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    min_distance: int = 5,
    use_skeleton_fallback: bool = True,
    merge_distance: float = 5.0,
) -> np.ndarray:
    """
    Detect junctions using heatmap (primary) and skeleton (fallback).

    Args:
        heatmap: (H, W) junction probability heatmap (0-1)
        skeleton: (H, W) binary skeleton mask (optional, for fallback)
        threshold: Minimum heatmap value for junction detection
        min_distance: Minimum distance between junctions
        use_skeleton_fallback: Also detect junctions from skeleton topology
        merge_distance: Distance threshold for merging heatmap/skeleton detections

    Returns:
        junctions: (N, 2) array of (x, y) junction coordinates
    """
    # Primary: detect from heatmap
    heatmap_junctions = detect_junctions_from_heatmap(
        heatmap,
        threshold=threshold,
        min_distance=min_distance,
    )

    # Fallback: detect from skeleton topology
    if use_skeleton_fallback and skeleton is not None:
        skeleton_junctions = detect_junctions_from_skeleton(skeleton)

        # Merge both detection methods
        junctions = merge_junction_detections(
            heatmap_junctions,
            skeleton_junctions,
            merge_distance=merge_distance,
        )
    else:
        junctions = heatmap_junctions

    return junctions


def add_boundary_vertices(
    junctions: np.ndarray,
    skeleton: np.ndarray,
    boundary_distance: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add vertices at skeleton endpoints that touch the image boundary.

    Args:
        junctions: (N, 2) array of (x, y) interior junction coordinates
        skeleton: (H, W) binary skeleton mask
        boundary_distance: Max distance from boundary to be considered a boundary vertex

    Returns:
        all_vertices: (M, 2) array including boundary vertices
        is_boundary: (M,) boolean array indicating boundary vertices
    """
    from .skeletonize import find_skeleton_endpoints

    h, w = skeleton.shape

    # Find skeleton endpoints
    endpoints = find_skeleton_endpoints(skeleton)  # (y, x) format

    if len(endpoints) == 0:
        is_boundary = np.zeros(len(junctions), dtype=bool)
        return junctions, is_boundary

    # Convert to (x, y) format
    endpoints_xy = endpoints[:, ::-1].astype(np.float32)

    # Check which endpoints are near the boundary
    near_left = endpoints_xy[:, 0] < boundary_distance
    near_right = endpoints_xy[:, 0] > w - boundary_distance
    near_top = endpoints_xy[:, 1] < boundary_distance
    near_bottom = endpoints_xy[:, 1] > h - boundary_distance

    is_boundary_endpoint = near_left | near_right | near_top | near_bottom
    boundary_vertices = endpoints_xy[is_boundary_endpoint]

    # Snap boundary vertices to the actual boundary
    for i, v in enumerate(boundary_vertices):
        x, y = v
        if x < boundary_distance:
            boundary_vertices[i, 0] = 0
        elif x > w - boundary_distance:
            boundary_vertices[i, 0] = w - 1
        if y < boundary_distance:
            boundary_vertices[i, 1] = 0
        elif y > h - boundary_distance:
            boundary_vertices[i, 1] = h - 1

    # Combine interior junctions with boundary vertices
    if len(boundary_vertices) > 0:
        all_vertices = np.vstack([junctions, boundary_vertices])
        is_boundary = np.concatenate([
            np.zeros(len(junctions), dtype=bool),
            np.ones(len(boundary_vertices), dtype=bool),
        ])
    else:
        all_vertices = junctions
        is_boundary = np.zeros(len(junctions), dtype=bool)

    return all_vertices, is_boundary
