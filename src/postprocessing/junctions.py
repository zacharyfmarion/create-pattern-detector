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
    skeleton: np.ndarray = None,
    use_skeleton_refinement: bool = True,
) -> np.ndarray:
    """
    Detect junction positions from a junction heatmap.

    When skeleton is provided and use_skeleton_refinement=True, also checks
    skeleton branch points to find additional junctions that were suppressed
    by NMS (common when multiple vertices are very close together).

    Args:
        heatmap: (H, W) junction probability heatmap (0-1)
        threshold: Minimum heatmap value for a valid junction
        min_distance: Minimum distance between detected junctions (NMS)
        sigma: Gaussian blur sigma for smoothing before peak detection
        skeleton: Optional (H, W) skeleton for branch point refinement
        use_skeleton_refinement: Whether to use skeleton branch points

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

    # Skeleton refinement: add branch points that have high heatmap values
    # but were suppressed by NMS (useful for closely-spaced junctions)
    if use_skeleton_refinement and skeleton is not None and len(junctions) > 0:
        branch_points = find_skeleton_branch_points(skeleton)  # (N, 2) as (y, x)

        if len(branch_points) > 0:
            from scipy.spatial.distance import cdist

            branch_xy = branch_points[:, ::-1].astype(np.float32)  # (x, y)

            # For each branch point, check if:
            # 1. It has high heatmap value (above threshold)
            # 2. It's not already near an existing junction
            additional = []
            dists_to_existing = cdist(branch_xy, junctions)

            for i, bp in enumerate(branch_xy):
                x, y = int(round(bp[0])), int(round(bp[1]))
                h, w = heatmap.shape

                if 0 <= y < h and 0 <= x < w:
                    hm_val = smoothed[y, x]

                    # Must have very high heatmap value (strong junction evidence)
                    # This is stricter than primary detection
                    if hm_val >= 0.7:  # Very high confidence only
                        # Must not be too close to existing junction
                        min_dist_to_existing = dists_to_existing[i].min()

                        # Add if it's 2-5px away from nearest junction
                        # (catches closely-spaced vertices that NMS suppressed)
                        # Only add if near existing junction (suggests NMS suppression)
                        if 2.0 <= min_dist_to_existing <= 5.0:
                            additional.append(bp)

            if additional:
                additional = np.array(additional, dtype=np.float32)
                # Cluster nearby additions
                additional = _cluster_points_xy(additional, cluster_distance=2)
                junctions = np.vstack([junctions, additional])

    return junctions


def _cluster_points_xy(points: np.ndarray, cluster_distance: float) -> np.ndarray:
    """Cluster nearby points (x, y format) and return centroids."""
    if len(points) <= 1:
        return points

    from scipy.spatial.distance import cdist

    distances = cdist(points, points)
    used = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if used[i]:
            continue
        nearby = np.where((distances[i] <= cluster_distance) & ~used)[0]
        used[nearby] = True
        cluster_points = points[nearby]
        centroid = cluster_points.mean(axis=0)
        clusters.append(centroid)

    return np.array(clusters, dtype=np.float32)


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
    use_skeleton_refinement: bool = True,
    merge_distance: float = 5.0,
) -> np.ndarray:
    """
    Detect junctions using heatmap (primary) and skeleton (fallback/refinement).

    Args:
        heatmap: (H, W) junction probability heatmap (0-1)
        skeleton: (H, W) binary skeleton mask (optional, for fallback)
        threshold: Minimum heatmap value for junction detection
        min_distance: Minimum distance between junctions
        use_skeleton_fallback: Also detect junctions from skeleton topology
        use_skeleton_refinement: Use skeleton to find NMS-suppressed junctions
        merge_distance: Distance threshold for merging heatmap/skeleton detections

    Returns:
        junctions: (N, 2) array of (x, y) junction coordinates
    """
    # Primary: detect from heatmap (with optional skeleton refinement)
    heatmap_junctions = detect_junctions_from_heatmap(
        heatmap,
        threshold=threshold,
        min_distance=min_distance,
        skeleton=skeleton if use_skeleton_refinement else None,
        use_skeleton_refinement=use_skeleton_refinement,
    )

    # Fallback: detect from skeleton topology (adds junctions not in heatmap)
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
    segmentation: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add vertices where crease lines meet the paper boundary.

    This detects boundary vertices in two ways:
    1. Skeleton endpoints near the image edges (old method)
    2. If segmentation is provided: endpoints of crease skeleton (M+V classes)
       that are near border pixels (class 3)

    Args:
        junctions: (N, 2) array of (x, y) interior junction coordinates
        skeleton: (H, W) binary skeleton mask
        boundary_distance: Max distance from boundary to be considered a boundary vertex
        segmentation: Optional (H, W) segmentation with classes BG=0, M=1, V=2, B=3, U=4

    Returns:
        all_vertices: (M, 2) array including boundary vertices
        is_boundary: (M,) boolean array indicating boundary vertices
    """
    from .skeletonize import find_skeleton_endpoints
    from skimage.morphology import skeletonize as skimage_skeletonize
    from scipy.ndimage import binary_dilation

    h, w = skeleton.shape
    boundary_vertices = []

    # Method 1: If we have segmentation, find where creases meet the border
    # AND find border-border intersections (paper corners)
    if segmentation is not None:
        border_mask = segmentation == 3

        # 1a. Find where creases meet the border
        crease_mask = (segmentation == 1) | (segmentation == 2)
        if crease_mask.any():
            crease_skeleton = skimage_skeletonize(crease_mask)
            crease_endpoints = find_skeleton_endpoints(crease_skeleton)  # (y, x) format

            if len(crease_endpoints) > 0:
                # Dilate border mask to catch nearby endpoints
                dilated_border = binary_dilation(border_mask, iterations=int(boundary_distance))

                endpoints_xy = crease_endpoints[:, ::-1].astype(np.float32)  # (x, y)
                for ep in endpoints_xy:
                    x, y = int(round(ep[0])), int(round(ep[1]))
                    if 0 <= y < h and 0 <= x < w and dilated_border[y, x]:
                        boundary_vertices.append(ep)

        # 1b. Find paper corners from the border mask
        # Look for the extreme points of the border (bounding box corners)
        if border_mask.any():
            border_coords = np.argwhere(border_mask)  # (y, x) format
            if len(border_coords) > 0:
                y_coords = border_coords[:, 0]
                x_coords = border_coords[:, 1]

                # Find the 4 corners as extreme points
                corners = []

                # Top-left: min x among min y, or min y among min x
                top_y = y_coords.min()
                left_x = x_coords.min()
                # Find point closest to (left_x, top_y)
                dists_to_tl = (x_coords - left_x)**2 + (y_coords - top_y)**2
                tl_idx = np.argmin(dists_to_tl)
                corners.append([x_coords[tl_idx], y_coords[tl_idx]])

                # Top-right
                right_x = x_coords.max()
                dists_to_tr = (x_coords - right_x)**2 + (y_coords - top_y)**2
                tr_idx = np.argmin(dists_to_tr)
                corners.append([x_coords[tr_idx], y_coords[tr_idx]])

                # Bottom-left
                bottom_y = y_coords.max()
                dists_to_bl = (x_coords - left_x)**2 + (y_coords - bottom_y)**2
                bl_idx = np.argmin(dists_to_bl)
                corners.append([x_coords[bl_idx], y_coords[bl_idx]])

                # Bottom-right
                dists_to_br = (x_coords - right_x)**2 + (y_coords - bottom_y)**2
                br_idx = np.argmin(dists_to_br)
                corners.append([x_coords[br_idx], y_coords[br_idx]])

                for corner in corners:
                    boundary_vertices.append(np.array(corner, dtype=np.float32))

    # Method 2: Also check skeleton endpoints near image edges (fallback)
    # Only used if we don't have segmentation with border info
    if segmentation is None:
        endpoints = find_skeleton_endpoints(skeleton)  # (y, x) format

        if len(endpoints) > 0:
            endpoints_xy = endpoints[:, ::-1].astype(np.float32)

            # Check which endpoints are near the image boundary
            near_left = endpoints_xy[:, 0] < boundary_distance
            near_right = endpoints_xy[:, 0] > w - boundary_distance
            near_top = endpoints_xy[:, 1] < boundary_distance
            near_bottom = endpoints_xy[:, 1] > h - boundary_distance

            is_near_edge = near_left | near_right | near_top | near_bottom
            edge_vertices = endpoints_xy[is_near_edge]

            # Snap to actual boundary
            for v in edge_vertices:
                x, y = v
                if x < boundary_distance:
                    v[0] = 0
                elif x > w - boundary_distance:
                    v[0] = w - 1
                if y < boundary_distance:
                    v[1] = 0
                elif y > h - boundary_distance:
                    v[1] = h - 1
                boundary_vertices.append(v.copy())

    # Deduplicate boundary vertices that are close together
    if len(boundary_vertices) > 0:
        boundary_vertices = np.array(boundary_vertices, dtype=np.float32)
        boundary_vertices = _deduplicate_vertices(boundary_vertices, min_distance=boundary_distance)
    else:
        boundary_vertices = np.empty((0, 2), dtype=np.float32)

    # Also remove boundary vertices that are too close to existing junctions
    if len(boundary_vertices) > 0 and len(junctions) > 0:
        from scipy.spatial.distance import cdist
        dists = cdist(boundary_vertices, junctions)
        far_from_junctions = dists.min(axis=1) > boundary_distance
        boundary_vertices = boundary_vertices[far_from_junctions]

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


def _deduplicate_vertices(vertices: np.ndarray, min_distance: float) -> np.ndarray:
    """Remove vertices that are too close together, keeping the first one."""
    if len(vertices) <= 1:
        return vertices

    from scipy.spatial.distance import cdist

    keep = np.ones(len(vertices), dtype=bool)
    dists = cdist(vertices, vertices)

    for i in range(len(vertices)):
        if not keep[i]:
            continue
        # Mark all later vertices within min_distance as duplicates
        for j in range(i + 1, len(vertices)):
            if keep[j] and dists[i, j] < min_distance:
                keep[j] = False

    return vertices[keep]
