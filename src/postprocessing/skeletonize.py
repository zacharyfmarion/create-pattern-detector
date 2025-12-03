"""
Skeletonization utilities for thinning segmentation masks to 1px wide lines.
"""

import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from typing import Tuple


def skeletonize_segmentation(
    segmentation: np.ndarray,
    preserve_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skeletonize a segmentation mask to 1px wide lines.

    Args:
        segmentation: (H, W) array with class labels
                     BG=0, M=1, V=2, B=3, U=4
        preserve_labels: If True, return skeleton with original class labels

    Returns:
        skeleton: (H, W) binary skeleton mask
        skeleton_labels: (H, W) skeleton with class labels preserved
                        (only if preserve_labels=True, else same as skeleton)
    """
    # Create binary mask of all creases (non-background)
    binary_mask = segmentation > 0

    # Apply morphological skeletonization
    skeleton = sk_skeletonize(binary_mask)

    if preserve_labels:
        # Preserve original class labels at skeleton positions
        skeleton_labels = np.zeros_like(segmentation)
        skeleton_labels[skeleton] = segmentation[skeleton]
    else:
        skeleton_labels = skeleton.astype(segmentation.dtype)

    return skeleton, skeleton_labels


def find_skeleton_branch_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Find branch points in a skeleton (pixels with >2 neighbors).

    These are candidate junction locations based on skeleton topology.

    Args:
        skeleton: (H, W) binary skeleton mask

    Returns:
        branch_points: (N, 2) array of (y, x) coordinates of branch points
    """
    from scipy.ndimage import convolve

    # Work on a copy to avoid any potential mutation issues
    skel = skeleton.copy()

    # Count 8-connected neighbors for each skeleton pixel
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

    # Branch points have >2 neighbors (junctions where lines meet)
    branch_mask = skel & (neighbor_count > 2)

    # Get coordinates
    branch_points = np.column_stack(np.where(branch_mask))

    return branch_points


def find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """
    Find endpoints in a skeleton (pixels with exactly 1 neighbor).

    These are dead ends or boundary connections.

    Args:
        skeleton: (H, W) binary skeleton mask

    Returns:
        endpoints: (N, 2) array of (y, x) coordinates of endpoints
    """
    from scipy.ndimage import convolve

    # Work on a copy to avoid any potential mutation issues
    skel = skeleton.copy()

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

    # Endpoints have exactly 1 neighbor
    endpoint_mask = skel & (neighbor_count == 1)

    endpoints = np.column_stack(np.where(endpoint_mask))

    return endpoints


def get_skeleton_neighbors(skeleton: np.ndarray, y: int, x: int) -> np.ndarray:
    """
    Get 8-connected skeleton neighbors of a pixel.

    Args:
        skeleton: (H, W) binary skeleton mask
        y, x: Pixel coordinates

    Returns:
        neighbors: (N, 2) array of (y, x) neighbor coordinates
    """
    h, w = skeleton.shape
    neighbors = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                neighbors.append([ny, nx])

    return np.array(neighbors) if neighbors else np.empty((0, 2), dtype=np.int64)
