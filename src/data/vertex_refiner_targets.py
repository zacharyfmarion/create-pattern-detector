"""Targets and source-image evidence for VertexRefinerV1 crops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ModuleNotFoundError:  # pragma: no cover - scipy is part of the normal dev env.
    distance_transform_edt = None

from src.models.vertex_refiner_contract import (
    BOUNDARY_SIDE_IDS,
    CROP_SIZE_PX,
    DEGREE_CLASS_NAMES,
    INCIDENT_RAY_BINS,
    VERTEX_KIND_NAMES,
    degree_class_for_count,
    ray_bin_for_delta,
)

VERTEX_KIND_IDS = {name: index for index, name in enumerate(VERTEX_KIND_NAMES)}


@dataclass(frozen=True)
class SquareFrame:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(1.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(1.0, self.y_max - self.y_min)


@dataclass(frozen=True)
class VertexRefinerTargetConfig:
    crop_size: int = CROP_SIZE_PX
    heatmap_sigma_px: float = 1.0
    offset_radius_px: float = 2.0
    class_radius_px: float = 2.0
    boundary_tolerance_px: float = 2.5


@dataclass(frozen=True)
class VertexRefinerTargets:
    vertex_heatmap: np.ndarray
    vertex_offset: np.ndarray
    vertex_offset_mask: np.ndarray
    vertex_kind: np.ndarray
    vertex_kind_mask: np.ndarray
    degree: np.ndarray
    degree_mask: np.ndarray
    incident_rays: np.ndarray
    incident_ray_mask: np.ndarray
    boundary_contact_heatmap: np.ndarray
    boundary_side: np.ndarray
    boundary_side_mask: np.ndarray
    local_vertices: np.ndarray
    vertex_indices: np.ndarray
    metadata: dict[str, Any]


def grayscale_image(image: np.ndarray) -> np.ndarray:
    """Return a float32 grayscale image normalized to 0..1."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float32)
    elif array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("image must have shape HxW or HxWx3")
    if gray.size == 0:
        return gray.astype(np.float32)
    if float(np.nanmax(gray)) > 1.5:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def source_ink_probability(image: np.ndarray) -> np.ndarray:
    """Estimate line/ink probability directly from source pixels.

    This intentionally does not use CPLineNet line probability. It looks for
    pixels that differ from the dominant background by luminance or chroma, so it
    handles black-on-white, colored assignment lines, and bright-on-dark renders.
    """
    gray = grayscale_image(image)
    median = float(np.median(gray))
    contrast = np.abs(gray - median)
    scale = max(float(np.percentile(contrast, 98)), 1e-3)
    contrast_prob = contrast / scale

    if np.asarray(image).ndim == 3 and np.asarray(image).shape[2] >= 3:
        rgb = np.asarray(image)[..., :3].astype(np.float32)
        if float(np.nanmax(rgb)) > 1.5:
            rgb = rgb / 255.0
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        chroma_scale = max(float(np.percentile(chroma, 98)), 1e-3)
        chroma_prob = chroma / chroma_scale
        prob = np.maximum(contrast_prob, chroma_prob)
    else:
        prob = contrast_prob

    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    if min(prob.shape) >= 3:
        prob = np.maximum(prob, cv2.GaussianBlur(prob, (3, 3), 0).astype(np.float32))
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def distance_to_ink_map(
    ink_probability: np.ndarray,
    *,
    ink_threshold: float = 0.2,
    normalize_by_px: float = CROP_SIZE_PX,
) -> np.ndarray:
    """Return a clipped distance-to-ink map normalized by ``normalize_by_px``."""
    ink = np.asarray(ink_probability, dtype=np.float32) >= float(ink_threshold)
    if not np.any(ink):
        return np.ones_like(ink_probability, dtype=np.float32)
    if distance_transform_edt is not None:
        distance = distance_transform_edt(~ink).astype(np.float32)
    else:  # pragma: no cover - scipy is available in normal test/dev envs.
        background = (~ink).astype(np.uint8)
        distance = cv2.distanceTransform(background, cv2.DIST_L2, 3).astype(np.float32)
    denom = max(float(normalize_by_px), 1.0)
    return np.clip(distance / denom, 0.0, 1.0).astype(np.float32)


def infer_square_frame(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image_size: int,
) -> SquareFrame:
    """Infer the rendered square frame from boundary edges."""
    vertices = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int8)
    border_edge_indices = np.flatnonzero(assignments == 2)
    if len(border_edge_indices) == 0:
        return SquareFrame(0.0, 0.0, float(image_size - 1), float(image_size - 1))
    border_vertex_indices = sorted(
        {int(vertex_idx) for edge_idx in border_edge_indices for vertex_idx in edges[int(edge_idx)]}
    )
    border_vertices = vertices[border_vertex_indices]
    return SquareFrame(
        x_min=float(np.min(border_vertices[:, 0])),
        y_min=float(np.min(border_vertices[:, 1])),
        x_max=float(np.max(border_vertices[:, 0])),
        y_max=float(np.max(border_vertices[:, 1])),
    )


def vertex_degrees(vertex_count: int, edges: np.ndarray) -> np.ndarray:
    degrees = np.zeros(vertex_count, dtype=np.int32)
    for v1, v2 in np.asarray(edges, dtype=np.int64):
        degrees[int(v1)] += 1
        degrees[int(v2)] += 1
    return degrees


def incident_ray_bins_for_vertex(
    vertex_index: int,
    vertices: np.ndarray,
    edges: np.ndarray,
) -> list[int]:
    """Return incident ray bins for all edges touching ``vertex_index``."""
    vertices = np.asarray(vertices, dtype=np.float32)
    bins: list[int] = []
    for v1, v2 in np.asarray(edges, dtype=np.int64):
        if int(v1) == vertex_index:
            delta = vertices[int(v2)] - vertices[int(v1)]
        elif int(v2) == vertex_index:
            delta = vertices[int(v1)] - vertices[int(v2)]
        else:
            continue
        if float(np.linalg.norm(delta)) <= 1e-8:
            continue
        bins.append(ray_bin_for_delta(float(delta[0]), float(delta[1])))
    return sorted(set(bins))


def classify_vertex_kind(
    vertex_index: int,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    frame: SquareFrame,
    *,
    image_size: int,
    boundary_tolerance_px: float = 2.5,
) -> str:
    """Classify a GT vertex into the V1 vertex-kind classes."""
    degrees = vertex_degrees(len(vertices), edges)
    boundary = boundary_position(
        vertices[vertex_index],
        frame,
        image_size=image_size,
        tolerance_px=boundary_tolerance_px,
    )
    if boundary is not None:
        return "corner" if boundary["is_corner"] else "boundary_contact"
    if int(degrees[vertex_index]) <= 1:
        return "endpoint_or_dangling"
    return "interior_junction"


def boundary_position(
    vertex: np.ndarray,
    frame: SquareFrame,
    *,
    image_size: int,
    tolerance_px: float = 2.5,
) -> dict[str, Any] | None:
    tolerance = max(float(tolerance_px), 4.0 * float(image_size) / 1024.0)
    x = float(vertex[0])
    y = float(vertex[1])
    near_left = abs(x - frame.x_min) <= tolerance
    near_right = abs(x - frame.x_max) <= tolerance
    near_top = abs(y - frame.y_min) <= tolerance
    near_bottom = abs(y - frame.y_max) <= tolerance
    if (near_left or near_right) and (near_top or near_bottom):
        if near_top and near_left:
            return {"side": "top", "coordinate": 0.0, "is_corner": True}
        if near_top and near_right:
            return {"side": "right", "coordinate": 1.0, "is_corner": True}
        if near_bottom and near_right:
            return {"side": "bottom", "coordinate": 1.0, "is_corner": True}
        return {"side": "left", "coordinate": 0.0, "is_corner": True}

    candidates = [
        ("top", abs(y - frame.y_min), (x - frame.x_min) / frame.width),
        ("right", abs(x - frame.x_max), (y - frame.y_min) / frame.height),
        ("bottom", abs(y - frame.y_max), (x - frame.x_min) / frame.width),
        ("left", abs(x - frame.x_min), (y - frame.y_min) / frame.height),
    ]
    side, distance, coordinate = min(candidates, key=lambda item: item[1])
    if float(distance) > tolerance:
        return None
    return {
        "side": side,
        "coordinate": float(np.clip(coordinate, 0.0, 1.0)),
        "is_corner": False,
    }


def build_vertex_refiner_targets(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    crop_origin_xy: tuple[float, float],
    image_size: int,
    square_frame: SquareFrame | None = None,
    config: VertexRefinerTargetConfig | None = None,
) -> VertexRefinerTargets:
    """Render VertexRefinerV1 dense targets for all GT vertices inside a crop."""
    cfg = config or VertexRefinerTargetConfig()
    vertices = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int8)
    frame = square_frame or infer_square_frame(vertices, edges, assignments, image_size)
    crop_size = int(cfg.crop_size)
    origin_x, origin_y = crop_origin_xy

    heatmap = np.zeros((crop_size, crop_size), dtype=np.float32)
    boundary_heatmap = np.zeros((crop_size, crop_size), dtype=np.float32)
    offset = np.zeros((crop_size, crop_size, 2), dtype=np.float32)
    offset_mask = np.zeros((crop_size, crop_size), dtype=bool)
    kind = np.zeros((crop_size, crop_size), dtype=np.int64)
    kind_mask = np.zeros((crop_size, crop_size), dtype=bool)
    boundary_side = np.zeros((crop_size, crop_size), dtype=np.int64)
    boundary_side_mask = np.zeros((crop_size, crop_size), dtype=bool)
    degree = np.zeros((crop_size, crop_size), dtype=np.int64)
    degree_mask = np.zeros((crop_size, crop_size), dtype=bool)
    rays = np.zeros((INCIDENT_RAY_BINS, crop_size, crop_size), dtype=np.float32)
    ray_mask = np.zeros((crop_size, crop_size), dtype=bool)
    best_distance = np.full((crop_size, crop_size), np.inf, dtype=np.float32)

    degrees = vertex_degrees(len(vertices), edges)
    local_vertices: list[tuple[float, float]] = []
    vertex_indices: list[int] = []
    kind_counts = {name: 0 for name in VERTEX_KIND_NAMES}
    close_pair_count = 0

    for vertex_index, vertex in enumerate(vertices):
        local_x = float(vertex[0]) - float(origin_x)
        local_y = float(vertex[1]) - float(origin_y)
        if not (0.0 <= local_x < crop_size and 0.0 <= local_y < crop_size):
            continue
        local_vertices.append((local_x, local_y))
        vertex_indices.append(vertex_index)
        _draw_gaussian(heatmap, local_x, local_y, sigma=cfg.heatmap_sigma_px)
        col = int(round(local_x))
        row = int(round(local_y))
        if 0 <= row < crop_size and 0 <= col < crop_size:
            heatmap[row, col] = 1.0

        kind_name = classify_vertex_kind(
            vertex_index,
            vertices,
            edges,
            assignments,
            frame,
            image_size=image_size,
            boundary_tolerance_px=cfg.boundary_tolerance_px,
        )
        kind_id = VERTEX_KIND_IDS[kind_name]
        kind_counts[kind_name] += 1
        boundary_info = boundary_position(
            vertex,
            frame,
            image_size=image_size,
            tolerance_px=cfg.boundary_tolerance_px,
        )
        boundary_side_id = None
        if kind_name == "boundary_contact" and boundary_info is not None:
            _draw_gaussian(boundary_heatmap, local_x, local_y, sigma=cfg.heatmap_sigma_px)
            if 0 <= row < crop_size and 0 <= col < crop_size:
                boundary_heatmap[row, col] = 1.0
            boundary_side_id = BOUNDARY_SIDE_IDS[str(boundary_info["side"])]
        degree_id = degree_class_for_count(int(degrees[vertex_index]))
        ray_bins = incident_ray_bins_for_vertex(vertex_index, vertices, edges)

        radius = int(math.ceil(max(cfg.offset_radius_px, cfg.class_radius_px)))
        col_min = max(0, int(math.floor(local_x)) - radius)
        col_max = min(crop_size - 1, int(math.ceil(local_x)) + radius)
        row_min = max(0, int(math.floor(local_y)) - radius)
        row_max = min(crop_size - 1, int(math.ceil(local_y)) + radius)
        cols = np.arange(col_min, col_max + 1, dtype=np.float32)
        rows = np.arange(row_min, row_max + 1, dtype=np.float32)
        grid_cols, grid_rows = np.meshgrid(cols, rows)
        dx = local_x - grid_cols
        dy = local_y - grid_rows
        dist = np.sqrt(dx * dx + dy * dy)
        window_best = best_distance[row_min : row_max + 1, col_min : col_max + 1]
        close_enough = dist <= max(cfg.offset_radius_px, cfg.class_radius_px)
        closer = close_enough & (dist < window_best)
        if not np.any(closer):
            continue
        window_best[closer] = dist[closer]
        offset_window = offset[row_min : row_max + 1, col_min : col_max + 1]
        offset_window[closer, 0] = dx[closer]
        offset_window[closer, 1] = dy[closer]
        mask_window = offset_mask[row_min : row_max + 1, col_min : col_max + 1]
        class_window = kind_mask[row_min : row_max + 1, col_min : col_max + 1]
        degree_window = degree_mask[row_min : row_max + 1, col_min : col_max + 1]
        boundary_side_mask_window = boundary_side_mask[row_min : row_max + 1, col_min : col_max + 1]
        ray_mask_window = ray_mask[row_min : row_max + 1, col_min : col_max + 1]
        offset_region = dist <= cfg.offset_radius_px
        class_region = dist <= cfg.class_radius_px
        offset_closer = closer & offset_region
        class_closer = closer & class_region
        mask_window[offset_closer] = True
        class_window[class_closer] = True
        degree_window[class_closer] = True
        ray_mask_window[class_closer] = True
        kind[row_min : row_max + 1, col_min : col_max + 1][class_closer] = kind_id
        degree[row_min : row_max + 1, col_min : col_max + 1][class_closer] = degree_id
        if boundary_side_id is not None:
            boundary_side_mask_window[class_closer] = True
            boundary_side[row_min : row_max + 1, col_min : col_max + 1][
                class_closer
            ] = boundary_side_id
        class_rows, class_cols = np.where(class_closer)
        absolute_rows = row_min + class_rows
        absolute_cols = col_min + class_cols
        rays[:, absolute_rows, absolute_cols] = 0.0
        for bin_index in ray_bins:
            rays[bin_index, absolute_rows, absolute_cols] = 1.0

    if len(local_vertices) >= 2:
        local_array = np.asarray(local_vertices, dtype=np.float32)
        distances = np.linalg.norm(local_array[:, None, :] - local_array[None, :, :], axis=2)
        distances[distances <= 1e-6] = np.inf
        close_pair_count = int(np.sum(distances < 8.0) // 2)

    return VertexRefinerTargets(
        vertex_heatmap=np.clip(heatmap, 0.0, 1.0).astype(np.float32),
        vertex_offset=offset.astype(np.float32),
        vertex_offset_mask=offset_mask,
        vertex_kind=kind,
        vertex_kind_mask=kind_mask,
        degree=degree,
        degree_mask=degree_mask,
        incident_rays=rays,
        incident_ray_mask=ray_mask,
        boundary_contact_heatmap=np.clip(boundary_heatmap, 0.0, 1.0).astype(np.float32),
        boundary_side=boundary_side,
        boundary_side_mask=boundary_side_mask,
        local_vertices=np.asarray(local_vertices, dtype=np.float32).reshape(-1, 2),
        vertex_indices=np.asarray(vertex_indices, dtype=np.int64),
        metadata={
            "vertex_count": len(vertex_indices),
            "kind_counts": {key: value for key, value in kind_counts.items() if value > 0},
            "close_pair_count": close_pair_count,
            "degree_class_count": len(DEGREE_CLASS_NAMES),
        },
    )


def _draw_gaussian(heatmap: np.ndarray, x: float, y: float, *, sigma: float) -> None:
    radius = max(2, int(round(3.0 * float(sigma))))
    height, width = heatmap.shape
    left = max(0, int(round(float(x))) - radius)
    right = min(width - 1, int(round(float(x))) + radius)
    top = max(0, int(round(float(y))) - radius)
    bottom = min(height - 1, int(round(float(y))) + radius)
    if left > right or top > bottom:
        return
    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    patch = np.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * sigma * sigma))
    heatmap[top : bottom + 1, left : right + 1] = np.maximum(
        heatmap[top : bottom + 1, left : right + 1],
        patch.astype(np.float32),
    )
