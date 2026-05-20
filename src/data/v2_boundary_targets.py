"""Boundary-contact targets for square-aware CPLineNet V2 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

V2_VERTEX_TYPE_IDS = {
    "background": 0,
    "corner": 1,
    "boundary_contact": 2,
    "interior_intersection": 3,
}
V2_VERTEX_TYPE_NAMES = {value: key for key, value in V2_VERTEX_TYPE_IDS.items()}
V2_BOUNDARY_SIDE_IDS = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}
V2_BOUNDARY_SIDE_NAMES = {value: key for key, value in V2_BOUNDARY_SIDE_IDS.items()}


@dataclass(frozen=True)
class V2BoundaryTargets:
    contact_heatmap: np.ndarray
    vertex_type: np.ndarray
    side: np.ndarray
    offset: np.ndarray
    mask: np.ndarray
    side_coord: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Frame:
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


def build_v2_boundary_targets(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image_size: int,
) -> V2BoundaryTargets:
    """Build square-boundary contact targets from rendered pixel geometry."""
    vertices = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int8)
    frame = _infer_frame(vertices, edges, assignments, image_size)
    incident = _incident_assignments(len(vertices), edges, assignments)

    contact_heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    vertex_type = np.zeros((image_size, image_size), dtype=np.uint8)
    side = np.full((image_size, image_size), -100, dtype=np.int16)
    offset = np.zeros((image_size, image_size, 2), dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=bool)
    side_coord = np.zeros((image_size, image_size), dtype=np.float32)

    radius = max(1, int(round(2.5 * image_size / 768)))
    sigma = max(1.0, 2.0 * image_size / 768)
    counts = {"corner": 0, "boundary_contact": 0, "interior_intersection": 0}
    side_counts = {name: 0 for name in V2_BOUNDARY_SIDE_IDS}
    contact_coords: list[dict[str, Any]] = []

    for vertex_idx, vertex in enumerate(vertices):
        boundary = _boundary_position(vertex, frame, image_size=image_size)
        has_crease = any(int(assignment) != 2 for assignment in incident[vertex_idx])
        vertex_kind = "interior_intersection"
        if boundary is not None:
            vertex_kind = "corner" if boundary["is_corner"] else "boundary_contact"
            if vertex_kind == "boundary_contact" and not has_crease:
                continue

        col = int(round(float(vertex[0])))
        row = int(round(float(vertex[1])))
        if not (0 <= row < image_size and 0 <= col < image_size):
            continue

        cv2.circle(vertex_type, (col, row), radius, V2_VERTEX_TYPE_IDS[vertex_kind], -1, cv2.LINE_8)
        counts[vertex_kind] += 1

        if vertex_kind != "boundary_contact" or boundary is None:
            continue
        _draw_gaussian(contact_heatmap, vertex[0], vertex[1], sigma=sigma)
        side_id = V2_BOUNDARY_SIDE_IDS[str(boundary["side"])]
        side[row, col] = side_id
        offset[row, col, 0] = float(vertex[0]) - col
        offset[row, col, 1] = float(vertex[1]) - row
        mask[row, col] = True
        side_coord[row, col] = float(boundary["coordinate"])
        side_counts[str(boundary["side"])] += 1
        contact_coords.append(
            {
                "vertex_id": vertex_idx,
                "side": str(boundary["side"]),
                "coordinate": round(float(boundary["coordinate"]), 6),
                "x": round(float(vertex[0]), 3),
                "y": round(float(vertex[1]), 3),
            }
        )

    contact_heatmap = np.clip(contact_heatmap, 0.0, 1.0)
    metadata = {
        "frame": {
            "x_min": frame.x_min,
            "y_min": frame.y_min,
            "x_max": frame.x_max,
            "y_max": frame.y_max,
        },
        "counts": counts,
        "side_counts": {key: value for key, value in side_counts.items() if value > 0},
        "contacts": contact_coords,
    }
    return V2BoundaryTargets(
        contact_heatmap=contact_heatmap,
        vertex_type=vertex_type,
        side=side,
        offset=offset,
        mask=mask,
        side_coord=side_coord,
        metadata=metadata,
    )


def _infer_frame(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image_size: int,
) -> _Frame:
    border_edge_indices = np.flatnonzero(assignments == 2)
    if len(border_edge_indices) == 0:
        return _Frame(0.0, 0.0, float(image_size - 1), float(image_size - 1))
    border_vertex_indices = sorted(
        {int(vertex_idx) for edge_idx in border_edge_indices for vertex_idx in edges[int(edge_idx)]}
    )
    border_vertices = vertices[border_vertex_indices]
    return _Frame(
        x_min=float(np.min(border_vertices[:, 0])),
        y_min=float(np.min(border_vertices[:, 1])),
        x_max=float(np.max(border_vertices[:, 0])),
        y_max=float(np.max(border_vertices[:, 1])),
    )


def _incident_assignments(
    vertex_count: int,
    edges: np.ndarray,
    assignments: np.ndarray,
) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for edge, assignment in zip(edges, assignments):
        incident[int(edge[0])].append(int(assignment))
        incident[int(edge[1])].append(int(assignment))
    return incident


def _boundary_position(
    vertex: np.ndarray,
    frame: _Frame,
    *,
    image_size: int,
) -> dict[str, Any] | None:
    tolerance = max(2.5, 4.0 * image_size / 1024)
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


def _draw_gaussian(heatmap: np.ndarray, x: float, y: float, *, sigma: float) -> None:
    radius = max(2, int(round(3.0 * sigma)))
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
