"""Render Phase 2 vectorizer evidence from FOLD geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.data.fold_parser import CreasePattern, transform_coords
from src.vectorization.planar_graph_builder import VectorizerEvidence


@dataclass(frozen=True)
class RenderedVectorizerEvidence:
    evidence: VectorizerEvidence
    pixel_vertices: np.ndarray
    edges: np.ndarray
    assignments: np.ndarray


def render_vectorizer_evidence(
    cp: CreasePattern,
    image_size: int = 1024,
    padding: int = 32,
    line_width: int = 2,
    junction_sigma: float = 2.0,
    junction_radius: float = 4.0,
) -> RenderedVectorizerEvidence:
    """Render label evidence for deterministic vectorization.

    This is intentionally separate from ``GroundTruthGenerator`` because Phase 2
    needs geometry evidence for U-only real FOLD files. Existing training labels
    currently treat only M/V edges as crease evidence for junctions, which hides
    nearly all vertices in the scraped native corpus.
    """
    pixel_vertices, _ = transform_coords(cp.vertices, image_size=image_size, padding=padding)
    line_prob = np.zeros((image_size, image_size), dtype=np.float32)
    assignment_labels = np.zeros((image_size, image_size), dtype=np.uint8)
    assignment_priority = np.zeros((image_size, image_size), dtype=np.uint8)
    angle = np.zeros((image_size, image_size, 2), dtype=np.float32)

    for edge_idx, (v1_idx, v2_idx) in enumerate(cp.edges):
        p0 = pixel_vertices[v1_idx]
        p1 = pixel_vertices[v2_idx]
        assignment_label = int(cp.assignments[edge_idx]) + 1
        start = (int(round(float(p0[0]))), int(round(float(p0[1]))))
        end = (int(round(float(p1[0]))), int(round(float(p1[1]))))

        line_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.line(line_mask, start, end, 255, line_width, lineType=cv2.LINE_AA)
        line_float = line_mask.astype(np.float32) / 255.0
        line_prob = np.maximum(line_prob, line_float)
        update = (line_mask > 0) & (_assignment_priority(assignment_label) >= assignment_priority)
        assignment_labels[update] = assignment_label
        assignment_priority[update] = _assignment_priority(assignment_label)

        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        theta = float(np.arctan2(direction[1], direction[0]))
        cos2 = np.cos(2.0 * theta)
        sin2 = np.sin(2.0 * theta)
        update = line_float > np.linalg.norm(angle, axis=2)
        angle[update, 0] = cos2
        angle[update, 1] = sin2

    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    degrees = np.zeros(len(pixel_vertices), dtype=np.int32)
    for v1_idx, v2_idx in cp.edges:
        degrees[v1_idx] += 1
        degrees[v2_idx] += 1

    for vertex_idx, vertex in enumerate(pixel_vertices):
        if degrees[vertex_idx] >= 1:
            _add_gaussian(heatmap, vertex, sigma=junction_sigma, radius=junction_radius)
            _add_impulse(heatmap, vertex)

    canonical_edges, canonical_assignments = _canonicalize_metric_edges(
        pixel_vertices,
        cp.edges,
        cp.assignments,
    )

    return RenderedVectorizerEvidence(
        evidence=VectorizerEvidence(
            line_prob=line_prob,
            angle=angle,
            junction_heatmap=heatmap,
            assignment_labels=assignment_labels,
        ),
        pixel_vertices=pixel_vertices,
        edges=canonical_edges,
        assignments=canonical_assignments,
    )


def _add_gaussian(
    heatmap: np.ndarray,
    center: np.ndarray,
    sigma: float,
    radius: float,
) -> None:
    x, y = float(center[0]), float(center[1])
    h, w = heatmap.shape
    size = int(radius * 3)
    x_min = max(0, int(x) - size)
    x_max = min(w, int(x) + size + 1)
    y_min = max(0, int(y) - size)
    y_max = min(h, int(y) + size + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    xs = np.arange(x_min, x_max)
    ys = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(xs, ys)
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    heatmap[y_min:y_max, x_min:x_max] = np.maximum(
        heatmap[y_min:y_max, x_min:x_max],
        gaussian.astype(np.float32),
    )


def _add_impulse(heatmap: np.ndarray, center: np.ndarray) -> None:
    h, w = heatmap.shape
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    if 0 <= x < w and 0 <= y < h:
        heatmap[y, x] = 1.0


def _assignment_priority(label: int) -> int:
    # labels are segmentation-style: 1=M, 2=V, 3=B, 4=U.
    if label in (1, 2):
        return 3
    if label == 4:
        return 2
    if label == 3:
        return 1
    return 0


def _canonicalize_metric_edges(
    pixel_vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    vertex_distance_px: float = 2.0,
    min_edge_length_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split and dedupe raw FOLD edges for metric comparisons.

    Some scraped FOLD files include long border or crease spans that pass
    through explicit vertices already connected by shorter edges. The
    deterministic builder emits the canonical planar adjacency graph, so metrics
    use that same adjacency convention while rendering still uses the original
    geometry as evidence.
    """
    edge_map: dict[tuple[int, int], int] = {}
    for edge, assignment in zip(edges, assignments):
        sequence = _vertices_on_segment(
            pixel_vertices,
            int(edge[0]),
            int(edge[1]),
            vertex_distance_px=vertex_distance_px,
            min_edge_length_px=min_edge_length_px,
        )
        for v1, v2 in zip(sequence[:-1], sequence[1:]):
            if v1 == v2:
                continue
            if np.linalg.norm(pixel_vertices[v1] - pixel_vertices[v2]) < min_edge_length_px:
                continue
            key = (min(v1, v2), max(v1, v2))
            previous = edge_map.get(key)
            if previous is None or _metric_assignment_priority(int(assignment)) > _metric_assignment_priority(previous):
                edge_map[key] = int(assignment)

    if not edge_map:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int8)
    return (
        np.array(list(edge_map.keys()), dtype=np.int64),
        np.array(list(edge_map.values()), dtype=np.int8),
    )


def _vertices_on_segment(
    vertices: np.ndarray,
    v1: int,
    v2: int,
    vertex_distance_px: float,
    min_edge_length_px: float,
) -> list[int]:
    p0 = vertices[v1]
    p1 = vertices[v2]
    segment = p1 - p0
    length = float(np.linalg.norm(segment))
    if length <= 1e-6:
        return [v1, v2]
    direction = segment / length
    rel = vertices - p0[None, :]
    projection = rel @ direction
    between = (projection > min_edge_length_px) & (projection < length - min_edge_length_px)
    perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
    close = perp <= vertex_distance_px
    close[v1] = False
    close[v2] = False
    intermediate = np.where(between & close)[0]
    if len(intermediate) == 0:
        return [v1, v2]
    ordered = sorted((int(idx) for idx in intermediate), key=lambda idx: float(projection[idx]))
    return [v1, *ordered, v2]


def _metric_assignment_priority(assignment: int) -> int:
    if assignment in (0, 1):
        return 3
    if assignment == 3:
        return 2
    if assignment == 2:
        return 1
    return 0
