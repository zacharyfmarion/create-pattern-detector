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

    return RenderedVectorizerEvidence(
        evidence=VectorizerEvidence(
            line_prob=line_prob,
            angle=angle,
            junction_heatmap=heatmap,
            assignment_labels=assignment_labels,
        ),
        pixel_vertices=pixel_vertices,
        edges=cp.edges,
        assignments=cp.assignments,
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


def _assignment_priority(label: int) -> int:
    # labels are segmentation-style: 1=M, 2=V, 3=B, 4=U.
    if label in (1, 2):
        return 3
    if label == 4:
        return 2
    if label == 3:
        return 1
    return 0
