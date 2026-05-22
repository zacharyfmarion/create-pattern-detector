"""Square-aware topology decoder for rectified crease-pattern images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from src.vectorization.planar_graph_builder import (
    LineHypothesis,
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    PlanarGraphResult,
    VectorizerEvidence,
    _line_direction,
    _line_intersection,
    _point_line_distances,
    _sample_segment_points,
)


@dataclass(frozen=True)
class SquareTopologyDecoderConfig:
    """Controls the square-domain CP topology decoder.

    The decoder assumes the image has already been rectified into the unit square
    domain represented by a square raster. Border edges are deterministic frame
    geometry; only interior crease carriers are extracted from image evidence.
    """

    image_size: int = 1024
    line_threshold: float = 0.65
    hough_threshold: int = 10
    hough_min_line_length: int = 6
    hough_max_line_gap: int = 4
    line_angle_merge_degrees: float = 2.5
    line_rho_merge_px: float = 3.0
    max_line_hypotheses: int = 240
    max_intersection_lines: int = 180
    add_intersection_vertices: bool = False
    junction_snap_px: float = 4.5
    min_edge_length_px: float = 3.0
    min_edge_support: float = 0.45
    edge_sample_step_px: float = 1.0
    edge_sample_width_px: int = 3
    dashed_support_weight: float = 0.35
    carrier_extent_padding_px: float = 24.0
    frame_epsilon_px: float = 1.0
    border_carrier_tolerance_px: float = 4.0
    vertex_merge_px: float = 2.0
    line_vertex_distance_px: float = 4.0
    junction_threshold: float = 0.20
    junction_nms_radius: int = 2
    boundary_contact_threshold: float = 0.25
    boundary_contact_nms_radius_px: float = 8.0
    boundary_contact_side_band_px: float | None = None
    non_crease_suppression_threshold: float = 0.65
    non_crease_strong_line_threshold: float = 0.85
    non_crease_suppression_scale: float = 0.15
    planar_cleanup: bool = True
    planar_cleanup_max_edges: int = 2500


@dataclass(frozen=True)
class _Carrier:
    line: LineHypothesis
    p0: np.ndarray
    p1: np.ndarray
    t_min: float
    t_max: float
    direction: np.ndarray


class SquareTopologyDecoder:
    """Decode a CP graph with a hard square-boundary prior."""

    def __init__(self, config: SquareTopologyDecoderConfig | None = None) -> None:
        self.config = config or SquareTopologyDecoderConfig()
        self._builder = PlanarGraphBuilder(
            PlanarGraphBuilderConfig(
                image_size=self.config.image_size,
                line_threshold=self.config.line_threshold,
                hough_threshold=self.config.hough_threshold,
                hough_min_line_length=self.config.hough_min_line_length,
                hough_max_line_gap=self.config.hough_max_line_gap,
                line_angle_merge_degrees=self.config.line_angle_merge_degrees,
                line_rho_merge_px=self.config.line_rho_merge_px,
                max_line_hypotheses=self.config.max_line_hypotheses,
                min_edge_length_px=self.config.min_edge_length_px,
                edge_sample_step_px=self.config.edge_sample_step_px,
                edge_sample_width_px=self.config.edge_sample_width_px,
                min_edge_support=self.config.min_edge_support,
                non_crease_suppression_threshold=self.config.non_crease_suppression_threshold,
                non_crease_strong_line_threshold=self.config.non_crease_strong_line_threshold,
                non_crease_suppression_scale=self.config.non_crease_suppression_scale,
                planar_cleanup_max_edges=self.config.planar_cleanup_max_edges,
            )
        )

    def build(self, evidence: VectorizerEvidence) -> PlanarGraphResult:
        line_prob = np.asarray(evidence.line_prob, dtype=np.float32)
        if line_prob.ndim != 2:
            raise ValueError("line_prob must be a 2D array")

        effective_line_prob, suppression_stats = self._builder._effective_line_prob(
            line_prob,
            evidence,
        )
        mask = self._builder._line_mask(effective_line_prob)
        raw_segments = self._builder._hough_segments(mask)
        raw_lines = self._builder._merge_segments(raw_segments)
        carriers = self._carriers_from_lines(raw_lines)

        candidate_vertices, vertex_meta = self._candidate_vertices(evidence, carriers, mask)
        vertices = self._merge_vertices(candidate_vertices)
        vertex_meta = self._refresh_vertex_meta(vertices, vertex_meta)

        interior_edges, interior_support, interior_assignments = self._interior_edges(
            vertices,
            carriers,
            effective_line_prob,
            evidence.assignment_labels,
        )
        vertices, interior_edges, vertex_support, used_boundary = self._drop_unused_non_border_vertices(
            vertices,
            interior_edges,
            interior_support,
            vertex_meta,
        )
        interior_support = self._support_for_edges(vertices, interior_edges, effective_line_prob)
        interior_assignments = self._assignments_for_edges(
            vertices,
            interior_edges,
            evidence.assignment_labels,
            default=3,
        )

        border_edges, border_support, border_assignments = self._border_chain(
            vertices,
            used_boundary,
            effective_line_prob,
        )
        edges, edge_support, edge_assignments = self._combine_edges(
            interior_edges,
            interior_support,
            interior_assignments,
            border_edges,
            border_support,
            border_assignments,
        )
        cleanup_stats: dict[str, int | bool] = {}
        if self.config.planar_cleanup:
            edges, edge_support, edge_assignments, cleanup_stats = self._builder._planar_cleanup(
                vertices,
                edges,
                edge_support,
                edge_assignments,
                line_prob=effective_line_prob,
            )
        vertices, edges, vertex_support = self._drop_unused_vertices_keep_corners(
            vertices,
            edges,
            vertex_support,
        )

        return PlanarGraphResult(
            vertices_coords=self._canonicalize_vertices(vertices),
            edges_vertices=edges.astype(np.int64),
            edges_assignment=edge_assignments.astype(np.int8),
            edge_support=edge_support.astype(np.float32),
            vertex_support=vertex_support.astype(np.float32),
            pixel_vertices=vertices.astype(np.float32),
            debug={
                "decoder": "square_topology",
                "mask": mask,
                "raw_segments": raw_segments,
                "lines": raw_lines,
                "carriers": carriers,
                "effective_line_prob": effective_line_prob,
                "v2_evidence": suppression_stats,
                "planar_cleanup": cleanup_stats,
                "square_topology": {
                    "raw_line_count": len(raw_lines),
                    "carrier_count": len(carriers),
                    "candidate_vertex_count": len(candidate_vertices),
                    "output_vertex_count": len(vertices),
                    "interior_edge_count": len(interior_edges),
                    "border_edge_count": len(border_edges),
                },
            },
        )

    def _carriers_from_lines(self, lines: list[LineHypothesis]) -> list[_Carrier]:
        carriers: list[_Carrier] = []
        frame = self._frame()
        for line in lines[: self.config.max_line_hypotheses]:
            if self._line_is_frame_border(line):
                continue
            clipped = self._clip_line_to_frame(line, frame)
            if clipped is None:
                continue
            p0, p1, square_t_min, square_t_max = clipped
            direction = _line_direction(line)
            line_t0 = float(np.dot(line.p0, direction))
            line_t1 = float(np.dot(line.p1, direction))
            finite_min = min(line_t0, line_t1) - self.config.carrier_extent_padding_px
            finite_max = max(line_t0, line_t1) + self.config.carrier_extent_padding_px
            t_min = max(square_t_min, finite_min)
            t_max = min(square_t_max, finite_max)
            if t_max - t_min < self.config.min_edge_length_px:
                continue
            start = direction * t_min + self._line_normal(line) * line.rho
            end = direction * t_max + self._line_normal(line) * line.rho
            if self._segment_is_frame_border(start, end):
                continue
            carriers.append(
                _Carrier(
                    line=line,
                    p0=start.astype(np.float32),
                    p1=end.astype(np.float32),
                    t_min=float(t_min),
                    t_max=float(t_max),
                    direction=direction.astype(np.float32),
                )
            )
        return carriers

    def _candidate_vertices(
        self,
        evidence: VectorizerEvidence,
        carriers: list[_Carrier],
        mask: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        points: list[np.ndarray] = []
        meta: list[str] = []

        for corner in self._corners():
            points.append(corner)
            meta.append("corner")

        for point in self._boundary_contact_points(evidence):
            points.append(point)
            meta.append("boundary_contact")

        intersections = self._carrier_intersections(carriers)
        junctions = self._junction_points(evidence, mask)
        for point in junctions:
            if self._point_on_any_carrier(point, carriers):
                points.append(self._snap_junction_to_intersection(point, intersections))
                meta.append("junction")

        if self.config.add_intersection_vertices:
            for point in intersections:
                points.append(point.astype(np.float32))
                meta.append("intersection")

        for carrier in carriers:
            if self._point_on_frame(carrier.p0):
                points.append(self._snap_to_frame(carrier.p0))
                meta.append("boundary_contact")
            if self._point_on_frame(carrier.p1):
                points.append(self._snap_to_frame(carrier.p1))
                meta.append("boundary_contact")

        if not points:
            return np.empty((0, 2), dtype=np.float32), []
        return np.asarray(points, dtype=np.float32), meta

    def _carrier_intersections(self, carriers: list[_Carrier]) -> np.ndarray:
        intersections: list[np.ndarray] = []
        max_lines = min(len(carriers), self.config.max_intersection_lines)
        for i in range(max_lines):
            for j in range(i + 1, max_lines):
                point = _line_intersection(carriers[i].line, carriers[j].line)
                if point is None:
                    continue
                if not self._point_in_frame(point):
                    continue
                if not (
                    self._point_within_carrier(point, carriers[i])
                    and self._point_within_carrier(point, carriers[j])
                ):
                    continue
                intersections.append(point.astype(np.float32))
        if not intersections:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(intersections, dtype=np.float32)

    def _snap_junction_to_intersection(
        self,
        point: np.ndarray,
        intersections: np.ndarray,
    ) -> np.ndarray:
        if len(intersections) == 0:
            return point.astype(np.float32)
        distances = np.linalg.norm(intersections - point[None, :], axis=1)
        best = int(np.argmin(distances))
        if float(distances[best]) <= self.config.junction_snap_px:
            return intersections[best].astype(np.float32)
        return point.astype(np.float32)

    def _junction_points(self, evidence: VectorizerEvidence, mask: np.ndarray) -> np.ndarray:
        heatmap = evidence.junction_heatmap
        if heatmap is None:
            return np.empty((0, 2), dtype=np.float32)
        heatmap = np.asarray(heatmap, dtype=np.float32)
        radius = max(1, int(self.config.junction_nms_radius))
        local_max = maximum_filter(heatmap, size=2 * radius + 1, mode="nearest")
        peaks = (heatmap >= self.config.junction_threshold) & (heatmap == local_max)
        peaks &= mask > 0
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        order = np.argsort(heatmap[ys, xs])[::-1]
        return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)

    def _boundary_contact_points(self, evidence: VectorizerEvidence) -> list[np.ndarray]:
        heatmap = evidence.boundary_contact_heatmap
        if heatmap is None:
            return []
        heatmap = np.asarray(heatmap, dtype=np.float32)
        band = self.config.boundary_contact_side_band_px
        if band is None:
            band = max(4.0, 0.04 * float(self.config.image_size))
        radius = max(1, int(round(self.config.boundary_contact_nms_radius_px / 2.0)))
        local_max = maximum_filter(heatmap, size=2 * radius + 1, mode="nearest")
        max_coord = float(self.config.image_size - 1)
        yy, xx = np.mgrid[0 : heatmap.shape[0], 0 : heatmap.shape[1]]
        side_masks = [
            np.abs(yy) <= band,
            np.abs(xx - max_coord) <= band,
            np.abs(yy - max_coord) <= band,
            np.abs(xx) <= band,
        ]
        points: list[np.ndarray] = []
        for side, side_mask in enumerate(side_masks):
            ys, xs = np.nonzero(
                side_mask
                & (heatmap == local_max)
                & (heatmap >= self.config.boundary_contact_threshold)
            )
            if len(xs) == 0:
                continue
            scores = heatmap[ys, xs]
            order = np.argsort(scores)[::-1]
            kept: list[np.ndarray] = []
            for idx in order:
                point = np.asarray([float(xs[idx]), float(ys[idx])], dtype=np.float32)
                point = self._snap_point_to_side(point, side)
                if any(np.linalg.norm(point - other) <= self.config.boundary_contact_nms_radius_px for other in kept):
                    continue
                kept.append(point)
            points.extend(kept)
        return points

    def _merge_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if len(vertices) == 0:
            return vertices.astype(np.float32)
        remaining = np.ones(len(vertices), dtype=bool)
        merged: list[np.ndarray] = []
        for idx, point in enumerate(vertices):
            if not remaining[idx]:
                continue
            distances = np.linalg.norm(vertices - point[None, :], axis=1)
            group = np.where(remaining & (distances <= self.config.vertex_merge_px))[0]
            remaining[group] = False
            merged.append(self._snap_to_frame(vertices[group].mean(axis=0)))
        return np.asarray(merged, dtype=np.float32)

    def _refresh_vertex_meta(self, vertices: np.ndarray, original_meta: list[str]) -> list[str]:
        # After merging, infer boundary/corner classes from geometry. The original
        # labels are only used to preserve list length when there are no vertices.
        if len(vertices) == 0:
            return original_meta[:0]
        meta: list[str] = []
        for vertex in vertices:
            if self._point_is_corner(vertex):
                meta.append("corner")
            elif self._point_on_frame(vertex):
                meta.append("boundary_contact")
            else:
                meta.append("junction")
        return meta

    def _interior_edges(
        self,
        vertices: np.ndarray,
        carriers: list[_Carrier],
        line_prob: np.ndarray,
        assignment_labels: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_map: dict[tuple[int, int], tuple[float, int]] = {}
        if len(vertices) < 2:
            return self._empty_edges()

        for carrier in carriers:
            distances = np.abs(_point_line_distances(vertices, carrier.line))
            projections = vertices @ carrier.direction
            on_line = np.where(
                (distances <= self.config.line_vertex_distance_px)
                & (projections >= carrier.t_min - self.config.vertex_merge_px)
                & (projections <= carrier.t_max + self.config.vertex_merge_px)
            )[0]
            if len(on_line) < 2:
                continue
            order = np.argsort(projections[on_line])
            sorted_vertices = on_line[order]
            for v1, v2 in zip(sorted_vertices[:-1], sorted_vertices[1:]):
                p0 = vertices[int(v1)]
                p1 = vertices[int(v2)]
                if np.linalg.norm(p1 - p0) < self.config.min_edge_length_px:
                    continue
                support = self._segment_support(p0, p1, line_prob)
                if support < self.config.min_edge_support:
                    continue
                key = (int(min(v1, v2)), int(max(v1, v2)))
                assignment = self._vote_assignment(p0, p1, assignment_labels, default=3)
                previous = edge_map.get(key)
                if previous is None or support > previous[0]:
                    edge_map[key] = (support, assignment)

        if not edge_map:
            return self._empty_edges()
        edges = np.asarray(list(edge_map.keys()), dtype=np.int64)
        support = np.asarray([value[0] for value in edge_map.values()], dtype=np.float32)
        assignments = np.asarray([value[1] for value in edge_map.values()], dtype=np.int8)
        return edges, support, assignments

    def _drop_unused_non_border_vertices(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        vertex_meta: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
        if len(vertices) == 0:
            return vertices, edges, np.empty(0, dtype=np.float32), set()
        used = set(int(v) for v in edges.reshape(-1)) if len(edges) else set()
        keep = [
            idx
            for idx, kind in enumerate(vertex_meta)
            if kind == "corner" or int(idx) in used
        ]
        if not keep:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                set(),
            )
        remap = {old: new for new, old in enumerate(keep)}
        remapped_edges = np.asarray(
            [[remap[int(a)], remap[int(b)]] for a, b in edges if int(a) in remap and int(b) in remap],
            dtype=np.int64,
        ).reshape(-1, 2)
        kept_vertices = vertices[np.asarray(keep, dtype=np.int64)]
        used_boundary = {
            remap[idx]
            for idx in used
            if idx in remap and self._point_on_frame(vertices[idx])
        }
        vertex_support = np.ones(len(kept_vertices), dtype=np.float32)
        return kept_vertices.astype(np.float32), remapped_edges, vertex_support, used_boundary

    def _border_chain(
        self,
        vertices: np.ndarray,
        used_boundary: set[int],
        line_prob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(vertices) < 4:
            return self._empty_edges()
        side_vertices = self._side_vertices(vertices, used_boundary)
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for side in ("top", "right", "bottom", "left"):
            ordered = side_vertices[side]
            for v1, v2 in zip(ordered[:-1], ordered[1:]):
                if v1 == v2:
                    continue
                key = (min(v1, v2), max(v1, v2))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((v1, v2))
        if not edges:
            return self._empty_edges()
        edge_arr = np.asarray(edges, dtype=np.int64)
        support = np.asarray(
            [max(0.99, self._segment_support(vertices[a], vertices[b], line_prob)) for a, b in edge_arr],
            dtype=np.float32,
        )
        assignments = np.full(len(edge_arr), 2, dtype=np.int8)
        return edge_arr, support, assignments

    def _side_vertices(self, vertices: np.ndarray, used_boundary: set[int]) -> dict[str, list[int]]:
        side_to_vertices = {"top": [], "right": [], "bottom": [], "left": []}
        corners = self._corner_indices(vertices)
        for side, pair in {
            "top": ("top_left", "top_right"),
            "right": ("top_right", "bottom_right"),
            "bottom": ("bottom_left", "bottom_right"),
            "left": ("top_left", "bottom_left"),
        }.items():
            for name in pair:
                if name in corners:
                    side_to_vertices[side].append(corners[name])
        for idx in sorted(used_boundary):
            if idx in corners.values():
                continue
            side = self._point_side(vertices[idx])
            if side is not None:
                side_to_vertices[side].append(idx)
        for side, indices in side_to_vertices.items():
            unique = sorted(set(indices), key=lambda idx: self._side_position(vertices[idx], side))
            if side == "bottom":
                unique = sorted(set(indices), key=lambda idx: self._side_position(vertices[idx], side))
            side_to_vertices[side] = unique
        return side_to_vertices

    def _combine_edges(
        self,
        interior_edges: np.ndarray,
        interior_support: np.ndarray,
        interior_assignments: np.ndarray,
        border_edges: np.ndarray,
        border_support: np.ndarray,
        border_assignments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_map: dict[tuple[int, int], tuple[float, int]] = {}
        for edges, support, assignments in (
            (interior_edges, interior_support, interior_assignments),
            (border_edges, border_support, border_assignments),
        ):
            for edge, item_support, assignment in zip(edges, support, assignments):
                v1, v2 = int(edge[0]), int(edge[1])
                if v1 == v2:
                    continue
                key = (min(v1, v2), max(v1, v2))
                previous = edge_map.get(key)
                if previous is None or int(assignment) == 2 or float(item_support) > previous[0]:
                    edge_map[key] = (float(item_support), int(assignment))
        if not edge_map:
            return self._empty_edges()
        return (
            np.asarray(list(edge_map.keys()), dtype=np.int64),
            np.asarray([value[0] for value in edge_map.values()], dtype=np.float32),
            np.asarray([value[1] for value in edge_map.values()], dtype=np.int8),
        )

    def _drop_unused_vertices_keep_corners(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        vertex_support: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(vertices) == 0 or len(edges) == 0:
            return vertices, edges, vertex_support
        used = set(int(v) for v in edges.reshape(-1))
        keep = sorted(used)
        remap = {old: new for new, old in enumerate(keep)}
        remapped_edges = np.asarray(
            [[remap[int(a)], remap[int(b)]] for a, b in edges],
            dtype=np.int64,
        )
        return vertices[keep].astype(np.float32), remapped_edges, vertex_support[keep].astype(np.float32)

    def _support_for_edges(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        line_prob: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [self._segment_support(vertices[int(a)], vertices[int(b)], line_prob) for a, b in edges],
            dtype=np.float32,
        )

    def _assignments_for_edges(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignment_labels: np.ndarray | None,
        *,
        default: int,
    ) -> np.ndarray:
        return np.asarray(
            [
                self._vote_assignment(vertices[int(a)], vertices[int(b)], assignment_labels, default=default)
                for a, b in edges
            ],
            dtype=np.int8,
        )

    def _segment_support(self, p0: np.ndarray, p1: np.ndarray, line_prob: np.ndarray) -> float:
        samples = _sample_segment_points(p0, p1, self.config.edge_sample_step_px)
        if len(samples) == 0:
            return 0.0
        h, w = line_prob.shape
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return 0.0
        perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
        half_width = self.config.edge_sample_width_px // 2
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
        coords = samples[:, None, :] + offsets[None, :, None] * perp[None, None, :]
        xs = np.rint(coords[:, :, 0]).astype(np.int32)
        ys = np.rint(coords[:, :, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(valid):
            return 0.0
        values = np.zeros(valid.shape, dtype=np.float32)
        values[valid] = line_prob[ys[valid], xs[valid]]
        max_values = np.max(values, axis=1)
        hit_fraction = float(np.mean(max_values >= self.config.line_threshold))
        mean_prob = float(np.mean(max_values))
        return float(max(hit_fraction, self.config.dashed_support_weight * mean_prob))

    def _vote_assignment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        assignment_labels: np.ndarray | None,
        *,
        default: int,
    ) -> int:
        if assignment_labels is None:
            return default
        labels = []
        h, w = assignment_labels.shape
        points = _sample_segment_points(p0, p1, self.config.edge_sample_step_px)
        if len(points) > 6:
            trim = max(1, len(points) // 10)
            points = points[trim:-trim]
        for point in points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < w and 0 <= y < h:
                label = int(assignment_labels[y, x])
                if label > 0:
                    labels.append(label - 1)
        if not labels:
            return default
        values, counts = np.unique(np.asarray(labels, dtype=np.int8), return_counts=True)
        best = int(np.argmax(counts))
        confidence = float(counts[best] / np.sum(counts))
        if confidence < 0.75:
            return default
        return int(values[best])

    def _clip_line_to_frame(
        self,
        line: LineHypothesis,
        frame: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        left, top, right, bottom = frame
        direction = _line_direction(line)
        points: list[np.ndarray] = []
        p0 = line.p0.astype(np.float64)
        p1 = line.p1.astype(np.float64)
        d = p1 - p0
        if abs(float(d[0])) > 1e-6:
            for x in (left, right):
                t = (x - float(p0[0])) / float(d[0])
                y = float(p0[1] + t * d[1])
                if top - self.config.frame_epsilon_px <= y <= bottom + self.config.frame_epsilon_px:
                    points.append(np.asarray([x, y], dtype=np.float32))
        if abs(float(d[1])) > 1e-6:
            for y in (top, bottom):
                t = (y - float(p0[1])) / float(d[1])
                x = float(p0[0] + t * d[0])
                if left - self.config.frame_epsilon_px <= x <= right + self.config.frame_epsilon_px:
                    points.append(np.asarray([x, y], dtype=np.float32))
        if len(points) < 2:
            return None
        unique = []
        for point in points:
            snapped = self._snap_to_frame(point)
            if not any(np.linalg.norm(snapped - other) <= 1.0 for other in unique):
                unique.append(snapped)
        if len(unique) < 2:
            return None
        ts = [float(np.dot(point, direction)) for point in unique]
        order = np.argsort(ts)
        a = unique[int(order[0])]
        b = unique[int(order[-1])]
        return a, b, float(min(ts)), float(max(ts))

    def _line_is_frame_border(self, line: LineHypothesis) -> bool:
        max_coord = float(self.config.image_size - 1)
        horizontal = abs(np.sin(line.theta)) <= np.sin(np.deg2rad(3.0))
        vertical = abs(np.cos(line.theta)) <= np.sin(np.deg2rad(3.0))
        p0, p1 = line.p0, line.p1
        tol = self.config.border_carrier_tolerance_px
        if horizontal and (abs(float(p0[1])) <= tol and abs(float(p1[1])) <= tol):
            return True
        if horizontal and (abs(float(p0[1] - max_coord)) <= tol and abs(float(p1[1] - max_coord)) <= tol):
            return True
        if vertical and (abs(float(p0[0])) <= tol and abs(float(p1[0])) <= tol):
            return True
        if vertical and (abs(float(p0[0] - max_coord)) <= tol and abs(float(p1[0] - max_coord)) <= tol):
            return True
        return False

    def _segment_is_frame_border(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        max_coord = float(self.config.image_size - 1)
        tol = self.config.border_carrier_tolerance_px
        if abs(float(p0[1])) <= tol and abs(float(p1[1])) <= tol:
            return True
        if abs(float(p0[1] - max_coord)) <= tol and abs(float(p1[1] - max_coord)) <= tol:
            return True
        if abs(float(p0[0])) <= tol and abs(float(p1[0])) <= tol:
            return True
        if abs(float(p0[0] - max_coord)) <= tol and abs(float(p1[0] - max_coord)) <= tol:
            return True
        return False

    def _line_normal(self, line: LineHypothesis) -> np.ndarray:
        return np.asarray([-np.sin(line.theta), np.cos(line.theta)], dtype=np.float32)

    def _point_within_carrier(self, point: np.ndarray, carrier: _Carrier) -> bool:
        t = float(np.dot(point, carrier.direction))
        return carrier.t_min - self.config.vertex_merge_px <= t <= carrier.t_max + self.config.vertex_merge_px

    def _point_on_any_carrier(self, point: np.ndarray, carriers: list[_Carrier]) -> bool:
        for carrier in carriers:
            if (
                self._point_within_carrier(point, carrier)
                and abs(_point_line_distances(point[None, :], carrier.line)[0])
                <= self.config.line_vertex_distance_px
            ):
                return True
        return False

    def _point_in_frame(self, point: np.ndarray) -> bool:
        max_coord = float(self.config.image_size - 1)
        eps = self.config.frame_epsilon_px
        return bool(-eps <= point[0] <= max_coord + eps and -eps <= point[1] <= max_coord + eps)

    def _point_on_frame(self, point: np.ndarray) -> bool:
        max_coord = float(self.config.image_size - 1)
        tol = self.config.frame_epsilon_px + self.config.vertex_merge_px
        x, y = float(point[0]), float(point[1])
        return bool(
            abs(x) <= tol
            or abs(y) <= tol
            or abs(x - max_coord) <= tol
            or abs(y - max_coord) <= tol
        )

    def _point_side(self, point: np.ndarray) -> str | None:
        max_coord = float(self.config.image_size - 1)
        x, y = float(point[0]), float(point[1])
        distances = {
            "top": abs(y),
            "right": abs(x - max_coord),
            "bottom": abs(y - max_coord),
            "left": abs(x),
        }
        side, distance = min(distances.items(), key=lambda item: item[1])
        return side if distance <= self.config.vertex_merge_px + self.config.frame_epsilon_px else None

    def _side_position(self, point: np.ndarray, side: str) -> float:
        if side in {"top", "bottom"}:
            return float(point[0])
        return float(point[1])

    def _snap_point_to_side(self, point: np.ndarray, side: int) -> np.ndarray:
        max_coord = float(self.config.image_size - 1)
        x, y = float(point[0]), float(point[1])
        if side == 0:
            return np.asarray([np.clip(x, 0.0, max_coord), 0.0], dtype=np.float32)
        if side == 1:
            return np.asarray([max_coord, np.clip(y, 0.0, max_coord)], dtype=np.float32)
        if side == 2:
            return np.asarray([np.clip(x, 0.0, max_coord), max_coord], dtype=np.float32)
        return np.asarray([0.0, np.clip(y, 0.0, max_coord)], dtype=np.float32)

    def _snap_to_frame(self, point: np.ndarray) -> np.ndarray:
        max_coord = float(self.config.image_size - 1)
        x, y = float(point[0]), float(point[1])
        tol = self.config.vertex_merge_px + self.config.frame_epsilon_px
        if abs(x) <= tol:
            x = 0.0
        elif abs(x - max_coord) <= tol:
            x = max_coord
        if abs(y) <= tol:
            y = 0.0
        elif abs(y - max_coord) <= tol:
            y = max_coord
        return np.asarray([np.clip(x, 0.0, max_coord), np.clip(y, 0.0, max_coord)], dtype=np.float32)

    def _point_is_corner(self, point: np.ndarray) -> bool:
        return any(np.linalg.norm(point - corner) <= self.config.vertex_merge_px for corner in self._corners())

    def _corners(self) -> list[np.ndarray]:
        max_coord = float(self.config.image_size - 1)
        return [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([max_coord, 0.0], dtype=np.float32),
            np.asarray([max_coord, max_coord], dtype=np.float32),
            np.asarray([0.0, max_coord], dtype=np.float32),
        ]

    def _corner_indices(self, vertices: np.ndarray) -> dict[str, int]:
        names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        corners: dict[str, int] = {}
        for name, corner in zip(names, self._corners()):
            distances = np.linalg.norm(vertices - corner[None, :], axis=1)
            idx = int(np.argmin(distances))
            if float(distances[idx]) <= self.config.vertex_merge_px:
                corners[name] = idx
        return corners

    def _frame(self) -> tuple[float, float, float, float]:
        max_coord = float(self.config.image_size - 1)
        return (0.0, 0.0, max_coord, max_coord)

    def _canonicalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        max_coord = max(1.0, float(self.config.image_size - 1))
        return (vertices / max_coord).clip(0.0, 1.0).astype(np.float32)

    def _empty_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )
