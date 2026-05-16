"""Deterministic baseline for turning dense label evidence into a planar graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VectorizerEvidence:
    """Dense evidence consumed by the deterministic vectorizer baseline."""

    line_prob: np.ndarray
    angle: np.ndarray | None = None
    junction_heatmap: np.ndarray | None = None
    assignment_labels: np.ndarray | None = None


@dataclass(frozen=True)
class PlanarGraphBuilderConfig:
    """Tunable geometry thresholds for the deterministic baseline."""

    image_size: int = 1024
    line_threshold: float = 0.5
    mask_dilate_iterations: int = 0
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 720.0
    hough_threshold: int = 24
    hough_min_line_length: int = 12
    hough_max_line_gap: int = 4
    max_hough_segments: int = 12000
    line_angle_merge_degrees: float = 2.5
    line_rho_merge_px: float = 3.0
    max_line_hypotheses: int = 3000
    max_intersection_lines: int = 250
    add_intersection_vertices: bool = False
    junction_threshold: float = 0.18
    junction_nms_radius: int = 2
    junction_snap_px: float = 4.5
    vertex_merge_px: float = 1.5
    line_vertex_distance_px: float = 3.5
    min_edge_length_px: float = 3.0
    edge_sample_step_px: float = 1.0
    edge_sample_width_px: int = 3
    min_edge_support: float = 0.58
    direct_edge_fallback: bool = True
    direct_edge_max_length_px: float = 96.0
    direct_edge_min_support: float = 0.9
    direct_edge_max_vertices: int = 800
    direct_edge_vertex_distance_px: float = 2.5
    assignment_min_confidence: float = 0.75
    planar_cleanup: bool = True
    planar_cleanup_max_edges: int = 3000
    planar_split_vertex_distance_px: float = 2.0
    planar_crossing_support_tie: float = 1e-4


@dataclass
class LineHypothesis:
    """Merged straight line support extracted from raster evidence."""

    p0: np.ndarray
    p1: np.ndarray
    theta: float
    rho: float
    support: float
    votes: int = 1


@dataclass
class PlanarGraphResult:
    """Vectorized planar graph plus debug intermediates."""

    vertices_coords: np.ndarray
    edges_vertices: np.ndarray
    edges_assignment: np.ndarray
    edge_support: np.ndarray
    vertex_support: np.ndarray
    pixel_vertices: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices_coords))

    @property
    def num_edges(self) -> int:
        return int(len(self.edges_vertices))


class PlanarGraphBuilder:
    """Build a graph from clean dense labels using deterministic geometry."""

    def __init__(self, config: PlanarGraphBuilderConfig | None = None):
        self.config = config or PlanarGraphBuilderConfig()

    def build(self, evidence: VectorizerEvidence) -> PlanarGraphResult:
        line_prob = np.asarray(evidence.line_prob, dtype=np.float32)
        if line_prob.ndim != 2:
            raise ValueError("line_prob must be a 2D array")

        mask = self._line_mask(line_prob)
        raw_segments = self._hough_segments(mask)
        lines = self._merge_segments(raw_segments)
        peaks = self._junction_peaks(evidence.junction_heatmap, mask)
        vertices = self._snap_vertices_to_intersections(peaks, lines, mask)
        vertices = self._merge_vertices(vertices)

        edges, edge_support, edge_assignments = self._build_edges(
            vertices,
            lines,
            line_prob,
            evidence.assignment_labels,
        )
        if self.config.direct_edge_fallback:
            edges, edge_support, edge_assignments = self._add_direct_edges(
                vertices,
                edges,
                edge_support,
                edge_assignments,
                line_prob,
                evidence.assignment_labels,
            )

        cleanup_stats: dict[str, int | bool] = {}
        if self.config.planar_cleanup:
            edges, edge_support, edge_assignments, cleanup_stats = self._planar_cleanup(
                vertices,
                edges,
                edge_support,
                edge_assignments,
            )

        vertices, edges, vertex_support = self._drop_unused_vertices(vertices, edges, lines)
        vertices_coords = self._canonicalize_vertices(vertices)

        return PlanarGraphResult(
            vertices_coords=vertices_coords.astype(np.float32),
            edges_vertices=edges.astype(np.int64),
            edges_assignment=edge_assignments.astype(np.int8),
            edge_support=edge_support.astype(np.float32),
            vertex_support=vertex_support.astype(np.float32),
            pixel_vertices=vertices.astype(np.float32),
            debug={
                "mask": mask,
                "raw_segments": raw_segments,
                "lines": lines,
                "junction_peaks": peaks,
                "planar_cleanup": cleanup_stats,
            },
        )

    def _line_mask(self, line_prob: np.ndarray) -> np.ndarray:
        mask = (line_prob >= self.config.line_threshold).astype(np.uint8) * 255
        if self.config.mask_dilate_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.config.mask_dilate_iterations)
        return mask

    def _hough_segments(self, mask: np.ndarray) -> np.ndarray:
        segments = cv2.HoughLinesP(
            mask,
            rho=self.config.hough_rho,
            theta=self.config.hough_theta,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )
        if segments is None:
            return np.empty((0, 4), dtype=np.float32)
        segments = segments.reshape(-1, 4).astype(np.float32)
        if len(segments) > self.config.max_hough_segments:
            lengths = np.linalg.norm(segments[:, 2:4] - segments[:, 0:2], axis=1)
            keep = np.argsort(lengths)[-self.config.max_hough_segments :]
            segments = segments[keep]
        return segments

    def _merge_segments(self, segments: np.ndarray) -> list[LineHypothesis]:
        if len(segments) == 0:
            return []

        groups: list[list[tuple[np.ndarray, np.ndarray, float, float, float]]] = []
        angle_tol = np.deg2rad(self.config.line_angle_merge_degrees)

        for x1, y1, x2, y2 in segments:
            p0 = np.array([x1, y1], dtype=np.float64)
            p1 = np.array([x2, y2], dtype=np.float64)
            direction = p1 - p0
            length = float(np.linalg.norm(direction))
            if length < self.config.min_edge_length_px:
                continue
            theta = float(np.arctan2(direction[1], direction[0]) % np.pi)
            normal = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
            rho = float(np.dot(normal, p0))

            matched = False
            for group in groups:
                _, _, group_theta, group_rho, _ = group[0]
                if (
                    _angle_delta(theta, group_theta) <= angle_tol
                    and abs(rho - group_rho) <= self.config.line_rho_merge_px
                ):
                    group.append((p0, p1, theta, rho, length))
                    matched = True
                    break
            if not matched:
                groups.append([(p0, p1, theta, rho, length)])

        lines: list[LineHypothesis] = []
        for group in groups:
            weights = np.array([item[4] for item in group], dtype=np.float64)
            theta = _weighted_bidirectional_angle(np.array([item[2] for item in group]), weights)
            rho = float(np.average([item[3] for item in group], weights=weights))
            direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            endpoints = np.array([point for item in group for point in item[:2]], dtype=np.float64)
            ts = endpoints @ direction
            center = direction * float((ts.min() + ts.max()) / 2.0)
            normal = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
            signed_center = center + normal * (rho - float(np.dot(normal, center)))
            p0 = signed_center + direction * float(ts.min() - np.dot(direction, signed_center))
            p1 = signed_center + direction * float(ts.max() - np.dot(direction, signed_center))
            lines.append(
                LineHypothesis(
                    p0=p0.astype(np.float32),
                    p1=p1.astype(np.float32),
                    theta=theta,
                    rho=rho,
                    support=float(weights.sum()),
                    votes=len(group),
                )
            )

        lines.sort(key=lambda line: line.support, reverse=True)
        return lines[: self.config.max_line_hypotheses]

    def _junction_peaks(self, heatmap: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
        if heatmap is None:
            return self._fallback_line_endpoints(mask)
        heatmap = np.asarray(heatmap, dtype=np.float32)
        radius = max(1, int(self.config.junction_nms_radius))
        local_max = maximum_filter(heatmap, size=2 * radius + 1, mode="nearest")
        peaks = (heatmap >= self.config.junction_threshold) & (heatmap == local_max)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return self._fallback_line_endpoints(mask)
        scores = heatmap[ys, xs]
        order = np.argsort(scores)[::-1]
        coords = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
        return coords

    def _fallback_line_endpoints(self, mask: np.ndarray) -> np.ndarray:
        segments = self._hough_segments(mask)
        if len(segments) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return segments[:, [0, 1, 2, 3]].reshape(-1, 2)

    def _snap_vertices_to_intersections(
        self,
        peaks: np.ndarray,
        lines: list[LineHypothesis],
        mask: np.ndarray,
    ) -> np.ndarray:
        intersections: list[np.ndarray] = []
        h, w = mask.shape
        max_pairs = min(len(lines), self.config.max_intersection_lines)
        for i in range(max_pairs):
            for j in range(i + 1, max_pairs):
                point = _line_intersection(lines[i], lines[j])
                if point is None:
                    continue
                if not (-2 <= point[0] <= w + 1 and -2 <= point[1] <= h + 1):
                    continue
                if not (
                    _point_within_line_extent(point, lines[i], self.config.junction_snap_px)
                    and _point_within_line_extent(point, lines[j], self.config.junction_snap_px)
                ):
                    continue
                if _local_mask_support(mask, point, radius=2):
                    intersections.append(point)

        if len(peaks) == 0 and not intersections:
            return np.empty((0, 2), dtype=np.float32)
        if not intersections:
            return peaks.astype(np.float32)

        intersection_arr = np.array(intersections, dtype=np.float32)
        snapped = []
        for peak in peaks:
            distances = np.linalg.norm(intersection_arr - peak[None, :], axis=1)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= self.config.junction_snap_px:
                snapped.append(intersection_arr[best_idx])
            else:
                snapped.append(peak)
        if snapped and self.config.add_intersection_vertices:
            return np.concatenate([np.array(snapped, dtype=np.float32), intersection_arr], axis=0)
        if snapped:
            return np.array(snapped, dtype=np.float32)
        return intersection_arr

    def _merge_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if len(vertices) == 0:
            return vertices.astype(np.float32)
        remaining = list(range(len(vertices)))
        merged: list[np.ndarray] = []
        while remaining:
            seed = remaining.pop(0)
            seed_point = vertices[seed]
            group = [seed]
            keep = []
            for idx in remaining:
                if np.linalg.norm(vertices[idx] - seed_point) <= self.config.vertex_merge_px:
                    group.append(idx)
                else:
                    keep.append(idx)
            remaining = keep
            merged.append(vertices[group].mean(axis=0))
        return np.array(merged, dtype=np.float32)

    def _build_edges(
        self,
        vertices: np.ndarray,
        lines: list[LineHypothesis],
        line_prob: np.ndarray,
        assignment_labels: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_map: dict[tuple[int, int], tuple[float, int]] = {}
        if len(vertices) < 2:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int8),
            )

        for line in lines:
            direction = _line_direction(line)
            distances = np.abs(_point_line_distances(vertices, line))
            projections = vertices @ direction
            line_t0 = float(np.dot(line.p0, direction))
            line_t1 = float(np.dot(line.p1, direction))
            line_min = min(line_t0, line_t1) - self.config.junction_snap_px
            line_max = max(line_t0, line_t1) + self.config.junction_snap_px
            on_line = np.where(
                (distances <= self.config.line_vertex_distance_px)
                & (projections >= line_min)
                & (projections <= line_max)
            )[0]
            if len(on_line) < 2:
                continue
            ts = projections[on_line]
            order = np.argsort(ts)
            sorted_vertices = on_line[order]
            for v1, v2 in zip(sorted_vertices[:-1], sorted_vertices[1:]):
                p0 = vertices[v1]
                p1 = vertices[v2]
                length = float(np.linalg.norm(p1 - p0))
                if length < self.config.min_edge_length_px:
                    continue
                support = self._segment_support(p0, p1, line_prob)
                if support < self.config.min_edge_support:
                    continue
                assignment = self._vote_assignment(p0, p1, assignment_labels)
                key = (int(min(v1, v2)), int(max(v1, v2)))
                previous = edge_map.get(key)
                if previous is None or support > previous[0]:
                    edge_map[key] = (support, assignment)

        if not edge_map:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int8),
            )

        edges = np.array(list(edge_map.keys()), dtype=np.int64)
        support = np.array([value[0] for value in edge_map.values()], dtype=np.float32)
        assignments = np.array([value[1] for value in edge_map.values()], dtype=np.int8)
        return edges, support, assignments

    def _add_direct_edges(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_assignments: np.ndarray,
        line_prob: np.ndarray,
        assignment_labels: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(vertices) < 2 or len(vertices) > self.config.direct_edge_max_vertices:
            return edges, edge_support, edge_assignments

        edge_map: dict[tuple[int, int], tuple[float, int]] = {
            (int(min(a, b)), int(max(a, b))): (float(support), int(assignment))
            for (a, b), support, assignment in zip(edges, edge_support, edge_assignments)
        }
        tree = cKDTree(vertices)
        pairs = tree.query_pairs(self.config.direct_edge_max_length_px, output_type="ndarray")

        for v1, v2 in pairs:
            key = (int(min(v1, v2)), int(max(v1, v2)))
            if key in edge_map:
                continue
            p0 = vertices[v1]
            p1 = vertices[v2]
            if self._has_intermediate_vertex(vertices, int(v1), int(v2)):
                continue
            support = self._segment_support(p0, p1, line_prob)
            if support < self.config.direct_edge_min_support:
                continue
            assignment = self._vote_assignment(p0, p1, assignment_labels)
            edge_map[key] = (support, assignment)

        new_edges = np.array(list(edge_map.keys()), dtype=np.int64)
        new_support = np.array([value[0] for value in edge_map.values()], dtype=np.float32)
        new_assignments = np.array([value[1] for value in edge_map.values()], dtype=np.int8)
        return new_edges, new_support, new_assignments

    def _has_intermediate_vertex(self, vertices: np.ndarray, v1: int, v2: int) -> bool:
        p0 = vertices[v1]
        p1 = vertices[v2]
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            return True
        direction = segment / length
        rel = vertices - p0[None, :]
        projection = rel @ direction
        between = (projection > self.config.min_edge_length_px) & (
            projection < length - self.config.min_edge_length_px
        )
        if not np.any(between):
            return False
        perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        close = perp <= self.config.direct_edge_vertex_distance_px
        close[v1] = False
        close[v2] = False
        return bool(np.any(between & close))

    def _planar_cleanup(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_assignments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | bool]]:
        stats: dict[str, int | bool] = {
            "input_edges": int(len(edges)),
            "split_edges": 0,
            "duplicate_edges_removed": 0,
            "crossing_edges_removed": 0,
            "skipped": False,
        }
        if len(edges) == 0:
            return edges, edge_support, edge_assignments, stats
        if len(edges) > self.config.planar_cleanup_max_edges:
            stats["skipped"] = True
            return edges, edge_support, edge_assignments, stats

        edges, edge_support, edge_assignments, split_edges = self._split_edges_at_intermediate_vertices(
            vertices,
            edges,
            edge_support,
            edge_assignments,
        )
        stats["split_edges"] = split_edges

        deduped_edges, deduped_support, deduped_assignments = self._dedupe_edges(
            edges,
            edge_support,
            edge_assignments,
        )
        stats["duplicate_edges_removed"] = int(len(edges) - len(deduped_edges))

        edges, edge_support, edge_assignments, removed_crossings = self._remove_crossing_edges(
            vertices,
            deduped_edges,
            deduped_support,
            deduped_assignments,
        )
        stats["crossing_edges_removed"] = removed_crossings
        stats["output_edges"] = int(len(edges))
        return edges, edge_support, edge_assignments, stats

    def _split_edges_at_intermediate_vertices(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_assignments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        new_edges: list[tuple[int, int]] = []
        new_support: list[float] = []
        new_assignments: list[int] = []
        split_edges = 0

        for edge, support, assignment in zip(edges, edge_support, edge_assignments):
            sequence = self._vertices_on_segment(vertices, int(edge[0]), int(edge[1]))
            if len(sequence) > 2:
                split_edges += 1
            for v1, v2 in zip(sequence[:-1], sequence[1:]):
                if v1 == v2:
                    continue
                if np.linalg.norm(vertices[v1] - vertices[v2]) < self.config.min_edge_length_px:
                    continue
                new_edges.append((int(min(v1, v2)), int(max(v1, v2))))
                new_support.append(float(support))
                new_assignments.append(int(assignment))

        if not new_edges:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int8),
                split_edges,
            )
        return (
            np.array(new_edges, dtype=np.int64),
            np.array(new_support, dtype=np.float32),
            np.array(new_assignments, dtype=np.int8),
            split_edges,
        )

    def _vertices_on_segment(self, vertices: np.ndarray, v1: int, v2: int) -> list[int]:
        p0 = vertices[v1]
        p1 = vertices[v2]
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            return [v1, v2]
        direction = segment / length
        rel = vertices - p0[None, :]
        projection = rel @ direction
        between = (projection > self.config.min_edge_length_px) & (
            projection < length - self.config.min_edge_length_px
        )
        perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        close = perp <= self.config.planar_split_vertex_distance_px
        close[v1] = False
        close[v2] = False
        intermediate = np.where(between & close)[0]
        if len(intermediate) == 0:
            return [v1, v2]
        ordered = sorted((int(idx) for idx in intermediate), key=lambda idx: float(projection[idx]))
        return [v1, *ordered, v2]

    def _dedupe_edges(
        self,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_assignments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_map: dict[tuple[int, int], tuple[float, int]] = {}
        for edge, support, assignment in zip(edges, edge_support, edge_assignments):
            v1, v2 = int(edge[0]), int(edge[1])
            if v1 == v2:
                continue
            key = (min(v1, v2), max(v1, v2))
            previous = edge_map.get(key)
            if previous is None or float(support) > previous[0]:
                edge_map[key] = (float(support), int(assignment))

        if not edge_map:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int8),
            )
        return (
            np.array(list(edge_map.keys()), dtype=np.int64),
            np.array([value[0] for value in edge_map.values()], dtype=np.float32),
            np.array([value[1] for value in edge_map.values()], dtype=np.int8),
        )

    def _remove_crossing_edges(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_assignments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if len(edges) < 2:
            return edges, edge_support, edge_assignments, 0

        keep = np.ones(len(edges), dtype=bool)
        removed = 0
        for i, edge_a in enumerate(edges):
            if not keep[i]:
                continue
            a0, a1 = int(edge_a[0]), int(edge_a[1])
            for j in range(i + 1, len(edges)):
                if not keep[j]:
                    continue
                b0, b1 = int(edges[j][0]), int(edges[j][1])
                if len({a0, a1, b0, b1}) < 4:
                    continue
                if not _proper_segments_intersect(vertices[a0], vertices[a1], vertices[b0], vertices[b1]):
                    continue
                loser = self._crossing_loser(vertices, edges, edge_support, i, j)
                keep[loser] = False
                removed += 1
                if loser == i:
                    break

        return edges[keep], edge_support[keep], edge_assignments[keep], removed

    def _crossing_loser(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_support: np.ndarray,
        edge_a: int,
        edge_b: int,
    ) -> int:
        support_delta = float(edge_support[edge_a] - edge_support[edge_b])
        if abs(support_delta) > self.config.planar_crossing_support_tie:
            return edge_b if support_delta > 0 else edge_a

        a0, a1 = edges[edge_a]
        b0, b1 = edges[edge_b]
        length_a = float(np.linalg.norm(vertices[a0] - vertices[a1]))
        length_b = float(np.linalg.norm(vertices[b0] - vertices[b1]))
        return edge_a if length_a >= length_b else edge_b

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
        hits = 0
        total = 0
        half_width = self.config.edge_sample_width_px // 2
        for point in samples:
            sample_hit = False
            for offset in range(-half_width, half_width + 1):
                sample = point + perp * offset
                x = int(round(float(sample[0])))
                y = int(round(float(sample[1])))
                if 0 <= x < w and 0 <= y < h:
                    total += 1
                    if line_prob[y, x] >= self.config.line_threshold:
                        sample_hit = True
            if sample_hit:
                hits += 1
        return hits / max(len(samples), 1) if total > 0 else 0.0

    def _vote_assignment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        assignment_labels: np.ndarray | None,
    ) -> int:
        if assignment_labels is None:
            return 3
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
            return 3
        values, counts = np.unique(np.array(labels, dtype=np.int8), return_counts=True)
        best = int(np.argmax(counts))
        confidence = float(counts[best] / np.sum(counts))
        if confidence < self.config.assignment_min_confidence:
            return 3
        return int(values[best])

    def _drop_unused_vertices(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        lines: list[LineHypothesis],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(edges) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        used = np.unique(edges.reshape(-1))
        remap = {int(old): new for new, old in enumerate(used)}
        new_edges = np.array([[remap[int(a)], remap[int(b)]] for a, b in edges], dtype=np.int64)
        new_vertices = vertices[used]
        support = np.ones(len(new_vertices), dtype=np.float32)
        if lines:
            for idx, vertex in enumerate(new_vertices):
                support[idx] = float(min(np.min(np.abs([_point_line_distance(vertex, line) for line in lines])), 1e6))
        return new_vertices.astype(np.float32), new_edges, support

    def _canonicalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if len(vertices) == 0:
            return np.empty((0, 2), dtype=np.float32)
        min_xy = vertices.min(axis=0)
        max_xy = vertices.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        return ((vertices - min_xy) / span).clip(0.0, 1.0)


def _angle_delta(a: float, b: float) -> float:
    delta = abs(a - b) % np.pi
    return float(min(delta, np.pi - delta))


def _weighted_bidirectional_angle(angles: np.ndarray, weights: np.ndarray) -> float:
    x = float(np.sum(np.cos(2.0 * angles) * weights))
    y = float(np.sum(np.sin(2.0 * angles) * weights))
    angle = 0.5 * float(np.arctan2(y, x))
    return angle % np.pi


def _line_direction(line: LineHypothesis) -> np.ndarray:
    return np.array([np.cos(line.theta), np.sin(line.theta)], dtype=np.float32)


def _line_intersection(a: LineHypothesis, b: LineHypothesis) -> np.ndarray | None:
    p = a.p0.astype(np.float64)
    r = (a.p1 - a.p0).astype(np.float64)
    q = b.p0.astype(np.float64)
    s = (b.p1 - b.p0).astype(np.float64)
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-6:
        return None
    qp = q - p
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
    return (p + t * r).astype(np.float32)


def _point_line_distance(point: np.ndarray, line: LineHypothesis) -> float:
    direction = (line.p1 - line.p0).astype(np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return float(np.linalg.norm(point - line.p0))
    delta = point - line.p0
    return float(abs(direction[0] * delta[1] - direction[1] * delta[0]) / length)


def _point_line_distances(points: np.ndarray, line: LineHypothesis) -> np.ndarray:
    direction = (line.p1 - line.p0).astype(np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return np.linalg.norm(points - line.p0[None, :], axis=1)
    cross = direction[0] * (points[:, 1] - line.p0[1]) - direction[1] * (points[:, 0] - line.p0[0])
    return np.abs(cross) / length


def _sample_segment_points(p0: np.ndarray, p1: np.ndarray, step: float) -> np.ndarray:
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    count = max(int(np.ceil(length / max(step, 1e-3))) + 1, 2)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


def _point_within_line_extent(point: np.ndarray, line: LineHypothesis, tolerance: float) -> bool:
    direction = _line_direction(line)
    t = float(np.dot(point, direction))
    t0 = float(np.dot(line.p0, direction))
    t1 = float(np.dot(line.p1, direction))
    lo, hi = min(t0, t1), max(t0, t1)
    return lo - tolerance <= t <= hi + tolerance


def _local_mask_support(mask: np.ndarray, point: np.ndarray, radius: int) -> bool:
    h, w = mask.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _proper_segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-6

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
