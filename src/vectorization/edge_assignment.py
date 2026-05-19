"""Stage 4 edge assignment attribution for vectorized crease graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.vectorization.planar_graph_builder import PlanarGraphResult

ASSIGNMENT_NAMES = {0: "M", 1: "V", 2: "B", 3: "U"}
ASSIGNMENT_TO_INT = {value: key for key, value in ASSIGNMENT_NAMES.items()}


@dataclass(frozen=True)
class EdgeAssignmentConfig:
    """Sampling and confidence policy for Stage 4 edge assignments."""

    sample_step_px: float = 1.0
    sample_width_px: int = 3
    endpoint_trim_fraction: float = 0.10
    min_samples_after_trim: int = 4
    observed_min_confidence: float = 0.60
    observed_min_margin: float = 0.12
    line_weight_floor: float = 0.05


@dataclass
class EdgeAssignmentResult:
    """Per-edge assignment attribution sampled from dense prediction fields."""

    assignments: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    source: list[str]
    probabilities: np.ndarray
    edge_support: np.ndarray
    sample_count: np.ndarray

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "assignment": [ASSIGNMENT_NAMES[int(value)] for value in self.assignments],
            "assignment_confidence": self.confidence.tolist(),
            "assignment_margin": self.margin.tolist(),
            "assignment_source": list(self.source),
            "edge_support": self.edge_support.tolist(),
            "sample_count": self.sample_count.tolist(),
        }


@dataclass
class AttributedPlanarGraph:
    """Planar graph plus Stage 4 assignment confidence/source metadata."""

    vertices_coords: np.ndarray
    edges_vertices: np.ndarray
    edges_assignment: np.ndarray
    edge_support: np.ndarray
    vertex_support: np.ndarray
    pixel_vertices: np.ndarray
    assignment_confidence: np.ndarray
    assignment_margin: np.ndarray
    assignment_source: list[str]
    assignment_probabilities: np.ndarray | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_planar_result(
        cls,
        result: PlanarGraphResult,
        assignment_result: EdgeAssignmentResult | None = None,
    ) -> AttributedPlanarGraph:
        if assignment_result is None:
            assignment_result = assignments_from_existing_result(result)
        return cls(
            vertices_coords=np.asarray(result.vertices_coords, dtype=np.float32).copy(),
            edges_vertices=np.asarray(result.edges_vertices, dtype=np.int64).copy(),
            edges_assignment=np.asarray(assignment_result.assignments, dtype=np.int8).copy(),
            edge_support=np.asarray(assignment_result.edge_support, dtype=np.float32).copy(),
            vertex_support=np.asarray(result.vertex_support, dtype=np.float32).copy(),
            pixel_vertices=np.asarray(result.pixel_vertices, dtype=np.float32).copy(),
            assignment_confidence=np.asarray(assignment_result.confidence, dtype=np.float32).copy(),
            assignment_margin=np.asarray(assignment_result.margin, dtype=np.float32).copy(),
            assignment_source=list(assignment_result.source),
            assignment_probabilities=(
                None
                if assignment_result.probabilities is None
                else np.asarray(assignment_result.probabilities, dtype=np.float32).copy()
            ),
            debug=dict(result.debug),
        )

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices_coords))

    @property
    def num_edges(self) -> int:
        return int(len(self.edges_vertices))

    def to_planar_result(self) -> PlanarGraphResult:
        return PlanarGraphResult(
            vertices_coords=np.asarray(self.vertices_coords, dtype=np.float32).copy(),
            edges_vertices=np.asarray(self.edges_vertices, dtype=np.int64).copy(),
            edges_assignment=np.asarray(self.edges_assignment, dtype=np.int8).copy(),
            edge_support=np.asarray(self.edge_support, dtype=np.float32).copy(),
            vertex_support=np.asarray(self.vertex_support, dtype=np.float32).copy(),
            pixel_vertices=np.asarray(self.pixel_vertices, dtype=np.float32).copy(),
            debug=dict(self.debug),
        )


def assignments_from_existing_result(result: PlanarGraphResult) -> EdgeAssignmentResult:
    """Create conservative attribution metadata from builder-voted labels."""
    assignments = np.asarray(result.edges_assignment, dtype=np.int8).copy()
    edge_support = np.asarray(result.edge_support, dtype=np.float32).copy()
    confidence = np.clip(edge_support, 0.0, 1.0).astype(np.float32)
    margin = confidence.copy()
    source = ["unknown" if int(value) == 3 else "observed" for value in assignments]
    probabilities = np.zeros((len(assignments), 4), dtype=np.float32)
    for idx, assignment in enumerate(assignments):
        probabilities[idx, int(assignment)] = confidence[idx]
        probabilities[idx, 3] = max(probabilities[idx, 3], 1.0 - confidence[idx])
    return EdgeAssignmentResult(
        assignments=assignments,
        confidence=confidence,
        margin=margin,
        source=source,
        probabilities=probabilities,
        edge_support=edge_support,
        sample_count=np.zeros(len(assignments), dtype=np.int32),
    )


def assign_edges_from_logits(
    result: PlanarGraphResult,
    assignment_logits: Any,
    *,
    line_prob: Any | None = None,
    config: EdgeAssignmentConfig | None = None,
) -> EdgeAssignmentResult:
    """Sample dense CPLineNet assignment logits along vectorized edges."""
    cfg = config or EdgeAssignmentConfig()
    probabilities = _coerce_probabilities(assignment_logits)
    line_weights = None if line_prob is None else np.asarray(_to_numpy(line_prob), dtype=np.float32)
    if line_weights is not None and line_weights.ndim == 3:
        line_weights = np.squeeze(line_weights)
    if line_weights is not None and line_weights.shape != probabilities.shape[1:]:
        raise ValueError("line_prob must have the same height/width as assignment logits")

    assignments: list[int] = []
    confidence: list[float] = []
    margin: list[float] = []
    sources: list[str] = []
    edge_probabilities: list[np.ndarray] = []
    support: list[float] = []
    sample_counts: list[int] = []

    vertices = np.asarray(result.pixel_vertices, dtype=np.float32)
    for edge_idx, edge in enumerate(np.asarray(result.edges_vertices, dtype=np.int64)):
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        points = _trim_endpoint_samples(
            _sample_segment_points(p0, p1, cfg.sample_step_px),
            trim_fraction=cfg.endpoint_trim_fraction,
            min_samples=cfg.min_samples_after_trim,
        )
        pooled, edge_support, count = _pool_assignment_probabilities(
            probabilities,
            points,
            p0=p0,
            p1=p1,
            line_prob=line_weights,
            sample_width_px=cfg.sample_width_px,
            line_weight_floor=cfg.line_weight_floor,
        )
        if count == 0:
            assignments.append(3)
            confidence.append(0.0)
            margin.append(0.0)
            sources.append("unknown")
            edge_probabilities.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            support.append(float(result.edge_support[edge_idx]) if len(result.edge_support) else 0.0)
            sample_counts.append(0)
            continue

        order = np.argsort(pooled)[::-1]
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        top_confidence = float(pooled[top])
        top_margin = float(pooled[top] - pooled[second]) if top != second else top_confidence
        observed = (
            top_confidence >= cfg.observed_min_confidence
            and top_margin >= cfg.observed_min_margin
        )
        if observed:
            assignment = top
            source = "observed"
        else:
            assignment = 3
            source = "unknown"

        assignments.append(assignment)
        confidence.append(top_confidence)
        margin.append(max(top_margin, 0.0))
        sources.append(source)
        edge_probabilities.append(pooled.astype(np.float32))
        support.append(edge_support)
        sample_counts.append(count)

    return EdgeAssignmentResult(
        assignments=np.asarray(assignments, dtype=np.int8),
        confidence=np.asarray(confidence, dtype=np.float32),
        margin=np.asarray(margin, dtype=np.float32),
        source=sources,
        probabilities=np.asarray(edge_probabilities, dtype=np.float32),
        edge_support=np.asarray(support, dtype=np.float32),
        sample_count=np.asarray(sample_counts, dtype=np.int32),
    )


def attribute_graph_from_logits(
    result: PlanarGraphResult,
    assignment_logits: Any,
    *,
    line_prob: Any | None = None,
    config: EdgeAssignmentConfig | None = None,
) -> AttributedPlanarGraph:
    assignment_result = assign_edges_from_logits(
        result,
        assignment_logits,
        line_prob=line_prob,
        config=config,
    )
    return AttributedPlanarGraph.from_planar_result(result, assignment_result)


def _coerce_probabilities(assignment_logits: Any) -> np.ndarray:
    array = np.asarray(_to_numpy(assignment_logits), dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Only batch-size-one assignment logits can be sampled directly")
        array = array[0]
    if array.ndim != 3:
        raise ValueError("assignment_logits must have shape (4,H,W), (H,W,4), or (1,4,H,W)")
    if array.shape[0] == 4:
        channel_first = array
    elif array.shape[-1] == 4:
        channel_first = np.moveaxis(array, -1, 0)
    else:
        raise ValueError("assignment_logits must include four M/V/B/U channels")

    sums = np.sum(channel_first, axis=0)
    if (
        np.all(channel_first >= 0.0)
        and np.all(channel_first <= 1.0)
        and np.nanmax(np.abs(sums - 1.0)) < 1e-3
    ):
        return channel_first.astype(np.float32)
    shifted = channel_first - np.max(channel_first, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-8)).astype(np.float32)


def _pool_assignment_probabilities(
    probabilities: np.ndarray,
    points: np.ndarray,
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    line_prob: np.ndarray | None,
    sample_width_px: int,
    line_weight_floor: float,
) -> tuple[np.ndarray, float, int]:
    h, w = probabilities.shape[1:]
    if len(points) == 0:
        return np.zeros(4, dtype=np.float32), 0.0, 0
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return np.zeros(4, dtype=np.float32), 0.0, 0
    perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
    half_width = max(0, int(sample_width_px) // 2)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)

    weighted = np.zeros(4, dtype=np.float64)
    total_weight = 0.0
    support_values: list[float] = []
    count = 0
    for point in points:
        for offset in offsets:
            sample = point + offset * perp
            x = int(round(float(sample[0])))
            y = int(round(float(sample[1])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            line_weight = 1.0 if line_prob is None else float(np.clip(line_prob[y, x], 0.0, 1.0))
            weight = max(line_weight, line_weight_floor)
            weighted += probabilities[:, y, x] * weight
            total_weight += weight
            support_values.append(line_weight)
            count += 1
    if total_weight <= 0.0:
        return np.zeros(4, dtype=np.float32), 0.0, 0
    pooled = weighted / total_weight
    pooled = pooled / max(float(np.sum(pooled)), 1e-8)
    support = float(np.mean(support_values)) if support_values else 0.0
    return pooled.astype(np.float32), support, count


def _sample_segment_points(p0: np.ndarray, p1: np.ndarray, step_px: float) -> np.ndarray:
    distance = float(np.linalg.norm(p1 - p0))
    if distance <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    steps = max(int(np.ceil(distance / max(step_px, 1e-6))) + 1, 2)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return (p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]).astype(np.float32)


def _trim_endpoint_samples(
    points: np.ndarray,
    *,
    trim_fraction: float,
    min_samples: int,
) -> np.ndarray:
    if len(points) <= min_samples:
        return points
    trim = int(np.floor(len(points) * max(0.0, trim_fraction)))
    max_trim = max(0, (len(points) - min_samples) // 2)
    trim = min(trim, max_trim)
    if trim <= 0:
        return points
    return points[trim:-trim]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
