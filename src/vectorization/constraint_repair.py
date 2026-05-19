"""Conservative Stage 4 repair for attributed planar crease graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.vectorization.edge_assignment import AttributedPlanarGraph


@dataclass(frozen=True)
class RepairConfig:
    """Thresholds for repairs that avoid inventing crease semantics."""

    image_size: int | None = None
    min_edge_length_px: float = 1.0
    weak_edge_support_threshold: float = 0.35
    low_assignment_confidence: float = 0.55
    low_assignment_margin: float = 0.08
    border_snap_px: float = 2.5
    border_endpoint_tolerance_px: float = 3.0
    border_completion_min_support: float = 0.88
    border_completion_width_px: int = 3
    infer_assignments: bool = False


@dataclass
class RepairAction:
    code: str
    message: str
    edge_indices: list[int] = field(default_factory=list)
    vertex_indices: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    graph: AttributedPlanarGraph
    actions: list[RepairAction]
    max_geometry_drift_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "max_geometry_drift_px": self.max_geometry_drift_px,
        }


def conservative_repair(
    graph: AttributedPlanarGraph,
    *,
    line_prob: np.ndarray | None = None,
    config: RepairConfig | None = None,
    infer_assignments: bool | None = None,
) -> RepairResult:
    """Apply small structural repairs without hallucinating M/V labels."""
    cfg = config or RepairConfig()
    if infer_assignments is not None:
        cfg = RepairConfig(**{**asdict(cfg), "infer_assignments": bool(infer_assignments)})

    repaired = _copy_graph(graph)
    original_vertices = repaired.pixel_vertices.copy()
    actions: list[RepairAction] = []

    repaired, edge_actions = _remove_zero_length_and_duplicate_edges(repaired, cfg)
    actions.extend(edge_actions)

    repaired, weak_actions = _drop_weak_edges(repaired, cfg)
    actions.extend(weak_actions)

    repaired, snap_action = _snap_near_border_vertices(repaired, cfg)
    if snap_action is not None:
        actions.append(snap_action)

    if line_prob is not None:
        repaired, completion_actions = _complete_supported_border_edges(repaired, line_prob, cfg)
        actions.extend(completion_actions)

    repaired, downgrade_actions = _downgrade_low_confidence_mv(repaired, cfg)
    actions.extend(downgrade_actions)

    if cfg.infer_assignments:
        repaired, inference_actions = _infer_local_mv_assignments(repaired)
        actions.extend(inference_actions)

    if len(original_vertices) == len(repaired.pixel_vertices):
        drift = float(np.max(np.linalg.norm(repaired.pixel_vertices - original_vertices, axis=1)))
    else:
        drift = 0.0

    return RepairResult(graph=repaired, actions=actions, max_geometry_drift_px=drift)


def _remove_zero_length_and_duplicate_edges(
    graph: AttributedPlanarGraph,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, list[RepairAction]]:
    keep_by_key: dict[tuple[int, int], int] = {}
    removed_zero: list[int] = []
    removed_duplicate: list[int] = []
    for edge_idx, edge in enumerate(graph.edges_vertices):
        v1, v2 = int(edge[0]), int(edge[1])
        if v1 == v2 or np.linalg.norm(graph.pixel_vertices[v1] - graph.pixel_vertices[v2]) < cfg.min_edge_length_px:
            removed_zero.append(edge_idx)
            continue
        key = (min(v1, v2), max(v1, v2))
        previous = keep_by_key.get(key)
        if previous is None:
            keep_by_key[key] = edge_idx
            continue
        if float(graph.edge_support[edge_idx]) > float(graph.edge_support[previous]):
            removed_duplicate.append(previous)
            keep_by_key[key] = edge_idx
        else:
            removed_duplicate.append(edge_idx)

    keep = sorted(keep_by_key.values())
    repaired = _filter_edges(graph, keep)
    actions: list[RepairAction] = []
    if removed_zero:
        actions.append(
            RepairAction(
                code="remove_zero_length_edges",
                message="Removed zero-length or near-zero-length edges.",
                edge_indices=removed_zero,
            )
        )
    if removed_duplicate:
        actions.append(
            RepairAction(
                code="remove_duplicate_edges",
                message="Removed duplicate edges, keeping the strongest supported copy.",
                edge_indices=removed_duplicate,
            )
        )
    return repaired, actions


def _drop_weak_edges(
    graph: AttributedPlanarGraph,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, list[RepairAction]]:
    weak = np.where(np.asarray(graph.edge_support) < cfg.weak_edge_support_threshold)[0]
    if len(weak) == 0:
        return graph, []
    keep = [idx for idx in range(graph.num_edges) if idx not in set(int(v) for v in weak)]
    action = RepairAction(
        code="drop_weak_edges",
        message="Dropped edges whose line support was below the conservative repair threshold.",
        edge_indices=[int(idx) for idx in weak],
        details={"threshold": cfg.weak_edge_support_threshold},
    )
    return _drop_unused_vertices(_filter_edges(graph, keep), cfg), [action]


def _snap_near_border_vertices(
    graph: AttributedPlanarGraph,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, RepairAction | None]:
    image_size = _image_size_for_graph(graph, cfg)
    if image_size <= 1:
        return graph, None
    max_coord = float(image_size - 1)
    vertices = graph.pixel_vertices.copy()
    snapped: list[int] = []
    before = vertices.copy()
    for idx, (x, y) in enumerate(vertices):
        new_x = _snap_coordinate(float(x), max_coord, cfg.border_snap_px)
        new_y = _snap_coordinate(float(y), max_coord, cfg.border_snap_px)
        if new_x != float(x) or new_y != float(y):
            vertices[idx] = [new_x, new_y]
            snapped.append(idx)
    if not snapped:
        return graph, None
    repaired = _replace_vertices(graph, vertices, image_size=image_size)
    drift = float(np.max(np.linalg.norm(vertices[snapped] - before[snapped], axis=1)))
    return repaired, RepairAction(
        code="snap_border_vertices",
        message="Snapped vertices close to the square boundary onto the boundary.",
        vertex_indices=snapped,
        details={"max_drift_px": drift, "threshold_px": cfg.border_snap_px},
    )


def _complete_supported_border_edges(
    graph: AttributedPlanarGraph,
    line_prob: np.ndarray,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, list[RepairAction]]:
    image_size = _image_size_for_graph(graph, cfg)
    if image_size <= 1 or len(graph.pixel_vertices) < 2:
        return graph, []
    max_coord = float(image_size - 1)
    corners = np.array(
        [[0.0, 0.0], [max_coord, 0.0], [max_coord, max_coord], [0.0, max_coord]],
        dtype=np.float32,
    )
    corner_vertices: list[int] = []
    for corner in corners:
        distances = np.linalg.norm(graph.pixel_vertices - corner[None, :], axis=1)
        vertex_idx = int(np.argmin(distances))
        if float(distances[vertex_idx]) > cfg.border_endpoint_tolerance_px:
            return graph, []
        corner_vertices.append(vertex_idx)

    existing = {tuple(sorted(map(int, edge))) for edge in graph.edges_vertices}
    additions: list[tuple[int, int, float]] = []
    for v1, v2 in zip(corner_vertices, [*corner_vertices[1:], corner_vertices[0]]):
        key = (min(v1, v2), max(v1, v2))
        if key in existing:
            continue
        support = _segment_support(
            graph.pixel_vertices[v1],
            graph.pixel_vertices[v2],
            line_prob,
            sample_width_px=cfg.border_completion_width_px,
        )
        if support >= cfg.border_completion_min_support:
            additions.append((v1, v2, support))

    if not additions:
        return graph, []

    repaired = _append_edges(
        graph,
        edges=[(v1, v2) for v1, v2, _ in additions],
        assignments=[2 for _ in additions],
        support=[support for _, _, support in additions],
        confidence=[support for _, _, support in additions],
        margin=[support for _, _, support in additions],
        source=["observed" for _ in additions],
        probabilities=[np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32) for _ in additions],
    )
    actions = [
        RepairAction(
            code="complete_supported_border_edge",
            message="Completed a missing square border edge with strong line evidence.",
            edge_indices=[graph.num_edges + idx],
            details={"support": float(support), "vertices": [int(v1), int(v2)]},
        )
        for idx, (v1, v2, support) in enumerate(additions)
    ]
    return repaired, actions


def _downgrade_low_confidence_mv(
    graph: AttributedPlanarGraph,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, list[RepairAction]]:
    assignments = graph.edges_assignment.copy()
    sources = list(graph.assignment_source)
    low: list[int] = []
    for edge_idx, assignment in enumerate(assignments):
        if int(assignment) not in (0, 1):
            continue
        if (
            float(graph.assignment_confidence[edge_idx]) < cfg.low_assignment_confidence
            or float(graph.assignment_margin[edge_idx]) < cfg.low_assignment_margin
        ):
            assignments[edge_idx] = 3
            sources[edge_idx] = "unknown"
            low.append(edge_idx)
    if not low:
        return graph, []
    repaired = _copy_graph(graph)
    repaired.edges_assignment = assignments.astype(np.int8)
    repaired.assignment_source = sources
    return repaired, [
        RepairAction(
            code="downgrade_low_confidence_mv",
            message="Downgraded low-confidence M/V labels to unassigned.",
            edge_indices=low,
            details={
                "confidence_threshold": cfg.low_assignment_confidence,
                "margin_threshold": cfg.low_assignment_margin,
            },
        )
    ]


def _infer_local_mv_assignments(
    graph: AttributedPlanarGraph,
) -> tuple[AttributedPlanarGraph, list[RepairAction]]:
    assignments = graph.edges_assignment.copy()
    sources = list(graph.assignment_source)
    adjacency = _edge_adjacency(graph)
    inferred: list[int] = []
    for edge_idx, assignment in enumerate(assignments):
        if int(assignment) != 3:
            continue
        candidates: set[int] | None = None
        for vertex_idx in graph.edges_vertices[edge_idx]:
            incident = [idx for idx in adjacency[int(vertex_idx)] if int(assignments[idx]) != 2]
            if len(incident) < 4 or len(incident) % 2 != 0:
                continue
            if edge_idx not in incident:
                continue
            unknown = [idx for idx in incident if int(assignments[idx]) == 3]
            if len(unknown) != 1:
                continue
            observed = [idx for idx in incident if int(assignments[idx]) in (0, 1)]
            if len(observed) != len(incident) - 1:
                continue
            possible = set()
            for candidate in (0, 1):
                m_count = sum(int(assignments[idx]) == 0 for idx in observed) + int(candidate == 0)
                v_count = sum(int(assignments[idx]) == 1 for idx in observed) + int(candidate == 1)
                if abs(m_count - v_count) == 2:
                    possible.add(candidate)
            candidates = possible if candidates is None else candidates & possible
        if candidates is not None and len(candidates) == 1:
            assignments[edge_idx] = int(next(iter(candidates)))
            sources[edge_idx] = "inferred"
            inferred.append(edge_idx)

    if not inferred:
        return graph, []
    repaired = _copy_graph(graph)
    repaired.edges_assignment = assignments.astype(np.int8)
    repaired.assignment_source = sources
    return repaired, [
        RepairAction(
            code="infer_assignments",
            message="Inferred locally forced M/V labels from Maekawa-compatible constraints.",
            edge_indices=inferred,
        )
    ]


def _copy_graph(graph: AttributedPlanarGraph) -> AttributedPlanarGraph:
    return AttributedPlanarGraph(
        vertices_coords=np.asarray(graph.vertices_coords, dtype=np.float32).copy(),
        edges_vertices=np.asarray(graph.edges_vertices, dtype=np.int64).copy(),
        edges_assignment=np.asarray(graph.edges_assignment, dtype=np.int8).copy(),
        edge_support=np.asarray(graph.edge_support, dtype=np.float32).copy(),
        vertex_support=np.asarray(graph.vertex_support, dtype=np.float32).copy(),
        pixel_vertices=np.asarray(graph.pixel_vertices, dtype=np.float32).copy(),
        assignment_confidence=np.asarray(graph.assignment_confidence, dtype=np.float32).copy(),
        assignment_margin=np.asarray(graph.assignment_margin, dtype=np.float32).copy(),
        assignment_source=list(graph.assignment_source),
        assignment_probabilities=(
            None
            if graph.assignment_probabilities is None
            else np.asarray(graph.assignment_probabilities, dtype=np.float32).copy()
        ),
        debug=dict(graph.debug),
    )


def _filter_edges(graph: AttributedPlanarGraph, keep: list[int]) -> AttributedPlanarGraph:
    keep_array = np.asarray(keep, dtype=np.int64)
    repaired = _copy_graph(graph)
    repaired.edges_vertices = repaired.edges_vertices[keep_array]
    repaired.edges_assignment = repaired.edges_assignment[keep_array]
    repaired.edge_support = repaired.edge_support[keep_array]
    repaired.assignment_confidence = repaired.assignment_confidence[keep_array]
    repaired.assignment_margin = repaired.assignment_margin[keep_array]
    repaired.assignment_source = [repaired.assignment_source[int(idx)] for idx in keep_array]
    if repaired.assignment_probabilities is not None:
        repaired.assignment_probabilities = repaired.assignment_probabilities[keep_array]
    return repaired


def _drop_unused_vertices(graph: AttributedPlanarGraph, cfg: RepairConfig) -> AttributedPlanarGraph:
    if graph.num_edges == 0:
        repaired = _copy_graph(graph)
        repaired.vertices_coords = np.empty((0, 2), dtype=np.float32)
        repaired.pixel_vertices = np.empty((0, 2), dtype=np.float32)
        repaired.vertex_support = np.empty(0, dtype=np.float32)
        return repaired
    used = np.unique(graph.edges_vertices.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used)}
    repaired = _copy_graph(graph)
    repaired.pixel_vertices = repaired.pixel_vertices[used]
    repaired.vertex_support = repaired.vertex_support[used]
    repaired.edges_vertices = np.array(
        [[remap[int(v1)], remap[int(v2)]] for v1, v2 in repaired.edges_vertices],
        dtype=np.int64,
    )
    image_size = _image_size_for_graph(graph, cfg)
    repaired.vertices_coords = _canonical_from_pixels(repaired.pixel_vertices, image_size)
    return repaired


def _replace_vertices(
    graph: AttributedPlanarGraph,
    vertices: np.ndarray,
    *,
    image_size: int,
) -> AttributedPlanarGraph:
    repaired = _copy_graph(graph)
    repaired.pixel_vertices = np.asarray(vertices, dtype=np.float32)
    repaired.vertices_coords = _canonical_from_pixels(repaired.pixel_vertices, image_size)
    return repaired


def _append_edges(
    graph: AttributedPlanarGraph,
    *,
    edges: list[tuple[int, int]],
    assignments: list[int],
    support: list[float],
    confidence: list[float],
    margin: list[float],
    source: list[str],
    probabilities: list[np.ndarray],
) -> AttributedPlanarGraph:
    repaired = _copy_graph(graph)
    repaired.edges_vertices = np.concatenate(
        [repaired.edges_vertices, np.asarray(edges, dtype=np.int64)],
        axis=0,
    )
    repaired.edges_assignment = np.concatenate(
        [repaired.edges_assignment, np.asarray(assignments, dtype=np.int8)],
        axis=0,
    )
    repaired.edge_support = np.concatenate(
        [repaired.edge_support, np.asarray(support, dtype=np.float32)],
        axis=0,
    )
    repaired.assignment_confidence = np.concatenate(
        [repaired.assignment_confidence, np.asarray(confidence, dtype=np.float32)],
        axis=0,
    )
    repaired.assignment_margin = np.concatenate(
        [repaired.assignment_margin, np.asarray(margin, dtype=np.float32)],
        axis=0,
    )
    repaired.assignment_source.extend(source)
    if repaired.assignment_probabilities is not None:
        repaired.assignment_probabilities = np.concatenate(
            [repaired.assignment_probabilities, np.asarray(probabilities, dtype=np.float32)],
            axis=0,
        )
    return repaired


def _image_size_for_graph(graph: AttributedPlanarGraph, cfg: RepairConfig) -> int:
    if cfg.image_size is not None:
        return int(cfg.image_size)
    if len(graph.pixel_vertices) == 0:
        return 0
    return int(round(float(np.max(graph.pixel_vertices)))) + 1


def _canonical_from_pixels(vertices: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 1 or len(vertices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return (np.asarray(vertices, dtype=np.float32) / float(image_size - 1)).clip(0.0, 1.0)


def _snap_coordinate(value: float, max_coord: float, threshold: float) -> float:
    if value <= threshold:
        return 0.0
    if abs(max_coord - value) <= threshold:
        return max_coord
    return value


def _edge_adjacency(graph: AttributedPlanarGraph) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(graph.num_vertices)]
    for edge_idx, (v1, v2) in enumerate(graph.edges_vertices):
        adjacency[int(v1)].append(edge_idx)
        adjacency[int(v2)].append(edge_idx)
    return adjacency


def _segment_support(
    p0: np.ndarray,
    p1: np.ndarray,
    line_prob: np.ndarray,
    *,
    sample_width_px: int,
) -> float:
    points = _sample_segment_points(p0, p1, step_px=1.0)
    if len(points) == 0:
        return 0.0
    h, w = line_prob.shape[:2]
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return 0.0
    perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
    half_width = max(0, sample_width_px // 2)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    hits = []
    for point in points:
        values = []
        for offset in offsets:
            sample = point + offset * perp
            x = int(round(float(sample[0])))
            y = int(round(float(sample[1])))
            if 0 <= x < w and 0 <= y < h:
                values.append(float(line_prob[y, x]))
        if values:
            hits.append(max(values))
    return float(np.mean(hits)) if hits else 0.0


def _sample_segment_points(p0: np.ndarray, p1: np.ndarray, step_px: float) -> np.ndarray:
    distance = float(np.linalg.norm(p1 - p0))
    if distance <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    steps = max(int(np.ceil(distance / max(step_px, 1e-6))) + 1, 2)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return (p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]).astype(np.float32)
