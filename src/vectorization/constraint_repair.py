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
    canonicalize_square_border: bool = True
    border_canonicalization_tolerance_px: float = 6.0
    border_canonicalization_min_sides: int = 3
    border_canonicalization_min_vertices_per_side: int = 2
    border_canonicalization_min_support: float = 0.70
    border_canonicalization_min_edge_fraction: float = 0.12
    border_canonicalization_min_probability: float = 0.45
    border_canonicalization_snap_vertices: bool = False
    border_canonicalization_max_snap_drift_px: float = 3.0
    border_canonicalization_rebuild_chains: bool = False
    border_square_aspect_tolerance: float = 0.08
    reconstruct_square_border_chain: bool = True
    border_chain_reconstruction_tolerance_px: float = 10.0
    border_chain_reconstruction_max_snap_drift_px: float = 6.0
    border_chain_min_existing_b_frame_fraction: float = 0.60
    border_chain_max_edge_growth_factor: float = 2.0
    border_chain_max_edge_growth_extra: int = 4
    border_chain_allow_corner_synthesis: bool = False
    border_chain_downgrade_off_frame_b: bool = True
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

    if cfg.canonicalize_square_border:
        repaired, border_action = _canonicalize_square_border(repaired, line_prob, cfg)
        if border_action is not None:
            actions.append(border_action)

    if cfg.reconstruct_square_border_chain:
        repaired, chain_action = _reconstruct_square_border_chain(repaired, line_prob, cfg)
        if chain_action is not None:
            actions.append(chain_action)

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


@dataclass(frozen=True)
class _BorderFrame:
    left: float
    right: float
    top: float
    bottom: float


def _canonicalize_square_border(
    graph: AttributedPlanarGraph,
    line_prob: np.ndarray | None,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, RepairAction | None]:
    if graph.num_vertices < 4 or graph.num_edges == 0:
        return graph, None

    image_size = _image_size_for_graph(graph, cfg)
    vertices = np.asarray(graph.pixel_vertices, dtype=np.float32)
    frame = _infer_border_frame(vertices, image_size, cfg)
    if frame is None:
        return graph, None
    tolerance = _effective_border_tolerance(
        max(frame.right - frame.left, frame.bottom - frame.top),
        cfg,
    )

    side_vertices = _side_vertices(vertices, frame, tolerance)
    eligible_sides = [
        side
        for side, indices in side_vertices.items()
        if len(indices) >= cfg.border_canonicalization_min_vertices_per_side
    ]
    if len(eligible_sides) < cfg.border_canonicalization_min_sides:
        return graph, None

    snapped_vertices_all, _ = _snap_vertices_to_frame(
        vertices,
        frame,
        tolerance,
    )
    side_vertices = _side_vertices(snapped_vertices_all, frame, tolerance)

    repaired = _copy_graph(graph)
    assignments = repaired.edges_assignment.copy()
    confidence = repaired.assignment_confidence.copy()
    margin = repaired.assignment_margin.copy()
    source = list(repaired.assignment_source)
    probabilities = (
        None
        if repaired.assignment_probabilities is None
        else repaired.assignment_probabilities.copy()
    )

    remove_indices: set[int] = set()
    forced_edges: set[int] = set()
    border_vertices_to_snap: set[int] = set()
    covered_segments: dict[tuple[int, int], float] = {}
    existing = {tuple(sorted(map(int, edge))): idx for idx, edge in enumerate(repaired.edges_vertices)}

    for side, indices in side_vertices.items():
        if len(indices) < 2:
            continue
        side_length = _side_length(frame, side)
        positions = {int(vertex_idx): order for order, vertex_idx in enumerate(indices)}
        for edge_idx, edge in enumerate(repaired.edges_vertices):
            if edge_idx in remove_indices:
                continue
            v1, v2 = int(edge[0]), int(edge[1])
            if v1 not in positions or v2 not in positions:
                continue
            if not _should_treat_as_border_edge(
                repaired,
                edge_idx,
                side_length=side_length,
                cfg=cfg,
            ):
                continue
            lo = min(positions[v1], positions[v2])
            hi = max(positions[v1], positions[v2])
            if hi == lo:
                if cfg.border_canonicalization_rebuild_chains:
                    remove_indices.add(edge_idx)
                continue
            if hi - lo > 1:
                if not cfg.border_canonicalization_rebuild_chains:
                    forced_edges.add(edge_idx)
                    border_vertices_to_snap.update((v1, v2))
                    continue
                remove_indices.add(edge_idx)
                support = float(repaired.edge_support[edge_idx])
                for chain_idx in range(lo, hi):
                    key = tuple(sorted((int(indices[chain_idx]), int(indices[chain_idx + 1]))))
                    covered_segments[key] = max(covered_segments.get(key, 0.0), support)
                continue
            forced_edges.add(edge_idx)
            border_vertices_to_snap.update((v1, v2))

    additions: list[tuple[int, int]] = []
    addition_support: list[float] = []
    added_keys: set[tuple[int, int]] = set()
    if cfg.border_canonicalization_rebuild_chains:
        for side, indices in side_vertices.items():
            if len(indices) < 2:
                continue
            side_length = _side_length(frame, side)
            for v1, v2 in zip(indices, indices[1:]):
                key = tuple(sorted((int(v1), int(v2))))
                existing_idx = existing.get(key)
                if existing_idx is not None and existing_idx not in remove_indices:
                    if _should_treat_as_border_edge(
                        repaired,
                        existing_idx,
                        side_length=side_length,
                        cfg=cfg,
                    ):
                        forced_edges.add(existing_idx)
                        border_vertices_to_snap.update((int(v1), int(v2)))
                    continue
                support = covered_segments.get(key, 0.0)
                if line_prob is not None:
                    support = max(
                        support,
                        _segment_support(
                            snapped_vertices_all[int(v1)],
                            snapped_vertices_all[int(v2)],
                            line_prob,
                            sample_width_px=cfg.border_completion_width_px,
                        ),
                    )
                if support < cfg.border_canonicalization_min_support or key in added_keys:
                    continue
                additions.append((int(v1), int(v2)))
                addition_support.append(float(support))
                added_keys.add(key)
                border_vertices_to_snap.update((int(v1), int(v2)))

    snapped_vertices = vertices.copy()
    snap_rejected_for_drift = 0
    if cfg.border_canonicalization_snap_vertices and border_vertices_to_snap:
        snap_rejected_for_drift = sum(
            1
            for idx in border_vertices_to_snap
            if np.linalg.norm(snapped_vertices_all[int(idx)] - vertices[int(idx)])
            > cfg.border_canonicalization_max_snap_drift_px
        )
        snap_indices = np.asarray(
            [
                idx
                for idx in sorted(border_vertices_to_snap)
                if np.linalg.norm(snapped_vertices_all[int(idx)] - vertices[int(idx)])
                <= cfg.border_canonicalization_max_snap_drift_px
            ],
            dtype=np.int64,
        )
        snapped_vertices[snap_indices] = snapped_vertices_all[snap_indices]
    snapped = [
        int(idx)
        for idx in sorted(border_vertices_to_snap)
        if not np.allclose(snapped_vertices[int(idx)], vertices[int(idx)])
    ]
    if snapped:
        repaired = _replace_vertices(repaired, snapped_vertices, image_size=image_size)

    if forced_edges:
        for edge_idx in forced_edges:
            assignments[edge_idx] = 2
            confidence[edge_idx] = max(float(confidence[edge_idx]), float(repaired.edge_support[edge_idx]))
            margin[edge_idx] = max(float(margin[edge_idx]), float(repaired.edge_support[edge_idx]))
            source[edge_idx] = "observed"
            if probabilities is not None:
                probabilities[edge_idx] = _assignment_probability_row(2, float(confidence[edge_idx]))

    repaired.edges_assignment = assignments.astype(np.int8)
    repaired.assignment_confidence = confidence.astype(np.float32)
    repaired.assignment_margin = margin.astype(np.float32)
    repaired.assignment_source = source
    repaired.assignment_probabilities = probabilities

    if remove_indices:
        keep = [idx for idx in range(repaired.num_edges) if idx not in remove_indices]
        repaired = _filter_edges(repaired, keep)

    if additions:
        repaired = _append_edges(
            repaired,
            edges=additions,
            assignments=[2 for _ in additions],
            support=addition_support,
            confidence=addition_support,
            margin=addition_support,
            source=["observed" for _ in additions],
            probabilities=[
                _assignment_probability_row(2, support) for support in addition_support
            ],
        )

    geometry_reverted = False
    if snapped and _introduces_illegal_crossing(graph, repaired):
        repaired = _replace_vertices(repaired, vertices, image_size=image_size)
        snapped = []
        snapped_vertices = vertices
        geometry_reverted = True

    if not snapped and not forced_edges and not remove_indices and not additions:
        return graph, None

    drift = (
        float(np.max(np.linalg.norm(snapped_vertices[snapped] - vertices[snapped], axis=1)))
        if snapped
        else 0.0
    )
    action = RepairAction(
        code="canonicalize_square_border",
        message=(
            "Straightened the inferred square border and forced border-chain edges to B."
            if snapped
            else "Forced inferred square-border edges to B."
        ),
        edge_indices=sorted(int(idx) for idx in forced_edges),
        vertex_indices=snapped,
        details={
            "snapped_vertices": len(snapped),
            "forced_border_edges": len(forced_edges),
            "removed_redundant_edges": len(remove_indices),
            "added_border_edges": len(additions),
            "side_vertex_counts": {side: len(indices) for side, indices in side_vertices.items()},
            "max_drift_px": drift,
            "snap_rejected_for_drift": snap_rejected_for_drift,
            "tolerance_px": tolerance,
            "geometry_reverted": geometry_reverted,
        },
    )
    return repaired, action


def _reconstruct_square_border_chain(
    graph: AttributedPlanarGraph,
    line_prob: np.ndarray | None,
    cfg: RepairConfig,
) -> tuple[AttributedPlanarGraph, RepairAction | None]:
    if graph.num_vertices < 4 or graph.num_edges == 0:
        return graph, None

    image_size = _image_size_for_graph(graph, cfg)
    vertices = np.asarray(graph.pixel_vertices, dtype=np.float32)
    frame = _infer_border_frame(vertices, image_size, cfg)
    if frame is None:
        return graph, None

    tolerance = _border_chain_tolerance(
        max(frame.right - frame.left, frame.bottom - frame.top),
        cfg,
    )
    snapped_all, _ = _snap_vertices_to_frame(vertices, frame, tolerance)
    seed_side_vertices = _side_vertices(snapped_all, frame, tolerance)
    eligible_sides = [
        side
        for side, indices in seed_side_vertices.items()
        if len(indices) >= cfg.border_canonicalization_min_vertices_per_side
    ]
    if len(eligible_sides) < cfg.border_canonicalization_min_sides:
        return graph, None

    selected_vertices = _border_chain_seed_vertices(
        graph,
        snapped_all,
        frame=frame,
        tolerance=tolerance,
    )
    snapped_vertices = vertices.copy()
    snap_rejected_for_drift = 0
    for vertex_idx in sorted(selected_vertices):
        drift = float(np.linalg.norm(snapped_all[int(vertex_idx)] - vertices[int(vertex_idx)]))
        if drift > cfg.border_chain_reconstruction_max_snap_drift_px:
            selected_vertices.remove(int(vertex_idx))
            snap_rejected_for_drift += 1
            continue
        snapped_vertices[int(vertex_idx)] = snapped_all[int(vertex_idx)]

    vertex_support = np.asarray(graph.vertex_support, dtype=np.float32).copy()
    corner_vertices: dict[str, int] = {}
    added_corner_vertices = 0
    for corner_name, corner in _frame_corners(frame).items():
        distances = np.linalg.norm(snapped_vertices - corner[None, :], axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        if nearest_distance <= cfg.border_chain_reconstruction_max_snap_drift_px:
            snapped_vertices[nearest_idx] = corner
            selected_vertices.add(nearest_idx)
            corner_vertices[corner_name] = nearest_idx
            continue

        if not cfg.border_chain_allow_corner_synthesis:
            return graph, None

        corner_idx = int(len(snapped_vertices))
        snapped_vertices = np.concatenate([snapped_vertices, corner[None, :]], axis=0)
        vertex_support = np.concatenate([vertex_support, np.asarray([1.0], dtype=np.float32)])
        selected_vertices.add(corner_idx)
        corner_vertices[corner_name] = corner_idx
        added_corner_vertices += 1

    selected_by_side = _selected_border_vertices_by_side(
        snapped_vertices,
        selected_vertices,
        frame=frame,
        tolerance=tolerance,
        cfg=cfg,
    )
    if any(len(indices) < 2 for indices in selected_by_side.values()):
        return graph, None

    chain_edges = _border_chain_edges(selected_by_side)
    if len(chain_edges) < 4:
        return graph, None

    frame_edge_indices = {
        edge_idx
        for edge_idx, edge in enumerate(graph.edges_vertices)
        if _edge_lies_on_any_frame_side(snapped_vertices, edge, frame, tolerance)
    }
    existing_b_indices = {
        edge_idx for edge_idx, assignment in enumerate(graph.edges_assignment) if int(assignment) == 2
    }
    if len(existing_b_indices) >= 4:
        frame_b_fraction = len(frame_edge_indices & existing_b_indices) / len(existing_b_indices)
        if frame_b_fraction < cfg.border_chain_min_existing_b_frame_fraction:
            return graph, None
        max_chain_edges = (
            cfg.border_chain_max_edge_growth_factor * len(existing_b_indices)
            + cfg.border_chain_max_edge_growth_extra
        )
        if len(chain_edges) > max_chain_edges:
            return graph, None

    remove_indices: set[int] = set()
    downgraded_off_frame_b: list[int] = []
    assignments = graph.edges_assignment.copy()
    confidence = graph.assignment_confidence.copy()
    margin = graph.assignment_margin.copy()
    source = list(graph.assignment_source)
    probabilities = (
        None
        if graph.assignment_probabilities is None
        else graph.assignment_probabilities.copy()
    )
    for edge_idx, edge in enumerate(graph.edges_vertices):
        if edge_idx in frame_edge_indices:
            remove_indices.add(edge_idx)
            continue
        if cfg.border_chain_downgrade_off_frame_b and int(assignments[edge_idx]) == 2:
            assignments[edge_idx] = 3
            confidence[edge_idx] = min(float(confidence[edge_idx]), 0.5)
            margin[edge_idx] = min(float(margin[edge_idx]), 0.0)
            source[edge_idx] = "unknown"
            downgraded_off_frame_b.append(edge_idx)
            if probabilities is not None:
                probabilities[edge_idx] = _assignment_probability_row(3, 1.0)

    working = _copy_graph(graph)
    working.pixel_vertices = snapped_vertices.astype(np.float32)
    working.vertices_coords = _canonical_from_pixels(working.pixel_vertices, image_size)
    working.vertex_support = vertex_support.astype(np.float32)
    working.edges_assignment = assignments.astype(np.int8)
    working.assignment_confidence = confidence.astype(np.float32)
    working.assignment_margin = margin.astype(np.float32)
    working.assignment_source = source
    working.assignment_probabilities = probabilities

    keep = [idx for idx in range(working.num_edges) if idx not in remove_indices]
    repaired = _filter_edges(working, keep)

    edge_support = [
        _border_chain_edge_support(
            snapped_vertices[int(v1)],
            snapped_vertices[int(v2)],
            line_prob,
            cfg,
        )
        for v1, v2 in chain_edges
    ]
    repaired = _append_edges(
        repaired,
        edges=chain_edges,
        assignments=[2 for _ in chain_edges],
        support=edge_support,
        confidence=[1.0 for _ in chain_edges],
        margin=[1.0 for _ in chain_edges],
        source=["inferred" for _ in chain_edges],
        probabilities=[_assignment_probability_row(2, 1.0) for _ in chain_edges],
    )
    repaired = _drop_unused_vertices(repaired, cfg)

    snapped_existing = [
        int(idx)
        for idx in sorted(selected_vertices)
        if idx < len(vertices) and not np.allclose(snapped_vertices[int(idx)], vertices[int(idx)])
    ]
    changed = bool(
        snapped_existing
        or remove_indices
        or downgraded_off_frame_b
        or added_corner_vertices
        or chain_edges
    )
    if not changed:
        return graph, None

    max_drift = (
        float(np.max(np.linalg.norm(snapped_vertices[snapped_existing] - vertices[snapped_existing], axis=1)))
        if snapped_existing
        else 0.0
    )
    action = RepairAction(
        code="reconstruct_square_border_chain",
        message="Rebuilt the inferred square border as a clean B chain.",
        edge_indices=[],
        vertex_indices=snapped_existing,
        details={
            "added_border_edges": len(chain_edges),
            "removed_frame_edges": len(remove_indices),
            "downgraded_off_frame_b_edges": len(downgraded_off_frame_b),
            "added_corner_vertices": added_corner_vertices,
            "corner_vertices": {key: int(value) for key, value in corner_vertices.items()},
            "side_vertex_counts": {
                side: len(indices) for side, indices in selected_by_side.items()
            },
            "max_drift_px": max_drift,
            "snap_rejected_for_drift": snap_rejected_for_drift,
            "tolerance_px": tolerance,
        },
    )
    return repaired, action


def _border_chain_tolerance(side_length: float, cfg: RepairConfig) -> float:
    scaled = max(1.0, 0.01 * float(side_length))
    return float(min(cfg.border_chain_reconstruction_tolerance_px, scaled))


def _frame_corners(frame: _BorderFrame) -> dict[str, np.ndarray]:
    return {
        "top_left": np.asarray([frame.left, frame.top], dtype=np.float32),
        "top_right": np.asarray([frame.right, frame.top], dtype=np.float32),
        "bottom_right": np.asarray([frame.right, frame.bottom], dtype=np.float32),
        "bottom_left": np.asarray([frame.left, frame.bottom], dtype=np.float32),
    }


def _border_chain_seed_vertices(
    graph: AttributedPlanarGraph,
    vertices: np.ndarray,
    *,
    frame: _BorderFrame,
    tolerance: float,
) -> set[int]:
    side_vertices = _side_vertices(vertices, frame, tolerance)
    return {int(idx) for indices in side_vertices.values() for idx in indices}


def _selected_border_vertices_by_side(
    vertices: np.ndarray,
    selected_vertices: set[int],
    *,
    frame: _BorderFrame,
    tolerance: float,
    cfg: RepairConfig,
) -> dict[str, list[int]]:
    side_vertices = _side_vertices(vertices, frame, tolerance)
    selected_by_side: dict[str, list[int]] = {}
    for side, indices in side_vertices.items():
        side_selected = [int(idx) for idx in indices if int(idx) in selected_vertices]
        selected_by_side[side] = _unique_side_vertices(
            side_selected,
            vertices,
            side=side,
            min_spacing_px=cfg.min_edge_length_px,
        )
    return selected_by_side


def _unique_side_vertices(
    indices: list[int],
    vertices: np.ndarray,
    *,
    side: str,
    min_spacing_px: float,
) -> list[int]:
    ordered = sorted(indices, key=lambda idx: (_side_position(vertices[int(idx)], side), int(idx)))
    unique: list[int] = []
    for vertex_idx in ordered:
        if unique:
            previous = unique[-1]
            if abs(
                _side_position(vertices[int(vertex_idx)], side)
                - _side_position(vertices[int(previous)], side)
            ) < min_spacing_px:
                continue
        unique.append(int(vertex_idx))
    return unique


def _border_chain_edges(selected_by_side: dict[str, list[int]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for side in ("top", "right", "bottom", "left"):
        indices = selected_by_side.get(side, [])
        for v1, v2 in zip(indices, indices[1:]):
            if int(v1) == int(v2):
                continue
            key = tuple(sorted((int(v1), int(v2))))
            if key in seen:
                continue
            seen.add(key)
            edges.append((int(v1), int(v2)))
    return edges


def _border_chain_edge_support(
    p0: np.ndarray,
    p1: np.ndarray,
    line_prob: np.ndarray | None,
    cfg: RepairConfig,
) -> float:
    if line_prob is None:
        return 1.0
    support = _segment_support(
        p0,
        p1,
        line_prob,
        sample_width_px=cfg.border_completion_width_px,
    )
    return float(max(support, cfg.border_canonicalization_min_support))


def _edge_lies_on_any_frame_side(
    vertices: np.ndarray,
    edge: np.ndarray,
    frame: _BorderFrame,
    tolerance: float,
) -> bool:
    return any(
        _edge_lies_on_frame_side(vertices, edge, frame, side, tolerance)
        for side in ("top", "right", "bottom", "left")
    )


def _edge_lies_on_frame_side(
    vertices: np.ndarray,
    edge: np.ndarray,
    frame: _BorderFrame,
    side: str,
    tolerance: float,
) -> bool:
    v1, v2 = int(edge[0]), int(edge[1])
    return _vertex_on_frame_side(vertices[v1], frame, side, tolerance) and _vertex_on_frame_side(
        vertices[v2],
        frame,
        side,
        tolerance,
    )


def _vertex_on_frame_side(
    vertex: np.ndarray,
    frame: _BorderFrame,
    side: str,
    tolerance: float,
) -> bool:
    x = float(vertex[0])
    y = float(vertex[1])
    if side == "top":
        return frame.left - tolerance <= x <= frame.right + tolerance and abs(y - frame.top) <= tolerance
    if side == "right":
        return frame.top - tolerance <= y <= frame.bottom + tolerance and abs(x - frame.right) <= tolerance
    if side == "bottom":
        return frame.left - tolerance <= x <= frame.right + tolerance and abs(y - frame.bottom) <= tolerance
    return frame.top - tolerance <= y <= frame.bottom + tolerance and abs(x - frame.left) <= tolerance


def _is_near_frame_corner(
    vertex: np.ndarray,
    frame: _BorderFrame,
    tolerance: float,
) -> bool:
    return any(
        np.linalg.norm(vertex - corner) <= tolerance for corner in _frame_corners(frame).values()
    )


def _side_position(vertex: np.ndarray, side: str) -> float:
    if side in {"top", "bottom"}:
        return float(vertex[0])
    return float(vertex[1])


def _infer_border_frame(
    vertices: np.ndarray,
    image_size: int,
    cfg: RepairConfig,
) -> _BorderFrame | None:
    if len(vertices) == 0:
        return None
    left = float(np.min(vertices[:, 0]))
    right = float(np.max(vertices[:, 0]))
    top = float(np.min(vertices[:, 1]))
    bottom = float(np.max(vertices[:, 1]))
    width = right - left
    height = bottom - top
    if width <= cfg.min_edge_length_px or height <= cfg.min_edge_length_px:
        return None

    max_side = max(width, height)
    tolerance = _effective_border_tolerance(max_side, cfg)
    aspect_delta = abs(width - height)
    aspect_limit = min(
        4.0 * tolerance,
        max(tolerance, cfg.border_square_aspect_tolerance * max_side),
    )
    if aspect_delta <= aspect_limit:
        side = 0.5 * (width + height)
        cx = 0.5 * (left + right)
        cy = 0.5 * (top + bottom)
        left = cx - 0.5 * side
        right = cx + 0.5 * side
        top = cy - 0.5 * side
        bottom = cy + 0.5 * side

    if image_size > 1:
        max_coord = float(image_size - 1)
        left = float(np.clip(left, 0.0, max_coord))
        right = float(np.clip(right, 0.0, max_coord))
        top = float(np.clip(top, 0.0, max_coord))
        bottom = float(np.clip(bottom, 0.0, max_coord))
    if right - left <= cfg.min_edge_length_px or bottom - top <= cfg.min_edge_length_px:
        return None
    return _BorderFrame(left=left, right=right, top=top, bottom=bottom)


def _effective_border_tolerance(side_length: float, cfg: RepairConfig) -> float:
    scaled = max(1.0, 0.02 * float(side_length))
    return float(min(cfg.border_canonicalization_tolerance_px, scaled))


def _side_length(frame: _BorderFrame, side: str) -> float:
    if side in {"top", "bottom"}:
        return max(frame.right - frame.left, 1.0)
    return max(frame.bottom - frame.top, 1.0)


def _should_treat_as_border_edge(
    graph: AttributedPlanarGraph,
    edge_idx: int,
    *,
    side_length: float,
    cfg: RepairConfig,
) -> bool:
    if int(graph.edges_assignment[edge_idx]) == 2:
        return True
    edge = graph.edges_vertices[edge_idx]
    p0 = graph.pixel_vertices[int(edge[0])]
    p1 = graph.pixel_vertices[int(edge[1])]
    length_fraction = float(np.linalg.norm(p1 - p0)) / max(float(side_length), 1.0)
    if length_fraction < cfg.border_canonicalization_min_edge_fraction:
        return False
    return _edge_border_probability(graph, edge_idx) >= cfg.border_canonicalization_min_probability


def _edge_border_probability(graph: AttributedPlanarGraph, edge_idx: int) -> float:
    if graph.assignment_probabilities is None:
        return 1.0 if int(graph.edges_assignment[edge_idx]) == 2 else 0.0
    probabilities = np.asarray(graph.assignment_probabilities[edge_idx], dtype=np.float32)
    if len(probabilities) <= 2:
        return 0.0
    return float(probabilities[2])


def _introduces_illegal_crossing(
    original: AttributedPlanarGraph,
    candidate: AttributedPlanarGraph,
) -> bool:
    from src.vectorization.metrics import validate_structure

    original_validity = validate_structure(original.to_planar_result())
    if not original_validity.no_illegal_crossings:
        return False
    candidate_validity = validate_structure(candidate.to_planar_result())
    return not candidate_validity.no_illegal_crossings


def _side_vertices(
    vertices: np.ndarray,
    frame: _BorderFrame,
    tolerance: float,
) -> dict[str, list[int]]:
    sides: dict[str, list[int]] = {"top": [], "right": [], "bottom": [], "left": []}
    for idx, (x_value, y_value) in enumerate(vertices):
        x = float(x_value)
        y = float(y_value)
        within_x = frame.left - tolerance <= x <= frame.right + tolerance
        within_y = frame.top - tolerance <= y <= frame.bottom + tolerance
        if within_x and abs(y - frame.top) <= tolerance:
            sides["top"].append(idx)
        if within_y and abs(x - frame.right) <= tolerance:
            sides["right"].append(idx)
        if within_x and abs(y - frame.bottom) <= tolerance:
            sides["bottom"].append(idx)
        if within_y and abs(x - frame.left) <= tolerance:
            sides["left"].append(idx)

    sides["top"].sort(key=lambda vertex_idx: float(vertices[vertex_idx, 0]))
    sides["bottom"].sort(key=lambda vertex_idx: float(vertices[vertex_idx, 0]))
    sides["left"].sort(key=lambda vertex_idx: float(vertices[vertex_idx, 1]))
    sides["right"].sort(key=lambda vertex_idx: float(vertices[vertex_idx, 1]))
    return sides


def _snap_vertices_to_frame(
    vertices: np.ndarray,
    frame: _BorderFrame,
    tolerance: float,
) -> tuple[np.ndarray, list[int]]:
    snapped = np.asarray(vertices, dtype=np.float32).copy()
    snapped_indices: list[int] = []
    for idx, (x_value, y_value) in enumerate(vertices):
        x = float(x_value)
        y = float(y_value)
        new_x = x
        new_y = y
        within_x = frame.left - tolerance <= x <= frame.right + tolerance
        within_y = frame.top - tolerance <= y <= frame.bottom + tolerance
        x_candidates = []
        if within_y and abs(x - frame.left) <= tolerance:
            x_candidates.append((abs(x - frame.left), frame.left))
        if within_y and abs(x - frame.right) <= tolerance:
            x_candidates.append((abs(x - frame.right), frame.right))
        y_candidates = []
        if within_x and abs(y - frame.top) <= tolerance:
            y_candidates.append((abs(y - frame.top), frame.top))
        if within_x and abs(y - frame.bottom) <= tolerance:
            y_candidates.append((abs(y - frame.bottom), frame.bottom))
        if x_candidates:
            new_x = min(x_candidates, key=lambda item: item[0])[1]
        if y_candidates:
            new_y = min(y_candidates, key=lambda item: item[0])[1]
        if new_x != x or new_y != y:
            snapped[idx] = [new_x, new_y]
            snapped_indices.append(idx)
    return snapped, snapped_indices


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


def _assignment_probability_row(assignment: int, confidence: float) -> np.ndarray:
    row = np.zeros(4, dtype=np.float32)
    row[int(assignment)] = float(np.clip(confidence, 0.0, 1.0))
    row[3] = max(row[3], 1.0 - row[int(assignment)])
    return row


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
