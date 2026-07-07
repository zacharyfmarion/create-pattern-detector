"""Global duplicate suppression for VertexRefiner crop predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.vertex_refiner_proposals import VertexProposal, crop_origin_for_center
from src.models.vertex_refiner import DecodedVertex
from src.models.vertex_refiner_contract import BOUNDARY_SIDE_NAMES, CROP_SIZE_PX, VERTEX_KIND_NAMES


@dataclass(frozen=True)
class MergedVertex(DecodedVertex):
    """One global vertex hypothesis merged from overlapping crop predictions."""

    support_count: int
    possible_support_count: int
    support_fraction: float
    mean_member_distance_px: float
    max_member_distance_px: float


@dataclass(frozen=True)
class VertexMergeConfig:
    radius_px: float = 3.0
    boundary_merge_radius_px: float | None = None
    min_score: float = 0.0
    min_member_score: float = 0.0
    min_support: int = 1
    min_support_fraction: float = 0.0
    ray_vote_fraction: float = 0.35


def merge_decoded_vertices(
    vertices: list[DecodedVertex],
    *,
    proposals: list[VertexProposal] | None = None,
    crop_size: int = CROP_SIZE_PX,
    config: VertexMergeConfig | None = None,
) -> list[MergedVertex]:
    """Cluster decoded crop vertices into canonical full-image vertices."""
    cfg = config or VertexMergeConfig()
    candidates = [
        vertex
        for vertex in vertices
        if float(vertex.score) >= float(cfg.min_member_score)
    ]
    if not candidates:
        return []
    clusters: list[list[DecodedVertex]] = []
    centers: list[tuple[float, float]] = []
    grid: dict[tuple[int, int], set[int]] = defaultdict(set)
    cluster_cells: list[tuple[int, int]] = []
    cell_size = max(float(cfg.radius_px), 1e-6)
    for vertex in sorted(candidates, key=lambda item: (-float(item.score), float(item.y), float(item.x))):
        candidate_indices = _nearby_cluster_indices(vertex, grid=grid, cell_size=cell_size)
        match_index = _nearest_cluster_index(
            vertex,
            centers,
            clusters,
            radius_px=cfg.radius_px,
            boundary_radius_px=cfg.boundary_merge_radius_px,
            candidate_indices=candidate_indices,
        )
        if match_index is None:
            clusters.append([vertex])
            centers.append((float(vertex.x), float(vertex.y)))
            cluster_index = len(clusters) - 1
            cell = _grid_key(float(vertex.x), float(vertex.y), cell_size=cell_size)
            cluster_cells.append(cell)
            grid[cell].add(cluster_index)
            continue
        old_cell = cluster_cells[match_index]
        clusters[match_index].append(vertex)
        centers[match_index] = _weighted_center(clusters[match_index])
        new_cell = _grid_key(*centers[match_index], cell_size=cell_size)
        if new_cell != old_cell:
            grid[old_cell].discard(match_index)
            grid[new_cell].add(match_index)
            cluster_cells[match_index] = new_cell

    merged = [
        _merge_cluster(
            cluster,
            proposals=proposals,
            crop_size=crop_size,
            ray_vote_fraction=cfg.ray_vote_fraction,
        )
        for cluster in clusters
        if len(cluster) >= int(cfg.min_support)
    ]
    filtered = [
        vertex
        for vertex in merged
        if float(vertex.score) >= float(cfg.min_score)
        and float(vertex.support_fraction) >= float(cfg.min_support_fraction)
    ]
    return sorted(filtered, key=lambda item: (-item.support_count, -item.score, item.y, item.x))


def summarize_merge(
    raw_vertices: list[DecodedVertex],
    merged_vertices: list[MergedVertex],
) -> dict[str, Any]:
    support_counts = [vertex.support_count for vertex in merged_vertices]
    support_fractions = [vertex.support_fraction for vertex in merged_vertices]
    return {
        "raw_predictions": len(raw_vertices),
        "merged_predictions": len(merged_vertices),
        "suppressed_predictions": max(len(raw_vertices) - len(merged_vertices), 0),
        "support_count": _summary(support_counts),
        "support_fraction": _float_summary(support_fractions),
    }


def _nearest_cluster_index(
    vertex: DecodedVertex,
    centers: list[tuple[float, float]],
    clusters: list[list[DecodedVertex]],
    *,
    radius_px: float,
    boundary_radius_px: float | None,
    candidate_indices: Iterable[int] | None = None,
) -> int | None:
    best_index: int | None = None
    best_distance = float(boundary_radius_px if _is_boundary_vertex(vertex) and boundary_radius_px is not None else radius_px)
    indices = range(len(centers)) if candidate_indices is None else candidate_indices
    for index in indices:
        distance = _cluster_distance(
            vertex,
            centers[index],
            clusters[index],
            boundary_radius_px=boundary_radius_px,
        )
        if distance <= best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _cluster_distance(
    vertex: DecodedVertex,
    center: tuple[float, float],
    cluster: list[DecodedVertex],
    *,
    boundary_radius_px: float | None,
) -> float:
    center_x, center_y = center
    vertex_boundary = _is_boundary_vertex(vertex)
    cluster_side = _cluster_boundary_side(cluster)
    if vertex_boundary or cluster_side is not None:
        if not vertex_boundary or cluster_side is None or vertex.boundary_side != cluster_side:
            return float("inf")
        if vertex.boundary_side in {"top", "bottom"}:
            return abs(float(vertex.x) - float(center_x))
        return abs(float(vertex.y) - float(center_y))
    return float(np.hypot(float(vertex.x) - center_x, float(vertex.y) - center_y))


def _is_boundary_vertex(vertex: DecodedVertex) -> bool:
    return vertex.kind == "boundary_contact" and vertex.boundary_side is not None


def _cluster_boundary_side(cluster: list[DecodedVertex]) -> str | None:
    sides = [vertex.boundary_side for vertex in cluster if _is_boundary_vertex(vertex)]
    if not sides:
        return None
    return Counter(sides).most_common(1)[0][0]


def _nearby_cluster_indices(
    vertex: DecodedVertex,
    *,
    grid: dict[tuple[int, int], set[int]],
    cell_size: float,
) -> list[int]:
    cell_x, cell_y = _grid_key(float(vertex.x), float(vertex.y), cell_size=cell_size)
    indices: set[int] = set()
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            indices.update(grid.get((cell_x + dx, cell_y + dy), ()))
    return sorted(indices)


def _grid_key(x: float, y: float, *, cell_size: float) -> tuple[int, int]:
    return (int(np.floor(float(x) / cell_size)), int(np.floor(float(y) / cell_size)))


def _weighted_center(cluster: list[DecodedVertex]) -> tuple[float, float]:
    weights = np.asarray([max(float(vertex.score), 1e-4) for vertex in cluster], dtype=np.float64)
    xs = np.asarray([float(vertex.x) for vertex in cluster], dtype=np.float64)
    ys = np.asarray([float(vertex.y) for vertex in cluster], dtype=np.float64)
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def _merge_cluster(
    cluster: list[DecodedVertex],
    *,
    proposals: list[VertexProposal] | None,
    crop_size: int,
    ray_vote_fraction: float,
) -> MergedVertex:
    x, y = _weighted_center(cluster)
    score = max(float(vertex.score) for vertex in cluster)
    kind_id = _weighted_mode((vertex.kind_id, vertex.score) for vertex in cluster)
    degree_class = _weighted_mode((vertex.degree_class, vertex.score) for vertex in cluster)
    boundary_side = _weighted_side_mode(cluster)
    boundary_side_id = (
        BOUNDARY_SIDE_NAMES.index(boundary_side) if boundary_side in BOUNDARY_SIDE_NAMES else None
    )
    ray_bins = _ray_vote(cluster, ray_vote_fraction=ray_vote_fraction)
    distances = [float(np.hypot(float(vertex.x) - x, float(vertex.y) - y)) for vertex in cluster]
    possible_support_count = _possible_support_count(
        x,
        y,
        proposals=proposals,
        crop_size=crop_size,
    )
    support_fraction = min(
        1.0,
        float(len(cluster)) / max(float(possible_support_count), 1.0),
    )
    return MergedVertex(
        x=x,
        y=y,
        score=score,
        kind_id=kind_id,
        kind=VERTEX_KIND_NAMES[kind_id],
        degree_class=degree_class,
        degree=degree_class,
        ray_bins=ray_bins,
        boundary_side_id=boundary_side_id,
        boundary_side=boundary_side,
        support_count=len(cluster),
        possible_support_count=possible_support_count,
        support_fraction=support_fraction,
        mean_member_distance_px=float(np.mean(distances)) if distances else 0.0,
        max_member_distance_px=float(np.max(distances)) if distances else 0.0,
    )


def _weighted_mode(items: Any) -> int:
    votes: dict[int, float] = defaultdict(float)
    for key, score in items:
        votes[int(key)] += max(float(score), 1e-4)
    if not votes:
        return 0
    return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]


def _weighted_side_mode(cluster: list[DecodedVertex]) -> str | None:
    votes: dict[str, float] = defaultdict(float)
    for vertex in cluster:
        if vertex.boundary_side is None:
            continue
        votes[str(vertex.boundary_side)] += max(float(vertex.score), 1e-4)
    if not votes:
        return None
    return max(votes.items(), key=lambda item: (item[1], item[0]))[0]


def _ray_vote(
    cluster: list[DecodedVertex],
    *,
    ray_vote_fraction: float,
) -> tuple[int, ...]:
    votes: Counter[int] = Counter()
    for vertex in cluster:
        votes.update(int(bin_index) for bin_index in vertex.ray_bins)
    required = max(1, int(np.ceil(float(ray_vote_fraction) * len(cluster))))
    return tuple(sorted(bin_index for bin_index, count in votes.items() if count >= required))


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(min(values)),
        "mean": float(np.mean(values)),
        "max": int(max(values)),
    }


def _float_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(np.mean(values)),
        "max": float(max(values)),
    }


def _possible_support_count(
    x: float,
    y: float,
    *,
    proposals: list[VertexProposal] | None,
    crop_size: int,
) -> int:
    if proposals is None:
        return 0
    return sum(
        _proposal_contains_xy(x, y, proposal, crop_size=crop_size)
        for proposal in proposals
    )


def _proposal_contains_xy(
    x: float,
    y: float,
    proposal: VertexProposal,
    *,
    crop_size: int,
) -> bool:
    origin_x, origin_y = crop_origin_for_center((proposal.x, proposal.y), crop_size=crop_size)
    return (
        origin_x <= float(x) < origin_x + int(crop_size)
        and origin_y <= float(y) < origin_y + int(crop_size)
    )
