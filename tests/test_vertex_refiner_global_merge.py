from __future__ import annotations

import numpy as np

from src.data.vertex_refiner_proposals import VertexProposal
from src.evaluation.vertex_refiner_eval import match_decoded_vertices
from src.evaluation.vertex_refiner_global_merge import (
    VertexMergeConfig,
    merge_decoded_vertices,
    summarize_merge,
)
from src.models.vertex_refiner import DecodedVertex


def test_merge_decoded_vertices_clusters_nearby_duplicates() -> None:
    vertices = [
        _vertex(10.0, 10.0, score=0.7, ray_bins=(0, 9)),
        _vertex(10.8, 9.9, score=0.9, ray_bins=(0, 18)),
        _vertex(50.0, 50.0, score=0.8, ray_bins=(4, 22)),
    ]

    merged = merge_decoded_vertices(
        vertices,
        config=VertexMergeConfig(radius_px=2.0, ray_vote_fraction=0.75),
    )

    assert len(merged) == 2
    duplicate = max(merged, key=lambda vertex: vertex.support_count)
    assert duplicate.support_count == 2
    assert np.hypot(duplicate.x - 10.45, duplicate.y - 9.95) < 0.2
    assert duplicate.score == 0.9
    assert duplicate.ray_bins == (0,)


def test_merge_decoded_vertices_can_require_multi_crop_support() -> None:
    vertices = [
        _vertex(10.0, 10.0, score=0.8),
        _vertex(10.5, 10.1, score=0.7),
        _vertex(90.0, 90.0, score=0.99),
    ]

    merged = merge_decoded_vertices(
        vertices,
        config=VertexMergeConfig(radius_px=2.0, min_support=2),
    )

    assert len(merged) == 1
    assert merged[0].support_count == 2


def test_merge_decoded_vertices_handles_clusters_crossing_grid_cells() -> None:
    vertices = [
        _vertex(2.4, 10.0, score=0.9),
        _vertex(4.8, 10.0, score=0.8),
        _vertex(5.9, 10.0, score=0.7),
        _vertex(20.0, 20.0, score=0.95),
    ]

    merged = merge_decoded_vertices(
        vertices,
        config=VertexMergeConfig(radius_px=2.5),
    )

    assert len(merged) == 2
    crossing_cluster = max(merged, key=lambda vertex: vertex.support_count)
    assert crossing_cluster.support_count == 3
    assert np.hypot(crossing_cluster.x - 4.21, crossing_cluster.y - 10.0) < 0.2


def test_merge_decoded_vertices_can_filter_by_visibility_normalized_support() -> None:
    vertices = [
        _vertex(10.0, 10.0, score=0.8),
        _vertex(80.0, 80.0, score=0.8),
    ]
    proposals = [
        VertexProposal(10.0, 10.0, 1.0, ("source_skeleton_branchpoint",)),
        VertexProposal(80.0, 80.0, 1.0, ("source_skeleton_branchpoint",)),
        VertexProposal(85.0, 80.0, 1.0, ("source_line_arrangement_intersection",)),
        VertexProposal(80.0, 85.0, 1.0, ("source_line_arrangement_intersection",)),
        VertexProposal(85.0, 85.0, 1.0, ("source_line_arrangement_intersection",)),
    ]

    merged = merge_decoded_vertices(
        vertices,
        proposals=proposals,
        crop_size=20,
        config=VertexMergeConfig(radius_px=2.0, min_support_fraction=0.5),
    )

    assert len(merged) == 1
    assert merged[0].x == 10.0
    assert merged[0].support_count == 1
    assert merged[0].possible_support_count == 1
    assert merged[0].support_fraction == 1.0


def test_merge_improves_duplicate_precision_without_losing_recall() -> None:
    gt = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
    raw = [
        _vertex(10.0, 10.0, score=0.9),
        _vertex(10.5, 10.2, score=0.8),
        _vertex(49.8, 50.1, score=0.95),
        _vertex(50.4, 49.7, score=0.7),
    ]

    raw_tp, raw_fp, raw_fn, _ = match_decoded_vertices(raw, gt, tolerance_px=1.0)
    merged = merge_decoded_vertices(raw, config=VertexMergeConfig(radius_px=2.0))
    merged_tp, merged_fp, merged_fn, _ = match_decoded_vertices(merged, gt, tolerance_px=1.0)

    assert (raw_tp, raw_fp, raw_fn) == (2, 2, 0)
    assert (merged_tp, merged_fp, merged_fn) == (2, 0, 0)
    assert summarize_merge(raw, merged)["suppressed_predictions"] == 2
    assert summarize_merge(raw, merged)["support_fraction"]["max"] == 1.0


def test_boundary_merge_uses_side_coordinate_instead_of_plain_euclidean_distance() -> None:
    vertices = [
        _vertex(10.0, 5.0, score=0.9, kind="boundary_contact", boundary_side="top"),
        _vertex(10.8, 5.0, score=0.8, kind="boundary_contact", boundary_side="top"),
        _vertex(10.2, 5.8, score=0.95, kind="boundary_contact", boundary_side="left"),
    ]

    merged = merge_decoded_vertices(
        vertices,
        config=VertexMergeConfig(radius_px=2.0, boundary_merge_radius_px=2.0),
    )

    assert len(merged) == 2
    top = [vertex for vertex in merged if vertex.boundary_side == "top"][0]
    left = [vertex for vertex in merged if vertex.boundary_side == "left"][0]
    assert top.support_count == 2
    assert left.support_count == 1


def _vertex(
    x: float,
    y: float,
    *,
    score: float,
    ray_bins: tuple[int, ...] = (),
    kind: str = "interior_junction",
    boundary_side: str | None = None,
) -> DecodedVertex:
    kind_id = 2 if kind == "boundary_contact" else 1
    return DecodedVertex(
        x=x,
        y=y,
        score=score,
        kind_id=kind_id,
        kind=kind,
        degree_class=4,
        degree=4,
        ray_bins=ray_bins,
        boundary_side_id=None if boundary_side is None else ("top", "right", "bottom", "left").index(boundary_side),
        boundary_side=boundary_side,
    )
