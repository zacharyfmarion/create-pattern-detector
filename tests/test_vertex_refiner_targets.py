from __future__ import annotations

import numpy as np

from src.models.vertex_refiner_contract import BOUNDARY_SIDE_IDS
from src.data.vertex_refiner_targets import (
    VERTEX_KIND_IDS,
    SquareFrame,
    VertexRefinerTargetConfig,
    build_vertex_refiner_targets,
    distance_to_ink_map,
    source_ink_probability,
)


def test_vertex_refiner_targets_label_kind_degree_and_incident_rays() -> None:
    vertices = np.array(
        [
            [8.0, 8.0],
            [88.0, 8.0],
            [88.0, 88.0],
            [8.0, 88.0],
            [48.0, 8.0],
            [48.0, 48.0],
            [48.0, 88.0],
            [8.0, 48.0],
            [88.0, 48.0],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 4],
            [4, 1],
            [1, 8],
            [8, 2],
            [2, 6],
            [6, 3],
            [3, 7],
            [7, 0],
            [4, 5],
            [5, 6],
            [7, 5],
            [5, 8],
        ],
        dtype=np.int64,
    )
    assignments = np.array([2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1], dtype=np.int8)

    targets = build_vertex_refiner_targets(
        vertices=vertices,
        edges=edges,
        assignments=assignments,
        crop_origin_xy=(0, 0),
        image_size=96,
        square_frame=SquareFrame(8.0, 8.0, 88.0, 88.0),
    )

    assert targets.vertex_kind[48, 48] == VERTEX_KIND_IDS["interior_junction"]
    assert targets.degree[48, 48] == 4
    assert np.flatnonzero(targets.incident_rays[:, 48, 48]).tolist() == [0, 9, 18, 27]

    assert targets.vertex_kind[8, 48] == VERTEX_KIND_IDS["boundary_contact"]
    assert targets.degree[8, 48] == 3
    assert targets.boundary_contact_heatmap[8, 48] == 1.0
    assert targets.boundary_side_mask[8, 48]
    assert targets.boundary_side[8, 48] == BOUNDARY_SIDE_IDS["top"]
    assert targets.vertex_kind[8, 8] == VERTEX_KIND_IDS["corner"]
    assert targets.metadata["kind_counts"]["interior_junction"] == 1
    assert targets.metadata["kind_counts"]["boundary_contact"] == 4


def test_close_pair_targets_preserve_two_peaks_and_record_close_pair() -> None:
    vertices = np.array([[30.0, 32.0], [35.0, 32.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int64)
    assignments = np.array([0], dtype=np.int8)

    targets = build_vertex_refiner_targets(
        vertices=vertices,
        edges=edges,
        assignments=assignments,
        crop_origin_xy=(0, 0),
        image_size=64,
        square_frame=SquareFrame(0.0, 0.0, 63.0, 63.0),
        config=VertexRefinerTargetConfig(crop_size=64, heatmap_sigma_px=1.0),
    )

    assert targets.vertex_heatmap[32, 30] == 1.0
    assert targets.vertex_heatmap[32, 35] == 1.0
    assert targets.vertex_heatmap[32, 32] < 0.2
    assert targets.metadata["close_pair_count"] == 1


def test_source_ink_probability_uses_source_pixels_for_light_and_dark_inputs() -> None:
    light = np.full((32, 32, 3), 255, dtype=np.uint8)
    light[16, 4:28] = 0
    light_prob = source_ink_probability(light)
    assert float(light_prob[16, 16]) > 0.8
    assert float(light_prob[2, 2]) < 0.1

    dark = np.full((32, 32, 3), 20, dtype=np.uint8)
    dark[16, 4:28] = 235
    dark_prob = source_ink_probability(dark)
    assert float(dark_prob[16, 16]) > 0.8
    assert float(dark_prob[2, 2]) < 0.1


def test_distance_to_ink_map_is_zero_on_ink_and_grows_away() -> None:
    ink = np.zeros((32, 32), dtype=np.float32)
    ink[8, :] = 1.0
    distance = distance_to_ink_map(ink, ink_threshold=0.5, normalize_by_px=32)

    assert distance[8, 16] == 0.0
    assert distance[20, 16] > distance[12, 16] > 0.0
    assert distance.max() <= 1.0
