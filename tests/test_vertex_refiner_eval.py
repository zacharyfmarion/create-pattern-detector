from __future__ import annotations

import numpy as np

from src.evaluation.vertex_refiner_eval import match_decoded_vertices, vertex_refiner_slice_names
from src.models.vertex_refiner import DecodedVertex


def test_match_decoded_vertices_greedily_matches_with_tolerance() -> None:
    pred = [
        DecodedVertex(10.2, 10.0, 0.9, 1, "interior_junction", 4, 4, (0, 9, 18, 27)),
        DecodedVertex(30.0, 30.0, 0.8, 1, "interior_junction", 2, 2, (0, 18)),
        DecodedVertex(80.0, 80.0, 0.7, 1, "interior_junction", 2, 2, (0, 18)),
    ]
    gt = np.array([[10.0, 10.0], [29.2, 30.0], [50.0, 50.0]], dtype=np.float32)

    tp, fp, fn, errors = match_decoded_vertices(pred, gt, tolerance_px=1.0)

    assert tp == 2
    assert fp == 1
    assert fn == 1
    assert max(errors) <= 1.0


def test_vertex_refiner_slice_names_include_hard_target_buckets() -> None:
    meta = {
        "proposal": {"provenance": ["gt_training_anchor"]},
        "target": {
            "vertex_count": 3,
            "close_pair_count": 1,
            "kind_counts": {"boundary_contact": 1, "corner": 1},
        },
    }

    names = vertex_refiner_slice_names(meta)

    assert names == [
        "positive",
        "close_pair",
        "boundary_contact",
        "corner",
        "gt_training_anchor",
    ]
