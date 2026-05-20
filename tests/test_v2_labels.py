from __future__ import annotations

import numpy as np

from src.data.v2_labels import build_v2_label_sidecar


def test_v2_label_sidecar_marks_boundary_contacts_and_carriers() -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
            [50.0, 0.0],
            [50.0, 50.0],
            [50.0, 100.0],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 4],
            [4, 1],
            [1, 2],
            [2, 6],
            [6, 3],
            [3, 0],
            [4, 5],
            [5, 6],
        ],
        dtype=np.int64,
    )
    assignments = np.array([2, 2, 2, 2, 2, 2, 0, 0], dtype=np.int8)

    sidecar = build_v2_label_sidecar(
        sample_id="sample",
        issue="dashed_line_support",
        image_size=101,
        vertices=vertices,
        edges=edges,
        assignments=assignments,
        oracle_mask="mask.png",
        oracle_mask_kind="dashed_target_support",
    )

    vertex_types = {vertex["id"]: vertex["type"] for vertex in sidecar["vertices"]}
    assert sum(value == "corner" for value in vertex_types.values()) == 4
    assert vertex_types[4] == "boundary_contact"
    assert vertex_types[6] == "boundary_contact"
    assert vertex_types[5] == "interior_intersection"

    assert len(sidecar["carriers"]) == 1
    assert sidecar["carriers"][0]["edge_indices"] == [6, 7]
    assert sidecar["edges"][6]["line_style"] == "dashed"
    assert sidecar["render_evidence"]["target_line_mask"] == "mask.png"


def test_v2_label_sidecar_marks_artifacts_and_ambiguous_assignments() -> None:
    vertices = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [50.0, 50.0]],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 2]], dtype=np.int64)
    assignments = np.array([2, 2, 2, 2, 0, 1], dtype=np.int8)

    artifact = build_v2_label_sidecar(
        sample_id="artifact",
        issue="text_false_positive",
        image_size=101,
        vertices=vertices,
        edges=edges,
        assignments=assignments,
        oracle_mask="text_mask.png",
        oracle_mask_kind="non_crease_artifact",
    )
    assert artifact["render_evidence"]["artifact_mask"] == "text_mask.png"
    assert artifact["render_evidence"]["non_crease_regions"][0]["training_target"] == "non_crease_line"

    ambiguous = build_v2_label_sidecar(
        sample_id="ambiguous",
        issue="ambiguous_mv",
        image_size=101,
        vertices=vertices,
        edges=edges,
        assignments=assignments,
    )
    non_border = [edge for edge in ambiguous["edges"] if edge["type"] == "crease"]
    assert {edge["observed_assignment"] for edge in non_border} == {"U"}
    assert {edge["latent_assignment"] for edge in non_border} == {"M", "V"}
