from __future__ import annotations

import numpy as np

from src.vectorization.diagnostics import (
    build_stage4_diagnostic_payload,
    compute_what_if_statuses,
)


def test_stage4_diagnostics_classifies_missing_and_extra_edges() -> None:
    gt_vertices = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
    gt_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    gt_assignments = np.array([0, 1], dtype=np.int8)
    pred_vertices = np.array([[0.5, 0], [10.2, 0.1], [0, 10]], dtype=np.float32)
    pred_edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
    pred_assignments = np.array([0, 3], dtype=np.int8)

    payload = build_stage4_diagnostic_payload(
        row={
            "id": "example",
            "profile": "clean",
            "sample_index": 0,
            "status": "ambiguous",
        },
        image_url="/api/assets/example.png",
        image_size=16,
        gt_vertices=gt_vertices,
        gt_edges=gt_edges,
        gt_assignments=gt_assignments,
        pred_vertices=pred_vertices,
        pred_edges=pred_edges,
        pred_assignments=pred_assignments,
        pred_edge_support=np.array([0.9, 0.8], dtype=np.float32),
        pred_assignment_confidence=np.array([0.95, 0.4], dtype=np.float32),
        pred_assignment_margin=np.array([0.8, 0.05], dtype=np.float32),
        pred_assignment_source=["observed", "unknown"],
        report={
            "status": "ambiguous",
            "warnings": [
                {
                    "code": "unknown_assignments",
                    "message": "unknown",
                    "severity": "warning",
                    "edge_indices": [1],
                    "vertex_indices": [],
                    "details": {},
                }
            ],
            "repair_actions": [],
            "structural_validity": {"parseable_fold": True, "valid": True},
        },
        metrics={"edge_recall": 0.5, "structural_validity": {"valid": True}},
        vertex_tolerance_px=1.0,
    )

    pred_edges_payload = payload["graph"]["prediction"]["edges"]
    gt_edges_payload = payload["graph"]["groundTruth"]["edges"]
    assert pred_edges_payload[0]["match"]["state"] == "matched"
    assert pred_edges_payload[1]["match"]["state"] == "extra"
    assert "extra_predicted_edge" in pred_edges_payload[1]["issues"]
    assert "unknown_assignments" in pred_edges_payload[1]["issues"]
    assert gt_edges_payload[1]["match"]["state"] == "missing"
    assert "missing_gt_edge" in gt_edges_payload[1]["issues"]


def test_stage4_diagnostics_serializes_v2_evidence_rasters() -> None:
    vertices = np.array([[0, 0], [10, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int64)
    assignments = np.array([0], dtype=np.int8)
    line_style_prob = np.zeros((4, 4, 4), dtype=np.float32)
    line_style_prob[..., 0] = 0.1
    line_style_prob[..., 1] = 0.9
    assignment_labels = np.zeros((4, 4), dtype=np.uint8)
    assignment_labels[1:3, 1:3] = 2

    payload = build_stage4_diagnostic_payload(
        row={"id": "example", "profile": "v2-dashed", "sample_index": 0, "status": "valid"},
        image_url="/api/assets/example.png",
        image_size=16,
        gt_vertices=vertices,
        gt_edges=edges,
        gt_assignments=assignments,
        pred_vertices=vertices,
        pred_edges=edges,
        pred_assignments=assignments,
        pred_edge_support=np.array([0.9], dtype=np.float32),
        pred_assignment_confidence=np.array([0.95], dtype=np.float32),
        pred_assignment_margin=np.array([0.8], dtype=np.float32),
        pred_assignment_source=["observed"],
        report={"status": "valid", "warnings": [], "repair_actions": [], "structural_validity": {}},
        metrics={"edge_recall": 1.0},
        vertex_tolerance_px=1.0,
        line_prob=np.ones((4, 4), dtype=np.float32),
        junction_heatmap=np.eye(4, dtype=np.float32),
        boundary_contact_heatmap=np.ones((4, 4), dtype=np.float32) * 0.5,
        non_crease_prob=np.zeros((4, 4), dtype=np.float32),
        line_style_prob=line_style_prob,
        assignment_labels=assignment_labels,
    )

    evidence = payload["evidence"]
    assert evidence["lineProb"]["kind"] == "float"
    assert evidence["lineProb"]["width"] == 4
    assert evidence["lineStyle"]["kind"] == "class"
    assert evidence["lineStyle"]["labels"] == ["solid", "dashed", "faint", "monochrome"]
    assert set(evidence["lineStyle"]["values"]) == {1}
    assert evidence["assignmentLabels"]["labels"] == ["none", "M", "V", "B", "U"]


def test_stage4_what_if_statuses_can_ignore_origami_diagnostics() -> None:
    statuses = compute_what_if_statuses(
        [
            {"code": "even_degree_failures"},
            {"code": "kawasaki_residuals"},
        ],
        [],
        structural_validity={"parseable_fold": True},
        edge_count=4,
    )

    assert statuses["current"] == "ambiguous"
    assert statuses["ignoreOrigamiDiagnostics"] == "valid"
    assert statuses["structuralOnly"] == "valid"


def test_stage4_what_if_preserves_failed_precedence() -> None:
    statuses = compute_what_if_statuses(
        [
            {"code": "illegal_crossings"},
            {"code": "kawasaki_residuals"},
        ],
        [],
        structural_validity={"parseable_fold": True},
        edge_count=4,
    )

    assert statuses["current"] == "failed"
    assert statuses["ignoreOrigamiDiagnostics"] == "failed"
    assert statuses["structuralOnly"] == "failed"
