import json
import subprocess
import sys

import cv2
import numpy as np

from src.data.fold_parser import FOLDParser
from src.vectorization import (
    AttributedPlanarGraph,
    EdgeAssignmentConfig,
    PlanarGraphResult,
    QualityReportConfig,
    RepairConfig,
    assign_edges_from_logits,
    build_quality_report,
    conservative_repair,
    graph_to_fold_dict,
)


def _result(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray | None = None,
    support: np.ndarray | None = None,
    *,
    image_size: int = 32,
) -> PlanarGraphResult:
    if assignments is None:
        assignments = np.full(len(edges), 3, dtype=np.int8)
    if support is None:
        support = np.ones(len(edges), dtype=np.float32)
    return PlanarGraphResult(
        vertices_coords=(vertices / float(image_size - 1)).astype(np.float32),
        pixel_vertices=vertices.astype(np.float32),
        edges_vertices=edges.astype(np.int64),
        edges_assignment=assignments.astype(np.int8),
        edge_support=support.astype(np.float32),
        vertex_support=np.ones(len(vertices), dtype=np.float32),
    )


def _attributed(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray | None = None,
    support: np.ndarray | None = None,
    *,
    confidence: np.ndarray | None = None,
    margin: np.ndarray | None = None,
    source: list[str] | None = None,
    image_size: int = 32,
) -> AttributedPlanarGraph:
    result = _result(vertices, edges, assignments, support, image_size=image_size)
    graph = AttributedPlanarGraph.from_planar_result(result)
    if confidence is not None:
        graph.assignment_confidence = confidence.astype(np.float32)
    if margin is not None:
        graph.assignment_margin = margin.astype(np.float32)
    if source is not None:
        graph.assignment_source = source
    return graph


def _probabilities(image_size: int, class_idx: int, p0: tuple[int, int], p1: tuple[int, int]):
    probs = np.zeros((4, image_size, image_size), dtype=np.float32)
    probs[3, :, :] = 1.0
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.line(mask, p0, p1, 255, 1, lineType=cv2.LINE_AA)
    active = mask > 0
    probs[:, active] = 0.0
    probs[class_idx, active] = 1.0
    line_prob = active.astype(np.float32)
    return probs, line_prob


def test_assignment_sampler_uses_trimmed_logits_and_reports_margin():
    vertices = np.array([[0.0, 5.0], [19.0, 5.0]], dtype=np.float32)
    result = _result(vertices, np.array([[0, 1]], dtype=np.int64), image_size=20)
    probs, line_prob = _probabilities(20, 0, (0, 5), (19, 5))
    probs[:, 0:4, :] = 0.0
    probs[1, 0:4, :] = 1.0

    assignment = assign_edges_from_logits(
        result,
        probs,
        line_prob=line_prob,
        config=EdgeAssignmentConfig(endpoint_trim_fraction=0.25),
    )

    assert assignment.assignments.tolist() == [0]
    assert assignment.source == ["observed"]
    assert assignment.confidence[0] > 0.95
    assert assignment.margin[0] > 0.90
    assert assignment.sample_count[0] > 0


def test_low_confidence_mv_is_downgraded_to_unknown_unassigned():
    vertices = np.array([[0.0, 5.0], [19.0, 5.0]], dtype=np.float32)
    result = _result(vertices, np.array([[0, 1]], dtype=np.int64), image_size=20)
    probs = np.zeros((4, 20, 20), dtype=np.float32)
    probs[0, :, :] = 0.52
    probs[1, :, :] = 0.48
    line_prob = np.ones((20, 20), dtype=np.float32)

    assignment = assign_edges_from_logits(result, probs, line_prob=line_prob)

    assert assignment.assignments.tolist() == [3]
    assert assignment.source == ["unknown"]
    assert 0.51 < assignment.confidence[0] < 0.53


def test_monochrome_no_color_logits_do_not_hallucinate_mv():
    vertices = np.array([[0.0, 5.0], [19.0, 5.0]], dtype=np.float32)
    result = _result(vertices, np.array([[0, 1]], dtype=np.int64), image_size=20)
    probs, line_prob = _probabilities(20, 3, (0, 5), (19, 5))

    assignment = assign_edges_from_logits(result, probs, line_prob=line_prob)

    assert assignment.assignments.tolist() == [3]
    assert assignment.source == ["observed"]
    assert assignment.confidence[0] > 0.95


def test_conservative_repair_removes_duplicate_zero_and_weak_edges():
    vertices = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array([[0, 1], [1, 0], [2, 2], [1, 2]], dtype=np.int64),
        np.array([3, 3, 3, 3], dtype=np.int8),
        np.array([0.95, 0.80, 0.99, 0.10], dtype=np.float32),
        image_size=11,
    )

    repaired = conservative_repair(
        graph,
        config=RepairConfig(image_size=11, weak_edge_support_threshold=0.35),
    )

    assert repaired.graph.num_edges == 1
    assert repaired.graph.edges_vertices.tolist() == [[0, 1]]
    assert {action.code for action in repaired.actions} == {
        "remove_zero_length_edges",
        "remove_duplicate_edges",
        "drop_weak_edges",
    }


def test_repair_snaps_border_vertices_and_completes_supported_border():
    vertices = np.array(
        [[0.6, 0.4], [9.7, 0.3], [9.6, 9.8], [0.4, 9.5]],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        np.array([2, 2, 2], dtype=np.int8),
        np.ones(3, dtype=np.float32),
        image_size=11,
    )
    line_prob = np.zeros((11, 11), dtype=np.float32)
    cv2.line(line_prob, (0, 0), (0, 10), 1.0, 1)

    repaired = conservative_repair(
        graph,
        line_prob=line_prob,
        config=RepairConfig(image_size=11, border_completion_min_support=0.80),
    )

    assert repaired.graph.num_edges == 4
    assert repaired.graph.pixel_vertices.tolist() == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert repaired.graph.edges_assignment.tolist().count(2) == 4
    assert "complete_supported_border_edge" in {action.code for action in repaired.actions}


def test_repair_canonicalizes_inset_border_chain_without_touching_creases():
    vertices = np.array(
        [
            [1.2, 1.0],
            [5.0, 1.5],
            [9.8, 1.1],
            [10.1, 9.9],
            [1.0, 10.2],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array([[0, 2], [2, 3], [3, 4], [4, 0], [5, 1]], dtype=np.int64),
        np.array([2, 2, 2, 2, 0], dtype=np.int8),
        np.ones(5, dtype=np.float32),
        image_size=12,
    )
    line_prob = np.zeros((12, 12), dtype=np.float32)
    cv2.line(line_prob, (1, 1), (10, 1), 1.0, 1)

    repaired = conservative_repair(
        graph,
        line_prob=line_prob,
        config=RepairConfig(
            image_size=12,
            border_snap_px=0.0,
            border_canonicalization_min_support=0.60,
            border_canonicalization_snap_vertices=True,
            border_canonicalization_rebuild_chains=True,
        ),
    )

    assert "canonicalize_square_border" in {action.code for action in repaired.actions}
    assert repaired.graph.edges_assignment.tolist().count(2) == 5
    repaired_edges = {tuple(sorted(edge)) for edge in repaired.graph.edges_vertices.tolist()}
    assert (0, 1) in repaired_edges
    assert (1, 2) in repaired_edges
    assert repaired.graph.edges_assignment[repaired.graph.edges_vertices.tolist().index([5, 1])] == 0
    assert repaired.graph.pixel_vertices[1, 1] == repaired.graph.pixel_vertices[0, 1]


def test_repair_canonicalizes_border_labels_without_default_geometry_snap():
    vertices = np.array(
        [[0.4, 0.3], [10.0, 0.6], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
        np.array([3, 2, 2, 2], dtype=np.int8),
        np.ones(4, dtype=np.float32),
        image_size=11,
    )
    graph.assignment_probabilities = np.array(
        [
            [0.05, 0.05, 0.80, 0.10],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    repaired = conservative_repair(
        graph,
        config=RepairConfig(
            image_size=11,
            border_snap_px=0.0,
            reconstruct_square_border_chain=False,
        ),
    )

    assert "canonicalize_square_border" in {action.code for action in repaired.actions}
    assert repaired.graph.edges_assignment.tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(repaired.graph.pixel_vertices, vertices)
    assert repaired.max_geometry_drift_px == 0.0


def test_repair_reconstructs_square_border_chain_from_boundary_junctions():
    vertices = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [5.0, 0.7],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]],
            dtype=np.int64,
        ),
        np.array([2, 2, 2, 2, 0], dtype=np.int8),
        np.ones(5, dtype=np.float32),
        image_size=11,
    )

    repaired = conservative_repair(
        graph,
        config=RepairConfig(image_size=11, border_snap_px=0.0),
    )

    assert "reconstruct_square_border_chain" in {action.code for action in repaired.actions}
    assert repaired.graph.edges_assignment.tolist().count(2) == 5
    border_edges = {
        tuple(sorted(edge))
        for edge, assignment in zip(
            repaired.graph.edges_vertices.tolist(),
            repaired.graph.edges_assignment.tolist(),
        )
        if assignment == 2
    }
    top_mid = int(np.argmin(np.linalg.norm(repaired.graph.pixel_vertices - [5.0, 0.0], axis=1)))
    top_left = int(np.argmin(np.linalg.norm(repaired.graph.pixel_vertices - [0.0, 0.0], axis=1)))
    top_right = int(np.argmin(np.linalg.norm(repaired.graph.pixel_vertices - [10.0, 0.0], axis=1)))
    assert tuple(sorted((top_left, top_mid))) in border_edges
    assert tuple(sorted((top_mid, top_right))) in border_edges
    np.testing.assert_allclose(repaired.graph.pixel_vertices[top_mid], [5.0, 0.0])


def test_optional_assignment_inference_marks_forced_label_as_inferred():
    vertices = np.array(
        [[5.0, 5.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0], [5.0, 0.0]],
        dtype=np.float32,
    )
    graph = _attributed(
        vertices,
        np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64),
        np.array([0, 1, 0, 3], dtype=np.int8),
        np.ones(4, dtype=np.float32),
        confidence=np.ones(4, dtype=np.float32),
        margin=np.ones(4, dtype=np.float32),
        source=["observed", "observed", "observed", "unknown"],
        image_size=11,
    )

    repaired = conservative_repair(
        graph,
        config=RepairConfig(image_size=11, infer_assignments=True),
    )

    assert repaired.graph.edges_assignment.tolist() == [0, 1, 0, 0]
    assert repaired.graph.assignment_source[3] == "inferred"


def test_quality_report_status_precedence_and_constraint_warnings():
    short_graph = _attributed(
        np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([3], dtype=np.int8),
        np.ones(1, dtype=np.float32),
        confidence=np.ones(1, dtype=np.float32),
        margin=np.ones(1, dtype=np.float32),
        source=["observed"],
        image_size=32,
    )
    assert build_quality_report(short_graph).status == "outside_v1_envelope"

    ambiguous_graph = _attributed(
        np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], dtype=np.float32),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int64),
        np.array([2, 2, 2, 2, 3], dtype=np.int8),
        np.ones(5, dtype=np.float32),
        confidence=np.array([1.0, 1.0, 1.0, 1.0, 0.2], dtype=np.float32),
        margin=np.array([1.0, 1.0, 1.0, 1.0, 0.05], dtype=np.float32),
        source=["observed", "observed", "observed", "observed", "unknown"],
        image_size=32,
    )
    assert build_quality_report(ambiguous_graph).status == "ambiguous"

    invalid_graph = _attributed(
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([3], dtype=np.int8),
        np.ones(1, dtype=np.float32),
        image_size=32,
    )
    assert build_quality_report(invalid_graph).status == "failed"

    center_graph = _attributed(
        np.array(
            [
                [0.0, 0.0],
                [31.0, 0.0],
                [31.0, 31.0],
                [0.0, 31.0],
                [16.0, 16.0],
                [28.0, 16.0],
                [22.0, 26.0],
                [4.0, 16.0],
                [16.0, 4.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [4, 6], [4, 7], [4, 8]],
            dtype=np.int64,
        ),
        np.array([2, 2, 2, 2, 0, 1, 0, 1], dtype=np.int8),
        np.ones(8, dtype=np.float32),
        confidence=np.ones(8, dtype=np.float32),
        margin=np.ones(8, dtype=np.float32),
        source=["observed"] * 8,
        image_size=32,
    )
    report = build_quality_report(
        center_graph,
        config=QualityReportConfig(short_edge_warning_px=1.0, crowded_junction_px=1.0),
    )
    codes = {warning.code for warning in report.warnings}
    assert "kawasaki_residuals" in codes
    assert "maekawa_failures" in codes
    assert report.status == "ambiguous"


def test_quality_report_warns_on_dense_geometry():
    xs, ys = np.meshgrid(np.linspace(0.0, 127.0, 20), np.linspace(0.0, 127.0, 20))
    vertices = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
    edges = []
    for y in range(20):
        for x in range(19):
            idx = y * 20 + x
            edges.append([idx, idx + 1])
    for y in range(19):
        for x in range(20):
            idx = y * 20 + x
            edges.append([idx, idx + 20])
    graph = _attributed(
        vertices,
        np.asarray(edges, dtype=np.int64),
        np.full(len(edges), 3, dtype=np.int8),
        np.ones(len(edges), dtype=np.float32),
        confidence=np.ones(len(edges), dtype=np.float32),
        margin=np.ones(len(edges), dtype=np.float32),
        source=["observed"] * len(edges),
        image_size=128,
    )

    report = build_quality_report(graph)
    codes = {warning.code for warning in report.warnings}

    assert "dense_geometry" in codes
    assert report.status == "outside_v1_envelope"


def test_fold_writer_exports_required_fields_and_stage4_metadata():
    graph = _attributed(
        np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([3], dtype=np.int8),
        np.ones(1, dtype=np.float32),
        confidence=np.ones(1, dtype=np.float32),
        margin=np.ones(1, dtype=np.float32),
        source=["observed"],
        image_size=11,
    )
    report = build_quality_report(
        graph,
        config=QualityReportConfig(short_edge_warning_px=1.0, crowded_junction_px=1.0),
    )

    fold = graph_to_fold_dict(graph, report=report)
    parsed = FOLDParser().parse_dict(fold)

    assert parsed.num_edges == 1
    assert fold["file_classes"] == ["singleModel"]
    assert fold["frame_classes"] == ["creasePattern"]
    assert fold["cp_detector"]["schema"] == "cp-detector/stage4/v1"
    assert len(fold["cp_detector"]["assignment_confidence"]) == len(fold["edges_vertices"])


def test_stage4_assignment_eval_script_smoke(tmp_path):
    output_dir = tmp_path / "stage4_eval"

    subprocess.run(
        [
            sys.executable,
            "scripts/evals/eval_stage4_assignment.py",
            "--output-dir",
            str(output_dir),
            "--image-size",
            "64",
            "--profiles",
            "clean,line-style",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["profiles"]) == 2
    assert summary["profiles"][0]["assignment_accuracy"] >= 0.95
    assert (output_dir / "clean.fold").exists()
