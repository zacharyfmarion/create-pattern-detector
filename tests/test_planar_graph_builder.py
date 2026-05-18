import numpy as np

from src.data.fold_parser import CreasePattern
from src.vectorization import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    PlanarGraphResult,
    VectorizerEvidence,
    render_vectorizer_evidence,
)
from src.vectorization.metrics import evaluate_graph, metrics_from_results, validate_structure


def simple_square_with_diagonals() -> CreasePattern:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
        ],
        dtype=np.int64,
    )
    assignments = np.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int8)
    return CreasePattern(vertices=vertices, edges=edges, assignments=assignments)


def test_planar_graph_builder_recovers_simple_rendered_labels():
    rendered = render_vectorizer_evidence(
        simple_square_with_diagonals(),
        image_size=256,
        padding=16,
        line_width=2,
        junction_sigma=2.0,
    )
    builder = PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            image_size=256,
            hough_threshold=12,
            hough_min_line_length=10,
            hough_max_line_gap=4,
            junction_threshold=0.15,
            min_edge_support=0.5,
        )
    )

    result = builder.build(rendered.evidence)
    metrics = evaluate_graph(
        result,
        gt_vertices=rendered.pixel_vertices,
        gt_edges=rendered.edges,
        gt_assignments=rendered.assignments,
        vertex_tolerance_px=5.0,
    )

    assert result.num_vertices >= 5
    assert result.num_edges >= 4
    assert metrics.vertex_recall >= 0.8
    assert metrics.structural_validity.parseable_fold


def test_graph_metrics_report_border_precision_and_recall():
    vertices = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    gt_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    pred_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int64)
    result = PlanarGraphResult(
        vertices_coords=vertices,
        pixel_vertices=vertices,
        edges_vertices=pred_edges,
        edges_assignment=np.array([2, 2, 2, 2, 2], dtype=np.int8),
        edge_support=np.ones(len(pred_edges), dtype=np.float32),
        vertex_support=np.ones(len(vertices), dtype=np.float32),
    )

    metrics = evaluate_graph(
        result,
        gt_vertices=vertices,
        gt_edges=gt_edges,
        gt_assignments=np.array([2, 2, 2, 2], dtype=np.int8),
        vertex_tolerance_px=1.0,
    )
    summary = metrics_from_results([metrics])

    assert metrics.border_precision == 0.8
    assert metrics.border_recall == 1.0
    assert metrics.edge_by_class["B"]["geometry_recall"] == 1.0
    assert summary["border_precision"] == 0.8
    assert summary["border_recall"] == 1.0


def test_rendered_evidence_canonicalizes_metric_edges():
    cp = CreasePattern(
        vertices=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32),
        edges=np.array([[0, 2], [0, 1], [1, 2]], dtype=np.int64),
        assignments=np.array([2, 3, 3], dtype=np.int8),
    )

    rendered = render_vectorizer_evidence(cp, image_size=128, padding=8)

    assert {tuple(edge) for edge in rendered.edges.tolist()} == {(0, 1), (1, 2)}
    assert rendered.assignments.tolist() == [3, 3]


def test_planar_cleanup_splits_edges_at_intermediate_vertices():
    vertices = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.2]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int64)
    support = np.array([0.95], dtype=np.float32)
    assignments = np.array([3], dtype=np.int8)
    builder = PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            min_edge_length_px=1.0,
            planar_split_vertex_distance_px=0.5,
        )
    )

    cleaned_edges, cleaned_support, cleaned_assignments, stats = builder._planar_cleanup(
        vertices,
        edges,
        support,
        assignments,
    )

    assert stats["split_edges"] == 1
    assert {tuple(edge) for edge in cleaned_edges.tolist()} == {(0, 2), (1, 2)}
    assert np.allclose(cleaned_support, 0.95)
    assert cleaned_assignments.tolist() == [3, 3]


def test_planar_cleanup_removes_weaker_crossing_edge():
    vertices = np.array(
        [[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
    support = np.array([0.95, 0.75], dtype=np.float32)
    assignments = np.array([3, 3], dtype=np.int8)
    builder = PlanarGraphBuilder(PlanarGraphBuilderConfig(min_edge_length_px=1.0))

    cleaned_edges, _, _, stats = builder._planar_cleanup(vertices, edges, support, assignments)

    assert stats["crossing_edges_removed"] == 1
    assert cleaned_edges.tolist() == [[0, 1]]


def test_collinear_contraction_merges_degree_two_chain():
    vertices = np.array([[0.0, 0.0], [5.0, 0.2], [10.0, 0.0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    support = np.array([0.95, 0.9], dtype=np.float32)
    assignments = np.array([3, 3], dtype=np.int8)
    line_prob = np.ones((16, 16), dtype=np.float32)
    builder = PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            min_edge_length_px=1.0,
            min_edge_support=0.5,
            edge_sample_step_px=1.0,
            collinear_contraction_angle_degrees=6.0,
            collinear_contraction_distance_px=0.5,
        )
    )

    contracted_edges, contracted_support, contracted_assignments, stats = (
        builder._contract_collinear_edges(
            vertices,
            edges,
            support,
            assignments,
            line_prob,
        )
    )

    assert contracted_edges.tolist() == [[0, 2]]
    assert contracted_assignments.tolist() == [3]
    assert contracted_support[0] >= 0.9
    assert stats["contracted_vertices"] == 1
    assert stats["contracted_edges_removed"] == 1


def test_collinear_contraction_preserves_assignment_changes():
    vertices = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    support = np.array([0.95, 0.95], dtype=np.float32)
    assignments = np.array([0, 1], dtype=np.int8)
    builder = PlanarGraphBuilder(PlanarGraphBuilderConfig(min_edge_length_px=1.0))

    contracted_edges, _, contracted_assignments, stats = builder._contract_collinear_edges(
        vertices,
        edges,
        support,
        assignments,
    )

    assert {tuple(edge) for edge in contracted_edges.tolist()} == {(0, 1), (1, 2)}
    assert contracted_assignments.tolist() == [0, 1]
    assert stats["contracted_vertices"] == 0


def test_collinear_contraction_preserves_branch_vertices():
    vertices = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    support = np.array([0.95, 0.95, 0.95], dtype=np.float32)
    assignments = np.array([3, 3, 3], dtype=np.int8)
    builder = PlanarGraphBuilder(PlanarGraphBuilderConfig(min_edge_length_px=1.0))

    contracted_edges, _, _, stats = builder._contract_collinear_edges(
        vertices,
        edges,
        support,
        assignments,
    )

    assert {tuple(edge) for edge in contracted_edges.tolist()} == {(0, 1), (1, 2), (1, 3)}
    assert stats["contracted_vertices"] == 0


def test_structural_validator_finds_crossing_with_spatial_index():
    base_vertices = np.array([[float(i), 0.0] for i in range(320)], dtype=np.float32)
    crossing_vertices = np.array(
        [[10.0, 10.0], [20.0, 20.0], [10.0, 20.0], [20.0, 10.0]],
        dtype=np.float32,
    )
    vertices = np.concatenate([base_vertices, crossing_vertices], axis=0)
    chain_edges = np.array([[i, i + 1] for i in range(319)], dtype=np.int64)
    crossing_edges = np.array([[320, 321], [322, 323]], dtype=np.int64)
    result = PlanarGraphBuilder().build(
        VectorizerEvidence(line_prob=np.zeros((32, 32), dtype=np.float32))
    )
    result.pixel_vertices = vertices
    result.vertices_coords = vertices
    result.edges_vertices = np.concatenate([chain_edges, crossing_edges], axis=0)
    result.edges_assignment = np.full(len(result.edges_vertices), 3, dtype=np.int8)

    structural = validate_structure(result)

    assert not structural.no_illegal_crossings
