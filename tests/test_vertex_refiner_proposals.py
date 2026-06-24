from __future__ import annotations

import cv2
import numpy as np

from src.data.vertex_refiner_proposals import (
    ProposalConfig,
    VertexProposal,
    boundary_contact_proposals,
    generate_vertex_refiner_proposals,
    junction_peak_proposals,
    line_arrangement_intersection_proposals,
    merge_proposals,
    select_vertex_refiner_proposals,
    skeleton_node_proposals,
    square_frame_corner_proposals,
)
from src.data.vertex_refiner_targets import SquareFrame


def test_junction_peak_proposals_apply_offsets() -> None:
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[10, 12] = 0.9
    offset = np.zeros((32, 32, 2), dtype=np.float32)
    offset[10, 12] = [0.5, -0.25]

    proposals = junction_peak_proposals(heatmap, junction_offset=offset, threshold=0.2)

    assert len(proposals) == 1
    assert np.isclose(proposals[0].x, 12.5)
    assert np.isclose(proposals[0].y, 9.75)
    assert proposals[0].provenance == ("cpline_junction_peak",)


def test_merge_proposals_preserves_provenance() -> None:
    merged = merge_proposals(
        [
            VertexProposal(10.0, 10.0, 0.5, ("a",)),
            VertexProposal(12.0, 10.0, 0.9, ("b",)),
            VertexProposal(40.0, 40.0, 0.7, ("c",)),
        ],
        merge_radius_px=3.0,
    )

    assert len(merged) == 2
    close = min(merged, key=lambda proposal: proposal.x)
    assert close.provenance == ("a", "b")
    assert close.score == 0.9


def test_merge_proposals_returns_quality_ranked_order() -> None:
    merged = merge_proposals(
        [
            VertexProposal(5.0, 5.0, 0.1, ("source_skeleton_endpoint",)),
            VertexProposal(40.0, 40.0, 0.95, ("source_line_arrangement_intersection",)),
        ],
        merge_radius_px=3.0,
    )

    assert merged[0].x == 40.0
    assert merged[0].y == 40.0


def test_select_vertex_refiner_proposals_spreads_tied_candidates() -> None:
    proposals = [
        VertexProposal(16.0, 16.0, 1.0, ("source_line_arrangement_intersection",)),
        VertexProposal(24.0, 18.0, 1.0, ("source_line_arrangement_intersection",)),
        VertexProposal(220.0, 220.0, 1.0, ("source_line_arrangement_intersection",)),
        VertexProposal(228.0, 224.0, 1.0, ("source_line_arrangement_intersection",)),
    ]

    selected = select_vertex_refiner_proposals(
        proposals,
        max_count=2,
        crop_size=64,
        image_shape=(256, 256),
    )

    assert len(selected) == 2
    assert abs(selected[0].x - selected[1].x) > 100.0
    assert abs(selected[0].y - selected[1].y) > 100.0


def test_skeleton_node_proposals_find_branchpoints_and_endpoints() -> None:
    ink = np.zeros((64, 64), dtype=np.float32)
    cv2.line(ink, (10, 32), (54, 32), 1.0, 1)
    cv2.line(ink, (32, 10), (32, 32), 1.0, 1)

    proposals = skeleton_node_proposals(ink, threshold=0.5)

    assert any(
        "source_skeleton_branchpoint" in proposal.provenance
        and np.hypot(proposal.x - 32.0, proposal.y - 32.0) <= 2.0
        for proposal in proposals
    )
    endpoint_count = sum("source_skeleton_endpoint" in proposal.provenance for proposal in proposals)
    assert endpoint_count >= 3


def test_line_arrangement_intersection_proposals_find_supported_crossing() -> None:
    ink = np.zeros((64, 64), dtype=np.float32)
    cv2.line(ink, (8, 32), (56, 32), 1.0, 1)
    cv2.line(ink, (32, 8), (32, 56), 1.0, 1)

    proposals = line_arrangement_intersection_proposals(
        ink,
        config=ProposalConfig(
            hough_threshold=8,
            hough_min_line_length_px=10,
            source_ink_threshold=0.5,
        ),
    )

    assert any(np.hypot(proposal.x - 32.0, proposal.y - 32.0) <= 2.0 for proposal in proposals)


def test_boundary_contact_proposals_find_frame_hits() -> None:
    ink = np.zeros((64, 64), dtype=np.float32)
    cv2.line(ink, (32, 10), (32, 40), 1.0, 1)

    proposals = boundary_contact_proposals(
        ink,
        square_frame=SquareFrame(10.0, 10.0, 54.0, 54.0),
        config=ProposalConfig(source_ink_threshold=0.5, boundary_band_px=1),
    )

    assert any(
        "boundary_contact_top" in proposal.provenance
        and np.hypot(proposal.x - 32.0, proposal.y - 10.0) <= 1.5
        for proposal in proposals
    )


def test_square_frame_corner_proposals_include_all_frame_corners() -> None:
    proposals = square_frame_corner_proposals(SquareFrame(10.0, 11.0, 50.0, 51.0))

    assert [(proposal.x, proposal.y) for proposal in proposals] == [
        (10.0, 11.0),
        (50.0, 11.0),
        (50.0, 51.0),
        (10.0, 51.0),
    ]
    assert all(proposal.provenance == ("square_frame_corner",) for proposal in proposals)
    assert all(proposal.score == 1.0 for proposal in proposals)


def test_combined_proposals_include_square_frame_corners_without_ink() -> None:
    ink = np.zeros((64, 64), dtype=np.float32)

    proposals = generate_vertex_refiner_proposals(
        source_ink_probability=ink,
        square_frame=SquareFrame(10.0, 10.0, 54.0, 54.0),
    )

    corner_proposals = [
        proposal for proposal in proposals if "square_frame_corner" in proposal.provenance
    ]
    assert len(corner_proposals) == 4
    assert {(proposal.x, proposal.y) for proposal in corner_proposals} == {
        (10.0, 10.0),
        (54.0, 10.0),
        (54.0, 54.0),
        (10.0, 54.0),
    }


def test_combined_proposals_can_include_gt_training_anchors() -> None:
    ink = np.zeros((32, 32), dtype=np.float32)
    gt_vertices = np.array([[14.0, 15.0]], dtype=np.float32)

    proposals = generate_vertex_refiner_proposals(
        source_ink_probability=ink,
        gt_vertices=gt_vertices,
        include_gt_training_anchors=True,
    )

    assert len(proposals) == 1
    assert proposals[0].provenance == ("gt_training_anchor",)
