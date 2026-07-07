from __future__ import annotations

import pytest

from src.models.vertex_refiner_contract import (
    AUXILIARY_CPLINE_CHANNEL_INDICES,
    BOUNDARY_SIDE_NAMES,
    CROP_SIZE_PX,
    INCIDENT_RAY_BINS,
    INPUT_CHANNELS,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
    V2_INPUT_CHANNELS,
    V2_INPUT_CHANNEL_COUNT,
    V2_ONNX_OUTPUT_NAMES,
    V3_INPUT_CHANNELS,
    V3_INPUT_CHANNEL_COUNT,
    V3_ONNX_OUTPUT_NAMES,
    VERTEX_KIND_NAMES,
    checkpoint_manifest_skeleton,
    degree_class_for_count,
    image_point_from_crop_peak,
    normalized_coord_grid,
    opposite_ray_bin,
    ray_bin_for_delta,
    validate_checkpoint_manifest,
    versioned_onnx_dir,
)


def test_vertex_refiner_input_and_output_names_are_stable() -> None:
    assert CROP_SIZE_PX == 96
    assert [channel.name for channel in INPUT_CHANNELS] == [
        "image_gray",
        "source_ink_probability",
        "source_distance_to_ink",
        "cpline_junction_probability",
        "cpline_junction_offset_dx",
        "cpline_junction_offset_dy",
        "crop_x_normalized",
        "crop_y_normalized",
    ]
    assert AUXILIARY_CPLINE_CHANNEL_INDICES == (3, 4, 5)
    assert ONNX_INPUT_NAME == "refiner_input"
    assert ONNX_OUTPUT_NAMES == (
        "vertex_heatmap",
        "vertex_offset",
        "vertex_kind",
        "degree",
        "incident_rays",
    )
    assert VERTEX_KIND_NAMES == (
        "background",
        "interior_junction",
        "boundary_contact",
        "corner",
        "endpoint_or_dangling",
    )
    assert V2_INPUT_CHANNEL_COUNT == 12
    assert [channel.name for channel in V2_INPUT_CHANNELS] == [
        "image_gray",
        "source_ink_probability",
        "source_distance_to_ink",
        "source_skeleton",
        "source_orientation_cos2",
        "source_orientation_sin2",
        "signed_distance_to_frame",
        "frame_edge_mask",
        "inside_paper_mask",
        "boundary_contact_prior",
        "crop_x_normalized",
        "crop_y_normalized",
    ]
    assert BOUNDARY_SIDE_NAMES == ("top", "right", "bottom", "left")
    assert V2_ONNX_OUTPUT_NAMES == (
        "vertex_heatmap",
        "vertex_offset",
        "vertex_kind",
        "degree",
        "incident_rays",
        "boundary_contact_heatmap",
        "boundary_side",
    )
    assert V3_INPUT_CHANNEL_COUNT == 11
    assert [channel.name for channel in V3_INPUT_CHANNELS] == [
        "image_gray",
        "source_ink_probability",
        "source_distance_to_ink",
        "source_orientation_cos2",
        "source_orientation_sin2",
        "signed_distance_to_frame",
        "frame_edge_mask",
        "inside_paper_mask",
        "boundary_contact_prior",
        "crop_x_normalized",
        "crop_y_normalized",
    ]
    assert "source_skeleton" not in [channel.name for channel in V3_INPUT_CHANNELS]
    assert V3_ONNX_OUTPUT_NAMES == V2_ONNX_OUTPUT_NAMES


def test_normalized_coord_grid_matches_crop_edges() -> None:
    x_grid, y_grid = normalized_coord_grid(crop_size=3)

    assert x_grid.shape == (3, 3)
    assert y_grid.shape == (3, 3)
    assert x_grid.tolist() == [[-1.0, 0.0, 1.0]] * 3
    assert y_grid.tolist() == [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_ray_bins_use_image_coordinate_clockwise_angles() -> None:
    assert INCIDENT_RAY_BINS == 36
    assert ray_bin_for_delta(1.0, 0.0) == 0
    assert ray_bin_for_delta(0.0, 1.0) == 9
    assert ray_bin_for_delta(-1.0, 0.0) == 18
    assert ray_bin_for_delta(0.0, -1.0) == 27
    assert opposite_ray_bin(0) == 18
    assert opposite_ray_bin(27) == 9


def test_ray_bins_reject_degenerate_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        ray_bin_for_delta(0.0, 0.0)


def test_degree_class_caps_at_eight_plus() -> None:
    assert degree_class_for_count(0) == 0
    assert degree_class_for_count(4) == 4
    assert degree_class_for_count(8) == 8
    assert degree_class_for_count(32) == 8
    with pytest.raises(ValueError, match="non-negative"):
        degree_class_for_count(-1)


def test_crop_peak_to_image_point_uses_pixel_offsets() -> None:
    assert image_point_from_crop_peak((100.0, 200.0), col=5, row=7) == (105.0, 207.0)
    assert image_point_from_crop_peak(
        (100.0, 200.0),
        col=5,
        row=7,
        offset_dx=0.25,
        offset_dy=-0.5,
    ) == (105.25, 206.5)


def test_checkpoint_manifest_skeleton_matches_contract() -> None:
    manifest = checkpoint_manifest_skeleton(
        model_id="smoke-20260622",
        created_at="2026-06-22",
    )

    validate_checkpoint_manifest(manifest)
    assert manifest["model"]["class"] == "VertexRefinerV1"
    assert manifest["model"]["cropSize"] == 96
    assert manifest["inference"]["onnxInputName"] == "refiner_input"
    assert manifest["inference"]["onnxOutputNames"] == list(ONNX_OUTPUT_NAMES)
    assert manifest["inference"]["versionedOnnxManifest"] == (
        "tree-maker-rust/apps/web/public/models/"
        "cp-vertex-refiner-v1-smoke-20260622/manifest.json"
    )
    assert versioned_onnx_dir("smoke-20260622").endswith("cp-vertex-refiner-v1-smoke-20260622")


def test_manifest_validation_rejects_contract_drift() -> None:
    manifest = checkpoint_manifest_skeleton(
        model_id="smoke-20260622",
        created_at="2026-06-22",
    )
    manifest["inference"]["onnxOutputNames"] = ["vertex_heatmap"]

    with pytest.raises(ValueError, match="onnxOutputNames"):
        validate_checkpoint_manifest(manifest)
