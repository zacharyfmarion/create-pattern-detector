"""Shared contract for the high-resolution vertex refiner.

This module is intentionally free of Torch dependencies so dataset builders,
training, export, and product-integration tests can all import the same tensor
and manifest names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CONTRACT_SCHEMA = "create-pattern-detector/vertex-refiner-v1-contract/v1"
CHECKPOINT_MANIFEST_SCHEMA = "create-pattern-detector/vertex-refiner-checkpoint/v1"
CURRENT_POINTER_SCHEMA = "create-pattern-detector/current-vertex-refiner-pointer/v1"

CROP_SIZE_PX = 96
INPUT_CHANNEL_COUNT = 8
V2_INPUT_CHANNEL_COUNT = 12
V3_INPUT_CHANNEL_COUNT = 11
AUXILIARY_CPLINE_CHANNEL_INDICES = (3, 4, 5)
COORD_CHANNEL_RANGE = (-1.0, 1.0)

ONNX_INPUT_NAME = "refiner_input"
ONNX_MODEL_FILENAME = "model.onnx"
STABLE_ONNX_DIR = "tree-maker-rust/apps/web/public/models/cp-vertex-refiner-v1"
VERSIONED_ONNX_DIR_TEMPLATE = (
    "tree-maker-rust/apps/web/public/models/cp-vertex-refiner-v1-{model_id}"
)
CURRENT_POINTER_PATH = "artifacts/checkpoints/current-vertex-refiner-model.json"

VERTEX_KIND_NAMES = (
    "background",
    "interior_junction",
    "boundary_contact",
    "corner",
    "endpoint_or_dangling",
)
DEGREE_CLASS_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8+")
INCIDENT_RAY_BINS = 36
INCIDENT_RAY_DEGREES_PER_BIN = 360.0 / INCIDENT_RAY_BINS
BOUNDARY_SIDE_NAMES = ("top", "right", "bottom", "left")
BOUNDARY_SIDE_IDS = {name: index for index, name in enumerate(BOUNDARY_SIDE_NAMES)}


@dataclass(frozen=True)
class ChannelSpec:
    index: int
    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    description: str


INPUT_CHANNELS = (
    ChannelSpec(0, "image_gray", "float32", "Original grayscale crop, normalized to 0..1."),
    ChannelSpec(
        1,
        "source_ink_probability",
        "float32",
        "Processed source-image ink or crease-line probability, normalized to 0..1.",
    ),
    ChannelSpec(
        2,
        "source_distance_to_ink",
        "float32",
        "Distance-to-ink map in crop-local pixels, normalized by crop size unless noted.",
    ),
    ChannelSpec(
        3,
        "cpline_junction_probability",
        "float32",
        "CPLineNet junction heatmap crop. Auxiliary only; do not use as a hard gate.",
    ),
    ChannelSpec(
        4,
        "cpline_junction_offset_dx",
        "float32",
        "CPLineNet junction x offset crop in radius-normalized decoder units.",
    ),
    ChannelSpec(
        5,
        "cpline_junction_offset_dy",
        "float32",
        "CPLineNet junction y offset crop in radius-normalized decoder units.",
    ),
    ChannelSpec(
        6,
        "crop_x_normalized",
        "float32",
        "Crop-local x coordinate channel spanning -1 at the left edge to +1 at the right edge.",
    ),
    ChannelSpec(
        7,
        "crop_y_normalized",
        "float32",
        "Crop-local y coordinate channel spanning -1 at the top edge to +1 at the bottom edge.",
    ),
)

V2_INPUT_CHANNELS = (
    ChannelSpec(0, "image_gray", "float32", "Original grayscale crop, normalized to 0..1."),
    ChannelSpec(
        1,
        "source_ink_probability",
        "float32",
        "Processed source-image ink or crease-line probability, normalized to 0..1.",
    ),
    ChannelSpec(
        2,
        "source_distance_to_ink",
        "float32",
        "Distance-to-ink map in crop-local pixels, normalized by crop size unless noted.",
    ),
    ChannelSpec(3, "source_skeleton", "float32", "Binary source-ink skeleton crop."),
    ChannelSpec(
        4,
        "source_orientation_cos2",
        "float32",
        "Source line tangent orientation cos(2 theta), zero away from supported ink.",
    ),
    ChannelSpec(
        5,
        "source_orientation_sin2",
        "float32",
        "Source line tangent orientation sin(2 theta), zero away from supported ink.",
    ),
    ChannelSpec(
        6,
        "signed_distance_to_frame",
        "float32",
        "Signed distance to the square paper frame, clipped and normalized by crop size.",
    ),
    ChannelSpec(7, "frame_edge_mask", "float32", "Mask for pixels close to the square frame."),
    ChannelSpec(8, "inside_paper_mask", "float32", "Mask for pixels inside the known paper frame."),
    ChannelSpec(
        9,
        "boundary_contact_prior",
        "float32",
        "Source ink probability restricted to a narrow band around the square frame.",
    ),
    ChannelSpec(
        10,
        "crop_x_normalized",
        "float32",
        "Crop-local x coordinate channel spanning -1 at the left edge to +1 at the right edge.",
    ),
    ChannelSpec(
        11,
        "crop_y_normalized",
        "float32",
        "Crop-local y coordinate channel spanning -1 at the top edge to +1 at the bottom edge.",
    ),
)

V3_INPUT_CHANNELS = (
    ChannelSpec(0, "image_gray", "float32", "Original grayscale crop, normalized to 0..1."),
    ChannelSpec(
        1,
        "source_ink_probability",
        "float32",
        "Processed source-image ink or crease-line probability, normalized to 0..1.",
    ),
    ChannelSpec(
        2,
        "source_distance_to_ink",
        "float32",
        "Distance-to-ink map in crop-local pixels, normalized by crop size unless noted.",
    ),
    ChannelSpec(
        3,
        "source_orientation_cos2",
        "float32",
        "Source line tangent orientation cos(2 theta), zero away from supported ink.",
    ),
    ChannelSpec(
        4,
        "source_orientation_sin2",
        "float32",
        "Source line tangent orientation sin(2 theta), zero away from supported ink.",
    ),
    ChannelSpec(
        5,
        "signed_distance_to_frame",
        "float32",
        "Signed distance to the square paper frame, clipped and normalized by crop size.",
    ),
    ChannelSpec(6, "frame_edge_mask", "float32", "Mask for pixels close to the square frame."),
    ChannelSpec(7, "inside_paper_mask", "float32", "Mask for pixels inside the known paper frame."),
    ChannelSpec(
        8,
        "boundary_contact_prior",
        "float32",
        "Source ink probability restricted to a narrow band around the square frame.",
    ),
    ChannelSpec(
        9,
        "crop_x_normalized",
        "float32",
        "Crop-local x coordinate channel spanning -1 at the left edge to +1 at the right edge.",
    ),
    ChannelSpec(
        10,
        "crop_y_normalized",
        "float32",
        "Crop-local y coordinate channel spanning -1 at the top edge to +1 at the bottom edge.",
    ),
)

OUTPUT_SPECS = (
    TensorSpec(
        "vertex_heatmap",
        (1, CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Vertex confidence logits or probabilities at crop resolution.",
    ),
    TensorSpec(
        "vertex_offset",
        (2, CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Subpixel dx, dy offsets in crop pixels, sampled at heatmap peaks.",
    ),
    TensorSpec(
        "vertex_kind",
        (len(VERTEX_KIND_NAMES), CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Per-kind logits ordered by VERTEX_KIND_NAMES.",
    ),
    TensorSpec(
        "degree",
        (len(DEGREE_CLASS_NAMES), CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Per-degree logits ordered as 0, 1, ..., 7, 8+.",
    ),
    TensorSpec(
        "incident_rays",
        (INCIDENT_RAY_BINS, CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Multi-label incident ray logits in image-coordinate clockwise angle bins.",
    ),
)
ONNX_OUTPUT_NAMES = tuple(spec.name for spec in OUTPUT_SPECS)

V2_OUTPUT_SPECS = (
    *OUTPUT_SPECS,
    TensorSpec(
        "boundary_contact_heatmap",
        (1, CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Boundary-contact confidence logits or probabilities at crop resolution.",
    ),
    TensorSpec(
        "boundary_side",
        (len(BOUNDARY_SIDE_NAMES), CROP_SIZE_PX, CROP_SIZE_PX),
        "float32",
        "Boundary-side logits ordered by BOUNDARY_SIDE_NAMES.",
    ),
)
V2_ONNX_OUTPUT_NAMES = tuple(spec.name for spec in V2_OUTPUT_SPECS)
V3_OUTPUT_SPECS = V2_OUTPUT_SPECS
V3_ONNX_OUTPUT_NAMES = V2_ONNX_OUTPUT_NAMES


def normalized_coord_grid(crop_size: int = CROP_SIZE_PX) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinate channels for a square crop."""
    if crop_size < 2:
        raise ValueError("crop_size must be at least 2")
    coords = np.linspace(
        COORD_CHANNEL_RANGE[0],
        COORD_CHANNEL_RANGE[1],
        crop_size,
        dtype=np.float32,
    )
    x_grid = np.tile(coords[None, :], (crop_size, 1))
    y_grid = np.tile(coords[:, None], (1, crop_size))
    return x_grid, y_grid


def ray_angle_degrees(dx: float, dy: float) -> float:
    """Return the image-coordinate clockwise angle for a ray delta.

    The image coordinate system is x right, y down: east is 0 degrees, south is
    90, west is 180, and north is 270.
    """
    if math.hypot(dx, dy) <= 1e-8:
        raise ValueError("ray delta must be non-zero")
    return math.degrees(math.atan2(dy, dx)) % 360.0


def ray_bin_for_delta(dx: float, dy: float) -> int:
    """Map an incident edge direction to the nearest 10-degree ray bin."""
    angle = ray_angle_degrees(dx, dy)
    return int(round(angle / INCIDENT_RAY_DEGREES_PER_BIN)) % INCIDENT_RAY_BINS


def opposite_ray_bin(bin_index: int) -> int:
    """Return the bin pointing in the opposite direction."""
    if not 0 <= bin_index < INCIDENT_RAY_BINS:
        raise ValueError(f"bin_index must be in [0, {INCIDENT_RAY_BINS})")
    return (bin_index + INCIDENT_RAY_BINS // 2) % INCIDENT_RAY_BINS


def degree_class_for_count(degree: int) -> int:
    """Return the degree class index, capping all degrees >= 8 into 8+."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return min(int(degree), len(DEGREE_CLASS_NAMES) - 1)


def image_point_from_crop_peak(
    crop_origin_xy: tuple[float, float],
    *,
    col: float,
    row: float,
    offset_dx: float = 0.0,
    offset_dy: float = 0.0,
) -> tuple[float, float]:
    """Map a crop heatmap peak back to full-image pixel coordinates."""
    origin_x, origin_y = crop_origin_xy
    return (
        float(origin_x) + float(col) + float(offset_dx),
        float(origin_y) + float(row) + float(offset_dy),
    )


def versioned_onnx_dir(model_id: str) -> str:
    """Return the versioned downstream ONNX directory for a refiner model id."""
    if not model_id:
        raise ValueError("model_id must be non-empty")
    return VERSIONED_ONNX_DIR_TEMPLATE.format(model_id=model_id)


def checkpoint_manifest_skeleton(*, model_id: str, created_at: str) -> dict[str, Any]:
    """Build the required top-level manifest shape for future checkpoints."""
    return {
        "schemaVersion": 1,
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "id": model_id,
        "status": "candidate",
        "phase": "vertex-refiner-v1",
        "registeredAt": created_at,
        "checkpoint": {
            "gitTracked": False,
            "relativePath": None,
            "sha256": None,
            "sizeBytes": None,
        },
        "model": {
            "class": "VertexRefinerV1",
            "cropSize": CROP_SIZE_PX,
            "inputChannels": [channel.name for channel in INPUT_CHANNELS],
            "outputNames": list(ONNX_OUTPUT_NAMES),
            "rayBins": INCIDENT_RAY_BINS,
        },
        "training": {},
        "inference": {
            "onnxInputName": ONNX_INPUT_NAME,
            "onnxOutputNames": list(ONNX_OUTPUT_NAMES),
            "stableOnnxManifest": f"{STABLE_ONNX_DIR}/manifest.json",
            "versionedOnnxManifest": f"{versioned_onnx_dir(model_id)}/manifest.json",
            "nmsRadiusPx": 2.0,
            "heatmapThreshold": 0.25,
            "duplicateMergeRadiusPx": 1.0,
        },
        "evaluation": {},
        "relatedDocs": [
            "implementation-plan/high-resolution-vertex-refiner-v1.md",
            "docs/vertex-refiner-v1-contract.md",
        ],
    }


def validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the Phase 0 portions of a vertex-refiner checkpoint manifest."""
    if manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ValueError(f"Expected schema {CHECKPOINT_MANIFEST_SCHEMA!r}")
    if manifest.get("schemaVersion") != 1:
        raise ValueError("Expected schemaVersion=1")
    if manifest.get("model", {}).get("class") != "VertexRefinerV1":
        raise ValueError("Expected model.class='VertexRefinerV1'")
    if manifest.get("model", {}).get("cropSize") != CROP_SIZE_PX:
        raise ValueError(f"Expected model.cropSize={CROP_SIZE_PX}")
    if tuple(manifest.get("model", {}).get("inputChannels", ())) != tuple(
        channel.name for channel in INPUT_CHANNELS
    ):
        raise ValueError("Manifest inputChannels do not match VertexRefinerV1 contract")
    inference = manifest.get("inference", {})
    if inference.get("onnxInputName") != ONNX_INPUT_NAME:
        raise ValueError(f"Expected inference.onnxInputName={ONNX_INPUT_NAME!r}")
    if tuple(inference.get("onnxOutputNames", ())) != ONNX_OUTPUT_NAMES:
        raise ValueError("Manifest onnxOutputNames do not match VertexRefinerV1 contract")
