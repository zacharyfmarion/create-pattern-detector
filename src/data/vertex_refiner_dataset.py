"""Crop dataset and extraction utilities for VertexRefinerV1."""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from skimage.morphology import skeletonize
except ModuleNotFoundError:  # pragma: no cover - scikit-image is part of the normal dev env.
    skeletonize = None

from src.data.cpline_dataset import (
    load_manifest_records,
    render_cpline_sample,
    resolve_fold_path,
    select_records,
)
from src.data.fold_parser import CreasePattern, FOLDParser, transform_coords
from src.data.vertex_refiner_proposals import (
    ProposalConfig,
    VertexProposal,
    crop_origin_for_center,
    generate_vertex_refiner_proposals,
    select_vertex_refiner_proposals,
)
from src.data.vertex_refiner_targets import (
    SquareFrame,
    VertexRefinerTargetConfig,
    build_vertex_refiner_targets,
    classify_vertex_kind,
    distance_to_ink_map,
    grayscale_image,
    infer_square_frame,
    source_ink_probability,
)
from src.models.vertex_refiner_contract import CROP_SIZE_PX, normalized_coord_grid

AuxiliaryJunctionMode = Literal["zero", "rendered-labels", "dense-cache"]
AUXILIARY_JUNCTION_MODES: tuple[str, ...] = ("zero", "rendered-labels", "dense-cache")
RefinerInputVersion = Literal["v1", "v2"]
REFINER_INPUT_VERSIONS: tuple[str, ...] = ("v1", "v2")


@dataclass(frozen=True)
class DenseJunctionMaps:
    junction_probability: np.ndarray
    junction_offset: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RenderedVertexRefinerSample:
    image: np.ndarray
    image_gray: np.ndarray
    source_ink_probability: np.ndarray
    source_distance_to_ink: np.ndarray
    source_skeleton: np.ndarray
    source_orientation_cos2: np.ndarray
    source_orientation_sin2: np.ndarray
    signed_distance_to_frame: np.ndarray
    frame_edge_mask: np.ndarray
    inside_paper_mask: np.ndarray
    boundary_contact_prior: np.ndarray
    cpline_junction_probability: np.ndarray
    cpline_junction_offset: np.ndarray
    pixel_vertices: np.ndarray
    edges: np.ndarray
    assignments: np.ndarray
    square_frame: SquareFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CropRef:
    record_index: int
    proposal: VertexProposal


class CropRefCacheError(ValueError):
    """Raised when a precomputed crop-ref cache does not match the dataset config."""


def render_vertex_refiner_sample(
    cp: CreasePattern,
    *,
    image_size: int,
    padding: int,
    line_width: int,
    augment_profile: str = "clean",
    seed: int | None = None,
    auxiliary_mode: AuxiliaryJunctionMode = "zero",
    dense_junction_maps: DenseJunctionMaps | None = None,
) -> RenderedVertexRefinerSample:
    """Render a CP sample and derive V1 source-image/auxiliary junction channels."""
    auxiliary_mode = validate_auxiliary_mode(auxiliary_mode)
    pixel_vertices, _ = transform_coords(cp.vertices, image_size=image_size, padding=padding)
    sample = render_cpline_sample(
        cp,
        image_size=image_size,
        padding=padding,
        line_width=line_width,
        augment_profile=augment_profile,
        seed=seed,
        base_pixel_vertices=pixel_vertices,
        junction_sigma_px=1.5,
        junction_offset_radius_px=3.0,
    )
    gray = grayscale_image(sample.image)
    source_ink = source_ink_probability(sample.image)
    source_distance = distance_to_ink_map(source_ink, normalize_by_px=CROP_SIZE_PX)
    source_skeleton = source_skeleton_mask(source_ink)
    orientation_cos2, orientation_sin2 = source_orientation_channels(source_ink)
    if auxiliary_mode == "zero":
        junction_probability = np.zeros((image_size, image_size), dtype=np.float32)
        junction_offset = np.zeros((image_size, image_size, 2), dtype=np.float32)
        dense_metadata: dict[str, Any] = {"source": "zero_source_only"}
    elif auxiliary_mode == "rendered-labels":
        junction_probability = sample.junction_heatmap
        junction_offset = sample.junction_offset
        dense_metadata = {"source": "rendered_cpline_labels_test_only"}
    else:
        if dense_junction_maps is None:
            raise ValueError("auxiliary_mode='dense-cache' requires dense_junction_maps")
        junction_probability = dense_junction_maps.junction_probability
        junction_offset = dense_junction_maps.junction_offset
        dense_metadata = dense_junction_maps.metadata
    square_frame = infer_square_frame(sample.pixel_vertices, sample.edges, sample.assignments, image_size)
    frame_maps = build_frame_feature_maps(
        source_ink_probability=source_ink,
        square_frame=square_frame,
        crop_size=CROP_SIZE_PX,
    )
    return RenderedVertexRefinerSample(
        image=sample.image,
        image_gray=gray,
        source_ink_probability=source_ink,
        source_distance_to_ink=source_distance,
        source_skeleton=source_skeleton,
        source_orientation_cos2=orientation_cos2,
        source_orientation_sin2=orientation_sin2,
        signed_distance_to_frame=frame_maps["signed_distance_to_frame"],
        frame_edge_mask=frame_maps["frame_edge_mask"],
        inside_paper_mask=frame_maps["inside_paper_mask"],
        boundary_contact_prior=frame_maps["boundary_contact_prior"],
        cpline_junction_probability=np.asarray(junction_probability, dtype=np.float32),
        cpline_junction_offset=np.asarray(junction_offset, dtype=np.float32),
        pixel_vertices=sample.pixel_vertices,
        edges=sample.edges,
        assignments=sample.assignments,
        square_frame=square_frame,
        metadata={
            "image_size": image_size,
            "augment_profile": augment_profile,
            "auxiliary_mode": auxiliary_mode,
            "render": sample.metadata,
            "dense_junction_maps": dense_metadata,
        },
    )


def build_vertex_refiner_input(
    sample: RenderedVertexRefinerSample,
    *,
    crop_origin_xy: tuple[int, int],
    crop_size: int = CROP_SIZE_PX,
    input_version: RefinerInputVersion = "v1",
) -> np.ndarray:
    """Build the versioned input tensor for a crop."""
    input_version = validate_input_version(input_version)
    x_grid, y_grid = normalized_coord_grid(crop_size)
    if input_version == "v1":
        channels = [
            crop_array(sample.image_gray, crop_origin_xy, crop_size, pad_value=1.0),
            crop_array(sample.source_ink_probability, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.source_distance_to_ink, crop_origin_xy, crop_size, pad_value=1.0),
            crop_array(sample.cpline_junction_probability, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.cpline_junction_offset[..., 0], crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.cpline_junction_offset[..., 1], crop_origin_xy, crop_size, pad_value=0.0),
            x_grid,
            y_grid,
        ]
    else:
        channels = [
            crop_array(sample.image_gray, crop_origin_xy, crop_size, pad_value=1.0),
            crop_array(sample.source_ink_probability, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.source_distance_to_ink, crop_origin_xy, crop_size, pad_value=1.0),
            crop_array(sample.source_skeleton, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.source_orientation_cos2, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.source_orientation_sin2, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.signed_distance_to_frame, crop_origin_xy, crop_size, pad_value=-1.0),
            crop_array(sample.frame_edge_mask, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.inside_paper_mask, crop_origin_xy, crop_size, pad_value=0.0),
            crop_array(sample.boundary_contact_prior, crop_origin_xy, crop_size, pad_value=0.0),
            x_grid,
            y_grid,
        ]
    return np.stack(channels, axis=0).astype(np.float32)


def extract_vertex_refiner_crop(
    sample: RenderedVertexRefinerSample,
    proposal: VertexProposal,
    *,
    crop_size: int = CROP_SIZE_PX,
    target_config: VertexRefinerTargetConfig | None = None,
    input_version: RefinerInputVersion = "v1",
) -> dict[str, Any]:
    origin = crop_origin_for_center((proposal.x, proposal.y), crop_size=crop_size)
    inputs = build_vertex_refiner_input(
        sample,
        crop_origin_xy=origin,
        crop_size=crop_size,
        input_version=input_version,
    )
    targets = build_vertex_refiner_targets(
        vertices=sample.pixel_vertices,
        edges=sample.edges,
        assignments=sample.assignments,
        crop_origin_xy=origin,
        image_size=int(sample.metadata["image_size"]),
        square_frame=sample.square_frame,
        config=target_config or VertexRefinerTargetConfig(crop_size=crop_size),
    )
    return {
        "input": inputs,
        "targets": targets,
        "proposal": proposal,
        "crop_origin_xy": origin,
        "metadata": {
            "proposal": {
                "x": proposal.x,
                "y": proposal.y,
                "score": proposal.score,
                "provenance": list(proposal.provenance),
            },
            "square_frame": square_frame_to_dict(sample.square_frame),
            "target": targets.metadata,
        },
    }


def crop_array(
    array: np.ndarray,
    crop_origin_xy: tuple[int, int],
    crop_size: int,
    *,
    pad_value: float,
) -> np.ndarray:
    """Crop a 2D array with constant padding outside the source image."""
    source = np.asarray(array, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("crop_array expects a 2D array")
    origin_x, origin_y = crop_origin_xy
    output = np.full((crop_size, crop_size), float(pad_value), dtype=np.float32)
    src_x0 = max(0, int(origin_x))
    src_y0 = max(0, int(origin_y))
    src_x1 = min(source.shape[1], int(origin_x) + crop_size)
    src_y1 = min(source.shape[0], int(origin_y) + crop_size)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output
    dst_x0 = src_x0 - int(origin_x)
    dst_y0 = src_y0 - int(origin_y)
    output[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = source[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return output


def source_skeleton_mask(
    source_ink_probability: np.ndarray,
    *,
    threshold: float = 0.25,
) -> np.ndarray:
    """Return a one-pixel-wide skeleton estimate from source ink probability."""
    ink = np.asarray(source_ink_probability, dtype=np.float32) >= float(threshold)
    if not np.any(ink):
        return np.zeros_like(source_ink_probability, dtype=np.float32)
    if skeletonize is None:
        return ink.astype(np.float32)
    return skeletonize(ink).astype(np.float32)


def source_orientation_channels(
    source_ink_probability: np.ndarray,
    *,
    threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local line tangent orientation channels from source ink."""
    ink = np.asarray(source_ink_probability, dtype=np.float32)
    if ink.ndim != 2:
        raise ValueError("source_orientation_channels expects a 2D ink map")
    if min(ink.shape) >= 3:
        smooth = cv2.GaussianBlur(ink, (3, 3), 0)
    else:
        smooth = ink
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    tangent = np.arctan2(grad_y, grad_x) + np.pi / 2.0
    support = (ink >= float(threshold)).astype(np.float32)
    cos2 = np.cos(2.0 * tangent).astype(np.float32) * support
    sin2 = np.sin(2.0 * tangent).astype(np.float32) * support
    return cos2.astype(np.float32), sin2.astype(np.float32)


def build_frame_feature_maps(
    *,
    source_ink_probability: np.ndarray,
    square_frame: SquareFrame,
    crop_size: int = CROP_SIZE_PX,
    frame_edge_band_px: float = 1.5,
    boundary_contact_band_px: float = 3.0,
) -> dict[str, np.ndarray]:
    """Build full-image frame channels used by the frame-aware V2 refiner."""
    ink = np.asarray(source_ink_probability, dtype=np.float32)
    if ink.ndim != 2:
        raise ValueError("build_frame_feature_maps expects a 2D ink map")
    height, width = ink.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    inside = (
        (cols >= float(square_frame.x_min))
        & (cols <= float(square_frame.x_max))
        & (rows >= float(square_frame.y_min))
        & (rows <= float(square_frame.y_max))
    )
    inside_distance = np.minimum.reduce(
        [
            cols - float(square_frame.x_min),
            float(square_frame.x_max) - cols,
            rows - float(square_frame.y_min),
            float(square_frame.y_max) - rows,
        ]
    )
    outside_dx = np.maximum.reduce(
        [
            float(square_frame.x_min) - cols,
            cols - float(square_frame.x_max),
            np.zeros_like(cols),
        ]
    )
    outside_dy = np.maximum.reduce(
        [
            float(square_frame.y_min) - rows,
            rows - float(square_frame.y_max),
            np.zeros_like(rows),
        ]
    )
    outside_distance = np.sqrt(outside_dx * outside_dx + outside_dy * outside_dy)
    signed = np.where(inside, inside_distance, -outside_distance)
    signed = np.clip(signed / max(float(crop_size), 1.0), -1.0, 1.0).astype(np.float32)

    edge_mask = _frame_band_mask(
        rows,
        cols,
        square_frame=square_frame,
        band_px=frame_edge_band_px,
    )
    contact_band = _frame_band_mask(
        rows,
        cols,
        square_frame=square_frame,
        band_px=boundary_contact_band_px,
    )
    return {
        "signed_distance_to_frame": signed,
        "frame_edge_mask": edge_mask.astype(np.float32),
        "inside_paper_mask": inside.astype(np.float32),
        "boundary_contact_prior": (ink * contact_band.astype(np.float32)).astype(np.float32),
    }


def _frame_band_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    square_frame: SquareFrame,
    band_px: float,
) -> np.ndarray:
    band = float(band_px)
    horizontal_span = (cols >= float(square_frame.x_min) - band) & (
        cols <= float(square_frame.x_max) + band
    )
    vertical_span = (rows >= float(square_frame.y_min) - band) & (
        rows <= float(square_frame.y_max) + band
    )
    top = (np.abs(rows - float(square_frame.y_min)) <= band) & horizontal_span
    bottom = (np.abs(rows - float(square_frame.y_max)) <= band) & horizontal_span
    left = (np.abs(cols - float(square_frame.x_min)) <= band) & vertical_span
    right = (np.abs(cols - float(square_frame.x_max)) <= band) & vertical_span
    return top | bottom | left | right


def load_dense_junction_maps(
    dense_manifest_path: str | Path,
    sample_id: str,
) -> DenseJunctionMaps:
    """Load CPLineNet junction probability and offset maps from a dense-cache manifest."""
    manifest_path = Path(dense_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = _find_dense_sample(manifest, sample_id)
    root = manifest_path.parent
    image_size = int(sample["image_size"])
    dims = sample.get("dims", {})
    junction_logits = _read_dense_f32(
        root / sample["junction_logits_f32_path"],
        dims.get("junction_logits", [1, 1, image_size, image_size]),
    )[0, 0]
    if "junction_offset_f32_path" in sample:
        offset_nchw = _read_dense_f32(
            root / sample["junction_offset_f32_path"],
            dims.get("junction_offset", [1, 2, image_size, image_size]),
        )
        junction_offset = np.transpose(offset_nchw[0], (1, 2, 0)).astype(np.float32)
    else:
        junction_offset = np.zeros((image_size, image_size, 2), dtype=np.float32)
    return DenseJunctionMaps(
        junction_probability=_sigmoid(junction_logits).astype(np.float32),
        junction_offset=junction_offset,
        metadata={
            "source": "dense_cache",
            "manifest": str(manifest_path),
            "sample_id": str(sample["id"]),
        },
    )


class VertexRefinerCropDataset(Dataset):
    """Deterministic crop dataset for VertexRefinerV1 training smoke runs."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        limit: int | None = None,
        max_edges: int | None = 1200,
        image_size: int = 1024,
        padding: int | None = None,
        line_width: int | None = None,
        augment_profile: str = "clean",
        seed: int = 0,
        proposals_per_sample: int | None = 64,
        include_gt_training_anchors: bool = False,
        boundary_gt_anchor_repeats: int = 0,
        boundary_gt_anchor_jitter_px: float = 6.0,
        auxiliary_mode: AuxiliaryJunctionMode = "zero",
        input_version: RefinerInputVersion = "v1",
        cache_rendered_samples: bool = True,
        rendered_sample_cache_size: int | None = None,
        crop_refs_path: str | Path | None = None,
        crop_ref_progress_every: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        self.padding = padding if padding is not None else max(8, int(32 * image_size / 1024))
        self.line_width = (
            line_width if line_width is not None else max(1, int(2 * image_size / 768))
        )
        self.augment_profile = augment_profile
        self.seed = int(seed)
        self.boundary_gt_anchor_repeats = int(boundary_gt_anchor_repeats)
        self.boundary_gt_anchor_jitter_px = float(boundary_gt_anchor_jitter_px)
        self.auxiliary_mode = validate_auxiliary_mode(auxiliary_mode)
        self.input_version = validate_input_version(input_version)
        self.cache_rendered_samples = bool(cache_rendered_samples)
        self.rendered_sample_cache_size = (
            None if rendered_sample_cache_size is None else max(0, int(rendered_sample_cache_size))
        )
        self.crop_refs_path = None if crop_refs_path is None else Path(crop_refs_path)
        self.crop_ref_progress_every = max(0, int(crop_ref_progress_every))
        self._sample_cache: OrderedDict[int, RenderedVertexRefinerSample] = OrderedDict()
        if self.auxiliary_mode == "dense-cache":
            raise NotImplementedError(
                "VertexRefinerCropDataset does not yet wire dense-cache lookup per record; "
                "use auxiliary_mode='zero' for source-only runs or "
                "'rendered-labels' for explicit label-leakage tests."
            )
        self.parser = FOLDParser()
        self.records = select_records(
            load_manifest_records(self.manifest_path),
            split=split,
            limit=limit,
            max_edges=max_edges,
            seed=seed,
        )
        if not self.records:
            raise ValueError(f"No records selected from {manifest_path} for split={split}")
        self.crop_ref_config = vertex_refiner_crop_ref_config(
            split=split,
            limit=limit,
            max_edges=max_edges,
            image_size=self.image_size,
            padding=self.padding,
            line_width=self.line_width,
            augment_profile=self.augment_profile,
            seed=self.seed,
            proposals_per_sample=proposals_per_sample,
            include_gt_training_anchors=include_gt_training_anchors,
            boundary_gt_anchor_repeats=self.boundary_gt_anchor_repeats,
            boundary_gt_anchor_jitter_px=self.boundary_gt_anchor_jitter_px,
            auxiliary_mode=self.auxiliary_mode,
            input_version=self.input_version,
        )
        if self.crop_refs_path is None:
            self.crop_refs = self._build_crop_refs(
                proposals_per_sample=proposals_per_sample,
                include_gt_training_anchors=include_gt_training_anchors,
                boundary_gt_anchor_repeats=self.boundary_gt_anchor_repeats,
                boundary_gt_anchor_jitter_px=self.boundary_gt_anchor_jitter_px,
            )
            self.crop_refs_source = "generated"
        else:
            self.crop_refs = load_vertex_refiner_crop_refs(
                self.crop_refs_path,
                expected_config=self.crop_ref_config,
                records=self.records,
            )
            self.crop_refs_source = self.crop_refs_path.as_posix()

    def __len__(self) -> int:
        return len(self.crop_refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        crop_ref = self.crop_refs[index]
        sample = self._render_record(crop_ref.record_index)
        crop = extract_vertex_refiner_crop(
            sample,
            crop_ref.proposal,
            input_version=self.input_version,
        )
        targets = crop["targets"]
        return {
            "input": torch.from_numpy(crop["input"]).float(),
            "vertex_heatmap": torch.from_numpy(targets.vertex_heatmap).unsqueeze(0).float(),
            "boundary_contact_heatmap": torch.from_numpy(
                targets.boundary_contact_heatmap
            ).unsqueeze(0).float(),
            "vertex_offset": torch.from_numpy(targets.vertex_offset).permute(2, 0, 1).float(),
            "vertex_offset_mask": torch.from_numpy(targets.vertex_offset_mask).bool(),
            "vertex_kind": torch.from_numpy(targets.vertex_kind).long(),
            "vertex_kind_mask": torch.from_numpy(targets.vertex_kind_mask).bool(),
            "boundary_side": torch.from_numpy(targets.boundary_side).long(),
            "boundary_side_mask": torch.from_numpy(targets.boundary_side_mask).bool(),
            "degree": torch.from_numpy(targets.degree).long(),
            "degree_mask": torch.from_numpy(targets.degree_mask).bool(),
            "incident_rays": torch.from_numpy(targets.incident_rays).float(),
            "incident_ray_mask": torch.from_numpy(targets.incident_ray_mask).bool(),
            "local_vertices": torch.from_numpy(targets.local_vertices).float(),
            "vertex_indices": torch.from_numpy(targets.vertex_indices).long(),
            "meta": {
                "record_id": str(self.records[crop_ref.record_index]["id"]),
                "crop_origin_xy": crop["crop_origin_xy"],
                **crop["metadata"],
            },
        }

    def _build_crop_refs(
        self,
        *,
        proposals_per_sample: int | None,
        include_gt_training_anchors: bool,
        boundary_gt_anchor_repeats: int,
        boundary_gt_anchor_jitter_px: float,
    ) -> list[CropRef]:
        refs: list[CropRef] = []
        start = perf_counter()
        total = len(self.records)
        for record_index in range(len(self.records)):
            sample = self._render_record(record_index)
            proposals = generate_vertex_refiner_proposals(
                source_ink_probability=sample.source_ink_probability,
                junction_probability=sample.cpline_junction_probability,
                junction_offset=sample.cpline_junction_offset,
                square_frame=sample.square_frame,
                gt_vertices=sample.pixel_vertices,
                include_gt_training_anchors=include_gt_training_anchors,
                config=ProposalConfig(crop_size=CROP_SIZE_PX),
            )
            proposals = select_vertex_refiner_proposals(
                proposals,
                max_count=proposals_per_sample,
                crop_size=CROP_SIZE_PX,
                image_shape=sample.source_ink_probability.shape,
            )
            refs.extend(CropRef(record_index=record_index, proposal=proposal) for proposal in proposals)
            if include_gt_training_anchors and boundary_gt_anchor_repeats > 0:
                refs.extend(
                    CropRef(record_index=record_index, proposal=proposal)
                    for proposal in boundary_gt_jitter_anchor_proposals(
                        sample,
                        repeats=boundary_gt_anchor_repeats,
                        jitter_px=boundary_gt_anchor_jitter_px,
                        seed=self.seed + 100_000 + record_index,
                    )
                )
            if self.crop_ref_progress_every > 0 and (
                record_index == 0
                or (record_index + 1) % self.crop_ref_progress_every == 0
                or record_index + 1 == total
            ):
                elapsed = perf_counter() - start
                print(
                    json.dumps(
                        {
                            "event": "vertex_refiner_crop_refs_progress",
                            "records_done": record_index + 1,
                            "records_total": total,
                            "crop_refs": len(refs),
                            "elapsed_seconds": elapsed,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        if not refs:
            raise ValueError("No vertex-refiner crop proposals were generated")
        return refs

    def _render_record(self, record_index: int) -> RenderedVertexRefinerSample:
        if self.cache_rendered_samples and record_index in self._sample_cache:
            sample = self._sample_cache.pop(record_index)
            self._sample_cache[record_index] = sample
            return sample
        record = self.records[record_index]
        cp = self.parser.parse(resolve_fold_path(record, self.manifest_path))
        sample = render_vertex_refiner_sample(
            cp,
            image_size=self.image_size,
            padding=self.padding,
            line_width=self.line_width,
            augment_profile=self.augment_profile,
            seed=self.seed + record_index,
            auxiliary_mode=self.auxiliary_mode,
        )
        if self.cache_rendered_samples and self.rendered_sample_cache_size != 0:
            self._sample_cache[record_index] = sample
            if self.rendered_sample_cache_size is not None:
                while len(self._sample_cache) > self.rendered_sample_cache_size:
                    self._sample_cache.popitem(last=False)
        return sample


def _find_dense_sample(manifest: dict[str, Any], sample_id: str) -> dict[str, Any]:
    for sample in manifest.get("samples", manifest.get("fixtures", [])):
        if sample.get("id") == sample_id or sample.get("source_id") == sample_id:
            return sample
    raise KeyError(f"sample_id {sample_id!r} not found in dense manifest")


def _read_dense_f32(path: Path, dims: list[int]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} float32 values, expected {expected}")
    return values.reshape(tuple(int(dim) for dim in dims))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def validate_auxiliary_mode(mode: str) -> AuxiliaryJunctionMode:
    if mode not in AUXILIARY_JUNCTION_MODES:
        expected = ", ".join(AUXILIARY_JUNCTION_MODES)
        raise ValueError(f"Unknown auxiliary_mode {mode!r}; expected one of: {expected}")
    return mode  # type: ignore[return-value]


def validate_input_version(version: str) -> RefinerInputVersion:
    if version not in REFINER_INPUT_VERSIONS:
        expected = ", ".join(REFINER_INPUT_VERSIONS)
        raise ValueError(f"Unknown input_version {version!r}; expected one of: {expected}")
    return version  # type: ignore[return-value]


def square_frame_to_dict(frame: SquareFrame) -> dict[str, float]:
    return {
        "x_min": float(frame.x_min),
        "y_min": float(frame.y_min),
        "x_max": float(frame.x_max),
        "y_max": float(frame.y_max),
    }


CROP_REF_CACHE_SCHEMA = "create-pattern-detector/vertex-refiner-crop-refs/v1"


def vertex_refiner_crop_ref_config(
    *,
    split: str,
    limit: int | None,
    max_edges: int | None,
    image_size: int,
    padding: int,
    line_width: int,
    augment_profile: str,
    seed: int,
    proposals_per_sample: int | None,
    include_gt_training_anchors: bool,
    boundary_gt_anchor_repeats: int,
    boundary_gt_anchor_jitter_px: float,
    auxiliary_mode: AuxiliaryJunctionMode,
    input_version: RefinerInputVersion,
) -> dict[str, Any]:
    """Return the config keys that make crop-ref caches valid."""
    return {
        "split": str(split),
        "limit": None if limit is None else int(limit),
        "max_edges": None if max_edges is None else int(max_edges),
        "image_size": int(image_size),
        "padding": int(padding),
        "line_width": int(line_width),
        "augment_profile": str(augment_profile),
        "seed": int(seed),
        "proposals_per_sample": None
        if proposals_per_sample is None
        else int(proposals_per_sample),
        "include_gt_training_anchors": bool(include_gt_training_anchors),
        "boundary_gt_anchor_repeats": int(boundary_gt_anchor_repeats),
        "boundary_gt_anchor_jitter_px": float(boundary_gt_anchor_jitter_px),
        "auxiliary_mode": validate_auxiliary_mode(auxiliary_mode),
        "input_version": validate_input_version(input_version),
        "crop_size": int(CROP_SIZE_PX),
    }


def save_vertex_refiner_crop_refs(
    path: str | Path,
    dataset: VertexRefinerCropDataset,
) -> None:
    """Persist selected crop refs for a dataset constructor-compatible run."""
    out = Path(path)
    payload = {
        "schema": CROP_REF_CACHE_SCHEMA,
        "config": dataset.crop_ref_config,
        "record_count": len(dataset.records),
        "crop_ref_count": len(dataset.crop_refs),
        "records": [
            {
                "record_index": index,
                "id": _record_id(record),
            }
            for index, record in enumerate(dataset.records)
        ],
        "crop_refs": [
            {
                "record_index": int(ref.record_index),
                "record_id": _record_id(dataset.records[int(ref.record_index)]),
                "proposal": vertex_proposal_to_dict(ref.proposal),
                "crop_origin_xy": list(
                    crop_origin_for_center(
                        (ref.proposal.x, ref.proposal.y),
                        crop_size=CROP_SIZE_PX,
                    )
                ),
            }
            for ref in dataset.crop_refs
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_vertex_refiner_crop_refs(
    path: str | Path,
    *,
    expected_config: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[CropRef]:
    """Load cached crop refs and verify they match the selected dataset."""
    cache_path = Path(path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != CROP_REF_CACHE_SCHEMA:
        raise CropRefCacheError(
            f"{cache_path} has schema {schema!r}, expected {CROP_REF_CACHE_SCHEMA!r}"
        )
    _verify_crop_ref_config(cache_path, payload.get("config", {}), expected_config)
    cached_records = payload.get("records")
    if not isinstance(cached_records, list):
        raise CropRefCacheError(f"{cache_path} is missing records list")
    expected_ids = [_record_id(record) for record in records]
    cached_ids = [str(item.get("id")) for item in cached_records if isinstance(item, dict)]
    if cached_ids != expected_ids:
        raise CropRefCacheError(
            f"{cache_path} selected records do not match current dataset selection"
        )

    refs: list[CropRef] = []
    crop_refs = payload.get("crop_refs")
    if not isinstance(crop_refs, list):
        raise CropRefCacheError(f"{cache_path} is missing crop_refs list")
    for item_index, item in enumerate(crop_refs):
        if not isinstance(item, dict):
            raise CropRefCacheError(f"{cache_path} crop_refs[{item_index}] is not an object")
        record_index = int(item["record_index"])
        if record_index < 0 or record_index >= len(records):
            raise CropRefCacheError(
                f"{cache_path} crop_refs[{item_index}] has invalid record_index={record_index}"
            )
        record_id = str(item.get("record_id"))
        if record_id != _record_id(records[record_index]):
            raise CropRefCacheError(
                f"{cache_path} crop_refs[{item_index}] record_id mismatch: {record_id!r}"
            )
        proposal = vertex_proposal_from_dict(item["proposal"])
        cached_origin = item.get("crop_origin_xy")
        if cached_origin is not None:
            expected_origin = crop_origin_for_center(
                (proposal.x, proposal.y),
                crop_size=CROP_SIZE_PX,
            )
            if [int(value) for value in cached_origin] != list(expected_origin):
                raise CropRefCacheError(
                    f"{cache_path} crop_refs[{item_index}] crop_origin_xy is stale"
                )
        refs.append(CropRef(record_index=record_index, proposal=proposal))
    if len(refs) != int(payload.get("crop_ref_count", len(refs))):
        raise CropRefCacheError(f"{cache_path} crop_ref_count does not match crop_refs length")
    if not refs:
        raise CropRefCacheError(f"{cache_path} does not contain any crop refs")
    return refs


def vertex_proposal_to_dict(proposal: VertexProposal) -> dict[str, Any]:
    return {
        "x": float(proposal.x),
        "y": float(proposal.y),
        "score": float(proposal.score),
        "provenance": list(proposal.provenance),
    }


def vertex_proposal_from_dict(data: dict[str, Any]) -> VertexProposal:
    if not isinstance(data, dict):
        raise CropRefCacheError("proposal entry is not an object")
    provenance = data.get("provenance", [])
    if not isinstance(provenance, list):
        raise CropRefCacheError("proposal provenance is not a list")
    return VertexProposal(
        x=float(data["x"]),
        y=float(data["y"]),
        score=float(data["score"]),
        provenance=tuple(str(item) for item in provenance),
    )


def _verify_crop_ref_config(
    cache_path: Path,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if not isinstance(actual, dict):
        raise CropRefCacheError(f"{cache_path} config is not an object")
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            errors.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if errors:
        detail = "; ".join(errors)
        raise CropRefCacheError(f"{cache_path} config does not match: {detail}")


def _record_id(record: dict[str, Any]) -> str:
    return str(record["id"])


def boundary_gt_jitter_anchor_proposals(
    sample: RenderedVertexRefinerSample,
    *,
    repeats: int,
    jitter_px: float,
    seed: int,
) -> list[VertexProposal]:
    """Create training-only jittered GT anchors for boundary contacts."""
    if repeats <= 0:
        return []
    rng = np.random.default_rng(seed)
    proposals: list[VertexProposal] = []
    for vertex_index, vertex in enumerate(sample.pixel_vertices):
        kind = classify_vertex_kind(
            vertex_index,
            sample.pixel_vertices,
            sample.edges,
            sample.assignments,
            sample.square_frame,
            image_size=int(sample.metadata["image_size"]),
        )
        if kind != "boundary_contact":
            continue
        for _ in range(int(repeats)):
            jitter = rng.normal(0.0, float(jitter_px), size=2)
            proposals.append(
                VertexProposal(
                    x=float(vertex[0]) + float(jitter[0]),
                    y=float(vertex[1]) + float(jitter[1]),
                    score=1.0,
                    provenance=("boundary_gt_jitter_anchor",),
                )
            )
    return proposals


def vertex_refiner_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    stack_keys = [
        "input",
        "vertex_heatmap",
        "boundary_contact_heatmap",
        "vertex_offset",
        "vertex_offset_mask",
        "vertex_kind",
        "vertex_kind_mask",
        "boundary_side",
        "boundary_side_mask",
        "degree",
        "degree_mask",
        "incident_rays",
        "incident_ray_mask",
    ]
    result = {key: torch.stack([item[key] for item in batch]) for key in stack_keys}
    result["local_vertices"] = [item["local_vertices"] for item in batch]
    result["vertex_indices"] = [item["vertex_indices"] for item in batch]
    result["meta"] = [item["meta"] for item in batch]
    return result
