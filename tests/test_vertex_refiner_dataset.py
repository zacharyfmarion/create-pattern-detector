from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data.vertex_refiner_dataset import (
    CropRefCacheError,
    VertexRefinerCropDataset,
    extract_vertex_refiner_crop,
    load_dense_junction_maps,
    save_vertex_refiner_crop_refs,
)
from src.data.vertex_refiner_proposals import VertexProposal


def test_vertex_refiner_crop_dataset_is_deterministic(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    first = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=6,
    )
    second = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=6,
    )

    assert len(first) == len(second) > 0
    assert first[0]["input"].shape == (8, 96, 96)
    assert first[0]["vertex_heatmap"].shape == (1, 96, 96)
    assert first[0]["incident_rays"].shape == (36, 96, 96)
    assert np.allclose(first[0]["input"][3:6].numpy(), 0.0)
    assert first[0]["meta"]["crop_origin_xy"] == second[0]["meta"]["crop_origin_xy"]
    assert first[0]["meta"]["proposal"] == second[0]["meta"]["proposal"]


def test_vertex_refiner_dataset_can_explicitly_use_rendered_label_auxiliary_mode(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)

    dataset = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=6,
        include_gt_training_anchors=True,
        auxiliary_mode="rendered-labels",
    )

    max_aux = max(float(dataset[index]["input"][3:6].abs().max()) for index in range(len(dataset)))
    assert max_aux > 0.0


def test_vertex_refiner_v2_input_channels_are_frame_aligned(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    dataset = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=1,
        input_version="v2",
    )
    sample = dataset._render_record(0)
    crop = extract_vertex_refiner_crop(
        sample,
        VertexProposal(64.0, 8.0, 1.0, ("test_top_boundary_contact",)),
        input_version="v2",
    )

    inputs = crop["input"]
    assert inputs.shape == (12, 96, 96)
    assert inputs[7, 48, 48] > 0.5
    assert inputs[8, 44, 48] == 0.0
    assert inputs[8, 52, 48] == 1.0
    assert inputs[6, 44, 48] < 0.0
    assert inputs[6, 52, 48] > 0.0


def test_boundary_gt_jitter_anchors_oversample_boundary_contacts(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    baseline = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=2,
        include_gt_training_anchors=True,
        boundary_gt_anchor_repeats=0,
    )
    oversampled = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=2,
        include_gt_training_anchors=True,
        boundary_gt_anchor_repeats=2,
        boundary_gt_anchor_jitter_px=1.0,
    )

    extra = len(oversampled.crop_refs) - len(baseline.crop_refs)
    assert extra == 8
    assert sum(
        "boundary_gt_jitter_anchor" in ref.proposal.provenance
        for ref in oversampled.crop_refs
    ) == 8


def test_vertex_refiner_dataset_can_load_precomputed_crop_refs(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    cache_path = tmp_path / "crop-refs.json"

    generated = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=3,
        include_gt_training_anchors=True,
        boundary_gt_anchor_repeats=1,
        boundary_gt_anchor_jitter_px=1.0,
        input_version="v2",
    )
    save_vertex_refiner_crop_refs(cache_path, generated)

    cached = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=3,
        include_gt_training_anchors=True,
        boundary_gt_anchor_repeats=1,
        boundary_gt_anchor_jitter_px=1.0,
        input_version="v2",
        crop_refs_path=cache_path,
    )

    assert cached.crop_refs_source == cache_path.as_posix()
    assert len(cached.crop_refs) == len(generated.crop_refs)
    assert [
        (ref.record_index, ref.proposal.x, ref.proposal.y, ref.proposal.provenance)
        for ref in cached.crop_refs
    ] == [
        (ref.record_index, ref.proposal.x, ref.proposal.y, ref.proposal.provenance)
        for ref in generated.crop_refs
    ]
    assert cached[0]["input"].shape == (12, 96, 96)


def test_vertex_refiner_dataset_rejects_mismatched_crop_ref_cache(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    cache_path = tmp_path / "crop-refs.json"
    generated = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=3,
    )
    save_vertex_refiner_crop_refs(cache_path, generated)

    with pytest.raises(CropRefCacheError, match="config does not match"):
        VertexRefinerCropDataset(
            manifest,
            split="train",
            image_size=128,
            padding=8,
            line_width=2,
            seed=18,
            proposals_per_sample=3,
            crop_refs_path=cache_path,
        )


def test_vertex_refiner_sample_preserves_canonical_boundary_segments(tmp_path: Path) -> None:
    manifest = _write_native_like_manifest_with_boundary_overlap(tmp_path)

    dataset = VertexRefinerCropDataset(
        manifest,
        split="train",
        image_size=128,
        padding=8,
        line_width=2,
        seed=17,
        proposals_per_sample=6,
    )
    sample = dataset._render_record(0)

    assert Counter(sample.assignments.tolist())[2] > 4
    assert sample.square_frame.x_min == 8.0
    assert sample.square_frame.y_min == 8.0
    assert sample.square_frame.x_max == 120.0
    assert sample.square_frame.y_max == 120.0


def test_dense_junction_map_loader_reads_tree_maker_dense_cache_shape(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    junction_logits = np.array([[[[-8.0, 0.0], [2.0, 8.0]]]], dtype=np.float32)
    junction_offset = np.array([[[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]]]], dtype=np.float32)
    (sample_dir / "junction_logits.f32").write_bytes(junction_logits.tobytes())
    (sample_dir / "junction_offset.f32").write_bytes(junction_offset.tobytes())
    manifest = {
        "schema": "oristudio/cp-detect-dense-cache/v1",
        "samples": [
            {
                "id": "sample-a",
                "source_id": "source-a",
                "image_size": 2,
                "dims": {
                    "junction_logits": [1, 1, 2, 2],
                    "junction_offset": [1, 2, 2, 2],
                },
                "junction_logits_f32_path": "sample/junction_logits.f32",
                "junction_offset_f32_path": "sample/junction_offset.f32",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    maps = load_dense_junction_maps(manifest_path, "source-a")

    assert maps.junction_probability.shape == (2, 2)
    assert np.isclose(float(maps.junction_probability[0, 1]), 0.5)
    assert maps.junction_probability[1, 1] > 0.99
    assert maps.junction_offset.shape == (2, 2, 2)
    assert np.isclose(float(maps.junction_offset[1, 0, 0]), 0.2)
    assert np.isclose(float(maps.junction_offset[1, 0, 1]), 0.6)


def _write_manifest(tmp_path: Path) -> Path:
    folds = tmp_path / "folds"
    folds.mkdir()
    fold_path = folds / "simple.fold"
    vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1.0],
        [0.0, 0.5],
        [1.0, 0.5],
    ]
    edges = [
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
    ]
    assignments = ["B", "B", "B", "B", "B", "B", "B", "B", "M", "M", "V", "V"]
    fold_path.write_text(
        json.dumps(
            {
                "file_spec": 1.1,
                "vertices_coords": vertices,
                "edges_vertices": edges,
                "edges_assignment": assignments,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "raw-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "simple",
                "split": "train",
                "foldPath": "folds/simple.fold",
                "edges": len(edges),
                "family": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_native_like_manifest_with_boundary_overlap(tmp_path: Path) -> Path:
    folds = tmp_path / "folds"
    folds.mkdir()
    fold_path = folds / "native-like.fold"
    vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.50, 0.0],
        [0.50, 0.50],
        [0.75, 0.0],
    ]
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 7],
        [5, 6],
    ]
    assignments = ["B", "B", "B", "B", "U", "U", "U"]
    fold_path.write_text(
        json.dumps(
            {
                "file_spec": 1.1,
                "vertices_coords": vertices,
                "edges_vertices": edges,
                "edges_assignment": assignments,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "raw-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "native-like",
                "split": "train",
                "foldPath": "folds/native-like.fold",
                "edges": len(edges),
                "family": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
