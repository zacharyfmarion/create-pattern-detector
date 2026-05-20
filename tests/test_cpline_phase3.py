import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cpline_augmentations import (
    AUGMENT_MIXES,
    DARK_MODE_STYLE_VARIANTS,
    NON_IDENTITY_SQUARE_SYMMETRIES,
)
from src.data.cpline_dataset import (
    CplineFoldDataset,
    cpline_collate,
    render_cpline_sample,
    render_input_image,
    select_records,
)
from src.data.fold_parser import CreasePattern, FOLDParser
from src.data.v2_augmentations import V2_LINE_STYLE_IDS
from src.data.v2_boundary_targets import V2_BOUNDARY_SIDE_IDS, V2_VERTEX_TYPE_IDS
from src.models import CPLineNet
from src.models.batchnorm import model_eval_with_batchnorm_mode
from src.models.losses import CPLineLoss, CPLineLossConfig
from src.vectorization import cpline_outputs_to_evidence


def simple_unassigned_cp() -> CreasePattern:
    return CreasePattern(
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        edges=np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
            dtype=np.int64,
        ),
        assignments=np.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int8),
    )


def simple_mv_cp() -> CreasePattern:
    cp = simple_unassigned_cp()
    return CreasePattern(
        vertices=cp.vertices.copy(),
        edges=cp.edges.copy(),
        assignments=np.array([2, 2, 2, 2, 0, 1, 0, 1], dtype=np.int8),
    )


def boundary_contact_cp() -> CreasePattern:
    return CreasePattern(
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        edges=np.array(
            [[0, 4], [4, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 2]],
            dtype=np.int64,
        ),
        assignments=np.array([2, 2, 2, 2, 2, 0, 1], dtype=np.int8),
    )


def asymmetric_mv_cp() -> CreasePattern:
    return CreasePattern(
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.23, 0.36],
                [0.82, 0.58],
            ],
            dtype=np.float32,
        ),
        edges=np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 5], [5, 2], [1, 5]],
            dtype=np.int64,
        ),
        assignments=np.array([2, 2, 2, 2, 0, 1, 0, 1], dtype=np.int8),
    )


def test_batchnorm_batch_stats_eval_preserves_running_buffers():
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
    batchnorm = model[0]
    batchnorm.running_mean.fill_(3.0)
    batchnorm.running_var.fill_(2.0)
    running_mean = batchnorm.running_mean.clone()
    running_var = batchnorm.running_var.clone()
    image = torch.full((1, 3, 4, 4), 10.0)

    with torch.no_grad(), model_eval_with_batchnorm_mode(model, batchnorm_mode="batch-stats"):
        output = model(image)

    assert model.training is True
    assert torch.allclose(batchnorm.running_mean, running_mean)
    assert torch.allclose(batchnorm.running_var, running_var)
    assert torch.allclose(output.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-5)


def test_cpline_renderer_keeps_unassigned_geometry_visible():
    sample = render_cpline_sample(
        simple_unassigned_cp(),
        image_size=128,
        padding=8,
        line_width=2,
    )

    assert sample.image.shape == (128, 128, 3)
    assert sample.line_prob.max() > 0.9
    assert sample.junction_heatmap.max() == 1.0
    assert np.any(sample.assignment == 3)
    assert np.any(sample.assignment == 2)
    assert np.linalg.norm(sample.angle[sample.line_prob > 0.5], axis=1).mean() > 0.9


def test_cpline_clean_profile_matches_compatibility_renderer():
    cp = simple_mv_cp()
    sample = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    image = render_input_image(cp, image_size=128, padding=8, line_width=2, render_noise="clean")

    assert np.array_equal(sample.image, image)
    assert sample.metadata["selected_profile"] == "clean"
    assert sample.metadata["line_width"] == 2


def test_cpline_dark_mode_preserves_geometry_targets():
    cp = simple_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    dark = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="dark-mode",
        seed=11,
        style_variant="dark-bright",
        square_symmetry="identity",
    )

    assert dark.metadata["grid_enabled"] is False
    assert dark.image.mean() < clean.image.mean()
    assert not np.array_equal(dark.image, clean.image)
    assert np.array_equal(dark.line_prob, clean.line_prob)
    assert np.array_equal(dark.junction_heatmap, clean.junction_heatmap)
    assert np.array_equal(dark.assignment, clean.assignment)


def test_cpline_dark_mode_has_no_grid_background():
    cp = simple_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    dark = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="dark-mode",
        seed=12,
        style_variant="dark-muted",
        square_symmetry="identity",
    )
    background = clean.line_prob < 0.01
    changed_background = np.any(dark.image[background] != clean.image[background], axis=1)

    assert dark.metadata["grid_enabled"] is False
    assert np.count_nonzero(changed_background) > 0
    assert np.array_equal(dark.line_prob, clean.line_prob)


def test_cpline_square_symmetry_transforms_vertices_and_targets():
    cp = asymmetric_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    rotated = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="square-symmetry",
        square_symmetry="rotate90",
        seed=3,
    )
    expected_vertices = np.stack(
        [127.0 - clean.pixel_vertices[:, 1], clean.pixel_vertices[:, 0]], axis=1
    )

    assert rotated.metadata["selected_profile"] == "square-symmetry"
    assert rotated.metadata["square_symmetry"] == "rotate90"
    assert rotated.metadata["geometry_applied"] is True
    assert np.allclose(rotated.pixel_vertices, expected_vertices, atol=1e-4)
    assert np.array_equal(rotated.assignments, clean.assignments)
    assert not np.array_equal(rotated.line_prob, clean.line_prob)
    assert rotated.junction_mask.sum() == clean.junction_mask.sum()


def test_cpline_photo_light_geometric_augmentation_recomputes_targets():
    cp = simple_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    photo = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="photo-light", seed=3
    )

    assert photo.metadata["geometry_applied"] is True
    assert not np.allclose(photo.pixel_vertices, clean.pixel_vertices)
    assert not np.array_equal(photo.line_prob, clean.line_prob)
    assert photo.junction_mask.sum() == clean.junction_mask.sum()


def test_cpline_photo_dark_combines_dark_style_and_geometry():
    cp = simple_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="clean"
    )
    photo = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="photo-dark", seed=4
    )

    assert photo.metadata["selected_profile"] == "photo-dark"
    assert photo.metadata["style_variant"] in DARK_MODE_STYLE_VARIANTS
    assert photo.metadata["grid_enabled"] is False
    assert photo.metadata["geometry_applied"] is True
    assert photo.image.mean() < clean.image.mean()
    assert not np.allclose(photo.pixel_vertices, clean.pixel_vertices)
    assert not np.array_equal(photo.line_prob, clean.line_prob)
    assert photo.junction_mask.sum() == clean.junction_mask.sum()


def test_cpline_seeded_augmentation_is_reproducible():
    cp = simple_mv_cp()
    first = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="print-medium", seed=9
    )
    second = render_cpline_sample(
        cp, image_size=128, padding=8, line_width=2, augment_profile="print-medium", seed=9
    )

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.line_prob, second.line_prob)
    assert first.metadata["params"] == second.metadata["params"]


def test_cpline_loss_hard_negative_penalizes_background_false_positives():
    targets = {
        "line_prob": torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        "angle": torch.zeros((1, 2, 16, 16), dtype=torch.float32),
        "junction_heatmap": torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        "junction_offset": torch.zeros((1, 2, 16, 16), dtype=torch.float32),
        "junction_mask": torch.zeros((1, 16, 16), dtype=torch.bool),
        "assignment": torch.full((1, 16, 16), -100, dtype=torch.long),
    }
    targets["line_prob"][:, :, 1, 1] = 1.0
    targets["angle"][:, 0, 1, 1] = 1.0
    targets["assignment"][:, 1, 1] = 0
    outputs = {
        "line_logits": torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        "angle": torch.zeros((1, 2, 16, 16), dtype=torch.float32),
        "junction_logits": torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        "junction_offset": torch.zeros((1, 2, 16, 16), dtype=torch.float32),
        "assignment_logits": torch.zeros((1, 4, 16, 16), dtype=torch.float32),
    }
    noisy_outputs = {key: value.clone() for key, value in outputs.items()}
    noisy_outputs["line_logits"][:, :, 4:12, 4:12] = 6.0
    criterion = CPLineLoss(
        CPLineLossConfig(
            line_hard_negative_weight=1.0,
            line_hard_negative_ratio=0.25,
            line_hard_negative_multiplier=8.0,
            line_hard_negative_min_pixels=8,
        )
    )

    clean_losses = criterion(outputs, targets)
    noisy_losses = criterion(noisy_outputs, targets)

    assert noisy_losses["line_hard_negative"] > clean_losses["line_hard_negative"]
    assert noisy_losses["total"] > clean_losses["total"]


def test_cpline_stage_base_samples_only_geometry_warmup_profiles():
    cp = simple_mv_cp()
    allowed = {entry[0] for entry in AUGMENT_MIXES["stage-base"]}
    seen = {
        render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="stage-base",
            seed=seed,
        ).metadata["selected_profile"]
        for seed in range(40)
    }

    assert seen <= allowed
    assert "dark-mode" not in seen
    assert "print-medium" not in seen
    assert "photo-light" not in seen


def test_cpline_stage_balanced_samples_dark_and_photo_dark_without_grid():
    cp = simple_mv_cp()
    dark_sample = None
    photo_dark_sample = None
    for seed in range(160):
        sample = render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="stage-balanced",
            seed=seed,
        )
        if sample.metadata["selected_profile"] == "dark-mode":
            dark_sample = sample
        if sample.metadata["selected_profile"] == "photo-dark":
            photo_dark_sample = sample

    assert dark_sample is not None
    assert dark_sample.metadata["style_variant"] in DARK_MODE_STYLE_VARIANTS
    assert dark_sample.metadata["grid_enabled"] is False
    assert photo_dark_sample is not None
    assert photo_dark_sample.metadata["style_variant"] in DARK_MODE_STYLE_VARIANTS
    assert photo_dark_sample.metadata["grid_enabled"] is False
    assert photo_dark_sample.metadata["geometry_applied"] is True


def test_cpline_style_profiles_do_not_apply_square_symmetry_by_default():
    cp = asymmetric_mv_cp()
    clean = render_cpline_sample(
        cp, image_size=96, padding=8, line_width=2, augment_profile="clean"
    )
    styled = render_cpline_sample(
        cp, image_size=96, padding=8, line_width=2, augment_profile="line-style", seed=2
    )

    assert styled.metadata["square_symmetry"] == "identity"
    assert styled.metadata["geometry_applied"] is False
    assert np.allclose(styled.pixel_vertices, clean.pixel_vertices)


def test_cpline_monochrome_style_does_not_hallucinate_mv_targets():
    cp = simple_mv_cp()
    sample = render_cpline_sample(
        cp,
        image_size=96,
        padding=8,
        line_width=2,
        augment_profile="line-style",
        seed=0,
        square_symmetry="identity",
    )

    assert sample.metadata["params"]["palette_kind"] == "monochrome"
    assert sample.metadata["params"]["assignment_target_mode"] == "mv_to_unassigned"
    assert 0 not in sample.assignment
    assert 1 not in sample.assignment
    assert 3 in sample.assignment


def test_v2_text_augmentation_emits_non_crease_targets():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-text",
        seed=4,
    )

    assert sample.metadata["selected_profile"] == "v2-text"
    assert sample.metadata["v2_augmentation"]["modes"] == ["text"]
    assert np.count_nonzero(sample.v2_non_crease_mask) > 0
    assert np.count_nonzero(sample.v2_target_line_mask) > 0
    assert sample.v2_line_style[sample.v2_target_line_mask > 0].max() == V2_LINE_STYLE_IDS["solid"]


def test_v2_dashed_augmentation_preserves_solid_carrier_target():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-dashed",
        seed=5,
    )

    assert "dashed" in sample.metadata["v2_augmentation"]["modes"]
    assert np.count_nonzero(sample.v2_line_style == V2_LINE_STYLE_IDS["dashed"]) > 0
    assert sample.line_prob.max() > 0.9
    assert np.count_nonzero(sample.v2_target_line_mask) == np.count_nonzero(sample.line_prob > 0.05)


def test_v2_ambiguous_mv_marks_observed_assignment_unknown():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-ambiguous-mv",
        seed=6,
    )

    assert "ambiguous_mv" in sample.metadata["v2_augmentation"]["modes"]
    assert "dashed" not in sample.metadata["v2_augmentation"]["modes"]
    assert np.count_nonzero(sample.v2_line_style == V2_LINE_STYLE_IDS["dashed"]) == 0
    assert np.count_nonzero(sample.v2_line_style == V2_LINE_STYLE_IDS["monochrome"]) > 0
    assert 0 not in sample.v2_observed_assignment
    assert 1 not in sample.v2_observed_assignment
    assert 3 in sample.v2_observed_assignment


def test_v2_dark_profile_combines_dark_mode_with_issue_targets():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-dark-text",
        seed=8,
    )

    assert sample.metadata["selected_profile"] == "v2-dark-text"
    assert sample.metadata["style_variant"] in DARK_MODE_STYLE_VARIANTS
    assert sample.metadata["v2_augmentation"]["dark_mode"] is True
    assert sample.image.mean() < 120
    assert np.count_nonzero(sample.v2_non_crease_mask) > 0


def test_v2_dark_ambiguous_keeps_readable_monochrome_contrast():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-dark-ambiguous-mv",
        seed=29,
    )
    grayscale = sample.image.astype(np.float32).mean(axis=2)
    target = sample.v2_target_line_mask > 0
    background = np.array(sample.metadata["params"]["background"], dtype=np.float32).mean()

    assert float(grayscale[target].mean()) - float(background) > 35.0


def test_v2_dark_faint_keeps_minimum_readable_contrast():
    sample = render_cpline_sample(
        simple_mv_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="v2-dark-faint",
        seed=31,
    )
    grayscale = sample.image.astype(np.float32).mean(axis=2)
    target = sample.v2_target_line_mask > 0
    background = np.array(sample.metadata["params"]["background"], dtype=np.float32).mean()

    assert float(grayscale[target].mean()) - float(background) > 20.0


def test_v2_issue_mix_samples_combined_profile():
    cp = simple_mv_cp()
    seen = {
        render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="v2-issue-mix",
            seed=seed,
        ).metadata["selected_profile"]
        for seed in range(80)
    }

    assert "v2-combined" in seen
    assert "v2-text" in seen


def test_v2_dark_issue_mix_samples_dark_profiles():
    cp = simple_mv_cp()
    seen = {
        render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="v2-dark-issue-mix",
            seed=seed,
        ).metadata["selected_profile"]
        for seed in range(80)
    }

    assert "v2-dark-combined" in seen
    assert "v2-dark-text" in seen


def test_v2_boundary_targets_mark_boundary_contact_side_and_coord():
    sample = render_cpline_sample(
        boundary_contact_cp(),
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="clean",
    )

    assert sample.v2_boundary_contact_heatmap.max() > 0.9
    assert np.count_nonzero(sample.v2_boundary_mask) == 1
    assert V2_VERTEX_TYPE_IDS["corner"] in np.unique(sample.v2_vertex_type)
    assert V2_VERTEX_TYPE_IDS["boundary_contact"] in np.unique(sample.v2_vertex_type)
    assert V2_VERTEX_TYPE_IDS["interior_intersection"] in np.unique(sample.v2_vertex_type)

    side_labels = sample.v2_boundary_side[sample.v2_boundary_mask]
    side_coords = sample.v2_boundary_coord[sample.v2_boundary_mask]
    assert side_labels.tolist() == [V2_BOUNDARY_SIDE_IDS["top"]]
    assert np.isclose(float(side_coords[0]), 0.5, atol=0.02)
    assert sample.metadata["v2_boundary"]["side_counts"] == {"top": 1}


def test_cpline_dataset_does_not_cache_random_augmented_tensors(tmp_path):
    manifest = _write_manifest(tmp_path, count=2)
    dataset = CplineFoldDataset(
        manifest,
        split="train",
        limit=1,
        max_edges=20,
        image_size=96,
        augment_profile="print-medium",
        seed=5,
    )

    first = dataset[0]["image"]
    second = dataset[0]["image"]

    assert not torch.equal(first, second)


def test_cpline_dataset_workers_use_independent_augmentation_rngs(tmp_path):
    manifest = _write_manifest(tmp_path, count=4)
    dataset = CplineFoldDataset(
        manifest,
        split="train",
        limit=2,
        max_edges=20,
        image_size=96,
        augment_profile="print-medium",
        seed=5,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=cpline_collate)

    first, second = [batch["image"][0] for batch in loader]

    assert not torch.equal(first, second)


def test_cpline_dataset_collates_v2_targets(tmp_path):
    manifest = _write_manifest(tmp_path, count=2)
    dataset = CplineFoldDataset(
        manifest,
        split="train",
        limit=1,
        max_edges=20,
        image_size=96,
        augment_profile="v2-watermark",
        seed=5,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=cpline_collate)
    batch = next(iter(loader))

    assert batch["v2_non_crease_mask"].shape == (1, 1, 96, 96)
    assert batch["v2_target_line_mask"].shape == (1, 1, 96, 96)
    assert batch["v2_line_style"].shape == (1, 96, 96)
    assert batch["v2_observed_assignment"].shape == (1, 96, 96)
    assert batch["v2_boundary_contact_heatmap"].shape == (1, 1, 96, 96)
    assert batch["v2_vertex_type"].shape == (1, 96, 96)
    assert batch["v2_boundary_side"].shape == (1, 96, 96)
    assert batch["v2_boundary_offset"].shape == (1, 2, 96, 96)
    assert batch["v2_boundary_mask"].shape == (1, 96, 96)
    assert batch["v2_boundary_coord"].shape == (1, 1, 96, 96)


def test_cpline_dataset_limit_samples_across_ordered_mixed_manifest():
    records = [
        {
            "id": f"tree-{idx}",
            "foldPath": f"tree-{idx}.fold",
            "split": "train",
            "family": "treemaker-tree",
            "edges": 8,
        }
        for idx in range(100)
    ] + [
        {
            "id": f"rabbit-{idx}",
            "foldPath": f"rabbit-{idx}.fold",
            "split": "train",
            "family": "rabbit-ear-fold-program",
            "edges": 8,
        }
        for idx in range(100)
    ]

    selected = select_records(records, split="train", limit=50, max_edges=20, seed=7)
    families = {record["family"] for record in selected}

    assert families == {"treemaker-tree", "rabbit-ear-fold-program"}


def test_cpline_dataset_balanced_family_sampling_oversamples_small_family():
    records = [
        {
            "id": f"tree-{idx}",
            "foldPath": f"tree-{idx}.fold",
            "split": "train",
            "family": "treemaker-tree",
            "edges": 8,
        }
        for idx in range(8)
    ] + [
        {
            "id": f"rabbit-{idx}",
            "foldPath": f"rabbit-{idx}.fold",
            "split": "train",
            "family": "rabbit-ear-fold-program",
            "edges": 8,
        }
        for idx in range(2)
    ]

    selected = select_records(
        records,
        split="train",
        limit=8,
        max_edges=20,
        seed=7,
        family_sampling="balanced",
    )
    family_counts = {
        family: sum(record["family"] == family for record in selected)
        for family in {"treemaker-tree", "rabbit-ear-fold-program"}
    }

    assert family_counts == {"treemaker-tree": 4, "rabbit-ear-fold-program": 4}


def test_cpline_augmentation_visualization_script_smoke(tmp_path):
    manifest = _write_manifest(tmp_path, count=1)
    output_dir = tmp_path / "visualizations"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            sys.executable,
            "scripts/visualize/cpline_augmentations.py",
            "--manifest",
            str(manifest),
            "--profiles",
            "dark-mode",
            "--num-samples",
            "1",
            "--max-edges",
            "20",
            "--image-size",
            "96",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=env,
    )

    assert (output_dir / "dark-mode" / "contact_sheet_96.png").exists()
    sidecar = output_dir / "dark-mode" / "contact_sheet_96.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 4
    assert all(not row["augmentation"]["grid_enabled"] for row in data["rows"])


def test_cpline_square_symmetry_visualization_script_smoke(tmp_path):
    manifest = _write_manifest(tmp_path, count=1)
    output_dir = tmp_path / "visualizations"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            sys.executable,
            "scripts/visualize/cpline_augmentations.py",
            "--manifest",
            str(manifest),
            "--profiles",
            "square-symmetry",
            "--num-samples",
            "1",
            "--max-edges",
            "20",
            "--image-size",
            "96",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=env,
    )

    assert (output_dir / "square-symmetry" / "contact_sheet_96.png").exists()
    sidecar = output_dir / "square-symmetry" / "contact_sheet_96.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(data["rows"]) == len(NON_IDENTITY_SQUARE_SYMMETRIES)
    assert {row["augmentation"]["square_symmetry"] for row in data["rows"]} == set(
        NON_IDENTITY_SQUARE_SYMMETRIES
    )


def test_cpline_training_script_dark_mode_smoke(tmp_path):
    manifest = _write_manifest(tmp_path, count=2)
    output_dir = tmp_path / "checkpoint"
    init_output_dir = tmp_path / "checkpoint_init"
    env = {**os.environ, "TQDM_DISABLE": "1"}
    subprocess.run(
        [
            sys.executable,
            "scripts/training/train_cpline_smoke.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--train-count",
            "1",
            "--val-count",
            "1",
            "--max-edges",
            "20",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--hidden-channels",
            "32",
            "--train-family-sampling",
            "balanced",
            "--augment-profile",
            "dark-mode",
            "--eval-thresholds",
            "0.8",
            "--graph-eval-count",
            "1",
        ],
        check=True,
        env=env,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["augment_profile"] == "dark-mode"
    assert summary["train_family_sampling"] == "balanced"
    assert summary["graph_eval_count"] == 1
    assert summary["val_graph_sweep"]["thresholds"]["0.80"]["files"] == 1
    assert (output_dir / "latest.pt").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/training/train_cpline_smoke.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(init_output_dir),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--train-count",
            "1",
            "--val-count",
            "1",
            "--max-edges",
            "20",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--hidden-channels",
            "32",
            "--augment-profile",
            "stage-base",
            "--eval-thresholds",
            "0.8",
            "--graph-eval-count",
            "1",
            "--init-checkpoint",
            str(output_dir / "latest.pt"),
        ],
        check=True,
        env=env,
    )
    init_summary = json.loads((init_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert init_summary["augment_profile"] == "stage-base"
    assert init_summary["init_checkpoint"] == str(output_dir / "latest.pt")


def test_cpline_net_outputs_roadmap_fields():
    model = CPLineNet(backbone="tiny", hidden_channels=32)
    outputs = model(torch.zeros(2, 3, 64, 64))

    assert outputs["line_logits"].shape == (2, 1, 64, 64)
    assert outputs["angle"].shape == (2, 2, 64, 64)
    assert outputs["junction_logits"].shape == (2, 1, 64, 64)
    assert outputs["junction_offset"].shape == (2, 2, 64, 64)
    assert outputs["assignment_logits"].shape == (2, 4, 64, 64)


def test_cpline_net_optional_v2_heads_and_losses():
    model = CPLineNet(backbone="tiny", hidden_channels=32, v2_heads=True)
    outputs = model(torch.zeros(2, 3, 64, 64))

    assert outputs["non_crease_logits"].shape == (2, 1, 64, 64)
    assert outputs["line_style_logits"].shape == (2, 4, 64, 64)
    assert outputs["boundary_contact_logits"].shape == (2, 1, 64, 64)
    assert outputs["vertex_type_logits"].shape == (2, 4, 64, 64)
    assert outputs["boundary_side_logits"].shape == (2, 4, 64, 64)
    assert outputs["boundary_offset"].shape == (2, 2, 64, 64)
    assert outputs["boundary_coord"].shape == (2, 1, 64, 64)

    targets = {
        "line_prob": torch.zeros((2, 1, 64, 64), dtype=torch.float32),
        "angle": torch.zeros((2, 2, 64, 64), dtype=torch.float32),
        "junction_heatmap": torch.zeros((2, 1, 64, 64), dtype=torch.float32),
        "junction_offset": torch.zeros((2, 2, 64, 64), dtype=torch.float32),
        "junction_mask": torch.zeros((2, 64, 64), dtype=torch.bool),
        "assignment": torch.full((2, 64, 64), -100, dtype=torch.long),
        "v2_non_crease_mask": torch.zeros((2, 1, 64, 64), dtype=torch.float32),
        "v2_line_style": torch.full((2, 64, 64), -100, dtype=torch.long),
        "v2_observed_assignment": torch.full((2, 64, 64), -100, dtype=torch.long),
        "v2_boundary_contact_heatmap": torch.zeros((2, 1, 64, 64), dtype=torch.float32),
        "v2_vertex_type": torch.zeros((2, 64, 64), dtype=torch.long),
        "v2_boundary_side": torch.full((2, 64, 64), -100, dtype=torch.long),
        "v2_boundary_offset": torch.zeros((2, 2, 64, 64), dtype=torch.float32),
        "v2_boundary_mask": torch.zeros((2, 64, 64), dtype=torch.bool),
        "v2_boundary_coord": torch.zeros((2, 1, 64, 64), dtype=torch.float32),
    }
    targets["line_prob"][:, :, 10:14, 10:14] = 1.0
    targets["angle"][:, 0, 10:14, 10:14] = 1.0
    targets["assignment"][:, 10:14, 10:14] = 3
    targets["v2_observed_assignment"][:, 10:14, 10:14] = 3
    targets["v2_non_crease_mask"][:, :, 30:34, 30:34] = 1.0
    targets["v2_line_style"][:, 10:14, 10:14] = V2_LINE_STYLE_IDS["dashed"]
    targets["v2_boundary_contact_heatmap"][:, :, 5, 5] = 1.0
    targets["v2_vertex_type"][:, 5, 5] = V2_VERTEX_TYPE_IDS["boundary_contact"]
    targets["v2_boundary_side"][:, 5, 5] = V2_BOUNDARY_SIDE_IDS["top"]
    targets["v2_boundary_mask"][:, 5, 5] = True
    targets["v2_boundary_coord"][:, :, 5, 5] = 0.5
    criterion = CPLineLoss(
        CPLineLossConfig(
            non_crease_weight=1.0,
            line_style_weight=1.0,
            use_observed_assignment_target=True,
            boundary_contact_weight=1.0,
            vertex_type_weight=1.0,
            boundary_side_weight=1.0,
            boundary_offset_weight=1.0,
            boundary_coord_weight=1.0,
        )
    )
    losses = criterion(outputs, targets)

    assert losses["non_crease"] > 0
    assert losses["line_style"] > 0
    assert losses["boundary_contact"] > 0
    assert losses["vertex_type"] > 0
    assert losses["boundary_side"] > 0
    assert torch.isfinite(losses["boundary_offset"])
    assert torch.isfinite(losses["boundary_coord"])


def test_cpline_outputs_convert_to_vectorizer_evidence():
    model = CPLineNet(backbone="tiny", hidden_channels=32)
    outputs = model(torch.zeros(1, 3, 64, 64))
    evidence = cpline_outputs_to_evidence(outputs, line_threshold=0.0)

    assert evidence.line_prob.shape == (64, 64)
    assert evidence.angle.shape == (64, 64, 2)
    assert evidence.junction_heatmap.shape == (64, 64)
    assert evidence.assignment_labels.shape == (64, 64)


def _write_manifest(tmp_path, *, count: int) -> str:
    records = []
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    for idx in range(count):
        fold_path = folds_dir / f"sample_{idx}.fold"
        FOLDParser.save_fold(simple_mv_cp(), fold_path)
        records.append(
            {
                "id": f"sample-{idx}",
                "foldPath": str(fold_path.relative_to(tmp_path)),
                "metadataPath": f"metadata/sample_{idx}.json",
                "split": "train" if idx % 2 == 0 else "val",
                "family": "test",
                "bucket": "test",
                "vertices": 5,
                "edges": 8,
            }
        )
    manifest = tmp_path / "raw-manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return str(manifest)
