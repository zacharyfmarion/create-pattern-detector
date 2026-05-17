import json
import os
import subprocess
import sys

import numpy as np
import torch

from src.data.cpline_dataset import CplineFoldDataset, render_cpline_sample, render_input_image, select_records
from src.data.cpline_augmentations import AUGMENT_MIXES, NON_IDENTITY_SQUARE_SYMMETRIES
from src.data.fold_parser import CreasePattern, FOLDParser
from src.models import CPLineNet
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
    sample = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="clean")
    image = render_input_image(cp, image_size=128, padding=8, line_width=2, render_noise="clean")

    assert np.array_equal(sample.image, image)
    assert sample.metadata["selected_profile"] == "clean"
    assert sample.metadata["line_width"] == 2


def test_cpline_dark_mode_preserves_geometry_targets():
    cp = simple_mv_cp()
    clean = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="clean")
    dark = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="dark-mode",
        seed=11,
        style_variant="dark-grid",
        square_symmetry="identity",
    )

    assert dark.image.mean() < clean.image.mean()
    assert not np.array_equal(dark.image, clean.image)
    assert np.array_equal(dark.line_prob, clean.line_prob)
    assert np.array_equal(dark.junction_heatmap, clean.junction_heatmap)
    assert np.array_equal(dark.assignment, clean.assignment)


def test_cpline_dark_mode_grid_is_not_line_target():
    cp = simple_mv_cp()
    clean = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="clean")
    dark = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="dark-mode",
        seed=12,
        style_variant="dark-grid",
        square_symmetry="identity",
    )
    background = clean.line_prob < 0.01
    changed_background = np.any(dark.image[background] != clean.image[background], axis=1)

    assert dark.metadata["grid_enabled"] is True
    assert np.count_nonzero(changed_background) > 0
    assert np.array_equal(dark.line_prob, clean.line_prob)


def test_cpline_square_symmetry_transforms_vertices_and_targets():
    cp = asymmetric_mv_cp()
    clean = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="clean")
    rotated = render_cpline_sample(
        cp,
        image_size=128,
        padding=8,
        line_width=2,
        augment_profile="square-symmetry",
        square_symmetry="rotate90",
        seed=3,
    )
    expected_vertices = np.stack([127.0 - clean.pixel_vertices[:, 1], clean.pixel_vertices[:, 0]], axis=1)

    assert rotated.metadata["selected_profile"] == "square-symmetry"
    assert rotated.metadata["square_symmetry"] == "rotate90"
    assert rotated.metadata["geometry_applied"] is True
    assert np.allclose(rotated.pixel_vertices, expected_vertices, atol=1e-4)
    assert np.array_equal(rotated.assignments, clean.assignments)
    assert not np.array_equal(rotated.line_prob, clean.line_prob)
    assert rotated.junction_mask.sum() == clean.junction_mask.sum()


def test_cpline_photo_light_geometric_augmentation_recomputes_targets():
    cp = simple_mv_cp()
    clean = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="clean")
    photo = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="photo-light", seed=3)

    assert photo.metadata["geometry_applied"] is True
    assert not np.allclose(photo.pixel_vertices, clean.pixel_vertices)
    assert not np.array_equal(photo.line_prob, clean.line_prob)
    assert photo.junction_mask.sum() == clean.junction_mask.sum()


def test_cpline_seeded_augmentation_is_reproducible():
    cp = simple_mv_cp()
    first = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="print-medium", seed=9)
    second = render_cpline_sample(cp, image_size=128, padding=8, line_width=2, augment_profile="print-medium", seed=9)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.line_prob, second.line_prob)
    assert first.metadata["params"] == second.metadata["params"]


def test_cpline_stage_light_samples_only_stage_one_profiles():
    cp = simple_mv_cp()
    allowed = {entry[0] for entry in AUGMENT_MIXES["stage-light"]}
    seen = {
        render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="stage-light",
            seed=seed,
        ).metadata["selected_profile"]
        for seed in range(40)
    }

    assert seen <= allowed
    assert "dark-mode" not in seen
    assert "print-medium" not in seen
    assert "photo-light" not in seen


def test_cpline_stage_dark_pins_dark_mode_without_grid():
    cp = simple_mv_cp()
    dark_sample = None
    for seed in range(80):
        sample = render_cpline_sample(
            cp,
            image_size=96,
            padding=8,
            line_width=2,
            augment_profile="stage-dark",
            seed=seed,
        )
        if sample.metadata["selected_profile"] == "dark-mode":
            dark_sample = sample
            break

    assert dark_sample is not None
    assert dark_sample.metadata["style_variant"] == "dark-no-grid"
    assert dark_sample.metadata["grid_enabled"] is False


def test_cpline_style_profiles_do_not_apply_square_symmetry_by_default():
    cp = asymmetric_mv_cp()
    clean = render_cpline_sample(cp, image_size=96, padding=8, line_width=2, augment_profile="clean")
    styled = render_cpline_sample(cp, image_size=96, padding=8, line_width=2, augment_profile="line-style", seed=2)

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


def test_cpline_dataset_limit_samples_across_ordered_mixed_manifest():
    records = [
        {"id": f"tree-{idx}", "foldPath": f"tree-{idx}.fold", "split": "train", "family": "treemaker-tree", "edges": 8}
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
    assert any(row["augmentation"]["grid_enabled"] for row in data["rows"])


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
    assert {row["augmentation"]["square_symmetry"] for row in data["rows"]} == set(NON_IDENTITY_SQUARE_SYMMETRIES)


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
            "stage-light",
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
    assert init_summary["augment_profile"] == "stage-light"
    assert init_summary["init_checkpoint"] == str(output_dir / "latest.pt")


def test_cpline_net_outputs_roadmap_fields():
    model = CPLineNet(backbone="tiny", hidden_channels=32)
    outputs = model(torch.zeros(2, 3, 64, 64))

    assert outputs["line_logits"].shape == (2, 1, 64, 64)
    assert outputs["angle"].shape == (2, 2, 64, 64)
    assert outputs["junction_logits"].shape == (2, 1, 64, 64)
    assert outputs["junction_offset"].shape == (2, 2, 64, 64)
    assert outputs["assignment_logits"].shape == (2, 4, 64, 64)


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
