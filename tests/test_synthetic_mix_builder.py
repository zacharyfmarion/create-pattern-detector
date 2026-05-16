import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_synthetic_training_mix.py"
SPEC = importlib.util.spec_from_file_location("build_synthetic_training_mix", SCRIPT_PATH)
assert SPEC and SPEC.loader
mix_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mix_builder)


def test_mix_builder_symlinks_sources_and_preserves_provenance(tmp_path):
    first = write_source(tmp_path, "treemaker_tree_v1", "tm-001", "treemaker-tree")
    second = write_source(tmp_path, "rabbit_ear_fold_program_v1", "re-001", "rabbit-ear-fold-program")
    out = tmp_path / "cp_training_mix_v1"

    summary = mix_builder.build_mix([first, second], out, recompute_splits=True)

    assert summary["accepted"] == 2
    assert summary["sourceDatasetCounts"] == {"rabbit_ear_fold_program_v1": 1, "treemaker_tree_v1": 1}
    rows = [json.loads(line) for line in (out / "raw-manifest.jsonl").read_text().splitlines()]
    assert {row["sourceDataset"] for row in rows} == {"treemaker_tree_v1", "rabbit_ear_fold_program_v1"}
    assert {row["family"] for row in rows} == {"treemaker-tree", "rabbit-ear-fold-program"}
    for row in rows:
        fold_link = out / row["foldPath"]
        metadata_link = out / row["metadataPath"]
        assert fold_link.is_symlink()
        assert metadata_link.is_symlink()
        assert fold_link.exists()
        assert metadata_link.exists()


def test_mix_builder_rejects_duplicate_sample_ids(tmp_path):
    first = write_source(tmp_path, "source_a", "dup", "treemaker-tree")
    second = write_source(tmp_path, "source_b", "dup", "rabbit-ear-fold-program")

    with pytest.raises(ValueError, match="Duplicate sample id"):
        mix_builder.build_mix([first, second], tmp_path / "mix")


def test_mix_builder_rejects_broken_manifest_paths(tmp_path):
    source = tmp_path / "broken"
    source.mkdir()
    (source / "raw-manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "broken-001",
                "family": "treemaker-tree",
                "split": "train",
                "foldPath": "folds/missing.fold",
                "metadataPath": "metadata/missing.json",
            }
        )
        + "\n"
    )

    with pytest.raises(FileNotFoundError, match="Broken manifest"):
        mix_builder.build_mix([source], tmp_path / "mix")


def write_source(tmp_path, name, sample_id, family):
    source = tmp_path / name
    folds = source / "folds"
    metadata = source / "metadata"
    folds.mkdir(parents=True)
    metadata.mkdir()
    fold_path = folds / f"{sample_id}.fold"
    metadata_path = metadata / f"{sample_id}.json"
    fold_path.write_text(json.dumps(square_fold()) + "\n")
    metadata_path.write_text(json.dumps({"id": sample_id}) + "\n")
    row = {
        "id": sample_id,
        "seed": 1,
        "family": family,
        "bucket": "small",
        "split": "train",
        "foldPath": str(fold_path.relative_to(source)),
        "metadataPath": str(metadata_path.relative_to(source)),
        "vertices": 4,
        "edges": 4,
        "assignments": {"B": 4},
        "validation": {"valid": True, "passed": [], "failed": [], "errors": []},
    }
    (source / "raw-manifest.jsonl").write_text(json.dumps(row) + "\n")
    return source


def square_fold():
    return {
        "file_spec": 1.1,
        "file_creator": "test",
        "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "edges_assignment": ["B", "B", "B", "B"],
    }
