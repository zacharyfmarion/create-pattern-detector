"""Tests for the local SEARCH-22.5 dataset adapter and its sampling preset."""

from __future__ import annotations

import json

from src.data.cpline_dataset import select_records
from src.data.search225.dataset_builder import (
    Combo,
    build_search225_dataset_from_folds,
    parse_fold_name,
)


def _square_fold(view_url: str) -> dict:
    """Minimal valid FOLD: unit square with one M diagonal and one V hinge."""
    return {
        "file_spec": 1.1,
        "file_creator": "SEARCH-22.5",
        "file_classes": ["creasePattern"],
        "file_source": view_url,
        "vertices_coords": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [2, 4], [1, 4]],
        "edges_assignment": ["B", "B", "B", "B", "M", "M", "F"],
    }


def test_parse_fold_name_roundtrip(tmp_path):
    path = tmp_path / "search225_6book4123.fold"
    parsed = parse_fold_name(path)
    assert parsed is not None
    combo, tiling_id = parsed
    assert combo == Combo(N=6, sym="book")
    assert tiling_id == 4123
    assert combo.view_id(tiling_id) == "6b4123"
    assert parse_fold_name(tmp_path / "not_a_search225_name.fold") is None


def test_build_from_folds_writes_standard_root(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in ("search225_3diag7.fold", "search225_6book12.fold"):
        (staging / name).write_text(json.dumps(_square_fold(f"view?id={name}")))
    # Duplicate (combo, id) in a second dir must be deduped.
    staging2 = tmp_path / "staging2"
    staging2.mkdir()
    (staging2 / "search225_3diag7.fold").write_text(json.dumps(_square_fold("dup")))
    (staging2 / "unrelated.fold").write_text(json.dumps(_square_fold("skip")))

    out = tmp_path / "root"
    summary = build_search225_dataset_from_folds(
        out, [staging, staging2], dataset_name="search225_test", progress=lambda _: None
    )

    assert summary.saved == 2
    assert summary.skipped_unparseable_name == 1
    assert summary.per_combo == {"3_diag": 1, "6_book": 1}

    rows = [
        json.loads(line)
        for line in (out / "raw-manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert {row["id"] for row in rows} == {"search225_test-3d7", "search225_test-6b12"}
    for row in rows:
        assert row["family"] == "search225-tiling"
        assert row["edges"] == 7
        assert row["split"] in {"train", "val", "test"}
        assert (out / row["foldPath"]).exists()
        assert (out / row["metadataPath"]).exists()
    assert (out / "recipe.json").exists()
    assert (out / "qa.json").exists()


def test_v4_search225_sampling_preset_quotas():
    def rows(family: str, source: str, count: int) -> list[dict]:
        return [
            {
                "id": f"{family}-{source}-{idx}",
                "foldPath": f"{idx}.fold",
                "split": "train",
                "family": family,
                "sourceDataset": source,
                "edges": 8,
            }
            for idx in range(count)
        ]

    records = (
        rows("treemaker-tree", "cp_training_mix_v1", 100)
        + rows("rabbit-ear-fold-program", "cp_training_mix_v1", 100)
        + rows("tessellation-fold-program", "tessellation_orthogonal_bp_grid_v2_15pct", 100)
        + rows("tessellation-fold-program", "tessellation_miura_ori_v2_15pct", 100)
        + rows("search225-tiling", "search225_v1", 100)
    )

    selected = select_records(
        records,
        split="train",
        limit=200,
        max_edges=20,
        seed=7,
        family_sampling="v4-search225-20pct",
    )
    family_counts = {
        family: sum(record["family"] == family for record in selected)
        for family in {
            "treemaker-tree",
            "rabbit-ear-fold-program",
            "tessellation-fold-program",
            "search225-tiling",
        }
    }
    assert family_counts == {
        "treemaker-tree": 65,
        "rabbit-ear-fold-program": 65,
        "tessellation-fold-program": 30,
        "search225-tiling": 40,
    }


def test_v5_bp_search225_sampling_preset_quotas():
    def rows(family: str, source: str, count: int) -> list[dict]:
        return [
            {
                "id": f"{family}-{source}-{idx}",
                "foldPath": f"{idx}.fold",
                "split": "train",
                "family": family,
                "sourceDataset": source,
                "edges": 8,
            }
            for idx in range(count)
        ]

    records = (
        rows("treemaker-tree", "cp_training_mix_v1", 100)
        + rows("rabbit-ear-fold-program", "cp_training_mix_v1", 100)
        + rows("tessellation-fold-program", "tessellation_orthogonal_bp_grid_v2_15pct", 100)
        + rows("tessellation-fold-program", "tessellation_miura_ori_v2_15pct", 100)
        + rows("search225-tiling", "search225_v1", 100)
        + rows("box-pleated", "box_pleated_v1", 100)
    )
    selected = select_records(
        records,
        split="train",
        limit=200,
        max_edges=20,
        seed=7,
        family_sampling="v5-bp-search225",
    )
    family_counts = {
        family: sum(record["family"] == family for record in selected)
        for family in {
            "treemaker-tree",
            "rabbit-ear-fold-program",
            "tessellation-fold-program",
            "search225-tiling",
            "box-pleated",
        }
    }
    assert family_counts == {
        "treemaker-tree": 45,
        "rabbit-ear-fold-program": 45,
        "tessellation-fold-program": 30,
        "search225-tiling": 40,
        "box-pleated": 40,
    }
