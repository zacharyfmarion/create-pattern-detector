#!/usr/bin/env python3
"""Merge fold-only synthetic generator shards.

This merges accepted `.fold` outputs and raw manifests. It does not render images
or perform augmentations; use it to assemble graph datasets before Phase 3.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Merged dataset root")
    parser.add_argument("--recompute-splits", action="store_true", help="Reassign train/val/test splits globally after merge")
    parser.add_argument("shards", type=Path, nargs="+", help="Shard roots containing raw-manifest.jsonl")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "folds").mkdir(exist_ok=True)
    (args.out / "metadata").mkdir(exist_ok=True)
    (args.out / "qa").mkdir(exist_ok=True)

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    recipe_copied = False

    for shard in args.shards:
        manifest_path = shard / "raw-manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard raw manifest: {manifest_path}")
        rows = load_jsonl(manifest_path)
        source_summaries.append({"root": str(shard), "rows": len(rows)})
        if not recipe_copied and (shard / "recipe.json").exists():
            shutil.copy2(shard / "recipe.json", args.out / "recipe.json")
            recipe_copied = True
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id across shards: {sample_id}")
            seen_ids.add(sample_id)
            fold_src = resolve_path(row.get("foldPath") or row.get("fold_path"), shard, manifest_path.parent)
            metadata_src = resolve_path(row.get("metadataPath") or row.get("metadata_path"), shard, manifest_path.parent)
            fold_dst = unique_destination(args.out / "folds", fold_src.name)
            metadata_dst = unique_destination(args.out / "metadata", metadata_src.name)
            shutil.copy2(fold_src, fold_dst)
            shutil.copy2(metadata_src, metadata_dst)
            row = dict(row)
            row["foldPath"] = str(fold_dst.relative_to(args.out))
            row["metadataPath"] = str(metadata_dst.relative_to(args.out))
            merged.append(row)

    if args.recompute_splits:
        for index, row in enumerate(merged):
            row["split"] = split_for_index(index, DEFAULT_SPLITS)

    raw_manifest = args.out / "raw-manifest.jsonl"
    write_jsonl(raw_manifest, merged)
    qa = build_qa(merged, source_summaries)
    (args.out / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Merged {len(merged)} accepted FOLD samples from {len(args.shards)} shards into {args.out}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_path(value: Any, root: Path, manifest_dir: Path) -> Path:
    if value is None:
        raise ValueError("manifest row has no path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    root_path = root / path
    if root_path.exists():
        return root_path
    return manifest_dir / path


def unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}-{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def split_for_index(index: int, splits: dict[str, float]) -> str:
    total = sum(splits.values())
    position = (index * 0.6180339887498949) % 1
    train_cutoff = splits["train"] / total
    val_cutoff = (splits["train"] + splits["val"]) / total
    if position < train_cutoff:
        return "train"
    if position < val_cutoff:
        return "val"
    return "test"


def build_qa(rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted": len(rows),
        "sourceShards": sources,
        "familyCounts": clean_counter(Counter(row.get("family") for row in rows)),
        "bucketCounts": clean_counter(Counter(row.get("bucket") for row in rows)),
        "splitCounts": clean_counter(Counter(row.get("split") for row in rows)),
        "treeMakerSymmetryCounts": clean_counter(Counter((row.get("treeMetadata") or {}).get("symmetryClass") for row in rows)),
        "treeMakerVariantCounts": clean_counter(Counter((row.get("treeMetadata") or {}).get("symmetryVariant") for row in rows)),
        "treeMakerArchetypeCounts": clean_counter(Counter((row.get("treeMetadata") or {}).get("archetype") for row in rows)),
        "treeMakerTopologyCounts": clean_counter(Counter((row.get("treeMetadata") or {}).get("topology") for row in rows)),
    }


def clean_counter(counter: Counter) -> dict[str, int]:
    items = [(str(key), int(value)) for key, value in counter.items() if key is not None]
    return {key: value for key, value in sorted(items)}


if __name__ == "__main__":
    main()
