#!/usr/bin/env python3
"""Build a fold-only synthetic training mix from external dataset roots.

The output contains symlinks to source `.fold` and metadata files, plus a merged
`raw-manifest.jsonl`. It intentionally does not render images or augment data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Mixed dataset root to create or update")
    parser.add_argument(
        "--recompute-splits",
        action="store_true",
        help="Reassign train/val/test splits globally after merging all source roots",
    )
    parser.add_argument("sources", type=Path, nargs="+", help="Synthetic dataset roots containing raw-manifest.jsonl")
    args = parser.parse_args()

    summary = build_mix(args.sources, args.out, recompute_splits=args.recompute_splits)
    print(
        f"Built synthetic training mix with {summary['accepted']} samples "
        f"from {len(summary['sourceRoots'])} roots at {args.out}"
    )


def build_mix(source_roots: list[Path], out: Path, *, recompute_splits: bool = False) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    folds_dir = out / "folds"
    metadata_dir = out / "metadata"
    qa_dir = out / "qa"
    folds_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    qa_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_summaries: list[dict[str, Any]] = []

    for source_root in source_roots:
        source_root = source_root.expanduser().resolve()
        manifest_path = source_root / "raw-manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing raw manifest: {manifest_path}")
        source_dataset = source_root.name
        source_rows = load_jsonl(manifest_path)
        source_summaries.append({"name": source_dataset, "root": str(source_root), "rows": len(source_rows)})

        for row in source_rows:
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id across sources: {sample_id}")
            seen_ids.add(sample_id)

            fold_src = resolve_path(row.get("foldPath") or row.get("fold_path"), source_root, manifest_path.parent)
            metadata_src = resolve_path(row.get("metadataPath") or row.get("metadata_path"), source_root, manifest_path.parent)
            if not fold_src.exists():
                raise FileNotFoundError(f"Broken manifest foldPath for {sample_id}: {fold_src}")
            if not metadata_src.exists():
                raise FileNotFoundError(f"Broken manifest metadataPath for {sample_id}: {metadata_src}")

            fold_dst = unique_destination(folds_dir, f"{source_dataset}--{fold_src.name}")
            metadata_dst = unique_destination(metadata_dir, f"{source_dataset}--{metadata_src.name}")
            recreate_symlink(fold_src, fold_dst)
            recreate_symlink(metadata_src, metadata_dst)

            mixed_row = dict(row)
            mixed_row["sourceDataset"] = source_dataset
            mixed_row["sourceFoldPath"] = str(fold_src)
            mixed_row["sourceMetadataPath"] = str(metadata_src)
            mixed_row["foldPath"] = str(fold_dst.relative_to(out))
            mixed_row["metadataPath"] = str(metadata_dst.relative_to(out))
            rows.append(mixed_row)

    if recompute_splits:
        for index, row in enumerate(rows):
            row["split"] = split_for_index(index, DEFAULT_SPLITS)

    manifest_path = out / "raw-manifest.jsonl"
    write_jsonl(manifest_path, rows)
    qa = build_qa(rows, source_summaries, recompute_splits=recompute_splits)
    (out / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (qa_dir / "mix-summary.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return qa


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
        return root_path.resolve()
    return (manifest_dir / path).resolve()


def recreate_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())
    if not destination.exists():
        raise FileNotFoundError(f"Created broken symlink: {destination} -> {source}")


def unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists() and not destination.is_symlink():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}-{index:03d}{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
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


def build_qa(rows: list[dict[str, Any]], sources: list[dict[str, Any]], *, recompute_splits: bool) -> dict[str, Any]:
    return {
        "accepted": len(rows),
        "sourceRoots": sources,
        "recomputeSplits": recompute_splits,
        "sourceDatasetCounts": clean_counter(Counter(row.get("sourceDataset") for row in rows)),
        "familyCounts": clean_counter(Counter(row.get("family") for row in rows)),
        "bucketCounts": clean_counter(Counter(row.get("bucket") for row in rows)),
        "splitCounts": clean_counter(Counter(row.get("split") for row in rows)),
        "assignmentTotals": merge_assignment_counts(rows),
    }


def merge_assignment_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for assignment, count in (row.get("assignments") or {}).items():
            counts[str(assignment)] += int(count)
    return clean_counter(counts)


def clean_counter(counter: Counter) -> dict[str, int]:
    items = [(str(key), int(value)) for key, value in counter.items() if key is not None]
    return {key: value for key, value in sorted(items)}


if __name__ == "__main__":
    main()
