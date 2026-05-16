#!/usr/bin/env python3
"""Smoke-test a shared fold-only synthetic dataset.

This intentionally avoids importing torch so it can run in lightweight
worktrees before the full training environment is installed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.data.annotations import GroundTruthGenerator
from src.data.fold_parser import FOLDParser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/generated/synthetic/treemaker_tree_v1"),
        help="Dataset root containing raw-manifest.jsonl and folds/",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--samples-per-split", type=int, default=2)
    args = parser.parse_args()

    manifest_path = args.root / "raw-manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing raw manifest: {manifest_path}")

    rows = load_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"No rows in {manifest_path}")

    parser_fold = FOLDParser()
    gt_generator = GroundTruthGenerator(
        image_size=args.image_size,
        padding=args.padding,
        line_width=args.line_width,
    )
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    checked: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        for row in split_rows[: args.samples_per_split]:
            fold_path = resolve_path(args.root, row["foldPath"])
            cp = parser_fold.parse(fold_path)
            gt = gt_generator.generate(cp)
            checked.append(
                {
                    "id": row["id"],
                    "split": split,
                    "vertices": int(cp.num_vertices),
                    "edges": int(cp.num_edges),
                    "segmentation_shape": list(gt["segmentation"].shape),
                    "orientation_shape": list(gt["orientation"].shape),
                    "junction_shape": list(gt["junction_heatmap"].shape),
                }
            )

    summary = {
        "root": str(args.root),
        "rows": len(rows),
        "splitCounts": dict(sorted(split_counts.items())),
        "checked": checked,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


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


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    resolved = root / path
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {resolved}")
    return resolved


if __name__ == "__main__":
    main()
