#!/usr/bin/env python3
"""Report graph-level distributions for synthetic FOLD datasets.

This is intentionally fold-only: it reads accepted raw manifest rows and their
canonical `.fold` files, then writes distribution metrics needed before image
augmentation or model training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Synthetic dataset root containing raw-manifest.jsonl")
    parser.add_argument("--manifest", type=Path, help="Raw manifest JSONL path. Defaults to <root>/raw-manifest.jsonl")
    parser.add_argument("--out", type=Path, help="Output JSON path. Defaults to <root>/qa/fold-distribution-report.json")
    parser.add_argument("--angle-bin-degrees", type=float, default=15.0)
    args = parser.parse_args()

    root = args.root or (args.manifest.parent if args.manifest else None)
    if root is None:
        raise SystemExit("--root or --manifest is required")
    manifest_path = args.manifest or root / "raw-manifest.jsonl"
    output_path = args.out or root / "qa" / "fold-distribution-report.json"

    rows = load_jsonl(manifest_path)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", f"row-{index}"))
        try:
            fold_path = resolve_path(row.get("foldPath") or row.get("fold_path"), root, manifest_path.parent)
            fold = json.loads(fold_path.read_text(encoding="utf-8"))
            samples.append(analyze_sample(sample_id, row, fold, args.angle_bin_degrees))
        except Exception as exc:  # noqa: BLE001 - report bad rows without hiding the rest
            skipped.append({"id": sample_id, "row": index, "reason": str(exc)})

    report = build_report(samples, skipped, manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote fold distribution report for {len(samples)} samples to {output_path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def resolve_path(value: Any, root: Path, manifest_dir: Path) -> Path:
    if value is None:
        raise ValueError("manifest row has no foldPath")
    path = Path(str(value))
    if path.is_absolute():
        return path
    root_path = root / path
    if root_path.exists():
        return root_path
    return manifest_dir / path


def analyze_sample(sample_id: str, row: dict[str, Any], fold: dict[str, Any], angle_bin_degrees: float) -> dict[str, Any]:
    vertices = [tuple(map(float, point[:2])) for point in fold["vertices_coords"]]
    edges = [tuple(map(int, edge[:2])) for edge in fold["edges_vertices"]]
    assignments = [str(value) for value in fold.get("edges_assignment", ["U"] * len(edges))]
    degree = [0 for _ in vertices]
    active_degree = [0 for _ in vertices]
    border_vertices: set[int] = set()
    border_intersections: set[int] = set()
    angles = Counter()
    lengths: list[float] = []
    assignment_counts = Counter(assignments)

    for (a, b), assignment in zip(edges, assignments):
        degree[a] += 1
        degree[b] += 1
        if assignment in {"M", "V", "F", "U"}:
            active_degree[a] += 1
            active_degree[b] += 1
        p1, p2 = vertices[a], vertices[b]
        length = math.dist(p1, p2)
        lengths.append(length)
        angles[angle_bucket(p1, p2, angle_bin_degrees)] += 1
        if assignment == "B":
            border_vertices.add(a)
            border_vertices.add(b)
        else:
            if on_boundary(p1):
                border_intersections.add(a)
            if on_boundary(p2):
                border_intersections.add(b)

    tree = row.get("treeMetadata") or row.get("tree_metadata") or {}
    density = row.get("densityMetadata") or row.get("density_metadata") or {}
    return {
        "id": sample_id,
        "family": row.get("family"),
        "bucket": row.get("bucket") or density.get("densityBucket"),
        "split": row.get("split"),
        "vertices": len(vertices),
        "edges": len(edges),
        "assignments": dict(sorted(assignment_counts.items())),
        "degree_histogram": histogram(degree),
        "active_degree_histogram": histogram(active_degree),
        "angle_histogram": dict(sorted(angles.items())),
        "border_intersections": len(border_intersections),
        "border_vertex_count": len(border_vertices),
        "edge_length": summarize(lengths),
        "tree": {
            "archetype": tree.get("archetype"),
            "symmetryClass": tree.get("symmetryClass"),
            "symmetryVariant": tree.get("symmetryVariant"),
            "topology": tree.get("topology"),
            "branchDepth": tree.get("branchDepth"),
            "terminalCount": tree.get("terminalCount"),
            "nodeCount": tree.get("nodeCount"),
        },
    }


def build_report(samples: list[dict[str, Any]], skipped: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    counters = {
        "family": Counter(sample.get("family") for sample in samples),
        "bucket": Counter(sample.get("bucket") for sample in samples),
        "split": Counter(sample.get("split") for sample in samples),
        "archetype": Counter(sample["tree"].get("archetype") for sample in samples),
        "symmetryClass": Counter(sample["tree"].get("symmetryClass") for sample in samples),
        "symmetryVariant": Counter(sample["tree"].get("symmetryVariant") for sample in samples),
        "topology": Counter(sample["tree"].get("topology") for sample in samples),
    }
    merged_assignments = merge_counters(sample["assignments"] for sample in samples)
    merged_degree = merge_counters(sample["degree_histogram"] for sample in samples)
    merged_active_degree = merge_counters(sample["active_degree_histogram"] for sample in samples)
    merged_angles = merge_counters(sample["angle_histogram"] for sample in samples)
    return {
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "skipped": skipped,
        "counts": {name: clean_counter(counter) for name, counter in counters.items()},
        "vertices": summarize(sample["vertices"] for sample in samples),
        "edges": summarize(sample["edges"] for sample in samples),
        "branch_depth": summarize(sample["tree"].get("branchDepth") for sample in samples if sample["tree"].get("branchDepth") is not None),
        "terminal_count": summarize(sample["tree"].get("terminalCount") for sample in samples if sample["tree"].get("terminalCount") is not None),
        "border_intersections": summarize(sample["border_intersections"] for sample in samples),
        "assignment_totals": clean_counter(merged_assignments),
        "degree_histogram": clean_counter(merged_degree),
        "active_degree_histogram": clean_counter(merged_active_degree),
        "angle_histogram": clean_counter(merged_angles),
        "sample_metrics": samples,
    }


def angle_bucket(p1: tuple[float, float], p2: tuple[float, float], width: float) -> str:
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
    bucket = int(math.floor(angle / width) * width)
    return f"{bucket:03d}-{int(bucket + width):03d}"


def on_boundary(point: tuple[float, float], eps: float = 1e-7) -> bool:
    x, y = point
    return abs(x) <= eps or abs(x - 1) <= eps or abs(y) <= eps or abs(y - 1) <= eps


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def merge_counters(items: Iterable[dict[str, int]]) -> Counter:
    counter: Counter = Counter()
    for item in items:
        counter.update({str(key): int(value) for key, value in item.items()})
    return counter


def clean_counter(counter: Counter) -> dict[str, int]:
    items = [(str(key), int(value)) for key, value in counter.items() if key is not None]
    return {key: value for key, value in sorted(items)}


def summarize(values: Iterable[Any]) -> dict[str, float | int | None]:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(nums), "max": max(nums), "mean": mean(nums)}


if __name__ == "__main__":
    main()
