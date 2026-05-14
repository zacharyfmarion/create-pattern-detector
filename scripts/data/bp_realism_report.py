#!/usr/bin/env python3
"""Generate BP realism QA metrics for synthetic/reference crease graphs.

The report is intentionally graph-first. It can run against generated
``raw-manifest.jsonl`` or rendered ``manifest.jsonl`` rows without importing PIL.
If a reference row has only an image, raster metrics are attempted only when
Pillow is installed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_MANIFEST = REPO_ROOT / "data" / "references" / "bp_clean_v1" / "manifest.jsonl"

ORIENTATION_BUCKETS = ("horizontal", "vertical", "diag_pos", "diag_neg", "other")
ORIENTATION_BITS = {
    "horizontal": 1,
    "vertical": 2,
    "diag_pos": 4,
    "diag_neg": 8,
    "other": 16,
}
GRAPH_SUMMARY_KEYS = (
    "vertex_count",
    "edge_count",
    "crease_edge_count",
    "boundary_edge_count",
    "empty_space_ratio",
    "local_density_variance",
    "local_density_cv",
    "orientation_entropy",
    "axis_aligned_ratio",
    "diagonal_ratio",
    "other_orientation_ratio",
    "diagonal_run_count",
    "diagonal_run_max_edges",
    "long_diagonal_run_count",
    "stair_proxy_count",
    "fan_proxy_count",
    "grid_lattice_penalty",
    "full_sheet_grid_line_penalty",
    "regular_grid_spacing_score",
    "repeated_cell_penalty",
    "triangle_cycle_count",
    "axis_axis_diagonal_triangle_count",
    "isolated_triangle_count",
    "isolated_triangle_presence",
    "plain_grid_triangle_failure_score",
)
COMPARISON_KEYS = (
    "empty_space_ratio",
    "local_density_cv",
    "orientation_entropy",
    "diagonal_ratio",
    "diagonal_run_max_edges",
    "long_diagonal_run_count",
    "stair_proxy_count",
    "grid_lattice_penalty",
    "repeated_cell_penalty",
    "isolated_triangle_count",
    "plain_grid_triangle_failure_score",
)


@dataclass(frozen=True)
class ManifestSample:
    id: str
    row_index: int
    row: Mapping[str, Any]
    fold_path: Path | None
    image_path: Path | None
    inline_fold: Mapping[str, Any] | None


@dataclass(frozen=True)
class EdgeInfo:
    edge_index: int
    a: int
    b: int
    assignment: str
    role: str | None
    length: float
    orientation: str


class OptionalMetricUnavailable(RuntimeError):
    """Raised when an optional image-only metric cannot be computed."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Generated dataset root.")
    parser.add_argument(
        "--manifest",
        help="Generated raw-manifest.jsonl or rendered manifest.jsonl. Defaults to raw-manifest, then manifest.",
    )
    parser.add_argument(
        "--reference-manifest",
        help="Optional BP reference manifest. Defaults to data/references/bp_clean_v1/manifest.jsonl when present.",
    )
    parser.add_argument(
        "--no-default-reference",
        action="store_true",
        help="Do not auto-load data/references/bp_clean_v1/manifest.jsonl.",
    )
    parser.add_argument("--out", help="Output JSON path. Defaults to <root>/qa/bp-realism-report.json.")
    parser.add_argument("--grid-cells", type=int, default=16, help="Grid resolution for local density metrics.")
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.65,
        help="Plain grid/isolated triangle score threshold for flagged samples.",
    )
    parser.add_argument("--top", type=int, default=25, help="Number of highest-risk samples to include.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    manifest_path = Path(args.manifest) if args.manifest else _default_generated_manifest(root)
    output_path = Path(args.out) if args.out else root / "qa" / "bp-realism-report.json"

    generated = analyze_manifest(
        manifest_path=manifest_path,
        root=root,
        kind="generated",
        grid_cells=args.grid_cells,
        flag_threshold=args.flag_threshold,
        top_n=args.top,
    )

    reference_path: Path | None
    if args.reference_manifest:
        reference_path = Path(args.reference_manifest)
    elif args.no_default_reference or not DEFAULT_REFERENCE_MANIFEST.exists():
        reference_path = None
    else:
        reference_path = DEFAULT_REFERENCE_MANIFEST

    references = None
    if reference_path is not None:
        references = analyze_manifest(
            manifest_path=reference_path,
            root=reference_path.parent,
            kind="reference",
            grid_cells=args.grid_cells,
            flag_threshold=args.flag_threshold,
            top_n=args.top,
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest_path": str(manifest_path),
        "reference_manifest_path": str(reference_path) if reference_path else None,
        "config": {
            "grid_cells": args.grid_cells,
            "flag_threshold": args.flag_threshold,
            "top_n": args.top,
        },
        "generated": generated,
        "references": references,
        "comparison": compare_summaries(generated, references) if references else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote BP realism report to {output_path}")


def analyze_manifest(
    manifest_path: Path,
    root: Path,
    kind: str,
    grid_cells: int,
    flag_threshold: float,
    top_n: int,
) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)
    samples, manifest_skips = manifest_samples(rows, manifest_path=manifest_path, root=root, kind=kind)

    analyzed: list[dict[str, Any]] = []
    skips = list(manifest_skips)
    for sample in samples:
        try:
            analyzed.append(analyze_sample(sample, grid_cells=grid_cells))
        except OptionalMetricUnavailable as exc:
            skips.append(skip_record(sample, str(exc)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skips.append(skip_record(sample, f"invalid_sample: {exc}"))

    return summarize_analysis(
        manifest_path=manifest_path,
        kind=kind,
        total_rows=len(rows),
        samples=analyzed,
        skipped=skips,
        flag_threshold=flag_threshold,
        top_n=top_n,
    )


def manifest_samples(
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    root: Path,
    kind: str,
) -> tuple[list[ManifestSample], list[dict[str, Any]]]:
    samples: list[ManifestSample] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    bases = (root, manifest_path.parent)

    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id") or row.get("graph_id") or row.get("graphId") or f"row-{index}")
        if row.get("ignore") is True:
            skipped.append({"id": sample_id, "row_index": index, "reason": "ignored_by_manifest"})
            continue

        fold_path = optional_path(row, ("fold_path", "foldPath", "fold"), bases)
        image_path = optional_path(row, ("image_path", "imagePath", "image"), bases)
        inline_fold = inline_fold_from_row(row)

        if kind == "reference" and row.get("placeholder") is True:
            has_existing_asset = bool((fold_path and fold_path.exists()) or (image_path and image_path.exists()) or inline_fold)
            if not has_existing_asset:
                skipped.append({"id": sample_id, "row_index": index, "reason": "placeholder_missing_files"})
                continue

        graph_key = str(row.get("graph_id") or row.get("graphId") or fold_path or sample_id)
        if kind == "generated" and graph_key in seen:
            skipped.append(
                {
                    "id": sample_id,
                    "row_index": index,
                    "reason": "duplicate_graph_row",
                    "first_row_index": seen[graph_key],
                }
            )
            continue
        seen[graph_key] = index

        samples.append(
            ManifestSample(
                id=sample_id,
                row_index=index,
                row=row,
                fold_path=fold_path,
                image_path=image_path,
                inline_fold=inline_fold,
            )
        )

    return samples, skipped


def analyze_sample(sample: ManifestSample, grid_cells: int) -> dict[str, Any]:
    fold = sample.inline_fold
    if fold is None and sample.fold_path is not None and sample.fold_path.exists():
        fold = load_json(sample.fold_path)
    if fold is not None:
        metrics = graph_metrics(fold, row=sample.row, grid_cells=grid_cells)
        metric_type = "graph"
    elif sample.image_path is not None and sample.image_path.exists():
        metrics = raster_metrics(sample.image_path, grid_cells=grid_cells)
        metric_type = "raster"
    elif sample.fold_path is not None:
        raise ValueError(f"missing fold file {sample.fold_path}")
    elif sample.image_path is not None:
        raise ValueError(f"missing image file {sample.image_path}")
    else:
        raise ValueError("no fold_path, image_path, or inline FOLD data")

    return {
        "id": sample.id,
        "row_index": sample.row_index,
        "metric_type": metric_type,
        "fold_path": str(sample.fold_path) if sample.fold_path else None,
        "image_path": str(sample.image_path) if sample.image_path else None,
        "family": string_or_none(sample.row.get("family")),
        "bucket": string_or_none(sample.row.get("bucket")),
        "split": string_or_none(sample.row.get("split")),
        "archetype": archetype_from_metadata(sample.row, fold if fold is not None else {}),
        "metrics": metrics,
    }


def graph_metrics(fold_data: Mapping[str, Any], row: Mapping[str, Any], grid_cells: int) -> dict[str, Any]:
    fold = nested_fold(fold_data)
    vertices = coerce_vertices(fold.get("vertices_coords", []))
    edges_raw = coerce_edges(fold.get("edges_vertices", []))
    assignments_raw = list(fold.get("edges_assignment") or [])
    roles_raw = list(fold.get("edges_bpRole") or [])

    assignments = [
        str(assignments_raw[index]) if index < len(assignments_raw) else "U"
        for index in range(len(edges_raw))
    ]
    roles = [
        str(roles_raw[index]) if index < len(roles_raw) and roles_raw[index] is not None else None
        for index in range(len(edges_raw))
    ]

    valid_edges: list[EdgeInfo] = []
    invalid_edge_count = 0
    for edge_index, (a, b) in enumerate(edges_raw):
        if a == b or a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            invalid_edge_count += 1
            continue
        p1 = vertices[a]
        p2 = vertices[b]
        length = distance(p1, p2)
        if length <= 1e-12 or not math.isfinite(length):
            invalid_edge_count += 1
            continue
        valid_edges.append(
            EdgeInfo(
                edge_index=edge_index,
                a=a,
                b=b,
                assignment=assignments[edge_index],
                role=roles[edge_index],
                length=length,
                orientation=orientation_bucket(p1, p2),
            )
        )

    analysis_edges = [edge for edge in valid_edges if edge.assignment != "B" and edge.role != "border"]
    if not analysis_edges:
        analysis_edges = list(valid_edges)

    bounds = graph_bounds(vertices)
    density = density_metrics(vertices, analysis_edges, bounds, grid_cells=grid_cells)
    orientation = orientation_metrics(analysis_edges)
    line = line_grid_metrics(vertices, analysis_edges, bounds)
    repeat = repeated_cell_metrics(density["cell_masks"])
    degree = degree_metrics(len(vertices), valid_edges)
    diagonal = diagonal_run_metrics(len(vertices), analysis_edges)
    motif = motif_metrics(len(vertices), analysis_edges)
    triangle = triangle_metrics(len(vertices), analysis_edges, density["occupied_cell_count"])

    role_counts = Counter(edge.role for edge in valid_edges if edge.role)
    row_role_counts = row.get("role_counts") or row.get("roleCounts") or {}
    role_counts.update({str(key): int(value) for key, value in dict(row_role_counts).items() if key not in role_counts})
    assignment_counts = Counter(edge.assignment for edge in valid_edges)

    low_density_variation = 1.0 - min(1.0, float(density["local_density_cv"]) / 1.25)
    long_diag_presence = min(1.0, diagonal["long_diagonal_run_count"] / 3.0)
    stair_presence = min(1.0, motif["stair_proxy_density"] * 8.0)
    isolated_triangle_presence = 1.0 if triangle["isolated_triangle_count"] > 0 else 0.0
    low_diagonal_structure = 1.0 - min(1.0, diagonal["diagonal_run_max_edges"] / 3.0)
    grid_lattice_penalty = clamp01(
        0.30 * orientation["axis_aligned_ratio"]
        + 0.25 * line["full_sheet_grid_line_penalty"]
        + 0.20 * line["regular_grid_spacing_score"]
        + 0.15 * repeat["repeated_cell_penalty"]
        + 0.10 * low_density_variation
        + 0.10 * (1.0 - orientation["orientation_entropy"])
        - 0.10 * min(1.0, orientation["other_orientation_ratio"] * 4.0)
        - 0.10 * long_diag_presence
    )
    plain_grid_core = clamp01(
        0.45 * line["full_sheet_grid_line_penalty"]
        + 0.25 * orientation["axis_aligned_ratio"]
        + 0.15 * grid_lattice_penalty
        + 0.15 * repeat["repeated_cell_penalty"]
    )
    plain_grid_triangle_failure_score = clamp01(
        0.55 * plain_grid_core
        + 0.20 * grid_lattice_penalty
        + 0.15 * isolated_triangle_presence
        + 0.10 * low_diagonal_structure
        + 0.10 * (1.0 - orientation["orientation_entropy"])
        - 0.10 * long_diag_presence
        - 0.05 * stair_presence
    )

    metrics: dict[str, Any] = {
        "vertex_count": len(vertices),
        "edge_count": len(valid_edges),
        "invalid_edge_count": invalid_edge_count,
        "crease_edge_count": sum(1 for edge in valid_edges if edge.assignment in {"M", "V", "F", "U"}),
        "boundary_edge_count": sum(1 for edge in valid_edges if edge.assignment == "B" or edge.role == "border"),
        "bounds": {
            "min_x": bounds[0],
            "min_y": bounds[1],
            "max_x": bounds[2],
            "max_y": bounds[3],
            "width": bounds[4],
            "height": bounds[5],
        },
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "degree_histogram": degree["degree_histogram"],
        "role_ratios": role_ratios(role_counts),
        "grid_cells": grid_cells,
        "grid_lattice_penalty": round_float(grid_lattice_penalty),
        "plain_grid_triangle_failure_score": round_float(plain_grid_triangle_failure_score),
        "grid_lattice_penalty_components": {
            "axis_aligned_ratio": round_float(orientation["axis_aligned_ratio"]),
            "full_sheet_grid_line_penalty": round_float(line["full_sheet_grid_line_penalty"]),
            "regular_grid_spacing_score": round_float(line["regular_grid_spacing_score"]),
            "repeated_cell_penalty": round_float(repeat["repeated_cell_penalty"]),
            "low_density_variation": round_float(low_density_variation),
            "low_orientation_entropy": round_float(1.0 - orientation["orientation_entropy"]),
            "long_diagonal_run_credit": round_float(long_diag_presence),
        },
        "plain_grid_triangle_failure_components": {
            "plain_grid_core": round_float(plain_grid_core),
            "isolated_triangle_presence": round_float(isolated_triangle_presence),
            "low_diagonal_structure": round_float(low_diagonal_structure),
            "low_orientation_entropy": round_float(1.0 - orientation["orientation_entropy"]),
            "long_diagonal_run_credit": round_float(long_diag_presence),
            "stair_proxy_credit": round_float(stair_presence),
        },
    }
    metrics.update(without_internal_density(density))
    metrics.update(orientation)
    metrics.update(line)
    metrics.update(repeat)
    metrics.update(diagonal)
    metrics.update(motif)
    metrics.update(triangle)
    return normalize_floats(metrics)


def density_metrics(
    vertices: Sequence[tuple[float, float]],
    edges: Sequence[EdgeInfo],
    bounds: tuple[float, float, float, float, float, float],
    grid_cells: int,
) -> dict[str, Any]:
    min_x, min_y, _, _, width, height = bounds
    total_cells = grid_cells * grid_cells
    densities = [0.0 for _ in range(total_cells)]
    masks = [0 for _ in range(total_cells)]
    max_extent = max(width, height, 1e-9)

    for edge in edges:
        p1 = vertices[edge.a]
        p2 = vertices[edge.b]
        samples = max(2, min(2048, int(edge.length / max_extent * grid_cells * 8) + 1))
        contribution = edge.length / samples
        bit = ORIENTATION_BITS[edge.orientation]
        for sample_index in range(samples):
            t = (sample_index + 0.5) / samples
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            cell = grid_cell(x, y, min_x, min_y, width, height, grid_cells)
            densities[cell] += contribution
            masks[cell] |= bit

    occupied = [value for value in densities if value > 0]
    occupied_count = len(occupied)
    mean_density = sum(densities) / max(1, total_cells)
    if mean_density > 0:
        variance = sum((value - mean_density) ** 2 for value in densities) / total_cells
        local_density_variance = variance / (mean_density**2)
        local_density_cv = math.sqrt(variance) / mean_density
    else:
        local_density_variance = 0.0
        local_density_cv = 0.0

    return {
        "empty_space_ratio": 1.0 - occupied_count / max(1, total_cells),
        "occupied_cell_ratio": occupied_count / max(1, total_cells),
        "occupied_cell_count": occupied_count,
        "local_density_variance": local_density_variance,
        "local_density_cv": local_density_cv,
        "cell_masks": masks,
    }


def orientation_metrics(edges: Sequence[EdgeInfo]) -> dict[str, Any]:
    length_by_bucket = {bucket: 0.0 for bucket in ORIENTATION_BUCKETS}
    count_by_bucket = {bucket: 0 for bucket in ORIENTATION_BUCKETS}
    for edge in edges:
        length_by_bucket[edge.orientation] += edge.length
        count_by_bucket[edge.orientation] += 1

    total_length = sum(length_by_bucket.values())
    if total_length <= 0:
        histogram = {bucket: 0.0 for bucket in ORIENTATION_BUCKETS}
    else:
        histogram = {bucket: length_by_bucket[bucket] / total_length for bucket in ORIENTATION_BUCKETS}

    entropy = shannon_entropy(histogram.values(), bucket_count=len(ORIENTATION_BUCKETS))
    return {
        "orientation_histogram": {bucket: round_float(histogram[bucket]) for bucket in ORIENTATION_BUCKETS},
        "orientation_counts": count_by_bucket,
        "orientation_entropy": round_float(entropy),
        "axis_aligned_ratio": round_float(histogram["horizontal"] + histogram["vertical"]),
        "diagonal_ratio": round_float(histogram["diag_pos"] + histogram["diag_neg"]),
        "other_orientation_ratio": round_float(histogram["other"]),
        "diagonal_balance": round_float(
            1.0
            - abs(histogram["diag_pos"] - histogram["diag_neg"])
            / max(histogram["diag_pos"] + histogram["diag_neg"], 1e-9)
        )
        if histogram["diag_pos"] + histogram["diag_neg"] > 0
        else 0.0,
    }


def line_grid_metrics(
    vertices: Sequence[tuple[float, float]],
    edges: Sequence[EdgeInfo],
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    min_x, min_y, _, _, width, height = bounds
    horizontal: dict[float, list[tuple[float, float]]] = defaultdict(list)
    vertical: dict[float, list[tuple[float, float]]] = defaultdict(list)

    for edge in edges:
        p1 = vertices[edge.a]
        p2 = vertices[edge.b]
        if edge.orientation == "horizontal":
            y_key = quantized_axis_coord((p1[1] + p2[1]) * 0.5, min_y, height)
            horizontal[y_key].append((min(p1[0], p2[0]), max(p1[0], p2[0])))
        elif edge.orientation == "vertical":
            x_key = quantized_axis_coord((p1[0] + p2[0]) * 0.5, min_x, width)
            vertical[x_key].append((min(p1[1], p2[1]), max(p1[1], p2[1])))

    horizontal_stats = line_group_stats(horizontal, width)
    vertical_stats = line_group_stats(vertical, height)
    total_groups = horizontal_stats["line_group_count"] + vertical_stats["line_group_count"]
    long_groups = horizontal_stats["long_line_group_count"] + vertical_stats["long_line_group_count"]
    long_line_ratio = long_groups / max(1, total_groups)
    regular_spacing = (spacing_regular_score(sorted(horizontal)) + spacing_regular_score(sorted(vertical))) / 2.0

    axis_edge_count = sum(1 for edge in edges if edge.orientation in {"horizontal", "vertical"})
    full_sheet_grid_line_penalty = clamp01(
        long_line_ratio
        * min(1.0, total_groups / 8.0)
        * min(1.0, axis_edge_count / max(1, len(edges)) * 1.2)
    )

    return {
        "axis_line_group_count": total_groups,
        "full_sheet_axis_line_count": long_groups,
        "full_sheet_grid_line_penalty": round_float(full_sheet_grid_line_penalty),
        "regular_grid_spacing_score": round_float(regular_spacing),
        "horizontal_line_groups": horizontal_stats,
        "vertical_line_groups": vertical_stats,
    }


def repeated_cell_metrics(cell_masks: Sequence[int]) -> dict[str, Any]:
    non_empty = [mask for mask in cell_masks if mask]
    if not non_empty:
        return {
            "repeated_cell_penalty": 0.0,
            "cell_signature_count": 0,
            "dominant_cell_signature_fraction": 0.0,
            "cell_signature_entropy": 0.0,
            "cell_signature_counts": {},
        }

    counts = Counter(non_empty)
    dominant_fraction = max(counts.values()) / len(non_empty)
    occupied_ratio = len(non_empty) / max(1, len(cell_masks))
    signature_entropy = shannon_entropy(
        (count / len(non_empty) for count in counts.values()),
        bucket_count=max(2, len(counts)),
    )
    penalty = clamp01(dominant_fraction * min(1.0, occupied_ratio * 2.0) * (1.0 - 0.20 * signature_entropy))
    return {
        "repeated_cell_penalty": round_float(penalty),
        "cell_signature_count": len(counts),
        "dominant_cell_signature_fraction": round_float(dominant_fraction),
        "cell_signature_entropy": round_float(signature_entropy),
        "cell_signature_counts": {
            signature_from_mask(mask): count for mask, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }


def degree_metrics(vertex_count: int, edges: Sequence[EdgeInfo]) -> dict[str, Any]:
    degrees = [0 for _ in range(vertex_count)]
    for edge in edges:
        degrees[edge.a] += 1
        degrees[edge.b] += 1
    return {"degree_histogram": {str(k): v for k, v in sorted(Counter(degrees).items())}}


def diagonal_run_metrics(vertex_count: int, edges: Sequence[EdgeInfo]) -> dict[str, Any]:
    diagonal_edges = [edge for edge in edges if edge.orientation in {"diag_pos", "diag_neg"}]
    by_index = {edge.edge_index: edge for edge in diagonal_edges}
    incident: dict[tuple[str, int], list[int]] = defaultdict(list)
    for edge in diagonal_edges:
        incident[(edge.orientation, edge.a)].append(edge.edge_index)
        incident[(edge.orientation, edge.b)].append(edge.edge_index)

    neighbors: dict[int, set[int]] = {edge.edge_index: set() for edge in diagonal_edges}
    for ids in incident.values():
        for edge_id in ids:
            neighbors[edge_id].update(other for other in ids if other != edge_id)

    seen: set[int] = set()
    run_edge_counts: list[int] = []
    run_lengths: list[float] = []
    for edge_id in by_index:
        if edge_id in seen:
            continue
        stack = [edge_id]
        seen.add(edge_id)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        run_edge_counts.append(len(component))
        run_lengths.append(sum(by_index[item].length for item in component))

    return {
        "diagonal_edge_count": len(diagonal_edges),
        "diagonal_run_count": len(run_edge_counts),
        "diagonal_run_max_edges": max(run_edge_counts, default=0),
        "diagonal_run_mean_edges": round_float(sum(run_edge_counts) / len(run_edge_counts)) if run_edge_counts else 0.0,
        "diagonal_run_max_length": round_float(max(run_lengths, default=0.0)),
        "long_diagonal_run_count": sum(1 for count in run_edge_counts if count >= 3),
    }


def motif_metrics(vertex_count: int, edges: Sequence[EdgeInfo]) -> dict[str, Any]:
    incident_orientations: list[Counter[str]] = [Counter() for _ in range(vertex_count)]
    for edge in edges:
        incident_orientations[edge.a][edge.orientation] += 1
        incident_orientations[edge.b][edge.orientation] += 1

    diagonal_turn_vertices = 0
    axis_diagonal_junctions = 0
    stair_proxy_count = 0
    fan_proxy_count = 0
    for orientations in incident_orientations:
        degree = sum(orientations.values())
        has_horizontal = orientations["horizontal"] > 0
        has_vertical = orientations["vertical"] > 0
        has_axis = has_horizontal or has_vertical
        has_diag_pos = orientations["diag_pos"] > 0
        has_diag_neg = orientations["diag_neg"] > 0
        has_diagonal = has_diag_pos or has_diag_neg
        orientation_kinds = sum(1 for bucket in ORIENTATION_BUCKETS if orientations[bucket] > 0)
        if has_diag_pos and has_diag_neg:
            diagonal_turn_vertices += 1
        if has_axis and has_diagonal:
            axis_diagonal_junctions += 1
        if has_axis and has_diagonal and degree >= 3:
            stair_proxy_count += 1
        if degree >= 5 and orientation_kinds >= 3:
            fan_proxy_count += 1

    return {
        "diagonal_turn_vertex_count": diagonal_turn_vertices,
        "axis_diagonal_junction_count": axis_diagonal_junctions,
        "stair_proxy_count": stair_proxy_count,
        "stair_proxy_density": round_float(stair_proxy_count / max(1, vertex_count)),
        "fan_proxy_count": fan_proxy_count,
        "fan_proxy_density": round_float(fan_proxy_count / max(1, vertex_count)),
    }


def triangle_metrics(vertex_count: int, edges: Sequence[EdgeInfo], occupied_cell_count: int) -> dict[str, Any]:
    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    edge_by_vertices: dict[tuple[int, int], EdgeInfo] = {}
    for edge in edges:
        key = sorted_edge_key(edge.a, edge.b)
        edge_by_vertices[key] = edge
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)

    triangles: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[str, str, str]]] = []
    for u in range(vertex_count):
        for v in adjacency[u]:
            if v <= u:
                continue
            for w in adjacency[u].intersection(adjacency[v]):
                if w <= v:
                    continue
                edge_keys = (sorted_edge_key(u, v), sorted_edge_key(u, w), sorted_edge_key(v, w))
                edge_infos = [edge_by_vertices[key] for key in edge_keys]
                orientations = tuple(sorted(edge.orientation for edge in edge_infos))
                triangles.append((edge_keys[0], edge_keys[1], edge_keys[2], orientations))

    edge_triangle_counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        edge_triangle_counts.update(triangle[:3])

    axis_axis_diag = 0
    isolated = 0
    for triangle in triangles:
        orientations = triangle[3]
        axis_count = sum(1 for item in orientations if item in {"horizontal", "vertical"})
        diag_count = sum(1 for item in orientations if item in {"diag_pos", "diag_neg"})
        if axis_count == 2 and diag_count == 1:
            axis_axis_diag += 1
            if max(edge_triangle_counts[key] for key in triangle[:3]) <= 2:
                isolated += 1

    pressure = min(1.0, isolated / max(1.0, occupied_cell_count * 0.15))
    return {
        "triangle_cycle_count": len(triangles),
        "axis_axis_diagonal_triangle_count": axis_axis_diag,
        "isolated_triangle_count": isolated,
        "isolated_triangle_presence": 1.0 if isolated > 0 else 0.0,
        "isolated_triangle_pressure": round_float(pressure),
    }


def raster_metrics(image_path: Path, grid_cells: int) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalMetricUnavailable("Pillow is unavailable for image-only reference row") from exc

    with Image.open(image_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = list(gray.getdata())
    if not pixels:
        raise ValueError(f"empty image {image_path}")

    mean = sum(pixels) / len(pixels)
    threshold = min(245, max(16, mean - 18))
    cell_ink = [0 for _ in range(grid_cells * grid_cells)]
    cell_pixels = [0 for _ in range(grid_cells * grid_cells)]

    for y in range(height):
        cy = min(grid_cells - 1, int(y / max(1, height) * grid_cells))
        for x in range(width):
            cx = min(grid_cells - 1, int(x / max(1, width) * grid_cells))
            cell = cy * grid_cells + cx
            cell_pixels[cell] += 1
            if pixels[y * width + x] < threshold:
                cell_ink[cell] += 1

    densities = [ink / max(1, total) for ink, total in zip(cell_ink, cell_pixels)]
    occupied = [value for value in densities if value > 0.002]
    mean_density = sum(densities) / len(densities)
    if mean_density > 0:
        variance = sum((value - mean_density) ** 2 for value in densities) / len(densities)
        local_density_variance = variance / (mean_density**2)
        local_density_cv = math.sqrt(variance) / mean_density
    else:
        local_density_variance = 0.0
        local_density_cv = 0.0

    return normalize_floats(
        {
            "image_width": width,
            "image_height": height,
            "grid_cells": grid_cells,
            "ink_ratio": sum(cell_ink) / max(1, sum(cell_pixels)),
            "empty_space_ratio": 1.0 - len(occupied) / max(1, len(densities)),
            "occupied_cell_ratio": len(occupied) / max(1, len(densities)),
            "local_density_variance": local_density_variance,
            "local_density_cv": local_density_cv,
        }
    )


def summarize_analysis(
    manifest_path: Path,
    kind: str,
    total_rows: int,
    samples: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    flag_threshold: float,
    top_n: int,
) -> dict[str, Any]:
    graph_samples = [sample for sample in samples if sample.get("metric_type") == "graph"]
    raster_samples = [sample for sample in samples if sample.get("metric_type") == "raster"]
    metrics_summary = summarize_metric_keys(graph_samples, GRAPH_SUMMARY_KEYS)

    archetypes = Counter(string_or_none(sample.get("archetype")) or "unknown" for sample in samples)
    families = Counter(string_or_none(sample.get("family")) or "unknown" for sample in samples)
    assignment_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    orientation_totals: Counter[str] = Counter()
    for sample in graph_samples:
        metrics = sample["metrics"]
        assignment_counts.update({str(k): int(v) for k, v in dict(metrics.get("assignment_counts", {})).items()})
        role_counts.update({str(k): int(v) for k, v in dict(metrics.get("role_counts", {})).items()})
        for bucket, value in dict(metrics.get("orientation_histogram", {})).items():
            orientation_totals[str(bucket)] += float(value)

    flagged = []
    for sample in graph_samples:
        metrics = sample["metrics"]
        score = float(metrics.get("plain_grid_triangle_failure_score", 0.0))
        if score >= flag_threshold:
            flagged.append(
                {
                    "id": sample["id"],
                    "score": score,
                    "grid_lattice_penalty": metrics.get("grid_lattice_penalty"),
                    "repeated_cell_penalty": metrics.get("repeated_cell_penalty"),
                    "isolated_triangle_count": metrics.get("isolated_triangle_count"),
                    "local_density_cv": metrics.get("local_density_cv"),
                    "orientation_entropy": metrics.get("orientation_entropy"),
                    "fold_path": sample.get("fold_path"),
                }
            )
    flagged = sorted(flagged, key=lambda item: item["score"], reverse=True)[:top_n]
    top_risk = sorted(
        (
            {
                "id": sample["id"],
                "score": sample["metrics"].get("plain_grid_triangle_failure_score", 0.0),
                "grid_lattice_penalty": sample["metrics"].get("grid_lattice_penalty"),
                "repeated_cell_penalty": sample["metrics"].get("repeated_cell_penalty"),
                "fold_path": sample.get("fold_path"),
            }
            for sample in graph_samples
        ),
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:top_n]

    return {
        "kind": kind,
        "manifest_path": str(manifest_path),
        "row_count": total_rows,
        "sample_count": len(samples),
        "graph_sample_count": len(graph_samples),
        "raster_sample_count": len(raster_samples),
        "skipped_count": len(skipped),
        "skipped": list(skipped)[:100],
        "metrics_summary": metrics_summary,
        "families": dict(sorted(families.items())),
        "archetypes": dict(sorted(archetypes.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "mean_orientation_histogram": {
            bucket: round_float(orientation_totals[bucket] / max(1, len(graph_samples)))
            for bucket in ORIENTATION_BUCKETS
        },
        "flag_threshold": flag_threshold,
        "flagged_plain_grid_triangle_samples": flagged,
        "top_plain_grid_triangle_risk": top_risk,
        "samples": list(samples),
    }


def compare_summaries(generated: Mapping[str, Any], references: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not references or not references.get("graph_sample_count"):
        return None
    generated_summary = generated.get("metrics_summary", {})
    reference_summary = references.get("metrics_summary", {})
    metrics: dict[str, Any] = {}
    for key in COMPARISON_KEYS:
        generated_stats = generated_summary.get(key)
        reference_stats = reference_summary.get(key)
        if not generated_stats or not reference_stats:
            continue
        generated_mean = float(generated_stats["mean"])
        reference_mean = float(reference_stats["mean"])
        metrics[key] = {
            "generated_mean": round_float(generated_mean),
            "reference_mean": round_float(reference_mean),
            "delta": round_float(generated_mean - reference_mean),
            "ratio": round_float(generated_mean / reference_mean) if abs(reference_mean) > 1e-12 else None,
        }
    return {"metrics": metrics}


def summarize_metric_keys(samples: Sequence[Mapping[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in keys:
        values = []
        for sample in samples:
            value = sample["metrics"].get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            summary[key] = numeric_summary(values)
    return summary


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round_float(ordered[0]),
        "p10": round_float(percentile(ordered, 0.10)),
        "median": round_float(percentile(ordered, 0.50)),
        "p90": round_float(percentile(ordered, 0.90)),
        "max": round_float(ordered[-1]),
        "mean": round_float(sum(ordered) / len(ordered)),
    }


def percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_number}")
            rows.append(row)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _default_generated_manifest(root: Path) -> Path:
    raw_manifest = root / "raw-manifest.jsonl"
    rendered_manifest = root / "manifest.jsonl"
    if raw_manifest.exists():
        return raw_manifest
    if rendered_manifest.exists():
        return rendered_manifest
    raise FileNotFoundError(f"No raw-manifest.jsonl or manifest.jsonl under {root}")


def optional_path(row: Mapping[str, Any], names: Sequence[str], bases: Sequence[Path]) -> Path | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value:
            return resolve_path(value, bases)
    return None


def resolve_path(value: str, bases: Sequence[Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [base / path for base in bases]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def inline_fold_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    fold = row.get("fold")
    if isinstance(fold, Mapping):
        return fold
    if "vertices_coords" in row and "edges_vertices" in row:
        return row
    return None


def nested_fold(fold_data: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = fold_data.get("fold")
    if isinstance(nested, Mapping) and "vertices_coords" in nested:
        return nested
    return fold_data


def coerce_vertices(raw: Any) -> list[tuple[float, float]]:
    vertices: list[tuple[float, float]] = []
    for vertex in raw:
        if not isinstance(vertex, Sequence) or len(vertex) < 2:
            raise ValueError("invalid vertices_coords entry")
        x = float(vertex[0])
        y = float(vertex[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("non-finite vertex coordinate")
        vertices.append((x, y))
    if not vertices:
        raise ValueError("FOLD has no vertices_coords")
    return vertices


def coerce_edges(raw: Any) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for edge in raw:
        if not isinstance(edge, Sequence) or len(edge) < 2:
            raise ValueError("invalid edges_vertices entry")
        edges.append((int(edge[0]), int(edge[1])))
    if not edges:
        raise ValueError("FOLD has no edges_vertices")
    return edges


def graph_bounds(vertices: Sequence[tuple[float, float]]) -> tuple[float, float, float, float, float, float]:
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    return (min_x, min_y, max_x, max_y, width, height)


def orientation_bucket(p1: tuple[float, float], p2: tuple[float, float], tolerance_degrees: float = 7.5) -> str:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    if min(angle, 180.0 - angle) <= tolerance_degrees:
        return "horizontal"
    if abs(angle - 90.0) <= tolerance_degrees:
        return "vertical"
    if abs(angle - 45.0) <= tolerance_degrees:
        return "diag_pos"
    if abs(angle - 135.0) <= tolerance_degrees:
        return "diag_neg"
    return "other"


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def grid_cell(
    x: float,
    y: float,
    min_x: float,
    min_y: float,
    width: float,
    height: float,
    grid_cells: int,
) -> int:
    ix = min(grid_cells - 1, max(0, int((x - min_x) / width * grid_cells)))
    iy = min(grid_cells - 1, max(0, int((y - min_y) / height * grid_cells)))
    return iy * grid_cells + ix


def quantized_axis_coord(value: float, origin: float, span: float) -> float:
    return round((value - origin) / max(span, 1e-9), 5)


def line_group_stats(groups: Mapping[float, Sequence[tuple[float, float]]], span: float) -> dict[str, Any]:
    coverages: list[float] = []
    for intervals in groups.values():
        coverages.append(merged_interval_length(intervals, tolerance=max(span, 1e-9) * 1e-7) / max(span, 1e-9))
    long_count = sum(1 for coverage in coverages if coverage >= 0.75)
    return {
        "line_group_count": len(groups),
        "long_line_group_count": long_count,
        "mean_coverage": round_float(sum(coverages) / len(coverages)) if coverages else 0.0,
        "max_coverage": round_float(max(coverages, default=0.0)),
    }


def merged_interval_length(intervals: Sequence[tuple[float, float]], tolerance: float) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + tolerance:
            current_end = max(current_end, end)
        else:
            total += max(0.0, current_end - current_start)
            current_start, current_end = start, end
    total += max(0.0, current_end - current_start)
    return total


def spacing_regular_score(coords: Sequence[float]) -> float:
    unique = sorted(set(coords))
    if len(unique) < 4:
        return 0.0
    gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    if len(gaps) < 3:
        return 0.0
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return 0.0
    variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    cv = math.sqrt(variance) / mean_gap
    return clamp01((1.0 - cv * 3.0) * min(1.0, len(unique) / 8.0))


def shannon_entropy(values: Iterable[float], bucket_count: int) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if not positive or bucket_count <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in positive)
    return clamp01(entropy / math.log(bucket_count))


def signature_from_mask(mask: int) -> str:
    parts = [bucket for bucket in ORIENTATION_BUCKETS if mask & ORIENTATION_BITS[bucket]]
    return "+".join(parts) if parts else "empty"


def sorted_edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def role_ratios(role_counts: Mapping[Any, int]) -> dict[str, float]:
    total = sum(int(value) for value in role_counts.values())
    if total <= 0:
        return {}
    return {str(key): round_float(int(value) / total) for key, value in sorted(role_counts.items())}


def without_internal_density(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "cell_masks"}


def archetype_from_metadata(row: Mapping[str, Any], fold: Mapping[str, Any]) -> str | None:
    for container in (
        row.get("design_tree"),
        row.get("designTree"),
        fold.get("design_tree"),
        fold.get("designTree"),
        row.get("metadata"),
    ):
        if isinstance(container, Mapping) and container.get("archetype") is not None:
            return str(container["archetype"])
    if row.get("archetype") is not None:
        return str(row["archetype"])
    return None


def skip_record(sample: ManifestSample, reason: str) -> dict[str, Any]:
    return {
        "id": sample.id,
        "row_index": sample.row_index,
        "reason": reason,
        "fold_path": str(sample.fold_path) if sample.fold_path else None,
        "image_path": str(sample.image_path) if sample.image_path else None,
    }


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def round_float(value: float) -> float:
    return round(float(value), 6)


def normalize_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round_float(value)
    if isinstance(value, dict):
        return {key: normalize_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_floats(item) for item in value]
    return value


if __name__ == "__main__":
    main()
