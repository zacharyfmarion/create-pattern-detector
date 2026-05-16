#!/usr/bin/env python3
"""Run the Phase 2 deterministic vectorizer on a fixture manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fold_parser import FOLDParser  # noqa: E402
from src.vectorization import PlanarGraphBuilder, PlanarGraphBuilderConfig, render_vectorizer_evidence  # noqa: E402
from src.vectorization.metrics import evaluate_graph, metrics_from_results  # noqa: E402


SEG_COLORS = np.array(
    [
        [255, 255, 255],
        [220, 40, 40],
        [40, 80, 220],
        [10, 10, 10],
        [140, 140, 140],
    ],
    dtype=np.uint8,
)

ASSIGNMENT_RGB = {
    0: (220, 40, 40),
    1: (40, 80, 220),
    2: (0, 0, 0),
    3: (110, 110, 110),
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["records"])


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def seg_to_rgb(segmentation: np.ndarray) -> Image.Image:
    rgb = SEG_COLORS[np.clip(segmentation, 0, len(SEG_COLORS) - 1)]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def font(size: int = 13) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_line_hypotheses(base: Image.Image, lines: list, peaks: np.ndarray) -> Image.Image:
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for line in lines[:500]:
        draw.line(
            (
                float(line.p0[0]),
                float(line.p0[1]),
                float(line.p1[0]),
                float(line.p1[1]),
            ),
            fill=(0, 160, 80, 130),
            width=1,
        )
    for x, y in peaks[:1200]:
        draw.ellipse((float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2), fill=(255, 170, 0, 180))
    return image


def draw_predicted_graph(base: Image.Image, result) -> Image.Image:
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    vertices = result.pixel_vertices
    for edge_idx, (v1_idx, v2_idx) in enumerate(result.edges_vertices):
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        color = ASSIGNMENT_RGB.get(int(result.edges_assignment[edge_idx]), (120, 120, 120))
        draw.line((float(v1[0]), float(v1[1]), float(v2[0]), float(v2[1])), fill=(*color, 220), width=2)
    for x, y in vertices:
        draw.ellipse((float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2), fill=(255, 180, 0, 230))
    return image


def labeled_panel(image: Image.Image, label: str, width: int) -> Image.Image:
    label_h = 56
    panel = Image.new("RGB", (width, width + label_h), "white")
    resized = image.copy()
    resized.thumbnail((width, width), Image.Resampling.LANCZOS)
    panel.paste(resized, ((width - resized.width) // 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, width, width, width + label_h), fill=(248, 248, 248))
    draw.multiline_text((8, width + 6), label, fill=(20, 20, 20), font=font(12), spacing=2)
    return panel


def save_overlay(path: Path, filename: str, gt_rgb: Image.Image, result, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 360
    line_panel = draw_line_hypotheses(gt_rgb, result.debug.get("lines", []), result.debug.get("junction_peaks", np.empty((0, 2))))
    graph_panel = draw_predicted_graph(gt_rgb, result)

    summary = (
        f"{filename[:42]}\n"
        f"V {metrics.matched_vertices}/{metrics.gt_vertices} "
        f"P {metrics.vertex_precision:.2f} R {metrics.vertex_recall:.2f}\n"
        f"E {metrics.matched_edges}/{metrics.gt_edges} "
        f"P {metrics.edge_precision:.2f} R {metrics.edge_recall:.2f}"
    )
    panels = [
        labeled_panel(gt_rgb, "Rendered GT labels", panel_w),
        labeled_panel(line_panel, f"Line hypotheses: {len(result.debug.get('lines', []))}", panel_w),
        labeled_panel(graph_panel, summary, panel_w),
    ]
    sheet = Image.new("RGB", (panel_w * 3, panels[0].height), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (idx * panel_w, 0))
    sheet.save(path)


def save_contact_sheet(overlay_paths: list[Path], output_path: Path, columns: int = 2) -> None:
    if not overlay_paths:
        return
    thumbs = []
    for path in overlay_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 280), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (720, 280), "white")
        canvas.paste(image, ((720 - image.width) // 2, (280 - image.height) // 2))
        thumbs.append(canvas)
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 720, rows * 280), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 720, (idx // columns) * 280))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_ranked_contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    sort_key: str,
    reverse: bool = False,
    limit: int = 12,
) -> None:
    ranked = sorted(rows, key=lambda row: float(row.get(sort_key, 0.0)), reverse=reverse)
    paths = [Path(row["overlay_path"]) for row in ranked[:limit]]
    save_contact_sheet(paths, output_path)


def copy_failure_overlays(rows: list[dict[str, Any]], output_dir: Path) -> None:
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        structural = row.get("structural_validity", {})
        failed_metrics = (
            float(row["vertex_recall"]) < 0.99
            or float(row["vertex_precision"]) < 0.99
            or float(row["edge_recall"]) < 0.98
            or float(row["edge_precision"]) < 0.98
            or float(row["assignment_accuracy"]) < 0.99
            or not bool(structural.get("valid", False))
        )
        if not failed_metrics:
            continue
        source = Path(row["overlay_path"])
        if source.exists():
            shutil.copy2(source, failures_dir / source.name)


def bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["bucket"])].append(row)
    return {bucket: summarize_rows(items) for bucket, items in sorted(by_bucket.items())}


def failure_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "vertex_precision_below_99": 0,
        "vertex_recall_below_99": 0,
        "edge_precision_below_98": 0,
        "edge_recall_below_98": 0,
        "assignment_accuracy_below_99": 0,
        "structural_invalid": 0,
    }
    structural_errors: dict[str, int] = defaultdict(int)
    for row in rows:
        summary["vertex_precision_below_99"] += int(float(row["vertex_precision"]) < 0.99)
        summary["vertex_recall_below_99"] += int(float(row["vertex_recall"]) < 0.99)
        summary["edge_precision_below_98"] += int(float(row["edge_precision"]) < 0.98)
        summary["edge_recall_below_98"] += int(float(row["edge_recall"]) < 0.98)
        summary["assignment_accuracy_below_99"] += int(float(row["assignment_accuracy"]) < 0.99)
        structural = row.get("structural_validity", {})
        if not bool(structural.get("valid", False)):
            summary["structural_invalid"] += 1
            for error in structural.get("errors", []):
                structural_errors[str(error)] += 1
    return {**summary, **{f"structural_error:{key}": value for key, value in structural_errors.items()}}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    totals = {
        "files": len(rows),
        "gt_vertices": sum(int(row["gt_vertices"]) for row in rows),
        "pred_vertices": sum(int(row["pred_vertices"]) for row in rows),
        "matched_vertices": sum(int(row["matched_vertices"]) for row in rows),
        "gt_edges": sum(int(row["gt_edges"]) for row in rows),
        "pred_edges": sum(int(row["pred_edges"]) for row in rows),
        "matched_edges": sum(int(row["matched_edges"]) for row in rows),
        "assignment_total": sum(int(row["assignment_total"]) for row in rows),
        "assignment_correct": sum(int(row["assignment_correct"]) for row in rows),
        "structurally_valid_files": sum(bool(row["structural_validity"]["valid"]) for row in rows),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
    }
    return {
        **totals,
        "vertex_precision": ratio(totals["matched_vertices"], totals["pred_vertices"]),
        "vertex_recall": ratio(totals["matched_vertices"], totals["gt_vertices"]),
        "edge_precision": ratio(totals["matched_edges"], totals["pred_edges"]),
        "edge_recall": ratio(totals["matched_edges"], totals["gt_edges"]),
        "assignment_accuracy": ratio(totals["assignment_correct"], totals["assignment_total"]),
        "structural_validity_rate": ratio(totals["structurally_valid_files"], totals["files"]),
        "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in rows])),
        "max_elapsed_seconds": float(np.max([float(row["elapsed_seconds"]) for row in rows])),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def run(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest)
    if args.max_edges is not None:
        records = [record for record in records if int(record["edges"]) <= args.max_edges]
    if args.max_files is not None:
        records = records[: args.max_files]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    parser = FOLDParser()
    builder = PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            image_size=args.image_size,
            hough_threshold=args.hough_threshold,
            hough_min_line_length=args.hough_min_line_length,
            hough_max_line_gap=args.hough_max_line_gap,
            min_edge_support=args.min_edge_support,
            line_vertex_distance_px=args.line_vertex_distance_px,
            junction_threshold=args.junction_threshold,
            junction_nms_radius=args.junction_nms_radius,
            vertex_merge_px=args.vertex_merge_px,
            max_intersection_lines=args.max_intersection_lines,
            add_intersection_vertices=args.add_intersection_vertices,
            direct_edge_fallback=not args.disable_direct_edge_fallback,
            direct_edge_max_length_px=args.direct_edge_max_length_px,
            direct_edge_min_support=args.direct_edge_min_support,
            direct_edge_max_vertices=args.direct_edge_max_vertices,
            direct_edge_short_max_length_px=args.direct_edge_short_max_length_px,
            direct_edge_short_max_vertices=args.direct_edge_short_max_vertices,
            planar_cleanup=not args.disable_planar_cleanup,
            planar_cleanup_max_edges=args.planar_cleanup_max_edges,
            planar_split_vertex_distance_px=args.planar_split_vertex_distance_px,
        )
    )

    metrics_rows = []
    metrics_objects = []
    overlay_paths: list[Path] = []

    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] starting {record['bucket']} "
            f"{record['vertices']}V/{record['edges']}E {record['id'][:48]}",
            flush=True,
        )
        fold_path = resolve_path(record["path"])
        start = perf_counter()
        cp = parser.parse(fold_path)
        rendered = render_vectorizer_evidence(
            cp,
            image_size=args.image_size,
            padding=args.padding,
            line_width=args.line_width,
            junction_sigma=args.junction_sigma,
        )
        result = builder.build(rendered.evidence)
        metrics = evaluate_graph(
            result,
            gt_vertices=rendered.pixel_vertices,
            gt_edges=rendered.edges,
            gt_assignments=rendered.assignments,
            vertex_tolerance_px=args.vertex_tolerance,
        )
        elapsed = perf_counter() - start
        metrics_objects.append(metrics)

        row = {
            "id": record["id"],
            "bucket": record["bucket"],
            "path": record["path"],
            "elapsed_seconds": elapsed,
            **metrics.to_dict(),
        }
        metrics_rows.append(row)

        overlay_path = overlays_dir / f"{index:03d}_{record['bucket']}_{record['id'][:80]}.png"
        save_overlay(
            overlay_path,
            record["id"],
            seg_to_rgb(rendered.evidence.assignment_labels),
            result,
            metrics,
        )
        overlay_paths.append(overlay_path)
        row["overlay_path"] = overlay_path.as_posix()
        print(
            f"[{index}/{len(records)}] {record['bucket']} "
            f"V-R {metrics.vertex_recall:.3f} E-R {metrics.edge_recall:.3f} "
            f"pred {result.num_vertices}V/{result.num_edges}E in {elapsed:.2f}s",
            flush=True,
        )

    summary = metrics_from_results(metrics_objects)
    summary["manifest"] = args.manifest.as_posix()
    summary["image_size"] = args.image_size
    summary["records"] = len(records)
    summary["config"] = {
        "padding": args.padding,
        "line_width": args.line_width,
        "junction_sigma": args.junction_sigma,
        "vertex_tolerance": args.vertex_tolerance,
        "hough_threshold": args.hough_threshold,
        "hough_min_line_length": args.hough_min_line_length,
        "hough_max_line_gap": args.hough_max_line_gap,
        "min_edge_support": args.min_edge_support,
        "line_vertex_distance_px": args.line_vertex_distance_px,
        "junction_threshold": args.junction_threshold,
        "junction_nms_radius": args.junction_nms_radius,
        "vertex_merge_px": args.vertex_merge_px,
        "max_intersection_lines": args.max_intersection_lines,
        "add_intersection_vertices": args.add_intersection_vertices,
        "direct_edge_fallback": not args.disable_direct_edge_fallback,
        "direct_edge_max_length_px": args.direct_edge_max_length_px,
        "direct_edge_min_support": args.direct_edge_min_support,
        "direct_edge_max_vertices": args.direct_edge_max_vertices,
        "direct_edge_short_max_length_px": args.direct_edge_short_max_length_px,
        "direct_edge_short_max_vertices": args.direct_edge_short_max_vertices,
        "planar_cleanup": not args.disable_planar_cleanup,
        "planar_cleanup_max_edges": args.planar_cleanup_max_edges,
        "planar_split_vertex_distance_px": args.planar_split_vertex_distance_px,
    }
    summary["bucket_summary"] = bucket_summary(metrics_rows)
    summary["failure_summary"] = failure_summary(metrics_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_file_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics_rows:
            handle.write(json.dumps(row) + "\n")
    save_contact_sheet(overlay_paths[: args.contact_sheet_limit], output_dir / "contact_sheet.png")
    save_ranked_contact_sheet(
        metrics_rows,
        output_dir / "worst_vertex_recall_contact_sheet.png",
        sort_key="vertex_recall",
        limit=args.worst_limit,
    )
    save_ranked_contact_sheet(
        metrics_rows,
        output_dir / "worst_edge_recall_contact_sheet.png",
        sort_key="edge_recall",
        limit=args.worst_limit,
    )
    save_ranked_contact_sheet(
        metrics_rows,
        output_dir / "slowest_contact_sheet.png",
        sort_key="elapsed_seconds",
        reverse=True,
        limit=args.worst_limit,
    )
    copy_failure_overlays(metrics_rows, output_dir)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("fixtures/phase2_real_folds/smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/phase2_vectorizer/smoke"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--junction-sigma", type=float, default=2.5)
    parser.add_argument("--vertex-tolerance", type=float, default=5.0)
    parser.add_argument("--hough-threshold", type=int, default=20)
    parser.add_argument("--hough-min-line-length", type=int, default=10)
    parser.add_argument("--hough-max-line-gap", type=int, default=5)
    parser.add_argument("--min-edge-support", type=float, default=0.55)
    parser.add_argument("--line-vertex-distance-px", type=float, default=4.0)
    parser.add_argument("--junction-threshold", type=float, default=0.16)
    parser.add_argument("--junction-nms-radius", type=int, default=2)
    parser.add_argument("--vertex-merge-px", type=float, default=1.5)
    parser.add_argument("--max-intersection-lines", type=int, default=250)
    parser.add_argument("--add-intersection-vertices", action="store_true")
    parser.add_argument("--disable-direct-edge-fallback", action="store_true")
    parser.add_argument("--direct-edge-max-length-px", type=float, default=1024.0)
    parser.add_argument("--direct-edge-min-support", type=float, default=0.9)
    parser.add_argument("--direct-edge-max-vertices", type=int, default=256)
    parser.add_argument("--direct-edge-short-max-length-px", type=float, default=180.0)
    parser.add_argument("--direct-edge-short-max-vertices", type=int, default=800)
    parser.add_argument("--disable-planar-cleanup", action="store_true")
    parser.add_argument("--planar-cleanup-max-edges", type=int, default=3000)
    parser.add_argument("--planar-split-vertex-distance-px", type=float, default=2.0)
    parser.add_argument("--contact-sheet-limit", type=int, default=24)
    parser.add_argument("--worst-limit", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
