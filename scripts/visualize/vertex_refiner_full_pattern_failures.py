#!/usr/bin/env python3
"""Visualize full-pattern VertexRefiner misses and false positives."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import VertexRefinerCropDataset, vertex_refiner_collate
from src.data.vertex_refiner_proposals import VertexProposal, crop_origin_for_center
from src.data.vertex_refiner_targets import SquareFrame, classify_vertex_kind
from src.evaluation.vertex_refiner_global_merge import VertexMergeConfig, merge_decoded_vertices
from src.evaluation.vertex_refiner_recall_diagnostics import (
    _match_points,
    proposal_contains_vertex,
)
from src.models import VertexRefinerV1, VertexRefinerV2, VertexRefinerV3
from src.models.vertex_refiner import DecodedVertex, decode_vertex_refiner_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--proposals-per-sample", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--model-version", choices=["v1", "v2", "v3"], default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--auxiliary-mode", choices=["zero", "rendered-labels"], default=None)
    parser.add_argument("--rendered-sample-cache-size", type=int, default=4)
    parser.add_argument("--crop-ref-progress-every", type=int, default=0)
    parser.add_argument("--heatmap-threshold", type=float, default=0.25)
    parser.add_argument("--match-tolerance-px", type=float, default=2.0)
    parser.add_argument("--merge-radius-px", type=float, default=2.0)
    parser.add_argument("--merge-boundary-radius-px", type=float, default=None)
    parser.add_argument("--merge-min-score", type=float, default=0.25)
    parser.add_argument("--merge-min-member-score", type=float, default=0.0)
    parser.add_argument("--merge-min-support", type=int, default=1)
    parser.add_argument("--merge-min-support-fraction", type=float, default=0.45)
    parser.add_argument("--merge-ray-vote-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-record-overlays", type=int, default=12)
    parser.add_argument("--max-zoom-items", type=int, default=36)
    parser.add_argument("--zoom-size", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _select_device(args.device)
    checkpoint_path = _resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = args.base_channels or int(config.get("base_channels", 48))
    model_version = args.model_version or str(config.get("model_version", "v1"))
    auxiliary_mode = args.auxiliary_mode or str(config.get("auxiliary_mode", "zero"))
    model_cls = _model_class(model_version)
    model = model_cls(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "record_overlays"
    zooms_dir = output_dir / "zoom_tiles"
    overlays_dir.mkdir(exist_ok=True)
    zooms_dir.mkdir(exist_ok=True)

    dataset = VertexRefinerCropDataset(
        manifest,
        split=args.split,
        limit=args.limit,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=False,
        auxiliary_mode=auxiliary_mode,
        input_version=model_version,
        rendered_sample_cache_size=args.rendered_sample_cache_size,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    predictions_by_record = _decode_predictions(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        heatmap_threshold=args.heatmap_threshold,
    )
    merge_config = VertexMergeConfig(
        radius_px=args.merge_radius_px,
        boundary_merge_radius_px=args.merge_boundary_radius_px,
        min_score=args.merge_min_score,
        min_member_score=args.merge_min_member_score,
        min_support=args.merge_min_support,
        min_support_fraction=args.merge_min_support_fraction,
        ray_vote_fraction=args.merge_ray_vote_fraction,
    )
    analysis = _analyze_records(
        dataset,
        predictions_by_record,
        merge_config=merge_config,
        match_tolerance_px=args.match_tolerance_px,
    )
    _write_visuals(
        dataset,
        analysis,
        output_dir=output_dir,
        overlays_dir=overlays_dir,
        zooms_dir=zooms_dir,
        max_record_overlays=args.max_record_overlays,
        max_zoom_items=args.max_zoom_items,
        zoom_size=args.zoom_size,
    )
    report = {
        "schema": "create-pattern-detector/vertex-refiner-full-pattern-failures/v1",
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest.as_posix(),
        "output_dir": output_dir.as_posix(),
        "split": args.split,
        "limit": args.limit,
        "max_edges": args.max_edges,
        "image_size": args.image_size,
        "base_channels": base_channels,
        "model_version": model_version,
        "auxiliary_mode": auxiliary_mode,
        "seed": args.seed,
        "heatmap_threshold": args.heatmap_threshold,
        "match_tolerance_px": args.match_tolerance_px,
        "merge_config": _to_jsonable(merge_config),
        **analysis["summary"],
        "records": analysis["records"],
        "false_negatives": analysis["false_negatives"],
        "false_positives": analysis["false_positives"],
        "visuals": analysis["visuals"],
    }
    report_path = output_dir / "failure_overlay_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": report_path.as_posix(), **analysis["summary"]}, indent=2), flush=True)
    return 0


@torch.no_grad()
def _decode_predictions(
    model: torch.nn.Module,
    dataset: VertexRefinerCropDataset,
    *,
    device: torch.device,
    batch_size: int,
    heatmap_threshold: float,
) -> dict[str, list[DecodedVertex]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vertex_refiner_collate,
    )
    predictions_by_record: dict[str, list[DecodedVertex]] = defaultdict(list)
    for batch in loader:
        outputs = model(batch["input"].to(device))
        decoded = decode_vertex_refiner_batch(
            outputs,
            crop_origins_xy=[tuple(meta["crop_origin_xy"]) for meta in batch["meta"]],
            square_frames=[_square_frame_from_meta(meta) for meta in batch["meta"]],
            heatmap_threshold=heatmap_threshold,
        )
        for vertices, meta in zip(decoded, batch["meta"], strict=True):
            predictions_by_record[str(meta["record_id"])].extend(vertices)
    return predictions_by_record


def _analyze_records(
    dataset: VertexRefinerCropDataset,
    predictions_by_record: dict[str, list[DecodedVertex]],
    *,
    merge_config: VertexMergeConfig,
    match_tolerance_px: float,
) -> dict[str, Any]:
    refs_by_record = _crop_refs_by_record(dataset)
    records: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    totals = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    decoded_total = 0
    merged_total = 0
    for record_index, record in enumerate(dataset.records):
        sample = dataset._render_record(record_index)
        record_id = str(record["id"])
        proposals = [ref.proposal for ref in refs_by_record.get(record_index, [])]
        raw_predictions = predictions_by_record.get(record_id, [])
        decoded_total += len(raw_predictions)
        predictions = merge_decoded_vertices(
            raw_predictions,
            proposals=proposals,
            config=merge_config,
        )
        merged_total += len(predictions)
        gt_xy = np.asarray(sample.pixel_vertices, dtype=np.float32).reshape(-1, 2)
        pred_xy = np.asarray([[vertex.x, vertex.y] for vertex in predictions], dtype=np.float32).reshape(
            -1, 2
        )
        matched_gt, matched_pred, errors = _match_points(
            gt_xy,
            pred_xy,
            tolerance_px=match_tolerance_px,
        )
        record_counts = Counter(
            gt=len(gt_xy),
            pred=len(predictions),
            matched=len(matched_gt),
            false_negative=len(gt_xy) - len(matched_gt),
            false_positive=len(predictions) - len(matched_pred),
        )
        record_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
        for vertex_index, vertex in enumerate(gt_xy):
            kind = classify_vertex_kind(
                vertex_index,
                sample.pixel_vertices,
                sample.edges,
                sample.assignments,
                sample.square_frame,
                image_size=int(sample.metadata["image_size"]),
            )
            covered = any(proposal_contains_vertex(vertex, proposal) for proposal in proposals)
            matched = vertex_index in matched_gt
            by_kind[kind]["total"] += 1
            record_by_kind[kind]["total"] += 1
            if covered:
                by_kind[kind]["covered"] += 1
                record_by_kind[kind]["covered"] += 1
            if matched:
                by_kind[kind]["matched"] += 1
                record_by_kind[kind]["matched"] += 1
            else:
                miss = {
                    "record_index": record_index,
                    "record_id": record_id,
                    "vertex_index": int(vertex_index),
                    "x": float(vertex[0]),
                    "y": float(vertex[1]),
                    "kind": kind,
                    "covered_by_crop": bool(covered),
                    "nearest_prediction_distance_px": _nearest_distance(vertex, pred_xy),
                    "nearest_proposal_center_distance_px": _nearest_proposal_center_distance(
                        vertex,
                        proposals,
                    ),
                    "nearest_proposal_box_distance_px": _nearest_proposal_box_distance(
                        vertex,
                        proposals,
                    ),
                }
                false_negatives.append(miss)
        for pred_index, prediction in enumerate(predictions):
            if pred_index in matched_pred:
                continue
            point = np.asarray([prediction.x, prediction.y], dtype=np.float32)
            false_positives.append(
                {
                    "record_index": record_index,
                    "record_id": record_id,
                    "prediction_index": int(pred_index),
                    "x": float(prediction.x),
                    "y": float(prediction.y),
                    "score": float(prediction.score),
                    "kind": prediction.kind,
                    "degree": int(prediction.degree),
                    "boundary_side": prediction.boundary_side,
                    "support_count": int(getattr(prediction, "support_count", 1)),
                    "support_fraction": float(getattr(prediction, "support_fraction", 1.0)),
                    "nearest_gt_distance_px": _nearest_distance(point, gt_xy),
                }
            )
        totals.update(record_counts)
        records.append(
            {
                "record_index": record_index,
                "record_id": record_id,
                "fold_path": str(record.get("foldPath", "")),
                "gt_vertices": int(record_counts["gt"]),
                "predictions": int(record_counts["pred"]),
                "matched": int(record_counts["matched"]),
                "false_negatives": int(record_counts["false_negative"]),
                "false_positives": int(record_counts["false_positive"]),
                "mean_match_error_px": float(np.mean(errors)) if errors else None,
                "proposal_count": len(proposals),
                "by_kind": _counter_tree_to_dict(record_by_kind),
            }
        )
    precision = _ratio(totals["matched"], totals["pred"])
    recall = _ratio(totals["matched"], totals["gt"])
    summary = {
        "summary": {
            "metrics": {
                "true_positive": int(totals["matched"]),
                "false_positive": int(totals["false_positive"]),
                "false_negative": int(totals["false_negative"]),
                "precision": precision,
                "recall": recall,
                "f1": _ratio(2 * precision * recall, precision + recall),
            },
            "decoded_predictions_total_before_global_nms": int(decoded_total),
            "decoded_predictions_total_after_global_merge": int(merged_total),
            "false_negative_split": {
                "due_to_no_crop_coverage": sum(
                    1 for item in false_negatives if not item["covered_by_crop"]
                ),
                "due_to_covered_but_not_matched": sum(
                    1 for item in false_negatives if item["covered_by_crop"]
                ),
            },
            "false_negatives_by_kind": dict(
                sorted(Counter(item["kind"] for item in false_negatives).items())
            ),
            "false_positives_by_kind": dict(
                sorted(Counter(item["kind"] for item in false_positives).items())
            ),
            "by_kind": {
                kind: {
                    "total": int(counts["total"]),
                    "covered": int(counts["covered"]),
                    "matched": int(counts["matched"]),
                    "proposal_coverage": _ratio(counts["covered"], counts["total"]),
                    "recall": _ratio(counts["matched"], counts["total"]),
                    "covered_conditional_recall": _ratio(counts["matched"], counts["covered"]),
                }
                for kind, counts in sorted(by_kind.items(), key=lambda item: item[0])
            },
        },
        "records": records,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "visuals": {},
    }
    return summary


def _write_visuals(
    dataset: VertexRefinerCropDataset,
    analysis: dict[str, Any],
    *,
    output_dir: Path,
    overlays_dir: Path,
    zooms_dir: Path,
    max_record_overlays: int,
    max_zoom_items: int,
    zoom_size: int,
) -> None:
    refs_by_record = _crop_refs_by_record(dataset)
    fns_by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
    fps_by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in analysis["false_negatives"]:
        fns_by_record[int(item["record_index"])].append(item)
    for item in analysis["false_positives"]:
        fps_by_record[int(item["record_index"])].append(item)

    ranked_records = sorted(
        analysis["records"],
        key=lambda row: (row["false_negatives"] + row["false_positives"], row["false_negatives"]),
        reverse=True,
    )[:max_record_overlays]
    overlay_paths: list[Path] = []
    for rank, record in enumerate(ranked_records):
        record_index = int(record["record_index"])
        sample = dataset._render_record(record_index)
        path = overlays_dir / f"record_{rank:02d}_{_slug(record['record_id'])}.png"
        _draw_record_overlay(
            sample.image,
            record=record,
            false_negatives=fns_by_record.get(record_index, []),
            false_positives=fps_by_record.get(record_index, []),
            output_path=path,
        )
        record["overlay_path"] = path.as_posix()
        overlay_paths.append(path)
    if overlay_paths:
        sheet = output_dir / "worst_record_overlays.png"
        _make_contact_sheet(overlay_paths, sheet, columns=3, tile_width=360)
        analysis["visuals"]["worst_record_overlays"] = sheet.as_posix()

    ranked_misses = sorted(
        analysis["false_negatives"],
        key=lambda item: (
            item["covered_by_crop"],
            item["kind"] != "boundary_contact",
            -(item["nearest_proposal_box_distance_px"] or 0.0),
            -(item["nearest_prediction_distance_px"] or 0.0),
        ),
    )[:max_zoom_items]
    miss_tiles = []
    for index, miss in enumerate(ranked_misses):
        record_index = int(miss["record_index"])
        sample = dataset._render_record(record_index)
        proposals = [ref.proposal for ref in refs_by_record.get(record_index, [])]
        tile = _draw_zoom_tile(
            sample.image,
            point=(float(miss["x"]), float(miss["y"])),
            label=_miss_label(miss),
            proposals=proposals,
            zoom_size=zoom_size,
            marker_color=(255, 42, 42, 255) if miss["covered_by_crop"] else (255, 138, 0, 255),
        )
        path = zooms_dir / f"miss_{index:02d}_{miss['kind']}_{_slug(miss['record_id'])}.png"
        tile.save(path)
        miss_tiles.append(path)
    if miss_tiles:
        sheet = output_dir / "miss_zoom_sheet.png"
        _make_contact_sheet(miss_tiles, sheet, columns=4, tile_width=260)
        analysis["visuals"]["miss_zoom_sheet"] = sheet.as_posix()

    ranked_fps = sorted(
        analysis["false_positives"],
        key=lambda item: (-(item["nearest_gt_distance_px"] or 0.0), -float(item["score"])),
    )[:max_zoom_items]
    fp_tiles = []
    for index, fp in enumerate(ranked_fps):
        record_index = int(fp["record_index"])
        sample = dataset._render_record(record_index)
        tile = _draw_zoom_tile(
            sample.image,
            point=(float(fp["x"]), float(fp["y"])),
            label=_fp_label(fp),
            proposals=[],
            zoom_size=zoom_size,
            marker_color=(214, 39, 180, 255),
        )
        path = zooms_dir / f"fp_{index:02d}_{fp['kind']}_{_slug(fp['record_id'])}.png"
        tile.save(path)
        fp_tiles.append(path)
    if fp_tiles:
        sheet = output_dir / "fp_zoom_sheet.png"
        _make_contact_sheet(fp_tiles, sheet, columns=4, tile_width=260)
        analysis["visuals"]["fp_zoom_sheet"] = sheet.as_posix()


def _draw_record_overlay(
    image: np.ndarray,
    *,
    record: dict[str, Any],
    false_negatives: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    output_path: Path,
) -> None:
    pil = _to_pil_rgb(image).convert("RGBA")
    draw = ImageDraw.Draw(pil, "RGBA")
    for miss in false_negatives:
        color = (255, 48, 48, 235) if miss["covered_by_crop"] else (255, 138, 0, 245)
        _draw_cross(draw, float(miss["x"]), float(miss["y"]), 8, color, width=3)
    for fp in false_positives:
        _draw_x(draw, float(fp["x"]), float(fp["y"]), 7, (214, 39, 180, 225), width=3)
    _draw_header(
        draw,
        pil.width,
        [
            _short(record["record_id"], 74),
            f"FN {record['false_negatives']}  FP {record['false_positives']}  pred {record['predictions']}  GT {record['gt_vertices']}",
            "orange=no crop, red=covered miss, magenta=false positive",
        ],
    )
    pil.convert("RGB").save(output_path)


def _draw_zoom_tile(
    image: np.ndarray,
    *,
    point: tuple[float, float],
    label: str,
    proposals: list[VertexProposal],
    zoom_size: int,
    marker_color: tuple[int, int, int, int],
) -> Image.Image:
    source = _to_pil_rgb(image).convert("RGBA")
    x, y = point
    half = int(zoom_size) // 2
    left = int(round(x)) - half
    top = int(round(y)) - half
    crop = Image.new("RGBA", (zoom_size, zoom_size), (255, 255, 255, 255))
    src_box = (
        max(left, 0),
        max(top, 0),
        min(left + zoom_size, source.width),
        min(top + zoom_size, source.height),
    )
    if src_box[2] > src_box[0] and src_box[3] > src_box[1]:
        crop.paste(source.crop(src_box), (src_box[0] - left, src_box[1] - top))
    draw = ImageDraw.Draw(crop, "RGBA")
    _draw_nearby_proposals(draw, proposals, left=left, top=top, zoom_size=zoom_size)
    _draw_cross(draw, x - left, y - top, 10, marker_color, width=3)
    draw.rectangle((0, 0, zoom_size - 1, zoom_size - 1), outline=(35, 35, 35, 255), width=1)

    label_height = 54
    tile = Image.new("RGBA", (zoom_size, zoom_size + label_height), (255, 255, 255, 255))
    tile.paste(crop, (0, 0))
    label_draw = ImageDraw.Draw(tile)
    for row, line in enumerate(_wrap(label, chars=34)[:3]):
        label_draw.text((5, zoom_size + 5 + row * 15), line, fill=(25, 25, 25), font=_font())
    return tile.convert("RGB")


def _draw_nearby_proposals(
    draw: ImageDraw.ImageDraw,
    proposals: list[VertexProposal],
    *,
    left: int,
    top: int,
    zoom_size: int,
) -> None:
    right = left + zoom_size
    bottom = top + zoom_size
    for proposal in proposals:
        origin_x, origin_y = crop_origin_for_center((proposal.x, proposal.y))
        rect = (origin_x, origin_y, origin_x + 96, origin_y + 96)
        if rect[2] < left or rect[0] > right or rect[3] < top or rect[1] > bottom:
            continue
        local = (rect[0] - left, rect[1] - top, rect[2] - left, rect[3] - top)
        draw.rectangle(local, outline=(0, 145, 255, 80), width=1)
        cx = float(proposal.x) - left
        cy = float(proposal.y) - top
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(0, 95, 255, 140))


def _make_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int,
    tile_width: int,
) -> None:
    thumbs: list[Image.Image] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        ratio = tile_width / max(image.width, 1)
        thumb = image.resize((tile_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    if not thumbs:
        return
    gap = 12
    rows = math.ceil(len(thumbs) / columns)
    row_heights = [
        max(thumb.height for thumb in thumbs[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    sheet = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * gap, sum(row_heights) + (rows + 1) * gap),
        (245, 247, 250),
    )
    y = gap
    for row in range(rows):
        x = gap
        for thumb in thumbs[row * columns : (row + 1) * columns]:
            sheet.paste(thumb, (x, y))
            x += tile_width + gap
        y += row_heights[row] + gap
    sheet.save(output_path)


def _crop_refs_by_record(dataset: VertexRefinerCropDataset) -> dict[int, list[Any]]:
    refs_by_record: dict[int, list[Any]] = defaultdict(list)
    for ref in dataset.crop_refs:
        refs_by_record[int(ref.record_index)].append(ref)
    return refs_by_record


def _square_frame_from_meta(meta: dict[str, Any]) -> SquareFrame | None:
    frame = meta.get("square_frame") if isinstance(meta, dict) else None
    if not isinstance(frame, dict):
        return None
    return SquareFrame(
        x_min=float(frame["x_min"]),
        y_min=float(frame["y_min"]),
        x_max=float(frame["x_max"]),
        y_max=float(frame["y_max"]),
    )


def _nearest_distance(point: np.ndarray, points: np.ndarray) -> float | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return None
    distances = np.linalg.norm(points - np.asarray(point, dtype=np.float32).reshape(1, 2), axis=1)
    return float(np.min(distances))


def _nearest_proposal_center_distance(
    point: np.ndarray,
    proposals: list[VertexProposal],
) -> float | None:
    if not proposals:
        return None
    x, y = float(point[0]), float(point[1])
    return float(min(math.hypot(x - proposal.x, y - proposal.y) for proposal in proposals))


def _nearest_proposal_box_distance(
    point: np.ndarray,
    proposals: list[VertexProposal],
) -> float | None:
    if not proposals:
        return None
    x, y = float(point[0]), float(point[1])
    distances = []
    for proposal in proposals:
        origin_x, origin_y = crop_origin_for_center((proposal.x, proposal.y))
        dx = max(float(origin_x) - x, 0.0, x - float(origin_x + 96))
        dy = max(float(origin_y) - y, 0.0, y - float(origin_y + 96))
        distances.append(math.hypot(dx, dy))
    return float(min(distances))


def _draw_header(draw: ImageDraw.ImageDraw, width: int, lines: list[str]) -> None:
    line_height = 16
    height = 8 + line_height * len(lines)
    draw.rectangle((0, 0, width, height), fill=(255, 255, 255, 220))
    for index, line in enumerate(lines):
        draw.text((6, 4 + index * line_height), line, fill=(20, 20, 20), font=_font())


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    color: tuple[int, int, int, int],
    *,
    width: int,
) -> None:
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=width)


def _draw_x(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    color: tuple[int, int, int, int],
    *,
    width: int,
) -> None:
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=1.0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array[..., :3], mode="RGB")


def _miss_label(miss: dict[str, Any]) -> str:
    coverage = "covered" if miss["covered_by_crop"] else "NO CROP"
    nearest_pred = _format_px(miss["nearest_prediction_distance_px"])
    nearest_box = _format_px(miss["nearest_proposal_box_distance_px"])
    return f"FN {miss['kind']} {coverage} pred={nearest_pred} cropbox={nearest_box}"


def _fp_label(fp: dict[str, Any]) -> str:
    nearest_gt = _format_px(fp["nearest_gt_distance_px"])
    return f"FP {fp['kind']} score={fp['score']:.2f} gt={nearest_gt}"


def _format_px(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}px"


def _wrap(text: str, *, chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _counter_tree_to_dict(tree: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        key: {inner_key: int(value) for inner_key, value in sorted(counter.items())}
        for key, counter in sorted(tree.items())
    }


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _slug(value: str, *, max_len: int = 52) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:max_len] or "record"


def _short(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _model_class(model_version: str) -> type[torch.nn.Module]:
    if model_version == "v1":
        return VertexRefinerV1
    if model_version == "v2":
        return VertexRefinerV2
    if model_version == "v3":
        return VertexRefinerV3
    raise ValueError(f"Unsupported model_version: {model_version}")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
