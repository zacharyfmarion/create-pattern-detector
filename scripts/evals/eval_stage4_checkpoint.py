#!/usr/bin/env python3
"""Checkpoint-backed Stage 4 validation with visual examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES  # noqa: E402
from src.data.cpline_dataset import CplineFoldDataset, cpline_collate  # noqa: E402
from src.models import CPLineNet  # noqa: E402
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode  # noqa: E402
from src.vectorization import (  # noqa: E402
    AttributedPlanarGraph,
    EdgeAssignmentConfig,
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    QualityReportConfig,
    RepairAction,
    RepairConfig,
    attribute_graph_from_logits,
    build_quality_report,
    conservative_repair,
    cpline_outputs_to_evidence,
    evaluate_graph,
)
from src.vectorization.metrics import GraphMetrics, metrics_from_results  # noqa: E402

ASSIGNMENT_COLORS = {
    0: "#e11d48",  # M
    1: "#2563eb",  # V
    2: "#111827",  # B
    3: "#6b7280",  # U
}
DARK_BACKGROUND_ASSIGNMENT_COLORS = {
    **ASSIGNMENT_COLORS,
    2: "#f8fafc",
    3: "#cbd5e1",
}
STATUS_ORDER = ("valid", "repaired", "ambiguous", "outside_v1_envelope", "failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/stage4_checkpoint_eval"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--samples-per-profile", type=int, default=24)
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["clean", "line-style", "print-light", "dark-mode", "photo-light", "photo-dark"],
        choices=AUGMENT_PROFILES,
    )
    parser.add_argument(
        "--family-sampling",
        choices=["natural", "balanced"],
        default="balanced",
        help="Record sampling strategy for each profile.",
    )
    parser.add_argument(
        "--batchnorm-mode",
        choices=BATCHNORM_MODES,
        default="batch-stats",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--repair-near-endpoint-crossings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the gated topology repair that snaps near-endpoint crossings before cleanup deletes edges.",
    )
    parser.add_argument("--max-visuals-per-profile", type=int, default=4)
    parser.add_argument("--infer-assignments", action="store_true")
    parser.add_argument(
        "--reconstruct-square-border-chain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic square-frame border-chain reconstruction.",
    )
    parser.add_argument(
        "--oracle-border-chain",
        action="store_true",
        help=(
            "Evaluation-only ablation: replace predicted B edges with the "
            "ground-truth border chain before metrics/reporting."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = _resolve_path(args.checkpoint)
    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = load_model(checkpoint, device)
    builder = make_builder(
        args.image_size,
        args.threshold,
        repair_near_endpoint_crossings=args.repair_near_endpoint_crossings,
    )
    assignment_config = EdgeAssignmentConfig()
    repair_config = RepairConfig(
        image_size=args.image_size,
        reconstruct_square_border_chain=args.reconstruct_square_border_chain,
    )
    report_config = QualityReportConfig(image_size=args.image_size)

    rows: list[dict[str, Any]] = []
    metric_items: list[tuple[str, str, GraphMetrics]] = []
    examples: list[dict[str, Any]] = []

    with torch.no_grad(), model_eval_with_batchnorm_mode(model, batchnorm_mode=args.batchnorm_mode):
        for profile in args.profiles:
            dataset = CplineFoldDataset(
                manifest,
                split=args.split,
                limit=args.samples_per_profile,
                max_edges=args.max_edges,
                image_size=args.image_size,
                augment_profile=profile,
                seed=args.seed,
                family_sampling=args.family_sampling,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=cpline_collate,
            )
            selector = ExampleSelector(args.max_visuals_per_profile)
            for sample_index, batch in enumerate(loader):
                row, metrics_obj, payload = evaluate_sample(
                    model=model,
                    batch=batch,
                    device=device,
                    builder=builder,
                    assignment_config=assignment_config,
                    repair_config=repair_config,
                    report_config=report_config,
                    infer_assignments=args.infer_assignments,
                    oracle_border_chain=args.oracle_border_chain,
                    image_size=args.image_size,
                    threshold=args.threshold,
                    profile=profile,
                    sample_index=sample_index,
                )
                rows.append(row)
                metric_items.append((profile, row["family"], metrics_obj))
                selector.consider(row, payload)
                print(json.dumps(row), flush=True)
            examples.extend(selector.selected())

    example_paths = write_examples(examples, examples_dir)
    summary = build_summary(
        rows,
        metric_items,
        checkpoint=checkpoint,
        manifest=manifest,
        args=args,
        example_paths=example_paths,
    )
    (output_dir / "per_sample_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_profile_metric_chart(summary, output_dir / "profile_metrics.png")
    write_status_chart(summary, output_dir / "status_counts.png")
    write_contact_sheet(example_paths, output_dir / "key_examples_contact_sheet.png")
    print(json.dumps({"summary": summary}), flush=True)


def evaluate_sample(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    builder: PlanarGraphBuilder,
    assignment_config: EdgeAssignmentConfig,
    repair_config: RepairConfig,
    report_config: QualityReportConfig,
    infer_assignments: bool,
    oracle_border_chain: bool,
    image_size: int,
    threshold: float,
    profile: str,
    sample_index: int,
) -> tuple[dict[str, Any], GraphMetrics, dict[str, Any]]:
    outputs = model(batch["image"].to(device))
    evidence = cpline_outputs_to_evidence(
        outputs,
        batch_index=0,
        line_threshold=threshold,
    )
    line_prob = evidence.line_prob
    junction_heatmap = evidence.junction_heatmap
    graph_result = builder.build(evidence)
    attributed = attribute_graph_from_logits(
        graph_result,
        outputs["assignment_logits"][0].detach().cpu(),
        line_prob=line_prob,
        config=assignment_config,
    )
    repair = conservative_repair(
        attributed,
        line_prob=line_prob,
        config=repair_config,
        infer_assignments=infer_assignments,
    )
    gt_graph = batch["graph"][0]
    graph_for_report = repair.graph
    repair_actions = list(repair.actions)
    if oracle_border_chain:
        graph_for_report, oracle_action = apply_oracle_border_chain(
            graph_for_report,
            gt_vertices=gt_graph["vertices"].numpy(),
            gt_edges=gt_graph["edges"].numpy(),
            gt_assignments=gt_graph["assignments"].numpy(),
            image_size=image_size,
        )
        if oracle_action is not None:
            repair_actions.append(oracle_action)
    report = build_quality_report(
        graph_for_report,
        repair_actions=repair_actions,
        config=report_config,
    )

    metrics_obj = evaluate_graph(
        graph_for_report.to_planar_result(),
        gt_vertices=gt_graph["vertices"].numpy(),
        gt_edges=gt_graph["edges"].numpy(),
        gt_assignments=gt_graph["assignments"].numpy(),
        vertex_tolerance_px=max(3.0, 5.0 * image_size / 768),
    )
    metrics = metrics_obj.to_dict()
    meta = batch["meta"][0]
    status = report.status
    warning_codes = [warning.code for warning in report.warnings]
    repair_codes = [action.code for action in repair_actions]
    source_counts = Counter(graph_for_report.assignment_source)
    row = {
        "id": meta["id"],
        "sample_index": sample_index,
        "family": meta.get("family", ""),
        "bucket": meta.get("bucket", ""),
        "profile": profile,
        "status": status,
        "warnings": warning_codes,
        "repairs": repair_codes,
        "observed_edges": int(source_counts.get("observed", 0)),
        "unknown_edges": int(source_counts.get("unknown", 0)),
        "inferred_edges": int(source_counts.get("inferred", 0)),
        "mean_edge_support": float(report.summary.get("mean_edge_support", 0.0)),
        "mean_assignment_confidence": float(report.summary.get("mean_assignment_confidence", 0.0)),
        "max_geometry_drift_px": float(repair.max_geometry_drift_px),
        "threshold": threshold,
        **metrics,
    }
    image = (batch["image"][0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    payload = {
        "row": row,
        "image": image,
        "line_prob": line_prob.astype(np.float32),
        "junction_heatmap": junction_heatmap.astype(np.float32),
        "pred_vertices": graph_for_report.pixel_vertices.copy(),
        "pred_edges": graph_for_report.edges_vertices.copy(),
        "pred_assignments": graph_for_report.edges_assignment.copy(),
        "pred_sources": list(graph_for_report.assignment_source),
        "pred_confidence": graph_for_report.assignment_confidence.copy(),
        "gt_vertices": gt_graph["vertices"].numpy(),
        "gt_edges": gt_graph["edges"].numpy(),
        "gt_assignments": gt_graph["assignments"].numpy(),
        "report": report.to_dict(),
    }
    return row, metrics_obj, payload


def apply_oracle_border_chain(
    graph: AttributedPlanarGraph,
    *,
    gt_vertices: np.ndarray,
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    image_size: int,
) -> tuple[AttributedPlanarGraph, RepairAction | None]:
    """Evaluation-only upper bound: use GT boundary vertices/edges as the border chain."""
    gt_vertices = np.asarray(gt_vertices, dtype=np.float32)
    gt_edges = np.asarray(gt_edges, dtype=np.int64)
    gt_assignments = np.asarray(gt_assignments, dtype=np.int8)
    border_edge_indices = np.flatnonzero(gt_assignments == 2)
    if len(border_edge_indices) == 0:
        return graph, None

    pixel_vertices = np.asarray(graph.pixel_vertices, dtype=np.float32).copy()
    vertex_support = np.asarray(graph.vertex_support, dtype=np.float32).copy()
    gt_to_pred: dict[int, int] = {}
    added_vertices = 0
    for gt_idx in sorted(set(int(v) for edge_idx in border_edge_indices for v in gt_edges[edge_idx])):
        gt_point = gt_vertices[gt_idx]
        exact_matches = np.flatnonzero(np.linalg.norm(pixel_vertices - gt_point, axis=1) <= 1e-4)
        if len(exact_matches):
            gt_to_pred[gt_idx] = int(exact_matches[0])
            continue
        gt_to_pred[gt_idx] = int(len(pixel_vertices))
        pixel_vertices = np.vstack([pixel_vertices, gt_point])
        vertex_support = np.concatenate([vertex_support, np.array([1.0], dtype=np.float32)])
        added_vertices += 1

    oracle_edges: list[tuple[int, int]] = []
    seen_oracle_edges: set[tuple[int, int]] = set()
    for edge_idx in border_edge_indices:
        gt_v1, gt_v2 = (int(value) for value in gt_edges[int(edge_idx)])
        pred_v1 = gt_to_pred[gt_v1]
        pred_v2 = gt_to_pred[gt_v2]
        if pred_v1 == pred_v2:
            continue
        key = (min(pred_v1, pred_v2), max(pred_v1, pred_v2))
        if key in seen_oracle_edges:
            continue
        oracle_edges.append((pred_v1, pred_v2))
        seen_oracle_edges.add(key)

    keep_existing = np.flatnonzero(np.asarray(graph.edges_assignment) != 2)
    kept_edges = np.asarray(graph.edges_vertices, dtype=np.int64)[keep_existing]
    kept_assignments = np.asarray(graph.edges_assignment, dtype=np.int8)[keep_existing]
    kept_support = np.asarray(graph.edge_support, dtype=np.float32)[keep_existing]
    kept_confidence = np.asarray(graph.assignment_confidence, dtype=np.float32)[keep_existing]
    kept_margin = np.asarray(graph.assignment_margin, dtype=np.float32)[keep_existing]
    kept_source = [graph.assignment_source[int(idx)] for idx in keep_existing]

    oracle_count = len(oracle_edges)
    oracle_edge_array = np.asarray(oracle_edges, dtype=np.int64).reshape(oracle_count, 2)
    edges_vertices = np.concatenate([oracle_edge_array, kept_edges], axis=0)
    edges_assignment = np.concatenate(
        [
            np.full(oracle_count, 2, dtype=np.int8),
            kept_assignments,
        ],
    )
    edge_support = np.concatenate(
        [
            np.ones(oracle_count, dtype=np.float32),
            kept_support,
        ],
    )
    assignment_confidence = np.concatenate(
        [
            np.ones(oracle_count, dtype=np.float32),
            kept_confidence,
        ],
    )
    assignment_margin = np.concatenate(
        [
            np.ones(oracle_count, dtype=np.float32),
            kept_margin,
        ],
    )
    assignment_source = ["oracle" for _ in range(oracle_count)] + kept_source
    probabilities = None
    if graph.assignment_probabilities is not None:
        oracle_probabilities = np.zeros((oracle_count, 4), dtype=np.float32)
        oracle_probabilities[:, 2] = 1.0
        probabilities = np.concatenate(
            [
                oracle_probabilities,
                np.asarray(graph.assignment_probabilities, dtype=np.float32)[keep_existing],
            ],
            axis=0,
        )

    max_coord = max(float(image_size - 1), 1.0)
    repaired = AttributedPlanarGraph(
        vertices_coords=np.clip(pixel_vertices / max_coord, 0.0, 1.0).astype(np.float32),
        edges_vertices=edges_vertices,
        edges_assignment=edges_assignment,
        edge_support=edge_support,
        vertex_support=vertex_support,
        pixel_vertices=pixel_vertices,
        assignment_confidence=assignment_confidence,
        assignment_margin=assignment_margin,
        assignment_source=assignment_source,
        assignment_probabilities=probabilities,
        debug={**graph.debug, "oracle_border_chain": True},
    )
    action = RepairAction(
        code="oracle_border_chain",
        message="Replaced predicted border edges with the ground-truth border chain for eval-only ablation.",
        edge_indices=list(range(oracle_count)),
        vertex_indices=[gt_to_pred[idx] for idx in sorted(gt_to_pred)],
        details={
            "added_vertices": added_vertices,
            "oracle_border_edges": oracle_count,
            "removed_predicted_border_edges": int(np.sum(np.asarray(graph.edges_assignment) == 2)),
        },
    )
    return repaired, action


class ExampleSelector:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._items: dict[str, dict[str, Any]] = {}
        self._worst_recall: tuple[float, dict[str, Any]] | None = None
        self._best_assignment: tuple[float, dict[str, Any]] | None = None

    def consider(self, row: dict[str, Any], payload: dict[str, Any]) -> None:
        if self.max_items <= 0:
            return
        if "first" not in self._items:
            self._items["first"] = payload
        if row["status"] != "valid" and "first_non_valid" not in self._items:
            self._items["first_non_valid"] = payload
        recall = float(row["edge_recall"])
        if self._worst_recall is None or recall < self._worst_recall[0]:
            self._worst_recall = (recall, payload)
        assignment = float(row["assignment_accuracy"])
        if self._best_assignment is None or assignment > self._best_assignment[0]:
            self._best_assignment = (assignment, payload)

    def selected(self) -> list[dict[str, Any]]:
        items = list(self._items.values())
        if self._worst_recall is not None:
            items.append(self._worst_recall[1])
        if self._best_assignment is not None:
            items.append(self._best_assignment[1])

        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for payload in items:
            row = payload["row"]
            key = (str(row["profile"]), int(row["sample_index"]))
            if key in seen:
                continue
            unique.append(payload)
            seen.add(key)
            if len(unique) >= self.max_items:
                break
        return unique


def load_model(checkpoint: Path, device: torch.device) -> CPLineNet:
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    config = loaded.get("config", {})
    model = CPLineNet(
        backbone=config.get("backbone", "hrnet_w18"),
        pretrained=False,
        hidden_channels=int(config.get("hidden_channels", 128)),
        v2_heads=bool(config.get("v2_heads", False)),
    ).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but torch.backends.mps.is_available() is false")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def make_builder(
    image_size: int,
    threshold: float,
    *,
    repair_near_endpoint_crossings: bool = False,
) -> PlanarGraphBuilder:
    return PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            image_size=image_size,
            line_threshold=threshold,
            hough_threshold=10,
            hough_min_line_length=6,
            hough_max_line_gap=4,
            min_edge_support=0.45,
            junction_threshold=0.20,
            junction_nms_radius=2,
            vertex_merge_px=max(1.0, 1.5 * image_size / 768),
            line_vertex_distance_px=max(2.0, 4.0 * image_size / 768),
            direct_edge_max_vertices=256,
            direct_edge_short_max_vertices=512,
            planar_cleanup_max_edges=2500,
            repair_near_endpoint_crossings=repair_near_endpoint_crossings,
        )
    )


def build_summary(
    rows: list[dict[str, Any]],
    metric_items: list[tuple[str, str, GraphMetrics]],
    *,
    checkpoint: Path,
    manifest: Path,
    args: argparse.Namespace,
    example_paths: list[Path],
) -> dict[str, Any]:
    by_profile: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    by_profile_family: dict[str, Any] = {}
    metrics_only = [metrics for _, _, metrics in metric_items]

    for profile in sorted({profile for profile, _, _ in metric_items}):
        profile_rows = [row for row in rows if row["profile"] == profile]
        profile_metrics = [metrics for item_profile, _, metrics in metric_items if item_profile == profile]
        by_profile[profile] = {
            **metrics_from_results(profile_metrics),
            **row_counts(profile_rows),
            "assignment_by_class": aggregate_assignment_by_class(profile_metrics),
        }

    for family in sorted({family or "unknown" for _, family, _ in metric_items}):
        family_rows = [row for row in rows if (row["family"] or "unknown") == family]
        family_metrics = [
            metrics
            for _, item_family, metrics in metric_items
            if (item_family or "unknown") == family
        ]
        by_family[family] = {
            **metrics_from_results(family_metrics),
            **row_counts(family_rows),
            "assignment_by_class": aggregate_assignment_by_class(family_metrics),
        }

    for profile in sorted({profile for profile, _, _ in metric_items}):
        for family in sorted({family or "unknown" for _, family, _ in metric_items}):
            rows_slice = [
                row
                for row in rows
                if row["profile"] == profile and (row["family"] or "unknown") == family
            ]
            metrics_slice = [
                metrics
                for item_profile, item_family, metrics in metric_items
                if item_profile == profile and (item_family or "unknown") == family
            ]
            if not rows_slice:
                continue
            by_profile_family[f"{profile}/{family}"] = {
                **metrics_from_results(metrics_slice),
                **row_counts(rows_slice),
                "assignment_by_class": aggregate_assignment_by_class(metrics_slice),
            }

    aggregate = {
        **metrics_from_results(metrics_only),
        **row_counts(rows),
        "assignment_by_class": aggregate_assignment_by_class(metrics_only),
    }
    return {
        "checkpoint": checkpoint.as_posix(),
        "manifest": manifest.as_posix(),
        "split": args.split,
        "profiles": list(args.profiles),
        "sample_count": len(rows),
        "samples_per_profile": args.samples_per_profile,
        "family_sampling": args.family_sampling,
        "threshold": args.threshold,
        "repair_near_endpoint_crossings": bool(args.repair_near_endpoint_crossings),
        "batchnorm_mode": args.batchnorm_mode,
        "image_size": args.image_size,
        "max_edges": args.max_edges,
        "seed": args.seed,
        "infer_assignments": bool(args.infer_assignments),
        "reconstruct_square_border_chain": bool(args.reconstruct_square_border_chain),
        "oracle_border_chain": bool(args.oracle_border_chain),
        "aggregate": aggregate,
        "by_profile": by_profile,
        "by_family": by_family,
        "by_profile_family": by_profile_family,
        "examples": [path.as_posix() for path in example_paths],
    }


def row_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows)
    warning_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    observed_edges = 0
    unknown_edges = 0
    inferred_edges = 0
    drift = []
    for row in rows:
        warning_counts.update(row["warnings"])
        repair_counts.update(row["repairs"])
        observed_edges += int(row["observed_edges"])
        unknown_edges += int(row["unknown_edges"])
        inferred_edges += int(row["inferred_edges"])
        drift.append(float(row["max_geometry_drift_px"]))
    return {
        "status_counts": {status: int(status_counts.get(status, 0)) for status in STATUS_ORDER},
        "warning_counts": dict(sorted(warning_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "observed_edges": observed_edges,
        "unknown_edges": unknown_edges,
        "inferred_edges": inferred_edges,
        "max_geometry_drift_px": max(drift) if drift else 0.0,
    }


def aggregate_assignment_by_class(metrics: list[GraphMetrics]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, int]] = {
        name: {"total": 0, "correct": 0} for name in ("M", "V", "B", "U")
    }
    for item in metrics:
        for name, stats in item.assignment_by_class.items():
            totals[name]["total"] += int(stats["total"])
            totals[name]["correct"] += int(stats["correct"])
    return {
        name: {
            "total": values["total"],
            "correct": values["correct"],
            "accuracy": _ratio(values["correct"], values["total"]),
        }
        for name, values in totals.items()
    }


def write_examples(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for payload in payloads:
        row = payload["row"]
        path = output_dir / f"{row['profile']}_{row['sample_index']:03d}_{safe_id(row['id'])}.png"
        render_example(payload, path)
        paths.append(path)
    return paths


def render_example(payload: dict[str, Any], save_path: Path) -> None:
    row = payload["row"]
    report = payload["report"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), constrained_layout=True)
    axes = axes.ravel()
    image = payload["image"]

    axes[0].imshow(image)
    axes[0].set_title("input")

    axes[1].imshow(image)
    axes[1].imshow(payload["line_prob"], cmap="magma", vmin=0.0, vmax=1.0, alpha=0.62)
    axes[1].set_title("line evidence")

    axes[2].imshow(image)
    draw_graph(
        axes[2],
        payload["pred_vertices"],
        payload["pred_edges"],
        payload["pred_assignments"],
        image,
        sources=payload["pred_sources"],
        confidences=payload["pred_confidence"],
    )
    axes[2].set_title(f"stage 4 graph: {len(payload['pred_edges'])} edges")

    axes[3].imshow(image)
    draw_graph(
        axes[3],
        payload["gt_vertices"],
        payload["gt_edges"],
        payload["gt_assignments"],
        image,
    )
    axes[3].set_title(f"ground truth: {len(payload['gt_edges'])} edges")

    axes[4].imshow(np.full_like(image, 248))
    draw_graph(
        axes[4],
        payload["gt_vertices"],
        payload["gt_edges"],
        payload["gt_assignments"],
        image,
        alpha=0.35,
        linewidth=2.0,
    )
    draw_graph(
        axes[4],
        payload["pred_vertices"],
        payload["pred_edges"],
        payload["pred_assignments"],
        image,
        sources=payload["pred_sources"],
        alpha=0.95,
        linewidth=1.1,
    )
    axes[4].set_title(
        f"overlay: edge P/R {row['edge_precision']:.2f}/{row['edge_recall']:.2f}, "
        f"assign {row['assignment_accuracy']:.2f}"
    )

    axes[5].axis("off")
    warnings = ", ".join(row["warnings"][:5]) if row["warnings"] else "none"
    repairs = ", ".join(row["repairs"][:4]) if row["repairs"] else "none"
    lines = [
        f"{row['profile']} | {row['family']}",
        f"id: {row['id']}",
        f"status: {row['status']}",
        f"warnings: {warnings}",
        f"repairs: {repairs}",
        f"edge P/R: {row['edge_precision']:.3f}/{row['edge_recall']:.3f}",
        f"vertex P/R: {row['vertex_precision']:.3f}/{row['vertex_recall']:.3f}",
        f"assignment acc: {row['assignment_accuracy']:.3f}",
        f"border P/R: {row['border_precision']:.3f}/{row['border_recall']:.3f}",
        (
            "sources: "
            f"observed={row['observed_edges']} "
            f"unknown={row['unknown_edges']} "
            f"inferred={row['inferred_edges']}"
        ),
        f"mean support/conf: {row['mean_edge_support']:.3f}/{row['mean_assignment_confidence']:.3f}",
        f"report summary: {json.dumps(report['summary'], sort_keys=True)[:170]}",
    ]
    axes[5].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        color="#111827",
        wrap=True,
    )

    for ax in axes[:5]:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("Stage 4 checkpoint validation example", fontsize=11)
    fig.savefig(save_path, dpi=125, facecolor="white")
    plt.close(fig)


def draw_graph(
    ax: plt.Axes,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image: np.ndarray,
    *,
    sources: list[str] | None = None,
    confidences: np.ndarray | None = None,
    alpha: float = 0.95,
    linewidth: float = 1.7,
) -> None:
    if len(vertices) == 0 or len(edges) == 0:
        return
    is_dark = float(np.mean(image)) < 100.0
    stroke_color = "white" if is_dark else "black"
    assignment_colors = DARK_BACKGROUND_ASSIGNMENT_COLORS if is_dark else ASSIGNMENT_COLORS
    effects = [pe.Stroke(linewidth=linewidth + 1.6, foreground=stroke_color, alpha=0.7), pe.Normal()]
    for edge_idx, (edge, assignment) in enumerate(zip(edges, assignments)):
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        assignment_int = int(assignment)
        source = sources[edge_idx] if sources is not None and edge_idx < len(sources) else "observed"
        confidence = (
            float(confidences[edge_idx])
            if confidences is not None and edge_idx < len(confidences)
            else 1.0
        )
        line_width = linewidth * (1.35 if assignment_int == 2 else 1.0)
        line_alpha = alpha * (0.45 + 0.55 * np.clip(confidence, 0.0, 1.0))
        linestyle = "--" if source == "unknown" else "-"
        (line,) = ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=assignment_colors.get(assignment_int, assignment_colors[3]),
            linewidth=line_width,
            alpha=line_alpha,
            linestyle=linestyle,
        )
        line.set_path_effects(effects)
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=7,
        c="yellow",
        edgecolors=stroke_color,
        linewidths=0.35,
        alpha=min(1.0, alpha + 0.05),
        zorder=5,
    )


def write_profile_metric_chart(summary: dict[str, Any], path: Path) -> None:
    profiles = list(summary["profiles"])
    metrics = ("edge_precision", "edge_recall", "assignment_accuracy", "structural_validity_rate")
    x = np.arange(len(profiles))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for offset, metric in enumerate(metrics):
        values = [float(summary["by_profile"][profile].get(metric, 0.0)) for profile in profiles]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=metric)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=20, ha="right")
    ax.set_title("Stage 4 validation metrics by render profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def write_status_chart(summary: dict[str, Any], path: Path) -> None:
    profiles = list(summary["profiles"])
    bottom = np.zeros(len(profiles), dtype=np.float32)
    colors = {
        "valid": "#10b981",
        "repaired": "#22c55e",
        "ambiguous": "#f59e0b",
        "outside_v1_envelope": "#f97316",
        "failed": "#ef4444",
    }
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for status in STATUS_ORDER:
        values = np.asarray(
            [summary["by_profile"][profile]["status_counts"].get(status, 0) for profile in profiles],
            dtype=np.float32,
        )
        ax.bar(profiles, values, bottom=bottom, label=status, color=colors[status])
        bottom += values
    ax.set_ylabel("files")
    ax.set_title("Stage 4 quality status by render profile")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=8)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def write_contact_sheet(paths: list[Path], save_path: Path) -> None:
    if not paths:
        return
    import matplotlib.image as mpimg

    cols = 2
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(5, rows * 5.4)), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(rows, cols)
    for ax, path in zip(axes_array.ravel(), paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    for ax in axes_array.ravel()[len(paths) :]:
        ax.axis("off")
    fig.savefig(save_path, dpi=115, facecolor="white")
    plt.close(fig)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    return cleaned[:80]


if __name__ == "__main__":
    main()
