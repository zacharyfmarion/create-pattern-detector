#!/usr/bin/env python3
"""Visualize CPLineNet graph extraction from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
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

from src.data.cpline_augmentations import AUGMENT_PROFILES
from src.data.cpline_dataset import CplineFoldDataset, cpline_collate
from src.models import CPLineNet
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode
from src.vectorization import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    cpline_outputs_to_evidence,
    evaluate_graph,
)


ASSIGNMENT_COLORS = {
    0: "#ff5f5f",  # M
    1: "#2094ff",  # V
    2: "#f2f2f2",  # B
    3: "#9aa0a6",  # U
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/cpline_graph_eval"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--augment-profile", choices=AUGMENT_PROFILES, default="clean")
    parser.add_argument(
        "--batchnorm-mode",
        choices=BATCHNORM_MODES,
        default="batch-stats",
        help="BatchNorm behavior during checkpoint visualization.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    config = loaded.get("config", {})
    model = CPLineNet(
        backbone=config.get("backbone", "hrnet_w18"),
        pretrained=False,
        hidden_channels=int(config.get("hidden_channels", 128)),
    ).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()

    dataset = CplineFoldDataset(
        manifest,
        split=args.split,
        limit=args.num_samples,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=args.augment_profile,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=cpline_collate)
    builder = make_builder(args.image_size, args.threshold)

    rows: list[dict[str, Any]] = []
    with torch.no_grad(), model_eval_with_batchnorm_mode(
        model,
        batchnorm_mode=args.batchnorm_mode,
    ):
        for sample_index, batch in enumerate(loader):
            outputs = model(batch["image"].to(device))
            evidence = cpline_outputs_to_evidence(outputs, batch_index=0, line_threshold=args.threshold)
            result = builder.build(evidence)
            graph = batch["graph"][0]
            metrics = evaluate_graph(
                result,
                gt_vertices=graph["vertices"].numpy(),
                gt_edges=graph["edges"].numpy(),
                gt_assignments=graph["assignments"].numpy(),
                vertex_tolerance_px=max(3.0, 5.0 * args.image_size / 768),
            ).to_dict()
            meta = batch["meta"][0]
            image = (batch["image"][0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            save_path = output_dir / f"{sample_index:02d}_{safe_id(meta['id'])}.png"
            render_sheet(
                image=image,
                evidence=evidence,
                pred_vertices=result.pixel_vertices,
                pred_edges=result.edges_vertices,
                pred_assignments=result.edges_assignment,
                gt_vertices=graph["vertices"].numpy(),
                gt_edges=graph["edges"].numpy(),
                gt_assignments=graph["assignments"].numpy(),
                metrics=metrics,
                title=(
                    f"{args.augment_profile} threshold={args.threshold:.2f} "
                    f"bn={args.batchnorm_mode} {meta['id']}"
                ),
                save_path=save_path,
            )
            row = {
                "id": meta["id"],
                "profile": args.augment_profile,
                "threshold": args.threshold,
                "image": save_path.name,
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


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


def make_builder(image_size: int, threshold: float) -> PlanarGraphBuilder:
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
        )
    )


def render_sheet(
    *,
    image: np.ndarray,
    evidence: Any,
    pred_vertices: np.ndarray,
    pred_edges: np.ndarray,
    pred_assignments: np.ndarray,
    gt_vertices: np.ndarray,
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("input")

    axes[1].imshow(image)
    axes[1].imshow(evidence.line_prob, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.62)
    axes[1].set_title("predicted line evidence")

    axes[2].imshow(evidence.junction_heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("predicted junction heatmap")

    axes[3].imshow(image)
    draw_graph(axes[3], gt_vertices, gt_edges, gt_assignments, image)
    axes[3].set_title(f"ground truth: {len(gt_edges)} edges")

    axes[4].imshow(image)
    draw_graph(axes[4], pred_vertices, pred_edges, pred_assignments, image)
    axes[4].set_title(f"pred graph: {len(pred_edges)} edges")

    axes[5].imshow(np.full_like(image, 248))
    draw_graph(axes[5], gt_vertices, gt_edges, gt_assignments, image, alpha=0.35, linewidth=2.0)
    draw_graph(axes[5], pred_vertices, pred_edges, pred_assignments, image, alpha=0.95, linewidth=1.1)
    axes[5].set_title(
        "overlay\n"
        f"edge P/R {metrics['edge_precision']:.2f}/{metrics['edge_recall']:.2f}, "
        f"vertex P/R {metrics['vertex_precision']:.2f}/{metrics['vertex_recall']:.2f}, "
        f"assign {metrics['assignment_accuracy']:.2f}"
    )

    for ax in axes:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.savefig(save_path, dpi=130, facecolor="white")
    plt.close(fig)


def draw_graph(
    ax: plt.Axes,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image: np.ndarray,
    *,
    alpha: float = 0.95,
    linewidth: float = 1.7,
) -> None:
    if len(vertices) == 0 or len(edges) == 0:
        return
    stroke_color = "white" if float(np.mean(image)) < 100.0 else "black"
    effects = [pe.Stroke(linewidth=linewidth + 1.6, foreground=stroke_color, alpha=0.7), pe.Normal()]
    for edge, assignment in zip(edges, assignments):
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        (line,) = ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=ASSIGNMENT_COLORS.get(int(assignment), ASSIGNMENT_COLORS[3]),
            linewidth=linewidth,
            alpha=alpha,
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


def safe_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    return cleaned[:80]


if __name__ == "__main__":
    main()
