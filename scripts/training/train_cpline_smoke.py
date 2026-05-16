#!/usr/bin/env python3
"""Local-first Phase 3 CPLineNet smoke training.

This is the roadmap-native training path. It trains dense CPLineNet fields from
real scraped `.fold` geometry and evaluates through PlanarGraphBuilder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES, normalize_augment_profile
from src.data.cpline_dataset import CplineFoldDataset, cpline_collate
from src.models import CPLineNet
from src.models.losses import CPLineLoss
from src.vectorization import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    cpline_outputs_to_evidence,
    evaluate_graph,
)
from src.vectorization.metrics import metrics_from_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("fixtures/phase2_real_folds/full_stress.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/phase3_local_smoke"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone", type=str, default="tiny")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--augment-profile", choices=AUGMENT_PROFILES, default="clean")
    parser.add_argument(
        "--render-noise",
        choices=["clean", "mild"],
        default=None,
        help="Deprecated compatibility alias. Use --augment-profile instead.",
    )
    parser.add_argument(
        "--eval-augment-profile",
        choices=AUGMENT_PROFILES,
        default=None,
        help="Optional augmented validation split in addition to clean validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-line-threshold", type=float, default=0.35)
    parser.add_argument(
        "--eval-thresholds",
        type=str,
        default="0.35,0.5,0.65,0.8",
        help="Comma-separated line thresholds to sweep through PlanarGraphBuilder.",
    )
    return parser.parse_args()


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_targets(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "line_prob": batch["line_prob"].to(device),
        "angle": batch["angle"].to(device),
        "junction_heatmap": batch["junction_heatmap"].to(device),
        "junction_offset": batch["junction_offset"].to(device),
        "junction_mask": batch["junction_mask"].to(device),
        "assignment": batch["assignment"].to(device),
    }


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "prediction_cache"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    augment_profile = normalize_augment_profile(args.augment_profile, render_noise=args.render_noise)

    train_dataset = CplineFoldDataset(
        manifest,
        split="train",
        train_count=args.train_count,
        val_count=args.val_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=augment_profile,
        seed=args.seed,
    )
    val_dataset = CplineFoldDataset(
        manifest,
        split="val",
        train_count=args.train_count,
        val_count=args.val_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile="clean",
        seed=args.seed + 1,
    )
    aug_val_dataset = None
    if args.eval_augment_profile:
        aug_val_dataset = CplineFoldDataset(
            manifest,
            split="val",
            train_count=args.train_count,
            val_count=args.val_count,
            max_edges=args.max_edges,
            image_size=args.image_size,
            augment_profile=args.eval_augment_profile,
            seed=args.seed + 2,
        )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=cpline_collate,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=cpline_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=cpline_collate,
    )
    aug_val_loader = None
    if aug_val_dataset is not None:
        aug_val_loader = DataLoader(
            aug_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=cpline_collate,
        )

    model = CPLineNet(
        backbone=args.backbone,
        pretrained=False,
        hidden_channels=args.hidden_channels,
    ).to(device)
    criterion = CPLineLoss()
    optimizer = torch.optim.AdamW(model.get_param_groups(args.lr), lr=args.lr, weight_decay=1e-4)

    run_config = {
        "device": str(device),
        "manifest": manifest.as_posix(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "backbone": args.backbone,
        "hidden_channels": args.hidden_channels,
        "augment_profile": augment_profile,
        "eval_augment_profile": args.eval_augment_profile,
        "render_noise": args.render_noise,
        "max_edges": args.max_edges,
        "lr": args.lr,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_config, indent=2), flush=True)

    model.train()
    history: list[dict[str, float]] = []
    start = perf_counter()
    iterator = cycle(train_loader)
    progress = tqdm(range(1, args.max_steps + 1), desc="CPLineNet local smoke")
    for step in progress:
        batch = next(iterator)
        images = batch["image"].to(device)
        targets = move_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = criterion(outputs, targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"step": float(step), **scalar_losses(losses)}
        history.append(row)
        progress.set_postfix({"loss": f"{row['total']:.4f}", "line": f"{row['line']:.4f}"})

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "history": history,
        },
        output_dir / "post_train.pt",
    )

    val_loss = evaluate_pixel_loss(model, val_loader, criterion, device)
    eval_thresholds = parse_thresholds(args.eval_thresholds, args.eval_line_threshold)
    train_graph_sweep = evaluate_vectorizer_sweep(
        model,
        train_eval_loader,
        device,
        image_size=args.image_size,
        line_thresholds=eval_thresholds,
        output_dir=predictions_dir / "train",
    )
    val_graph_sweep = evaluate_vectorizer_sweep(
        model,
        val_loader,
        device,
        image_size=args.image_size,
        line_thresholds=eval_thresholds,
        output_dir=predictions_dir / "val",
    )
    aug_val_loss = None
    aug_val_graph_sweep = None
    if aug_val_loader is not None:
        aug_val_loss = evaluate_pixel_loss(model, aug_val_loader, criterion, device)
        aug_val_graph_sweep = evaluate_vectorizer_sweep(
            model,
            aug_val_loader,
            device,
            image_size=args.image_size,
            line_thresholds=eval_thresholds,
            output_dir=predictions_dir / "val_augmented",
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": run_config,
        "history": history,
        "val_loss": val_loss,
        "aug_val_loss": aug_val_loss,
        "train_graph_sweep": train_graph_sweep,
        "val_graph_sweep": val_graph_sweep,
        "aug_val_graph_sweep": aug_val_graph_sweep,
    }
    torch.save(checkpoint, output_dir / "latest.pt")

    summary = {
        **run_config,
        "elapsed_seconds": perf_counter() - start,
        "first_train_loss": history[0]["total"] if history else None,
        "last_train_loss": history[-1]["total"] if history else None,
        "val_loss": val_loss,
        "aug_val_loss": aug_val_loss,
        "train_graph_sweep": train_graph_sweep,
        "val_graph_sweep": val_graph_sweep,
        "aug_val_graph_sweep": aug_val_graph_sweep,
        "checkpoint": (output_dir / "latest.pt").as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@torch.no_grad()
def evaluate_pixel_loss(
    model: CPLineNet,
    loader: DataLoader,
    criterion: CPLineLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        losses = criterion(model(images), move_targets(batch, device))
        for key, value in scalar_losses(losses).items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    model.train()
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_vectorizer_sweep(
    model: CPLineNet,
    loader: DataLoader,
    device: torch.device,
    *,
    image_size: int,
    line_thresholds: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_threshold: dict[str, Any] = {}
    best_key = ""
    best_score = -1.0
    for threshold in line_thresholds:
        threshold_dir = output_dir / f"threshold_{threshold:.2f}"
        summary = evaluate_vectorizer(
            model,
            loader,
            device,
            image_size=image_size,
            line_threshold=threshold,
            output_dir=threshold_dir,
        )
        key = f"{threshold:.2f}"
        summary["edge_f1"] = f1(summary.get("edge_precision", 0.0), summary.get("edge_recall", 0.0))
        summary["vertex_f1"] = f1(summary.get("vertex_precision", 0.0), summary.get("vertex_recall", 0.0))
        by_threshold[key] = summary
        score = summary["edge_f1"] + 0.1 * summary.get("structural_validity_rate", 0.0)
        if score > best_score:
            best_score = score
            best_key = key
    payload = {"best_threshold": best_key, "thresholds": by_threshold}
    (output_dir / "threshold_sweep_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


@torch.no_grad()
def evaluate_vectorizer(
    model: CPLineNet,
    loader: DataLoader,
    device: torch.device,
    *,
    image_size: int,
    line_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    builder = PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            image_size=image_size,
            line_threshold=line_threshold,
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
    metrics = []
    rows = []
    sample_index = 0
    for batch in loader:
        outputs = model(batch["image"].to(device))
        for i, graph in enumerate(batch["graph"]):
            evidence = cpline_outputs_to_evidence(
                outputs,
                batch_index=i,
                line_threshold=line_threshold,
            )
            result = builder.build(evidence)
            item_metrics = evaluate_graph(
                result,
                gt_vertices=graph["vertices"].numpy(),
                gt_edges=graph["edges"].numpy(),
                gt_assignments=graph["assignments"].numpy(),
                vertex_tolerance_px=max(3.0, 5.0 * image_size / 768),
            )
            metrics.append(item_metrics)
            meta = batch["meta"][i]
            np.savez_compressed(
                output_dir / f"{sample_index:03d}_{meta['id'][:72]}.npz",
                line_prob=evidence.line_prob,
                angle=evidence.angle,
                junction_heatmap=evidence.junction_heatmap,
                assignment_labels=evidence.assignment_labels,
                pred_vertices=result.pixel_vertices,
                pred_edges=result.edges_vertices,
            )
            rows.append({"id": meta["id"], **item_metrics.to_dict()})
            sample_index += 1

    summary = metrics_from_results(metrics)
    (output_dir / "per_file_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    model.train()
    return summary


def parse_thresholds(raw: str, fallback: float) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        values = [fallback]
    return sorted(set(values))


def f1(precision: float, recall: float) -> float:
    precision = float(precision)
    recall = float(recall)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
