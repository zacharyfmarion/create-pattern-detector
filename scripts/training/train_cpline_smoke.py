#!/usr/bin/env python3
"""Local-first Phase 3 CPLineNet smoke training.

This is the roadmap-native training path. It trains dense CPLineNet fields from
fold-only raw-manifest `.fold` geometry and evaluates through
PlanarGraphBuilder.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import random
import resource
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
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode
from src.models.losses import CPLineLoss, CPLineLossConfig
from src.vectorization import (
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    cpline_outputs_to_evidence,
    evaluate_graph,
)
from src.vectorization.metrics import metrics_from_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
        help="CPLine raw-manifest JSONL path. Rows must include foldPath, split, id, and edges.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/phase3_local_smoke"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=250)
    parser.add_argument(
        "--train-family-sampling",
        choices=["natural", "balanced"],
        default="natural",
        help="Record sampling for the training split. Balanced gives each manifest family an equal share.",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable CUDA DataLoader pinned-memory staging. Useful for memory-capped pods.",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Include process and CUDA memory stats in logged train_step rows.",
    )
    parser.add_argument(
        "--graph-eval-count",
        type=int,
        default=None,
        help="Optional cap per split for expensive PlanarGraphBuilder eval; pixel loss still uses the full validation slice.",
    )
    parser.add_argument(
        "--skip-graph-eval",
        action="store_true",
        help="Skip PlanarGraphBuilder eval and write pixel-loss summaries only.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help=(
            "Skip post-training pixel/vector validation. Useful for chunked "
            "continuation runs where only a resumable checkpoint is needed."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--line-hard-negative-weight",
        type=float,
        default=0.25,
        help="Extra loss weight for highest-loss background line pixels.",
    )
    parser.add_argument(
        "--line-hard-negative-ratio",
        type=float,
        default=0.05,
        help="Maximum background fraction mined for hard-negative line loss.",
    )
    parser.add_argument(
        "--line-hard-negative-multiplier",
        type=float,
        default=4.0,
        help="Maximum hard-negative pixels as a multiple of positive line pixels.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Write a JSONL training row every N steps. Set to 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Write latest_train.pt every N steps during training. Set to 0 to "
            "only save after the training loop finishes."
        ),
    )
    parser.add_argument("--backbone", type=str, default="tiny")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument(
        "--v2-heads",
        action="store_true",
        help="Enable V2 auxiliary non-crease and line-style heads.",
    )
    parser.add_argument("--non-crease-weight", type=float, default=0.0)
    parser.add_argument("--line-style-weight", type=float, default=0.0)
    parser.add_argument("--boundary-contact-weight", type=float, default=0.0)
    parser.add_argument("--boundary-contact-pos-weight", type=float, default=50.0)
    parser.add_argument("--boundary-contact-corner-negative-weight", type=float, default=4.0)
    parser.add_argument("--boundary-contact-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--boundary-contact-hard-negative-ratio", type=float, default=0.02)
    parser.add_argument("--boundary-contact-hard-negative-multiplier", type=float, default=8.0)
    parser.add_argument("--boundary-contact-hard-negative-min-pixels", type=int, default=256)
    parser.add_argument("--vertex-type-weight", type=float, default=0.0)
    parser.add_argument(
        "--vertex-type-class-weights",
        type=str,
        default="0.05,4.0,8.0,1.5",
        help="Comma-separated class weights for V2 vertex background, corner, contact, interior.",
    )
    parser.add_argument("--vertex-type-focal-gamma", type=float, default=1.5)
    parser.add_argument("--boundary-side-weight", type=float, default=0.0)
    parser.add_argument("--boundary-offset-weight", type=float, default=0.0)
    parser.add_argument("--boundary-coord-weight", type=float, default=0.0)
    parser.add_argument(
        "--use-v2-observed-assignment",
        action="store_true",
        help="Train assignment logits against observed labels, marking ambiguous M/V as U.",
    )
    parser.add_argument(
        "--batchnorm-mode",
        choices=BATCHNORM_MODES,
        default="eval",
        help=(
            "BatchNorm behavior during validation/vectorization. Use batch-stats "
            "for small-batch CPLine runs with strong light/dark/photo style mixing."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model weights initialize this run; optimizer starts fresh.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint whose model and optimizer state resume this run. "
            "The dataloader still starts a fresh local stream."
        ),
    )
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


def parse_float_tuple(value: str, *, expected: int) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != expected:
        raise SystemExit(f"Expected {expected} comma-separated floats, got {len(result)}: {value}")
    return result


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
    targets = {
        "line_prob": batch["line_prob"].to(device),
        "angle": batch["angle"].to(device),
        "junction_heatmap": batch["junction_heatmap"].to(device),
        "junction_offset": batch["junction_offset"].to(device),
        "junction_mask": batch["junction_mask"].to(device),
        "assignment": batch["assignment"].to(device),
    }
    for key in [
        "v2_non_crease_mask",
        "v2_target_line_mask",
        "v2_line_style",
        "v2_observed_assignment",
        "v2_boundary_contact_heatmap",
        "v2_vertex_type",
        "v2_boundary_side",
        "v2_boundary_offset",
        "v2_boundary_mask",
        "v2_boundary_coord",
    ]:
        if key in batch:
            targets[key] = batch[key].to(device)
    return targets


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def process_rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def process_max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def memory_stats(device: torch.device) -> dict[str, float]:
    stats = {
        "process_rss_mb": process_rss_mb(),
        "process_max_rss_mb": process_max_rss_mb(),
    }
    if device.type == "cuda":
        stats.update(
            {
                "cuda_allocated_mb": float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0),
                "cuda_reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0),
                "cuda_max_allocated_mb": float(torch.cuda.max_memory_allocated(device))
                / (1024.0 * 1024.0),
                "cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved(device))
                / (1024.0 * 1024.0),
            }
        )
    return stats


def save_training_checkpoint(
    output_dir: Path,
    *,
    model: CPLineNet,
    optimizer: torch.optim.Optimizer,
    run_config: dict[str, Any],
    history: list[dict[str, float]],
    step: int,
    elapsed_seconds: float,
    filename: str = "latest_train.pt",
) -> None:
    tmp_path = output_dir / f".{filename}.tmp"
    final_path = output_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {**run_config, "checkpoint_step": step},
            "history": history,
            "checkpoint_step": step,
            "elapsed_seconds": elapsed_seconds,
        },
        tmp_path,
    )
    tmp_path.replace(final_path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "prediction_cache"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    augment_profile = normalize_augment_profile(
        args.augment_profile, render_noise=args.render_noise
    )

    train_dataset = CplineFoldDataset(
        manifest,
        split="train",
        limit=args.train_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=augment_profile,
        seed=args.seed,
        family_sampling=args.train_family_sampling,
    )
    val_dataset = CplineFoldDataset(
        manifest,
        split="val",
        limit=args.val_count,
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
            limit=args.val_count,
            max_edges=args.max_edges,
            image_size=args.image_size,
            augment_profile=args.eval_augment_profile,
            seed=args.seed + 2,
        )
    pin_memory = device.type == "cuda" and not args.no_pin_memory
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
        v2_heads=args.v2_heads,
    ).to(device)
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("Use only one of --init-checkpoint or --resume-checkpoint.")

    init_checkpoint = args.init_checkpoint
    resume_checkpoint = args.resume_checkpoint
    checkpoint_path = init_checkpoint or resume_checkpoint
    loaded_checkpoint = None
    if checkpoint_path is not None:
        checkpoint_path = (
            checkpoint_path if checkpoint_path.is_absolute() else REPO_ROOT / checkpoint_path
        )
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        loaded = loaded_checkpoint
        model.load_state_dict(loaded["model_state_dict"], strict=not args.v2_heads)
    criterion = CPLineLoss(
        CPLineLossConfig(
            line_hard_negative_weight=args.line_hard_negative_weight,
            line_hard_negative_ratio=args.line_hard_negative_ratio,
            line_hard_negative_multiplier=args.line_hard_negative_multiplier,
            non_crease_weight=args.non_crease_weight,
            line_style_weight=args.line_style_weight,
            use_observed_assignment_target=args.use_v2_observed_assignment,
            boundary_contact_weight=args.boundary_contact_weight,
            boundary_contact_pos_weight=args.boundary_contact_pos_weight,
            boundary_contact_corner_negative_weight=args.boundary_contact_corner_negative_weight,
            boundary_contact_hard_negative_weight=args.boundary_contact_hard_negative_weight,
            boundary_contact_hard_negative_ratio=args.boundary_contact_hard_negative_ratio,
            boundary_contact_hard_negative_multiplier=args.boundary_contact_hard_negative_multiplier,
            boundary_contact_hard_negative_min_pixels=args.boundary_contact_hard_negative_min_pixels,
            vertex_type_weight=args.vertex_type_weight,
            vertex_type_class_weights=parse_float_tuple(args.vertex_type_class_weights, expected=4),
            vertex_type_focal_gamma=args.vertex_type_focal_gamma,
            boundary_side_weight=args.boundary_side_weight,
            boundary_offset_weight=args.boundary_offset_weight,
            boundary_coord_weight=args.boundary_coord_weight,
        )
    )
    optimizer = torch.optim.AdamW(model.get_param_groups(args.lr), lr=args.lr, weight_decay=1e-4)
    if resume_checkpoint is not None and loaded_checkpoint is not None:
        optimizer_state = loaded_checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError(f"Checkpoint has no optimizer_state_dict: {checkpoint_path}")
        optimizer.load_state_dict(optimizer_state)

    run_config = {
        "device": str(device),
        "manifest": manifest.as_posix(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "max_steps": args.max_steps,
        "log_every": args.log_every,
        "log_memory": args.log_memory,
        "checkpoint_every": args.checkpoint_every,
        "graph_eval_count": args.graph_eval_count,
        "skip_graph_eval": args.skip_graph_eval,
        "skip_final_eval": args.skip_final_eval,
        "backbone": args.backbone,
        "hidden_channels": args.hidden_channels,
        "v2_heads": args.v2_heads,
        "batchnorm_mode": args.batchnorm_mode,
        "init_checkpoint": init_checkpoint.as_posix() if init_checkpoint is not None else None,
        "resume_checkpoint": (
            resume_checkpoint.as_posix() if resume_checkpoint is not None else None
        ),
        "loaded_checkpoint": checkpoint_path.as_posix() if checkpoint_path is not None else None,
        "augment_profile": augment_profile,
        "eval_augment_profile": args.eval_augment_profile,
        "render_noise": args.render_noise,
        "max_edges": args.max_edges,
        "train_family_sampling": args.train_family_sampling,
        "lr": args.lr,
        "line_hard_negative_weight": args.line_hard_negative_weight,
        "line_hard_negative_ratio": args.line_hard_negative_ratio,
        "line_hard_negative_multiplier": args.line_hard_negative_multiplier,
        "non_crease_weight": args.non_crease_weight,
        "line_style_weight": args.line_style_weight,
        "boundary_contact_weight": args.boundary_contact_weight,
        "boundary_contact_pos_weight": args.boundary_contact_pos_weight,
        "boundary_contact_corner_negative_weight": args.boundary_contact_corner_negative_weight,
        "boundary_contact_hard_negative_weight": args.boundary_contact_hard_negative_weight,
        "boundary_contact_hard_negative_ratio": args.boundary_contact_hard_negative_ratio,
        "boundary_contact_hard_negative_multiplier": args.boundary_contact_hard_negative_multiplier,
        "boundary_contact_hard_negative_min_pixels": args.boundary_contact_hard_negative_min_pixels,
        "vertex_type_weight": args.vertex_type_weight,
        "vertex_type_class_weights": parse_float_tuple(args.vertex_type_class_weights, expected=4),
        "vertex_type_focal_gamma": args.vertex_type_focal_gamma,
        "boundary_side_weight": args.boundary_side_weight,
        "boundary_offset_weight": args.boundary_offset_weight,
        "boundary_coord_weight": args.boundary_coord_weight,
        "use_v2_observed_assignment": args.use_v2_observed_assignment,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2), flush=True)
    history_path = output_dir / "train_history.jsonl"
    history_path.write_text("", encoding="utf-8")

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
        if args.log_every > 0 and (
            step == 1 or step % args.log_every == 0 or step == args.max_steps
        ):
            log_row = {**row, "elapsed_seconds": perf_counter() - start}
            if args.log_memory:
                log_row.update(memory_stats(device))
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row) + "\n")
            print(json.dumps({"event": "train_step", **log_row}), flush=True)
        if args.checkpoint_every > 0 and (
            step % args.checkpoint_every == 0 or step == args.max_steps
        ):
            elapsed_seconds = perf_counter() - start
            save_training_checkpoint(
                output_dir,
                model=model,
                optimizer=optimizer,
                run_config=run_config,
                history=history,
                step=step,
                elapsed_seconds=elapsed_seconds,
            )
            print(
                json.dumps(
                    {
                        "event": "training_checkpoint",
                        "step": step,
                        "path": (output_dir / "latest_train.pt").as_posix(),
                        "elapsed_seconds": elapsed_seconds,
                    }
                ),
                flush=True,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "history": history,
        },
        output_dir / "post_train.pt",
    )
    if args.skip_final_eval:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "history": history,
            "val_loss": None,
            "aug_val_loss": None,
            "train_graph_sweep": None,
            "val_graph_sweep": None,
            "aug_val_graph_sweep": None,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        summary = {
            **run_config,
            "elapsed_seconds": perf_counter() - start,
            "first_train_loss": history[0]["total"] if history else None,
            "last_train_loss": history[-1]["total"] if history else None,
            "val_loss": None,
            "aug_val_loss": None,
            "train_graph_sweep": None,
            "val_graph_sweep": None,
            "aug_val_graph_sweep": None,
            "checkpoint": (output_dir / "latest.pt").as_posix(),
            "training_checkpoint": (output_dir / "latest_train.pt").as_posix(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    val_loss = evaluate_pixel_loss(
        model,
        val_loader,
        criterion,
        device,
        batchnorm_mode=args.batchnorm_mode,
    )
    eval_thresholds = parse_thresholds(args.eval_thresholds, args.eval_line_threshold)
    train_graph_sweep = None
    val_graph_sweep = None
    if not args.skip_graph_eval:
        train_graph_sweep = evaluate_vectorizer_sweep(
            model,
            train_eval_loader,
            device,
            image_size=args.image_size,
            line_thresholds=eval_thresholds,
            output_dir=predictions_dir / "train",
            batchnorm_mode=args.batchnorm_mode,
            max_samples=args.graph_eval_count,
        )
        val_graph_sweep = evaluate_vectorizer_sweep(
            model,
            val_loader,
            device,
            image_size=args.image_size,
            line_thresholds=eval_thresholds,
            output_dir=predictions_dir / "val",
            batchnorm_mode=args.batchnorm_mode,
            max_samples=args.graph_eval_count,
        )
    aug_val_loss = None
    aug_val_graph_sweep = None
    if aug_val_loader is not None:
        aug_val_loss = evaluate_pixel_loss(
            model,
            aug_val_loader,
            criterion,
            device,
            batchnorm_mode=args.batchnorm_mode,
        )
        if not args.skip_graph_eval:
            aug_val_graph_sweep = evaluate_vectorizer_sweep(
                model,
                aug_val_loader,
                device,
                image_size=args.image_size,
                line_thresholds=eval_thresholds,
                output_dir=predictions_dir / "val_augmented",
                batchnorm_mode=args.batchnorm_mode,
                max_samples=args.graph_eval_count,
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
    *,
    batchnorm_mode: str = "eval",
) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    with model_eval_with_batchnorm_mode(model, batchnorm_mode=batchnorm_mode):
        for batch in loader:
            images = batch["image"].to(device)
            losses = criterion(model(images), move_targets(batch, device))
            for key, value in scalar_losses(losses).items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
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
    batchnorm_mode: str = "eval",
    max_samples: int | None = None,
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
            batchnorm_mode=batchnorm_mode,
            max_samples=max_samples,
        )
        key = f"{threshold:.2f}"
        summary["edge_f1"] = f1(summary.get("edge_precision", 0.0), summary.get("edge_recall", 0.0))
        summary["vertex_f1"] = f1(
            summary.get("vertex_precision", 0.0), summary.get("vertex_recall", 0.0)
        )
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
    batchnorm_mode: str = "eval",
    max_samples: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    with model_eval_with_batchnorm_mode(model, batchnorm_mode=batchnorm_mode):
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
                if max_samples is not None and sample_index >= max_samples:
                    break
            if max_samples is not None and sample_index >= max_samples:
                break

    summary = metrics_from_results(metrics)
    (output_dir / "per_file_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "graph_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
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
