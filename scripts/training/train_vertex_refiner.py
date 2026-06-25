#!/usr/bin/env python3
"""Local smoke and probe training for VertexRefiner crop models."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.vertex_refiner_dataset import VertexRefinerCropDataset, vertex_refiner_collate
from src.evaluation.vertex_refiner_eval import (
    evaluate_vertex_refiner,
    vertex_refiner_targets_to_device,
)
from src.models import VertexRefinerV1, VertexRefinerV2, VertexRefinerV3
from src.models.losses import VertexRefinerLoss, VertexRefinerLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/vertex_refiner_local_smoke"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model weights initialize this run.",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--proposals-per-sample", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--shuffle-train-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle crop refs in the training loader. Disable for record-ordered cached runs.",
    )
    parser.add_argument(
        "--rendered-sample-cache-size",
        type=int,
        default=None,
        help="Optional max number of rendered full-pattern samples cached per dataset.",
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--model-version", choices=["v1", "v2", "v3"], default="v1")
    parser.add_argument(
        "--train-crop-refs",
        type=Path,
        default=None,
        help="Optional precomputed crop-ref cache for the selected train split.",
    )
    parser.add_argument(
        "--val-crop-refs",
        type=Path,
        default=None,
        help="Optional precomputed crop-ref cache for the selected val split.",
    )
    parser.add_argument(
        "--crop-ref-progress-every",
        type=int,
        default=0,
        help="Print dataset crop-ref construction progress every N selected records.",
    )
    parser.add_argument(
        "--auxiliary-mode",
        choices=["zero", "rendered-labels"],
        default="zero",
        help=(
            "Legacy CPLineNet auxiliary source. Use zero for source-only training; "
            "model-version=v2/v3 selects source/frame channels separately."
        ),
    )
    parser.add_argument("--auxiliary-dropout-p", type=float, default=0.5)
    parser.add_argument(
        "--include-gt-training-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GT-centered proposals in the training split for positive coverage.",
    )
    parser.add_argument(
        "--boundary-gt-anchor-repeats",
        type=int,
        default=0,
        help="Training-only jittered boundary-contact GT anchors per boundary vertex.",
    )
    parser.add_argument(
        "--boundary-gt-anchor-jitter-px",
        type=float,
        default=6.0,
        help="Stddev for training-only boundary-contact GT anchor jitter.",
    )
    parser.add_argument(
        "--include-val-gt-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include GT-centered proposals in validation. Keep false for product-style metrics.",
    )
    parser.add_argument("--heatmap-threshold", type=float, default=0.25)
    parser.add_argument("--match-tolerance-px", type=float, default=2.0)
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Cap before/after metric batches. Useful for paid probes with many generated crops.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--abort-loss-threshold",
        type=float,
        default=100000.0,
        help="Early-terminate if total loss is non-finite or exceeds this value.",
    )
    parser.add_argument(
        "--early-eval-every",
        type=int,
        default=0,
        help="Run validation during training every N steps. Disabled at 0.",
    )
    parser.add_argument(
        "--early-stop-min-val-f1",
        type=float,
        default=None,
        help="Early-terminate if an intermediate validation F1 is below this value.",
    )
    parser.add_argument(
        "--early-stop-after-step",
        type=int,
        default=0,
        help="Do not apply validation-based early termination before this step.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = select_device(args.device)
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_crop_refs = _optional_repo_path(args.train_crop_refs)
    val_crop_refs = _optional_repo_path(args.val_crop_refs)

    train_dataset = VertexRefinerCropDataset(
        manifest,
        split="train",
        limit=args.train_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=args.include_gt_training_anchors,
        boundary_gt_anchor_repeats=args.boundary_gt_anchor_repeats,
        boundary_gt_anchor_jitter_px=args.boundary_gt_anchor_jitter_px,
        auxiliary_mode=args.auxiliary_mode,
        input_version=args.model_version,
        rendered_sample_cache_size=args.rendered_sample_cache_size,
        crop_refs_path=train_crop_refs,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    val_dataset = VertexRefinerCropDataset(
        manifest,
        split="val",
        limit=args.val_count,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed + 1000,
        proposals_per_sample=args.proposals_per_sample,
        include_gt_training_anchors=args.include_val_gt_anchors,
        boundary_gt_anchor_repeats=0,
        auxiliary_mode=args.auxiliary_mode,
        input_version=args.model_version,
        rendered_sample_cache_size=args.rendered_sample_cache_size,
        crop_refs_path=val_crop_refs,
        crop_ref_progress_every=args.crop_ref_progress_every,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train_crops,
        num_workers=args.num_workers,
        collate_fn=vertex_refiner_collate,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vertex_refiner_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vertex_refiner_collate,
    )

    model_cls = _model_class(args.model_version)
    model = model_cls(base_channels=args.base_channels).to(device)
    init_checkpoint_path = None
    if args.init_checkpoint is not None:
        init_checkpoint_path = (
            args.init_checkpoint
            if args.init_checkpoint.is_absolute()
            else REPO_ROOT / args.init_checkpoint
        )
        init_checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(init_checkpoint["model_state_dict"])
    criterion = VertexRefinerLoss(VertexRefinerLossConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    preview_batch = next(iter(val_loader))
    run_config = {
        "device": str(device),
        "manifest": manifest.as_posix(),
        "init_checkpoint": None if init_checkpoint_path is None else init_checkpoint_path.as_posix(),
        "image_size": args.image_size,
        "train_records": args.train_count,
        "val_records": args.val_count,
        "train_crops": len(train_dataset),
        "val_crops": len(val_dataset),
        "max_edges": args.max_edges,
        "proposals_per_sample": args.proposals_per_sample,
        "batch_size": args.batch_size,
        "shuffle_train_crops": args.shuffle_train_crops,
        "rendered_sample_cache_size": args.rendered_sample_cache_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "model_version": args.model_version,
        "train_crop_refs": None if train_crop_refs is None else train_crop_refs.as_posix(),
        "val_crop_refs": None if val_crop_refs is None else val_crop_refs.as_posix(),
        "train_crop_refs_source": train_dataset.crop_refs_source,
        "val_crop_refs_source": val_dataset.crop_refs_source,
        "crop_ref_progress_every": args.crop_ref_progress_every,
        "auxiliary_mode": args.auxiliary_mode,
        "auxiliary_dropout_p": args.auxiliary_dropout_p,
        "include_gt_training_anchors": args.include_gt_training_anchors,
        "boundary_gt_anchor_repeats": args.boundary_gt_anchor_repeats,
        "boundary_gt_anchor_jitter_px": args.boundary_gt_anchor_jitter_px,
        "include_val_gt_anchors": args.include_val_gt_anchors,
        "heatmap_threshold": args.heatmap_threshold,
        "match_tolerance_px": args.match_tolerance_px,
        "eval_max_batches": args.eval_max_batches,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "abort_loss_threshold": args.abort_loss_threshold,
        "early_eval_every": args.early_eval_every,
        "early_stop_min_val_f1": args.early_stop_min_val_f1,
        "early_stop_after_step": args.early_stop_after_step,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    start = perf_counter()
    before_train = evaluate_vertex_refiner(
        model,
        train_eval_loader,
        device=device,
        criterion=criterion,
        heatmap_threshold=args.heatmap_threshold,
        match_tolerance_px=args.match_tolerance_px,
        max_batches=args.eval_max_batches,
    )
    before_val = evaluate_vertex_refiner(
        model,
        val_loader,
        device=device,
        criterion=criterion,
        heatmap_threshold=args.heatmap_threshold,
        match_tolerance_px=args.match_tolerance_px,
        max_batches=args.eval_max_batches,
    )
    with torch.no_grad():
        preview_before = {
            key: value.detach().cpu()
            for key, value in model(preview_batch["input"].to(device)).items()
        }

    model.train()
    history: list[dict[str, float]] = []
    early_evaluations: list[dict[str, Any]] = []
    early_stop_reason: str | None = None
    loader_iter = iter(train_loader)
    history_path = output_dir / "train_history.jsonl"
    history_path.write_text("", encoding="utf-8")
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        inputs = batch["input"].to(device)
        targets = vertex_refiner_targets_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, auxiliary_dropout_p=args.auxiliary_dropout_p)
        losses = criterion(outputs, targets)
        total_loss_value = float(losses["total"].detach().cpu())
        row = {"step": float(step), **scalar_losses(losses)}
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if not np.isfinite(total_loss_value) or total_loss_value > float(args.abort_loss_threshold):
            early_stop_reason = (
                f"bad_loss_at_step_{step}: total={total_loss_value:g}, "
                f"threshold={float(args.abort_loss_threshold):g}"
            )
            print(json.dumps({"step": step, "early_stop_reason": early_stop_reason}), flush=True)
            break
        losses["total"].backward()
        optimizer.step()
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            print(json.dumps(row), flush=True)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_vertex_refiner_checkpoint(
                output_dir / "latest_train.pt",
                model=model,
                optimizer=optimizer,
                run_config=run_config,
                history=history,
                elapsed_seconds=perf_counter() - start,
                early_stop_reason=early_stop_reason,
            )
        if args.early_eval_every > 0 and step % args.early_eval_every == 0:
            intermediate_val = evaluate_vertex_refiner(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                heatmap_threshold=args.heatmap_threshold,
                match_tolerance_px=args.match_tolerance_px,
                max_batches=args.eval_max_batches,
            )
            eval_row = {"step": step, "val": intermediate_val}
            early_evaluations.append(eval_row)
            print(json.dumps({"step": step, "intermediate_val": intermediate_val}), flush=True)
            if (
                args.early_stop_min_val_f1 is not None
                and step >= int(args.early_stop_after_step)
                and float(intermediate_val["f1"]) < float(args.early_stop_min_val_f1)
            ):
                early_stop_reason = (
                    f"low_val_f1_at_step_{step}: f1={float(intermediate_val['f1']):g}, "
                    f"minimum={float(args.early_stop_min_val_f1):g}"
                )
                print(json.dumps({"step": step, "early_stop_reason": early_stop_reason}), flush=True)
                break

    after_train = evaluate_vertex_refiner(
        model,
        train_eval_loader,
        device=device,
        criterion=criterion,
        heatmap_threshold=args.heatmap_threshold,
        match_tolerance_px=args.match_tolerance_px,
        max_batches=args.eval_max_batches,
    )
    after_val = evaluate_vertex_refiner(
        model,
        val_loader,
        device=device,
        criterion=criterion,
        heatmap_threshold=args.heatmap_threshold,
        match_tolerance_px=args.match_tolerance_px,
        max_batches=args.eval_max_batches,
    )
    with torch.no_grad():
        model.eval()
        preview_after = {
            key: value.detach().cpu()
            for key, value in model(preview_batch["input"].to(device)).items()
        }
    qualitative_overlay_path = output_dir / "qualitative_before_after.png"
    write_qualitative_before_after_overlay(
        preview_batch,
        before_outputs=preview_before,
        after_outputs=preview_after,
        path=qualitative_overlay_path,
        heatmap_threshold=args.heatmap_threshold,
    )
    elapsed_seconds = perf_counter() - start
    checkpoint_path = output_dir / "latest.pt"
    save_vertex_refiner_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        run_config=run_config,
        history=history,
        elapsed_seconds=elapsed_seconds,
        early_stop_reason=early_stop_reason,
    )
    summary = {
        "schema": "create-pattern-detector/vertex-refiner-local-smoke/v1",
        "run_config": run_config,
        "before": {"train": before_train, "val": before_val},
        "after": {"train": after_train, "val": after_val},
        "loss_delta": {
            "train": _nullable_delta(before_train["loss"], after_train["loss"]),
            "val": _nullable_delta(before_val["loss"], after_val["loss"]),
        },
        "first_step_loss": None if not history else history[0]["total"],
        "last_step_loss": None if not history else history[-1]["total"],
        "early_stop_reason": early_stop_reason,
        "early_evaluations": early_evaluations,
        "checkpoint": checkpoint_path.as_posix(),
        "qualitative_overlay": qualitative_overlay_path.as_posix(),
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _model_class(model_version: str) -> type[torch.nn.Module]:
    if model_version == "v1":
        return VertexRefinerV1
    if model_version == "v2":
        return VertexRefinerV2
    if model_version == "v3":
        return VertexRefinerV3
    raise ValueError(f"Unsupported model_version: {model_version}")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def save_vertex_refiner_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_config: dict[str, Any],
    history: list[dict[str, float]],
    elapsed_seconds: float,
    early_stop_reason: str | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "history": history,
            "elapsed_seconds": elapsed_seconds,
            "early_stop_reason": early_stop_reason,
        },
        path,
    )


def _nullable_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _optional_repo_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def write_qualitative_before_after_overlay(
    batch: dict[str, Any],
    *,
    before_outputs: dict[str, torch.Tensor],
    after_outputs: dict[str, torch.Tensor],
    path: Path,
    heatmap_threshold: float,
    max_items: int = 8,
) -> None:
    """Write a compact source/target/before/after validation crop sheet."""
    from PIL import Image, ImageDraw

    inputs = batch["input"].detach().cpu().numpy()
    targets = batch["vertex_heatmap"].detach().cpu().numpy()
    before_heatmaps = torch.sigmoid(before_outputs["vertex_heatmap"]).detach().cpu().numpy()
    after_heatmaps = torch.sigmoid(after_outputs["vertex_heatmap"]).detach().cpu().numpy()
    local_vertices = batch["local_vertices"]

    item_count = min(int(inputs.shape[0]), int(max_items))
    if item_count <= 0:
        return

    panels: list[list[np.ndarray]] = []
    for index in range(item_count):
        gray = np.clip(inputs[index, 0], 0.0, 1.0)
        vertices = local_vertices[index].detach().cpu().numpy()
        source = _draw_vertices(_gray_to_rgb(gray), vertices, color=(0, 120, 255))
        target = _draw_vertices(
            _overlay_heatmap(gray, targets[index, 0], color=(0, 190, 80)),
            vertices,
            color=(0, 120, 255),
        )
        before = _draw_vertices(
            _overlay_heatmap(gray, before_heatmaps[index, 0], color=(230, 70, 70)),
            vertices,
            color=(0, 120, 255),
            threshold_map=before_heatmaps[index, 0],
            threshold=heatmap_threshold,
            threshold_color=(255, 255, 255),
        )
        after = _draw_vertices(
            _overlay_heatmap(gray, after_heatmaps[index, 0], color=(70, 160, 240)),
            vertices,
            color=(0, 120, 255),
            threshold_map=after_heatmaps[index, 0],
            threshold=heatmap_threshold,
            threshold_color=(255, 255, 255),
        )
        panels.append([source, target, before, after])

    tile_h, tile_w = panels[0][0].shape[:2]
    gutter = 4
    label_h = 12
    sheet_w = 4 * tile_w + 3 * gutter
    sheet_h = item_count * (tile_h + label_h) + max(0, item_count - 1) * gutter
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)
    labels = ("source", "target", "before", "after")
    pil = Image.fromarray(sheet)
    draw = ImageDraw.Draw(pil)
    for row_index, row_panels in enumerate(panels):
        y0 = row_index * (tile_h + label_h + gutter)
        for col_index, panel in enumerate(row_panels):
            x0 = col_index * (tile_w + gutter)
            sheet_panel = Image.fromarray(panel)
            pil.paste(sheet_panel, (x0, y0 + label_h))
            draw.text((x0 + 2, y0), labels[col_index], fill=(35, 35, 35))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    value = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(value[:, :, None], 3, axis=2)


def _overlay_heatmap(gray: np.ndarray, heatmap: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    rgb = _gray_to_rgb(gray).astype(np.float32)
    alpha = np.clip(heatmap.astype(np.float32), 0.0, 1.0) ** 0.7
    alpha = np.clip(alpha * 0.7, 0.0, 0.7)[:, :, None]
    color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
    blended = rgb * (1.0 - alpha) + color_arr * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _draw_vertices(
    image: np.ndarray,
    vertices: np.ndarray,
    *,
    color: tuple[int, int, int],
    threshold_map: np.ndarray | None = None,
    threshold: float = 0.25,
    threshold_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    if threshold_map is not None:
        ys, xs = np.where(np.asarray(threshold_map, dtype=np.float32) >= float(threshold))
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((int(x), int(y)), fill=threshold_color)
    for x, y in np.asarray(vertices, dtype=np.float32).reshape(-1, 2):
        radius = 2
        draw.ellipse(
            (
                float(x) - radius,
                float(y) - radius,
                float(x) + radius,
                float(y) + radius,
            ),
            outline=color,
            width=1,
        )
    return np.asarray(pil, dtype=np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
