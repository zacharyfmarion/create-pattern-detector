#!/usr/bin/env python3
"""
Evaluate pixel head checkpoint on validation set.

Usage:
    python scripts/evals/eval_pixel_head.py --checkpoint checkpoints/checkpoint_epoch_14.pt
    python scripts/evals/eval_pixel_head.py --checkpoint checkpoints/checkpoint_epoch_8.pt --checkpoint2 checkpoints/checkpoint_epoch_14.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from src.data.dataset import CreasePatternDataset, collate_fn
from src.models import CreasePatternDetector
from src.models.losses import PixelLoss


def compute_seg_iou(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute IoU for each class and mean IoU."""
    pred = logits.argmax(dim=1)
    class_names = {0: 'BG', 1: 'M', 2: 'V', 3: 'B', 4: 'U'}
    ious = {}

    for c in range(5):
        pred_c = pred == c
        target_c = targets == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            ious[class_names[c]] = (intersection / union).item()
        else:
            ious[class_names[c]] = float('nan')

    # Mean IoU excluding BG and NaN
    crease_ious = [v for k, v in ious.items() if k != 'BG' and not np.isnan(v)]
    ious['mean_crease'] = sum(crease_ious) / len(crease_ious) if crease_ious else 0.0
    ious['mean_all'] = np.nanmean(list(ious.values())[:5])

    return ious


def compute_seg_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute per-class and overall accuracy."""
    pred = logits.argmax(dim=1)
    class_names = {0: 'BG', 1: 'M', 2: 'V', 3: 'B', 4: 'U'}

    accs = {}
    for c in range(5):
        mask = targets == c
        if mask.sum() > 0:
            correct = (pred[mask] == c).sum().float()
            accs[class_names[c]] = (correct / mask.sum()).item()
        else:
            accs[class_names[c]] = float('nan')

    # Overall accuracy
    accs['overall'] = (pred == targets).float().mean().item()

    return accs


def compute_junction_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute precision, recall, F1 for junction detection."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def compute_orientation_error(pred: torch.Tensor, target: torch.Tensor, seg_target: torch.Tensor) -> dict:
    """Compute orientation error on crease pixels only."""
    # Create mask for crease pixels (classes 1-4)
    crease_mask = (seg_target >= 1) & (seg_target <= 4)

    if crease_mask.sum() == 0:
        return {'mean_angle_error': float('nan'), 'cos_similarity': float('nan')}

    # Normalize predictions
    pred_norm = torch.nn.functional.normalize(pred, dim=1)
    target_norm = torch.nn.functional.normalize(target, dim=1)

    # Compute cosine similarity on masked pixels
    cos_sim = (pred_norm * target_norm).sum(dim=1)  # [B, H, W]

    # Apply mask
    cos_sim_masked = cos_sim[crease_mask]

    # Angle error (handle potential NaN from acos)
    cos_sim_clamped = torch.clamp(cos_sim_masked.abs(), 0, 1)  # abs because orientation is bidirectional
    angle_error = torch.acos(cos_sim_clamped) * 180 / np.pi  # Convert to degrees

    return {
        'mean_angle_error': angle_error.mean().item(),
        'cos_similarity': cos_sim_clamped.mean().item(),
    }


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, fold_dir: str, image_size: int = 1024, batch_size: int = 4, max_batches: int = None) -> dict:
    """Evaluate a checkpoint on validation set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create model
    model = CreasePatternDetector(
        backbone="hrnet_w32",
        num_seg_classes=5,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataset
    dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=image_size,
        padding=50,
        transform=None,
        split="val",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Validation set size: {len(dataset)}")
    if max_batches:
        print(f"Limiting to {max_batches} batches ({max_batches * batch_size} samples)")

    # Loss function
    criterion = PixelLoss(
        seg_weight=1.0,
        orient_weight=0.5,
        junction_weight=1.0,
        junction_pos_weight=50.0,
    )

    # Aggregate metrics
    all_metrics = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if max_batches and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        targets = {
            "segmentation": batch["segmentation"].to(device),
            "orientation": batch["orientation"].to(device),
            "junction_heatmap": batch["junction_heatmap"].to(device),
        }

        # Forward pass
        outputs = model(images)

        # Compute losses
        loss_dict = criterion(outputs, targets)
        all_metrics['loss'].append(loss_dict['total'].item())
        all_metrics['seg_loss'].append(loss_dict['seg'].item())
        all_metrics['orient_loss'].append(loss_dict['orient'].item())
        all_metrics['junction_loss'].append(loss_dict['junction'].item())

        # Segmentation metrics
        seg_iou = compute_seg_iou(outputs['segmentation'], targets['segmentation'])
        for k, v in seg_iou.items():
            all_metrics[f'seg_iou_{k}'].append(v)

        seg_acc = compute_seg_accuracy(outputs['segmentation'], targets['segmentation'])
        for k, v in seg_acc.items():
            all_metrics[f'seg_acc_{k}'].append(v)

        # Junction metrics
        junction_metrics = compute_junction_metrics(outputs['junction'], targets['junction_heatmap'])
        for k, v in junction_metrics.items():
            all_metrics[f'junction_{k}'].append(v)

        # Orientation metrics
        orient_metrics = compute_orientation_error(
            outputs['orientation'],
            targets['orientation'],
            targets['segmentation'],
        )
        for k, v in orient_metrics.items():
            all_metrics[f'orient_{k}'].append(v)

    # Average metrics
    results = {}
    for k, v in all_metrics.items():
        results[k] = np.nanmean(v)

    return results


def print_comparison(results1: dict, results2: dict, name1: str, name2: str):
    """Print side-by-side comparison of two checkpoint results."""
    print("\n" + "=" * 80)
    print(f"COMPARISON: {name1} vs {name2}")
    print("=" * 80)

    def format_val(v):
        if np.isnan(v):
            return "N/A"
        return f"{v:.4f}"

    def format_diff(v1, v2, higher_better=True):
        if np.isnan(v1) or np.isnan(v2):
            return ""
        diff = v2 - v1
        pct = (diff / abs(v1) * 100) if v1 != 0 else 0
        symbol = "+" if (diff > 0) == higher_better else "-"
        return f"{symbol} {abs(diff):.4f} ({abs(pct):.1f}%)"

    # Losses (lower is better)
    print("\n--- Losses (lower is better) ---")
    for key in ['loss', 'seg_loss', 'orient_loss', 'junction_loss']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=False)}")

    # Segmentation IoU (higher is better)
    print("\n--- Segmentation IoU (higher is better) ---")
    for key in ['seg_iou_BG', 'seg_iou_M', 'seg_iou_V', 'seg_iou_B', 'seg_iou_U', 'seg_iou_mean_crease']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=True)}")

    # Segmentation accuracy (higher is better)
    print("\n--- Segmentation Accuracy (higher is better) ---")
    for key in ['seg_acc_BG', 'seg_acc_M', 'seg_acc_V', 'seg_acc_B', 'seg_acc_U', 'seg_acc_overall']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=True)}")

    # Junction metrics (higher is better for precision/recall/f1)
    print("\n--- Junction Detection (higher is better) ---")
    for key in ['junction_precision', 'junction_recall', 'junction_f1']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=True)}")

    # Orientation metrics
    print("\n--- Orientation (lower angle error, higher cos_sim is better) ---")
    for key in ['orient_mean_angle_error']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=False)}")
    for key in ['orient_cos_similarity']:
        v1, v2 = results1.get(key, float('nan')), results2.get(key, float('nan'))
        print(f"  {key:20s}: {format_val(v1):>10s} -> {format_val(v2):>10s}  {format_diff(v1, v2, higher_better=True)}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pixel head checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Second checkpoint for comparison")
    parser.add_argument("--fold-dir", type=str, default="data/training/full-training/fold", help="FOLD directory")
    parser.add_argument("--image-size", type=int, default=1024, help="Image size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to evaluate (for quick testing)")
    args = parser.parse_args()

    # Evaluate first checkpoint
    print("\n" + "=" * 80)
    print("EVALUATING CHECKPOINT 1")
    print("=" * 80)
    results1 = evaluate_checkpoint(args.checkpoint, args.fold_dir, args.image_size, args.batch_size, args.max_batches)

    # Print results
    print("\n--- Results ---")
    for k, v in sorted(results1.items()):
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: N/A")

    # Evaluate second checkpoint if provided
    if args.checkpoint2:
        print("\n" + "=" * 80)
        print("EVALUATING CHECKPOINT 2")
        print("=" * 80)
        results2 = evaluate_checkpoint(args.checkpoint2, args.fold_dir, args.image_size, args.batch_size, args.max_batches)

        # Print comparison
        name1 = Path(args.checkpoint).stem
        name2 = Path(args.checkpoint2).stem
        print_comparison(results1, results2, name1, name2)


if __name__ == "__main__":
    main()
