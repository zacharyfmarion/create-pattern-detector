#!/usr/bin/env python3
"""
Quick visual comparison of two pixel head checkpoints.

Usage:
    python scripts/visualize/compare_pixel_head_checkpoints.py \
        --checkpoint1 checkpoints/checkpoint_epoch_8.pt \
        --checkpoint2 checkpoints/checkpoint_epoch_14.pt \
        --num-samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.data.dataset import CreasePatternDataset
from src.models import CreasePatternDetector


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CreasePatternDetector(
        backbone="hrnet_w32",
        num_seg_classes=5,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch = checkpoint.get('epoch', 'unknown')
    return model, epoch


@torch.no_grad()
def compare_checkpoints(
    checkpoint1: str,
    checkpoint2: str,
    fold_dir: str,
    num_samples: int = 5,
    image_size: int = 512,
    output_dir: str = "comparison_output",
):
    """Compare two checkpoints visually."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading checkpoint 1: {checkpoint1}")
    model1, epoch1 = load_model(checkpoint1, device)
    print(f"  Epoch: {epoch1}")

    print(f"\nLoading checkpoint 2: {checkpoint2}")
    model2, epoch2 = load_model(checkpoint2, device)
    print(f"  Epoch: {epoch2}")

    # Load dataset
    dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=image_size,
        padding=50,
        transform=None,
        split="val",
    )
    print(f"\nDataset size: {len(dataset)}")

    # Sample random indices
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Color maps
    seg_colors = ['white', 'red', 'blue', 'black', 'gray']
    seg_cmap = ListedColormap(seg_colors)
    class_names = ['BG', 'M', 'V', 'B', 'U']

    for i, idx in enumerate(indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (idx={idx})...")

        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_seg = sample['segmentation'].numpy()
        gt_junction = sample['junction_heatmap'].squeeze().numpy()

        # Get predictions
        out1 = model1(image)
        out2 = model2(image)

        pred_seg1 = out1['segmentation'][0].argmax(dim=0).cpu().numpy()
        pred_seg2 = out2['segmentation'][0].argmax(dim=0).cpu().numpy()

        pred_junction1 = torch.sigmoid(out1['junction'][0, 0]).cpu().numpy()
        pred_junction2 = torch.sigmoid(out2['junction'][0, 0]).cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        # Row 1: Input and GT
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_seg, cmap=seg_cmap, vmin=0, vmax=4)
        axes[0, 1].set_title('GT Segmentation')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gt_junction, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('GT Junction')
        axes[0, 2].axis('off')

        # Compute metrics for this sample
        iou1 = compute_iou(pred_seg1, gt_seg)
        iou2 = compute_iou(pred_seg2, gt_seg)
        axes[0, 3].text(0.1, 0.8, f"Epoch {epoch1} mIoU: {iou1:.3f}", fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.6, f"Epoch {epoch2} mIoU: {iou2:.3f}", fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.4, f"Improvement: {iou2-iou1:+.3f}", fontsize=12,
                        color='green' if iou2 > iou1 else 'red', transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')
        axes[0, 3].set_title('Metrics')

        # Row 2: Checkpoint 1 predictions
        axes[1, 0].imshow(pred_seg1, cmap=seg_cmap, vmin=0, vmax=4)
        axes[1, 0].set_title(f'Epoch {epoch1} Segmentation')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_junction1, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Epoch {epoch1} Junction')
        axes[1, 1].axis('off')

        # Diff: where checkpoint1 differs from GT
        diff1 = (pred_seg1 != gt_seg).astype(float)
        axes[1, 2].imshow(diff1, cmap='Reds', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Epoch {epoch1} Errors')
        axes[1, 2].axis('off')

        # Per-class accuracy
        acc_text1 = ""
        for c in range(5):
            mask = gt_seg == c
            if mask.sum() > 0:
                acc = (pred_seg1[mask] == c).mean()
                acc_text1 += f"{class_names[c]}: {acc:.2f}\n"
        axes[1, 3].text(0.1, 0.9, acc_text1, fontsize=10, transform=axes[1, 3].transAxes, verticalalignment='top')
        axes[1, 3].axis('off')
        axes[1, 3].set_title(f'Epoch {epoch1} Per-class Acc')

        # Row 3: Checkpoint 2 predictions
        axes[2, 0].imshow(pred_seg2, cmap=seg_cmap, vmin=0, vmax=4)
        axes[2, 0].set_title(f'Epoch {epoch2} Segmentation')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(pred_junction2, cmap='hot', vmin=0, vmax=1)
        axes[2, 1].set_title(f'Epoch {epoch2} Junction')
        axes[2, 1].axis('off')

        # Diff: where checkpoint2 differs from GT
        diff2 = (pred_seg2 != gt_seg).astype(float)
        axes[2, 2].imshow(diff2, cmap='Reds', vmin=0, vmax=1)
        axes[2, 2].set_title(f'Epoch {epoch2} Errors')
        axes[2, 2].axis('off')

        # Per-class accuracy
        acc_text2 = ""
        for c in range(5):
            mask = gt_seg == c
            if mask.sum() > 0:
                acc = (pred_seg2[mask] == c).mean()
                acc_text2 += f"{class_names[c]}: {acc:.2f}\n"
        axes[2, 3].text(0.1, 0.9, acc_text2, fontsize=10, transform=axes[2, 3].transAxes, verticalalignment='top')
        axes[2, 3].axis('off')
        axes[2, 3].set_title(f'Epoch {epoch2} Per-class Acc')

        plt.suptitle(f'Comparison: Epoch {epoch1} vs Epoch {epoch2} (Sample {idx})', fontsize=14)
        plt.tight_layout()

        out_file = output_path / f"comparison_{i:02d}_idx{idx}.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_file}")

    print(f"\n Done! Results saved to {output_path}/")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean IoU for crease classes (1-4)."""
    ious = []
    for c in range(1, 5):
        pred_c = pred == c
        gt_c = gt == c
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare two pixel head checkpoints visually")
    parser.add_argument("--checkpoint1", type=str, required=True, help="First checkpoint")
    parser.add_argument("--checkpoint2", type=str, required=True, help="Second checkpoint")
    parser.add_argument("--fold-dir", type=str, default="data/training/full-training/fold", help="FOLD directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to compare")
    parser.add_argument("--image-size", type=int, default=512, help="Image size (smaller = faster)")
    parser.add_argument("--output-dir", type=str, default="visualizations/checkpoint_comparisons", help="Output directory")
    args = parser.parse_args()

    compare_checkpoints(
        args.checkpoint1,
        args.checkpoint2,
        args.fold_dir,
        args.num_samples,
        args.image_size,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
