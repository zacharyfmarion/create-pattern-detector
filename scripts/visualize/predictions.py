#!/usr/bin/env python3
"""
Visualize model predictions vs ground truth.

Usage:
    python scripts/visualize/predictions.py --checkpoint checkpoints/best_model.pt --fold-dir data_small --num-samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import CreasePatternDataset, collate_fn
from src.data.transforms import get_val_transform
from src.models import CreasePatternDetector


# Color map for segmentation classes
# BG=0 (white), M=1 (red), V=2 (blue), B=3 (black), U=4 (gray)
SEG_COLORS = np.array([
    [255, 255, 255],  # Background - white
    [255, 0, 0],      # M - red
    [0, 0, 255],      # V - blue
    [0, 0, 0],        # B - black
    [128, 128, 128],  # U - gray
], dtype=np.uint8)


def seg_to_color(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB image."""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(5):
        mask = seg == c
        rgb[mask] = SEG_COLORS[c]
    return rgb


def visualize_sample(
    image: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    gt_junction: np.ndarray,
    pred_junction: np.ndarray,
    filename: str,
    save_path: Path = None,
):
    """Visualize a single sample with predictions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Input, GT Segmentation, Pred Segmentation
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_to_color(gt_seg))
    axes[0, 1].set_title("Ground Truth Segmentation")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(seg_to_color(pred_seg))
    axes[0, 2].set_title("Predicted Segmentation")
    axes[0, 2].axis("off")

    # Row 2: Overlay, GT Junctions, Pred Junctions
    # Create overlay of prediction on input
    overlay = image.copy()
    pred_color = seg_to_color(pred_seg)
    # Blend where there are predictions (non-background)
    mask = pred_seg > 0
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * pred_color[mask]).astype(np.uint8)

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Prediction Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_junction, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Ground Truth Junctions")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_junction, cmap="hot", vmin=0, vmax=1)
    axes[1, 2].set_title("Predicted Junctions")
    axes[1, 2].axis("off")

    plt.suptitle(f"Sample: {filename}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--fold-dir",
        type=str,
        required=True,
        help="Directory containing .fold files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (should match training)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Get image size from checkpoint config or use argument
    image_size = config.get("image_size", args.image_size)
    padding = config.get("padding", int(50 * image_size / 1024))
    line_width = config.get("line_width", max(1, int(2 * image_size / 1024)))

    print(f"Image size: {image_size}")

    # Create model
    model = CreasePatternDetector(
        backbone=config.get("backbone", "hrnet_w32"),
        pretrained=False,
        hidden_channels=config.get("hidden_channels", 256),
        num_seg_classes=config.get("num_seg_classes", 5),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Create dataset
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=image_size,
        padding=padding,
        line_width=line_width,
        transform=get_val_transform(image_size),
        split=args.split,
    )

    print(f"Dataset size ({args.split}): {len(dataset)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]

            # Get input image
            image = sample["image"].unsqueeze(0).to(device)

            # Run inference
            outputs = model(image)

            # Get predictions
            pred_seg = outputs["segmentation"].argmax(dim=1).cpu().numpy()[0]
            pred_junction = torch.sigmoid(outputs["junction"]).cpu().numpy()[0, 0]

            # Get ground truth
            gt_seg = sample["segmentation"].numpy()
            gt_junction = sample["junction_heatmap"].numpy()[0]

            # Convert image to numpy (H, W, C) for display
            img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Get filename
            filename = sample["meta"]["filename"]

            # Visualize
            save_path = output_dir / f"{filename}_prediction.png"
            visualize_sample(
                image=img_np,
                gt_seg=gt_seg,
                pred_seg=pred_seg,
                gt_junction=gt_junction,
                pred_junction=pred_junction,
                filename=filename,
                save_path=save_path,
            )

    print(f"\nDone! Visualizations saved to: {output_dir}")
    print(f"View with: ls {output_dir}")


if __name__ == "__main__":
    main()
