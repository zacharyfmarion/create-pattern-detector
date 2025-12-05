#!/usr/bin/env python3
"""
Visualize data augmentations to validate they work correctly.

Usage:
    python scripts/visualize/augmentations.py --fold-dir data/output/synthetic/raw/tier-a-small --num-samples 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import CreasePatternDataset
from src.data.transforms import (
    CreasePatternTransform,
    DarkMode,
    HueShiftCreases,
    get_train_transform,
    get_val_transform,
)


def visualize_individual_augmentations(sample_image: np.ndarray, output_dir: Path):
    """Show effect of each augmentation individually."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(sample_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Dark Mode
    dark_mode = DarkMode(always_apply=True)
    dark_img = dark_mode(image=sample_image)["image"]
    axes[0, 1].imshow(dark_img)
    axes[0, 1].set_title("Dark Mode")
    axes[0, 1].axis("off")

    # Hue Shift (apply multiple times to show variation)
    hue_shift = HueShiftCreases(hue_shift_limit=20, always_apply=True)
    shifted_img = hue_shift(image=sample_image)["image"]
    axes[0, 2].imshow(shifted_img)
    axes[0, 2].set_title("Hue Shift")
    axes[0, 2].axis("off")

    # Dark Mode + Hue Shift combined
    dark_shifted = dark_mode(image=shifted_img)["image"]
    axes[1, 0].imshow(dark_shifted)
    axes[1, 0].set_title("Dark Mode + Hue Shift")
    axes[1, 0].axis("off")

    # Another hue shift variation
    shifted_img2 = hue_shift(image=sample_image)["image"]
    axes[1, 1].imshow(shifted_img2)
    axes[1, 1].set_title("Hue Shift (different)")
    axes[1, 1].axis("off")

    # Another hue shift variation
    shifted_img3 = hue_shift(image=sample_image)["image"]
    axes[1, 2].imshow(shifted_img3)
    axes[1, 2].set_title("Hue Shift (different)")
    axes[1, 2].axis("off")

    plt.suptitle("Individual Augmentation Effects", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "augmentation_effects.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def visualize_training_augmentations(
    dataset: CreasePatternDataset,
    num_samples: int,
    output_dir: Path,
):
    """Show multiple augmented versions of the same sample."""
    # Get a sample without augmentation first
    dataset_no_aug = CreasePatternDataset(
        fold_dir=dataset.fold_dir,
        image_size=dataset.image_size,
        padding=dataset.padding,
        line_width=dataset.line_width,
        transform=get_val_transform(dataset.image_size),
        split="train",
    )

    for sample_idx in range(min(num_samples, len(dataset_no_aug))):
        # Get original (no augmentation)
        original_sample = dataset_no_aug[sample_idx]
        original_img = (original_sample["image"].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        filename = original_sample["meta"]["filename"]

        # Show individual augmentation effects for first sample
        if sample_idx == 0:
            visualize_individual_augmentations(original_img, output_dir)

        # Create figure with original + 5 augmented versions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        # Generate 5 augmented versions
        for i, ax in enumerate(axes.flat[1:]):
            # Re-fetch with augmentation (each call applies random augmentations)
            aug_sample = dataset[sample_idx]
            aug_img = (aug_sample["image"].permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )

            ax.imshow(aug_img)
            ax.set_title(f"Augmented #{i+1}")
            ax.axis("off")

        plt.suptitle(f"Training Augmentations: {filename}", fontsize=14)
        plt.tight_layout()

        save_path = output_dir / f"{filename}_augmentations.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


def visualize_segmentation_consistency(
    dataset: CreasePatternDataset,
    output_dir: Path,
):
    """Verify that segmentation masks transform correctly with images."""
    # Color map for segmentation
    seg_colors = np.array(
        [
            [255, 255, 255],  # BG - white
            [255, 0, 0],  # M - red
            [0, 0, 255],  # V - blue
            [0, 0, 0],  # B - black
            [128, 128, 128],  # U - gray
        ],
        dtype=np.uint8,
    )

    def seg_to_color(seg):
        h, w = seg.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(5):
            rgb[seg == c] = seg_colors[c]
        return rgb

    # Get a few samples
    for sample_idx in range(min(3, len(dataset))):
        sample = dataset[sample_idx]
        img = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        seg = sample["segmentation"].numpy()
        junction = sample["junction_heatmap"].numpy()[0]
        filename = sample["meta"]["filename"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img)
        axes[0].set_title("Augmented Image")
        axes[0].axis("off")

        axes[1].imshow(seg_to_color(seg))
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        # Overlay segmentation on image
        overlay = img.copy()
        seg_rgb = seg_to_color(seg)
        mask = seg > 0
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * seg_rgb[mask]).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (should align)")
        axes[2].axis("off")

        axes[3].imshow(junction, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Junction Heatmap")
        axes[3].axis("off")

        plt.suptitle(f"Augmentation Consistency: {filename}", fontsize=14)
        plt.tight_layout()

        save_path = output_dir / f"{filename}_consistency.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize data augmentations")
    parser.add_argument(
        "--fold-dir",
        type=str,
        required=True,
        help="Directory containing .fold files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="augmentation_vis",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset with training augmentations
    padding = int(50 * args.image_size / 1024)
    line_width = max(1, int(2 * args.image_size / 1024))

    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        padding=padding,
        line_width=line_width,
        transform=get_train_transform(args.image_size, strength="medium"),
        split="train",
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Output directory: {output_dir}")
    print()

    print("=" * 60)
    print("1. TRAINING AUGMENTATION VARIATIONS")
    print("=" * 60)
    visualize_training_augmentations(dataset, args.num_samples, output_dir)

    print()
    print("=" * 60)
    print("2. SEGMENTATION CONSISTENCY CHECK")
    print("=" * 60)
    visualize_segmentation_consistency(dataset, output_dir)

    print()
    print(f"Done! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
