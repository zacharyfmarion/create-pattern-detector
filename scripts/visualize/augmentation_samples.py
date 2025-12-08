#!/usr/bin/env python3
"""
Visualize data augmentations to validate they work correctly before training.

Usage:
    python scripts/visualize/augmentation_samples.py --data-dir data/training/full-training --num-samples 5 --output-dir visualizations/augmentations
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data.dataset import CreasePatternDataset
from src.data.transforms import CreasePatternTransform


# Segmentation class colors for visualization
SEG_COLORS = {
    0: [1.0, 1.0, 1.0],    # BG - white
    1: [1.0, 0.0, 0.0],    # M - red
    2: [0.0, 0.0, 1.0],    # V - blue
    3: [0.0, 0.0, 0.0],    # B - black
    4: [0.5, 0.5, 0.5],    # U - gray
}

SEG_NAMES = {
    0: "Background",
    1: "Mountain",
    2: "Valley",
    3: "Border",
    4: "Unassigned",
}


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB for visualization."""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id, color in SEG_COLORS.items():
        mask = seg == cls_id
        rgb[mask] = color
    return rgb


def visualize_sample(
    original_image: np.ndarray,
    original_seg: np.ndarray,
    augmented_image: np.ndarray,
    augmented_seg: np.ndarray,
    sample_idx: int,
    iteration: int,
    output_dir: Path,
):
    """Create side-by-side comparison of original vs augmented."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # Original segmentation
    axes[0, 1].imshow(seg_to_rgb(original_seg))
    axes[0, 1].set_title("Original Segmentation", fontsize=12)
    axes[0, 1].axis("off")

    # Augmented image
    axes[1, 0].imshow(augmented_image)
    axes[1, 0].set_title("Augmented Image", fontsize=12)
    axes[1, 0].axis("off")

    # Augmented segmentation
    axes[1, 1].imshow(seg_to_rgb(augmented_seg))
    axes[1, 1].set_title("Augmented Segmentation", fontsize=12)
    axes[1, 1].axis("off")

    # Add legend
    patches = [mpatches.Patch(color=color, label=SEG_NAMES[cls_id])
               for cls_id, color in SEG_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10)

    # Count class pixels
    orig_counts = {name: np.sum(original_seg == cls_id) for cls_id, name in SEG_NAMES.items()}
    aug_counts = {name: np.sum(augmented_seg == cls_id) for cls_id, name in SEG_NAMES.items()}

    # Check for changes
    changes = []
    if orig_counts["Mountain"] > 0 and aug_counts["Mountain"] == 0:
        changes.append("M→U")
    if orig_counts["Valley"] > 0 and aug_counts["Valley"] == 0:
        changes.append("V→U")
    if aug_counts["Unassigned"] > orig_counts["Unassigned"]:
        changes.append(f"U increased: {orig_counts['Unassigned']}→{aug_counts['Unassigned']}")

    title = f"Sample {sample_idx}, Iteration {iteration}"
    if changes:
        title += f" | Changes: {', '.join(changes)}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    output_path = output_dir / f"sample_{sample_idx}_iter_{iteration}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return changes


def main():
    parser = argparse.ArgumentParser(description="Visualize augmentations")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--iterations", type=int, default=5, help="Augmentation iterations per sample")
    parser.add_argument("--output-dir", type=str, default="visualizations/augmentations")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strength", type=str, default="medium", choices=["light", "medium", "heavy"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset WITHOUT transforms first (to get originals)
    print(f"Loading dataset from {args.data_dir}...")
    dataset_no_aug = CreasePatternDataset(
        fold_dir=args.data_dir,
        image_size=args.image_size,
        transform=None,
        split="train",
    )

    # Create transform for augmentation
    transform = CreasePatternTransform(
        image_size=args.image_size,
        strength=args.strength,
    )

    print(f"Visualizing {args.num_samples} samples × {args.iterations} iterations...")
    print(f"Augmentation strength: {args.strength}")
    print(f"Output directory: {output_dir}")
    print()

    # Track augmentation statistics
    stats = {
        "assignment_removal": 0,
        "text_overlay": 0,  # Hard to detect automatically
        "dark_mode": 0,
        "gray_background": 0,
        "total": 0,
    }

    for sample_idx in range(min(args.num_samples, len(dataset_no_aug))):
        # Get original sample
        original = dataset_no_aug[sample_idx]
        original_image = original["image"].permute(1, 2, 0).numpy()  # CHW -> HWC
        original_image = (original_image * 255).astype(np.uint8)
        original_seg = original["segmentation"].numpy()

        print(f"Sample {sample_idx}")

        # Get graph data
        graph = original["graph"]
        vertices = graph["vertices"].numpy()
        edges = graph["edges"].numpy()
        assignments = graph["assignments"].numpy()

        # Get orientation and junction heatmap (need to transpose from CHW to HWC/HW)
        orientation = original["orientation"].permute(1, 2, 0).numpy()  # CHW -> HWC
        junction_heatmap = original["junction_heatmap"].squeeze(0).numpy()  # 1HW -> HW

        for iteration in range(args.iterations):
            stats["total"] += 1

            # Apply transform manually to the raw arrays
            aug_result = transform(
                image=original_image.copy(),
                segmentation=original_seg.copy(),
                orientation=orientation.copy(),
                junction_heatmap=junction_heatmap.copy(),
                vertices=vertices.copy(),
                edges=edges.copy(),
                assignments=assignments.copy(),
            )

            augmented_image = aug_result["image"]
            augmented_seg = aug_result["segmentation"]

            # Detect specific augmentations
            orig_m_pixels = np.sum(original_seg == 1)
            orig_v_pixels = np.sum(original_seg == 2)
            aug_m_pixels = np.sum(augmented_seg == 1)
            aug_v_pixels = np.sum(augmented_seg == 2)

            # Assignment removal detection
            if (orig_m_pixels > 0 and aug_m_pixels == 0) or (orig_v_pixels > 0 and aug_v_pixels == 0):
                stats["assignment_removal"] += 1

            # Dark mode detection (check if background is dark)
            bg_mask = augmented_seg == 0
            if np.sum(bg_mask) > 0:
                bg_mean = np.mean(augmented_image[bg_mask])
                if bg_mean < 100:
                    stats["dark_mode"] += 1
                elif bg_mean < 220:
                    stats["gray_background"] += 1

            changes = visualize_sample(
                original_image,
                original_seg,
                augmented_image,
                augmented_seg,
                sample_idx,
                iteration,
                output_dir,
            )

            if changes:
                print(f"  Iter {iteration}: {', '.join(changes)}")

        print()

    # Print statistics
    print("=" * 50)
    print("Augmentation Statistics:")
    print(f"  Total iterations: {stats['total']}")
    print(f"  Assignment removal: {stats['assignment_removal']} ({100*stats['assignment_removal']/stats['total']:.1f}%)")
    print(f"  Dark mode: {stats['dark_mode']} ({100*stats['dark_mode']/stats['total']:.1f}%)")
    print(f"  Gray background: {stats['gray_background']} ({100*stats['gray_background']/stats['total']:.1f}%)")
    print()
    print(f"Saved {stats['total']} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
