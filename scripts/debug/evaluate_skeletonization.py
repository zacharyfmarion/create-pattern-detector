#!/usr/bin/env python3
"""
Evaluate and optimize skeletonization quality using ground truth.

Computes metrics:
- Edge recall: % of GT crease pixels covered by skeleton
- Edge precision: % of skeleton pixels that are on GT creases
- Junction recall: % of GT junctions found in skeleton
- Topology score: connectivity preservation

Usage:
    python scripts/debug/evaluate_skeletonization.py --fold-dir data/output/synthetic/raw/tier-a --num-samples 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize, thin
from tqdm import tqdm

from src.data.dataset import CreasePatternDataset
from src.data.transforms import get_val_transform


def extract_skeleton(seg_mask: np.ndarray, method: str = "skeletonize") -> np.ndarray:
    """
    Extract skeleton from segmentation mask.

    Args:
        seg_mask: (H, W) segmentation with classes 0=BG, 1-4=creases
        method: 'skeletonize' or 'thin'

    Returns:
        (H, W) binary skeleton
    """
    # Create binary mask of all creases (classes 1-4)
    binary = seg_mask > 0

    if method == "skeletonize":
        skeleton = skeletonize(binary)
    elif method == "thin":
        skeleton = thin(binary)
    else:
        raise ValueError(f"Unknown method: {method}")

    return skeleton.astype(np.uint8)


def find_skeleton_junctions(skeleton: np.ndarray) -> np.ndarray:
    """
    Find junction points in skeleton (pixels with 3+ neighbors).

    Returns:
        (N, 2) array of (y, x) junction coordinates
    """
    from scipy.ndimage import convolve

    # Ensure skeleton is binary
    skel_binary = (skeleton > 0).astype(np.uint8)

    # Count neighbors for each skeleton pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = convolve(skel_binary, kernel, mode='constant')

    # Junctions have 3+ neighbors and are skeleton pixels
    junctions = (neighbor_count >= 3) & (skel_binary > 0)

    ys, xs = np.where(junctions)
    return np.stack([ys, xs], axis=1) if len(ys) > 0 else np.zeros((0, 2), dtype=np.int32)


def compute_metrics(
    gt_seg: np.ndarray,
    gt_junction_heatmap: np.ndarray,
    skeleton: np.ndarray,
    distance_threshold: float = 3.0,
    junction_threshold: float = 0.5,
) -> dict:
    """
    Compute skeletonization quality metrics.

    Args:
        gt_seg: (H, W) ground truth segmentation
        gt_junction_heatmap: (H, W) ground truth junction heatmap
        skeleton: (H, W) binary skeleton
        distance_threshold: max distance for a skeleton pixel to count as covering GT
        junction_threshold: threshold for GT junction detection

    Returns:
        Dictionary of metrics
    """
    # Binary GT crease mask
    gt_crease = gt_seg > 0
    skeleton_bool = skeleton.astype(bool)

    # 1. Edge Recall: % of GT crease pixels within distance_threshold of skeleton
    if skeleton_bool.sum() > 0:
        dist_to_skeleton = distance_transform_edt(~skeleton_bool)
        gt_covered = gt_crease & (dist_to_skeleton <= distance_threshold)
        edge_recall = gt_covered.sum() / max(gt_crease.sum(), 1)
    else:
        edge_recall = 0.0

    # 2. Edge Precision: % of skeleton pixels within distance_threshold of GT crease
    if gt_crease.sum() > 0:
        dist_to_gt = distance_transform_edt(~gt_crease)
        skel_on_gt = skeleton_bool & (dist_to_gt <= distance_threshold)
        edge_precision = skel_on_gt.sum() / max(skeleton_bool.sum(), 1)
    else:
        edge_precision = 0.0

    # 3. F1 score
    if edge_precision + edge_recall > 0:
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)
    else:
        edge_f1 = 0.0

    # 4. Junction recall: % of GT junctions found near skeleton junctions
    gt_junction_peaks = gt_junction_heatmap > junction_threshold
    gt_junction_coords = np.stack(np.where(gt_junction_peaks), axis=1) if gt_junction_peaks.sum() > 0 else np.zeros((0, 2))

    skel_junctions = find_skeleton_junctions(skeleton)

    if len(gt_junction_coords) > 0 and len(skel_junctions) > 0:
        # For each GT junction, check if there's a skeleton junction within threshold
        from scipy.spatial.distance import cdist
        distances = cdist(gt_junction_coords, skel_junctions)
        min_distances = distances.min(axis=1) if distances.size > 0 else np.array([])
        junction_recall = (min_distances <= distance_threshold * 2).sum() / len(gt_junction_coords)
    elif len(gt_junction_coords) == 0:
        junction_recall = 1.0  # No junctions to find
    else:
        junction_recall = 0.0

    # 5. Skeleton junction precision
    if len(skel_junctions) > 0 and len(gt_junction_coords) > 0:
        from scipy.spatial.distance import cdist
        distances = cdist(skel_junctions, gt_junction_coords)
        min_distances = distances.min(axis=1)
        junction_precision = (min_distances <= distance_threshold * 2).sum() / len(skel_junctions)
    elif len(skel_junctions) == 0:
        junction_precision = 1.0
    else:
        junction_precision = 0.0

    # 6. Connectivity: count connected components in skeleton vs GT
    gt_labels, gt_num_components = label(gt_crease)
    skel_labels, skel_num_components = label(skeleton)

    # Ideally skeleton should have same or fewer components (no fragmentation)
    connectivity_ratio = skel_num_components / max(gt_num_components, 1)

    # 7. Skeleton density (pixels per unit length - lower is better for thin skeletons)
    skeleton_pixels = skeleton_bool.sum()

    return {
        "edge_recall": edge_recall,
        "edge_precision": edge_precision,
        "edge_f1": edge_f1,
        "junction_recall": junction_recall,
        "junction_precision": junction_precision,
        "connectivity_ratio": connectivity_ratio,
        "gt_components": gt_num_components,
        "skel_components": skel_num_components,
        "skeleton_pixels": int(skeleton_pixels),
        "gt_crease_pixels": int(gt_crease.sum()),
        "num_gt_junctions": len(gt_junction_coords),
        "num_skel_junctions": len(skel_junctions),
    }


def visualize_comparison(
    image: np.ndarray,
    gt_seg: np.ndarray,
    skeleton: np.ndarray,
    gt_junction_heatmap: np.ndarray,
    metrics: dict,
    save_path: Path = None,
):
    """Visualize GT vs skeleton comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Color map for segmentation
    seg_colors = np.array([
        [255, 255, 255],  # BG
        [255, 0, 0],      # M
        [0, 0, 255],      # V
        [0, 0, 0],        # B
        [128, 128, 128],  # U
    ], dtype=np.uint8)

    def seg_to_rgb(seg):
        h, w = seg.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(5):
            rgb[seg == c] = seg_colors[c]
        return rgb

    # Row 1: Input, GT Segmentation, Skeleton
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_to_rgb(gt_seg))
    axes[0, 1].set_title("GT Segmentation")
    axes[0, 1].axis("off")

    # Show skeleton - invert so lines are black on white
    axes[0, 2].imshow(1 - skeleton, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Skeleton ({int(metrics['skeleton_pixels'])} px)")
    axes[0, 2].axis("off")

    # Row 2: Overlay, Junction comparison, Metrics
    # Overlay skeleton on GT
    overlay = seg_to_rgb(gt_seg).copy()
    skel_mask = skeleton.astype(bool)
    overlay[skel_mask] = [0, 255, 0]  # Green skeleton
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Skeleton (green) on GT")
    axes[1, 0].axis("off")

    # Junction comparison
    skel_junctions = find_skeleton_junctions(skeleton)
    axes[1, 1].imshow(gt_junction_heatmap, cmap="hot", vmin=0, vmax=1)
    if len(skel_junctions) > 0:
        axes[1, 1].scatter(skel_junctions[:, 1], skel_junctions[:, 0],
                          c='lime', s=30, marker='x', linewidths=1)
    axes[1, 1].set_title(f"GT Junctions + Skeleton Junctions (x)\nRecall: {metrics['junction_recall']:.2f}")
    axes[1, 1].axis("off")

    # Metrics text
    metrics_text = (
        f"Edge Recall: {metrics['edge_recall']:.3f}\n"
        f"Edge Precision: {metrics['edge_precision']:.3f}\n"
        f"Edge F1: {metrics['edge_f1']:.3f}\n"
        f"\n"
        f"Junction Recall: {metrics['junction_recall']:.3f}\n"
        f"Junction Precision: {metrics['junction_precision']:.3f}\n"
        f"\n"
        f"GT Components: {metrics['gt_components']}\n"
        f"Skel Components: {metrics['skel_components']}\n"
        f"Connectivity Ratio: {metrics['connectivity_ratio']:.2f}\n"
        f"\n"
        f"GT Junctions: {metrics['num_gt_junctions']}\n"
        f"Skel Junctions: {metrics['num_skel_junctions']}"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Metrics")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate skeletonization quality")
    parser.add_argument("--fold-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="visualizations/skeletonization")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--visualize", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--method", type=str, default="skeletonize", choices=["skeletonize", "thin"])
    parser.add_argument("--distance-threshold", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.fold_dir}...")
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        transform=get_val_transform(args.image_size),
        split="val",
    )

    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples with method='{args.method}'")
    print(f"Distance threshold: {args.distance_threshold} pixels")
    print()

    # Collect metrics
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Processing"):
        sample = dataset[i]

        gt_seg = sample["segmentation"].numpy()
        gt_junction = sample["junction_heatmap"].numpy()[0]
        image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Extract skeleton
        skeleton = extract_skeleton(gt_seg, method=args.method)

        # Compute metrics
        metrics = compute_metrics(
            gt_seg, gt_junction, skeleton,
            distance_threshold=args.distance_threshold,
        )
        all_metrics.append(metrics)

        # Visualize first N samples
        if i < args.visualize:
            filename = sample["meta"]["filename"]
            save_path = output_dir / f"{filename}_skeleton.png"
            visualize_comparison(image, gt_seg, skeleton, gt_junction, metrics, save_path)
            print(f"  Saved: {save_path}")

    # Aggregate metrics
    print("\n" + "=" * 60)
    print("SKELETONIZATION QUALITY METRICS")
    print("=" * 60)

    metric_names = ["edge_recall", "edge_precision", "edge_f1",
                    "junction_recall", "junction_precision", "connectivity_ratio"]

    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{name:25s}: {mean:.4f} ± {std:.4f}")

    # Additional stats
    avg_gt_junctions = np.mean([m["num_gt_junctions"] for m in all_metrics])
    avg_skel_junctions = np.mean([m["num_skel_junctions"] for m in all_metrics])
    avg_compression = np.mean([m["skeleton_pixels"] / max(m["gt_crease_pixels"], 1) for m in all_metrics])

    print()
    print(f"Avg GT junctions per sample:   {avg_gt_junctions:.1f}")
    print(f"Avg Skel junctions per sample: {avg_skel_junctions:.1f}")
    print(f"Avg skeleton compression:      {avg_compression:.3f} (skeleton/GT pixels)")

    print()
    print(f"Visualizations saved to: {output_dir}")

    # Save metrics to JSON
    import json
    metrics_summary = {
        "method": args.method,
        "distance_threshold": args.distance_threshold,
        "num_samples": num_samples,
        "metrics": {
            name: {
                "mean": float(np.mean([m[name] for m in all_metrics])),
                "std": float(np.std([m[name] for m in all_metrics])),
            }
            for name in metric_names
        }
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
