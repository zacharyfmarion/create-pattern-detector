#!/usr/bin/env python3
"""
Visualize graph extraction from ground truth segmentation/orientation.

Tests the graph extraction pipeline using GT data to establish baseline quality.

Usage:
    python scripts/visualize/gt_graph_extraction.py --fold-dir data/output/scraped/raw --num-samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import CreasePatternDataset
from src.data.transforms import get_val_transform
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


# Color map for segmentation classes
SEG_COLORS = np.array([
    [255, 255, 255],  # BG - white
    [255, 0, 0],      # M - red
    [0, 0, 255],      # V - blue
    [0, 0, 0],        # B - black
    [128, 128, 128],  # U - gray
], dtype=np.uint8)

# Colors for graph assignments
ASSIGNMENT_COLORS = {
    0: 'red',      # M
    1: 'blue',     # V
    2: 'black',    # B
    3: 'gray',     # U
}


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB image."""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(5):
        rgb[seg == c] = SEG_COLORS[c]
    return rgb


def visualize_extraction(
    image: np.ndarray,
    gt_seg: np.ndarray,
    graph,
    filename: str,
    save_path: Path = None,
):
    """Visualize GT vs extracted graph."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    h, w = gt_seg.shape

    # 1. Input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2. GT Segmentation
    axes[1].imshow(seg_to_rgb(gt_seg))
    axes[1].set_title("GT Segmentation")
    axes[1].axis("off")

    # 3. Extracted Graph (clean white background)
    axes[2].set_facecolor('white')
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)

    # Draw edges
    for i, (v1_idx, v2_idx) in enumerate(graph.edges):
        v1 = graph.vertices[v1_idx]
        v2 = graph.vertices[v2_idx]
        color = ASSIGNMENT_COLORS.get(int(graph.assignments[i]), 'gray')
        axes[2].plot([v1[0], v2[0]], [v1[1], v2[1]],
                    color=color, linewidth=1.5, alpha=0.9)

    # Draw vertices
    for i, v in enumerate(graph.vertices):
        marker_color = 'lime' if graph.is_boundary[i] else 'orange'
        axes[2].plot(v[0], v[1], 'o', color=marker_color, markersize=3,
                    markeredgecolor='black', markeredgewidth=0.3)

    axes[2].set_title(f"Extracted Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    axes[2].set_aspect('equal')
    axes[2].axis("off")

    plt.suptitle(f"Graph Extraction: {filename}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize graph extraction from GT")
    parser.add_argument("--fold-dir", type=str, required=True,
                       help="Directory containing .fold files")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="visualizations/gt_graph_extraction",
                       help="Directory to save visualizations")
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image size")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.fold_dir}...")
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        transform=get_val_transform(args.image_size),
        split=args.split,
    )

    num_samples = min(args.num_samples, len(dataset))
    print(f"Processing {num_samples} samples")

    # Create graph extractor
    extractor = GraphExtractor(GraphExtractorConfig())

    for i in range(num_samples):
        sample = dataset[i]
        filename = sample["meta"]["filename"]
        print(f"Processing {i+1}/{num_samples}: {filename}")

        gt_seg = sample["segmentation"].numpy()
        gt_junction = sample["junction_heatmap"].numpy()[0]
        gt_orientation = sample["orientation"].permute(1, 2, 0).numpy()  # CHW -> HWC
        image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Extract graph using GT
        graph = extractor.extract(
            segmentation=gt_seg,
            junction_heatmap=gt_junction,
            orientation=gt_orientation,
        )

        print(f"  Extracted: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

        # Visualize
        save_path = output_dir / f"{filename}_graph.png"
        visualize_extraction(
            image=image,
            gt_seg=gt_seg,
            graph=graph,
            filename=filename,
            save_path=save_path,
        )

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
