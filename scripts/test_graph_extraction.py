#!/usr/bin/env python3
"""
Test graph extraction on model predictions.

Usage:
    python scripts/test_graph_extraction.py --checkpoint checkpoints/best_model.pt --fold-dir data_small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import CreasePatternDataset
from src.data.transforms import get_val_transform
from src.models import CreasePatternDetector
from src.postprocessing import GraphExtractor, GraphExtractorConfig


# Color map for assignments
ASSIGNMENT_COLORS = {
    0: 'red',      # M
    1: 'blue',     # V
    2: 'black',    # B
    3: 'gray',     # U
}


def visualize_extraction(
    image: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    gt_junction: np.ndarray,
    pred_junction: np.ndarray,
    extracted_graph,
    filename: str,
    save_path: Path = None,
):
    """Visualize graph extraction results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Input, Pred Seg, Pred Junction
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # Segmentation colormap
    seg_colors = np.array([
        [255, 255, 255],  # BG
        [255, 0, 0],      # M
        [0, 0, 255],      # V
        [0, 0, 0],        # B
        [128, 128, 128],  # U
    ], dtype=np.uint8)

    def seg_to_color(seg):
        h, w = seg.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(5):
            rgb[seg == c] = seg_colors[c]
        return rgb

    axes[0, 1].imshow(seg_to_color(pred_seg))
    axes[0, 1].set_title("Predicted Segmentation")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_junction, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title("Predicted Junction Heatmap")
    axes[0, 2].axis("off")

    # Row 2: GT Seg, Extracted Graph, Overlay
    axes[1, 0].imshow(seg_to_color(gt_seg))
    axes[1, 0].set_title("Ground Truth Segmentation")
    axes[1, 0].axis("off")

    # Extracted graph on white background
    h, w = image.shape[:2]
    axes[1, 1].set_xlim(0, w)
    axes[1, 1].set_ylim(h, 0)  # Flip y-axis for image coordinates
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_facecolor('white')

    # Draw edges
    for i, (v1_idx, v2_idx) in enumerate(extracted_graph.edges):
        v1 = extracted_graph.vertices[v1_idx]
        v2 = extracted_graph.vertices[v2_idx]
        assignment = extracted_graph.assignments[i]
        color = ASSIGNMENT_COLORS.get(assignment, 'gray')
        axes[1, 1].plot([v1[0], v2[0]], [v1[1], v2[1]], color=color, linewidth=2)

    # Draw vertices
    for i, (x, y) in enumerate(extracted_graph.vertices):
        if extracted_graph.is_boundary[i]:
            marker = 's'  # Square for boundary
            size = 50
        else:
            marker = 'o'  # Circle for interior
            size = 80
        axes[1, 1].scatter(x, y, c='black', marker=marker, s=size, zorder=5)

    axes[1, 1].set_title(f"Extracted Graph ({extracted_graph.num_vertices()} verts, {extracted_graph.num_edges()} edges)")

    # Overlay on input
    axes[1, 2].imshow(image)
    for i, (v1_idx, v2_idx) in enumerate(extracted_graph.edges):
        v1 = extracted_graph.vertices[v1_idx]
        v2 = extracted_graph.vertices[v2_idx]
        assignment = extracted_graph.assignments[i]
        color = ASSIGNMENT_COLORS.get(assignment, 'gray')
        axes[1, 2].plot([v1[0], v2[0]], [v1[1], v2[1]], color=color, linewidth=3, alpha=0.8)

    for i, (x, y) in enumerate(extracted_graph.vertices):
        axes[1, 2].scatter(x, y, c='lime', edgecolors='black', s=100, zorder=5)

    axes[1, 2].set_title("Extracted Graph Overlay")
    axes[1, 2].axis("off")

    plt.suptitle(f"Graph Extraction: {filename}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test graph extraction")
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
        help="Number of samples to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graph_extraction_test",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size",
    )
    parser.add_argument(
        "--junction-threshold",
        type=float,
        default=0.3,
        help="Junction detection threshold",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

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
        split="val",
    )

    print(f"Dataset size: {len(dataset)}")

    # Create graph extractor
    extractor_config = GraphExtractorConfig(
        junction_threshold=args.junction_threshold,
        junction_min_distance=5,
        min_edge_length=10,
    )
    extractor = GraphExtractor(extractor_config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    num_samples = min(args.num_samples, len(dataset))

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            filename = sample["meta"]["filename"]
            print(f"\nProcessing {i+1}/{num_samples}: {filename}")

            # Get input image
            image = sample["image"].unsqueeze(0).to(device)

            # Run inference
            outputs = model(image)

            # Get predictions as numpy
            pred_seg = outputs["segmentation"].argmax(dim=1).cpu().numpy()[0]
            pred_junction = torch.sigmoid(outputs["junction"]).cpu().numpy()[0, 0]
            pred_orientation = outputs["orientation"].cpu().numpy()[0].transpose(1, 2, 0)

            # Get ground truth
            gt_seg = sample["segmentation"].numpy()
            gt_junction = sample["junction_heatmap"].numpy()[0]

            # Extract graph
            graph = extractor.extract(pred_seg, pred_junction, pred_orientation)
            print(f"  Extracted: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

            # Get image for display
            img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Visualize
            save_path = output_dir / f"{filename}_extraction.png"
            visualize_extraction(
                image=img_np,
                gt_seg=gt_seg,
                pred_seg=pred_seg,
                gt_junction=gt_junction,
                pred_junction=pred_junction,
                extracted_graph=graph,
                filename=filename,
                save_path=save_path,
            )

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
