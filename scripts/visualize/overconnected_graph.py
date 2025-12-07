#!/usr/bin/env python3
"""
Visualize the overconnected graph extracted from pixel head predictions.

Usage:
    python scripts/visualize/overconnected_graph.py --checkpoint checkpoints/pixel_head_attempt_2/checkpoint_epoch_10.pt --image-dir /tmp/maypel_benchmark --num-samples 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models import CreasePatternDetector
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


def load_and_preprocess_image(
    image_path: Path, image_size: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Load image and preprocess for model input."""
    img = Image.open(image_path).convert("RGB")
    original_np = np.array(img)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    tensor = transform(img)
    return tensor, original_np


def visualize_graph(
    image: np.ndarray,
    pred_seg: np.ndarray,
    graph,
    filename: str,
    save_path: Path = None,
):
    """Visualize the extracted overconnected graph."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    h, w = pred_seg.shape
    img_resized = np.array(Image.fromarray(image).resize((w, h)))

    # Color map for assignments
    assignment_colors = {
        0: 'red',      # M
        1: 'blue',     # V
        2: 'green',    # B (using green for visibility)
        3: 'gray',     # U
    }

    # 1. Segmentation
    from scripts.visualize.pixel_head_predictions import seg_to_color
    axes[0].imshow(seg_to_color(pred_seg))
    axes[0].set_title("Predicted Segmentation")
    axes[0].axis("off")

    # 2. Graph on segmentation
    axes[1].imshow(seg_to_color(pred_seg), alpha=0.5)

    # Draw edges
    for i, (v1_idx, v2_idx) in enumerate(graph.edges):
        v1 = graph.vertices[v1_idx]
        v2 = graph.vertices[v2_idx]
        color = assignment_colors.get(graph.assignments[i], 'gray')
        axes[1].plot([v1[0], v2[0]], [v1[1], v2[1]],
                    color=color, linewidth=1.5, alpha=0.8)

    # Draw vertices
    for i, v in enumerate(graph.vertices):
        marker_color = 'lime' if graph.is_boundary[i] else 'yellow'
        axes[1].plot(v[0], v[1], 'o', color=marker_color, markersize=4,
                    markeredgecolor='black', markeredgewidth=0.5)

    axes[1].set_title(f"Overconnected Graph: {len(graph.vertices)} vertices, {len(graph.edges)} edges")
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)
    axes[1].axis("off")

    # 3. Graph on original image
    axes[2].imshow(img_resized)

    # Draw edges
    for i, (v1_idx, v2_idx) in enumerate(graph.edges):
        v1 = graph.vertices[v1_idx]
        v2 = graph.vertices[v2_idx]
        color = assignment_colors.get(graph.assignments[i], 'gray')
        axes[2].plot([v1[0], v2[0]], [v1[1], v2[1]],
                    color=color, linewidth=2, alpha=0.9)

    # Draw vertices
    for i, v in enumerate(graph.vertices):
        marker_color = 'lime' if graph.is_boundary[i] else 'yellow'
        axes[2].plot(v[0], v[1], 'o', color=marker_color, markersize=5,
                    markeredgecolor='black', markeredgewidth=0.5)

    axes[2].set_title("Graph on Image")
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)
    axes[2].axis("off")

    plt.suptitle(f"Overconnected Graph: {filename}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize overconnected graph from pixel predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images",
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
        default="visualizations/overconnected",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Image size (defaults to checkpoint config)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Get image size from checkpoint config or use argument
    image_size = args.image_size or config.get("image_size", 512)
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

    # Create graph extractor
    extractor_config = GraphExtractorConfig()
    extractor = GraphExtractor(extractor_config)
    print(f"Graph extractor config: junction_threshold={extractor_config.junction_threshold}")

    # Find images
    image_dir = Path(args.image_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions and not f.name.startswith(".")]
    )

    print(f"Found {len(image_files)} images")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize samples
    num_samples = min(args.num_samples, len(image_files))

    with torch.no_grad():
        for i in range(num_samples):
            image_path = image_files[i]
            print(f"Processing {i+1}/{num_samples}: {image_path.name}")

            # Load and preprocess image
            image_tensor, original_np = load_and_preprocess_image(image_path, image_size)
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Run inference
            outputs = model(image_batch)

            # Get predictions
            pred_seg = outputs["segmentation"].argmax(dim=1).cpu().numpy()[0]
            pred_junction = torch.sigmoid(outputs["junction"]).cpu().numpy()[0, 0]
            pred_orientation = outputs["orientation"].cpu().numpy()[0]  # (2, H, W)
            # Transpose to (H, W, 2) for edge tracing
            pred_orientation = np.transpose(pred_orientation, (1, 2, 0))

            # Extract graph
            graph = extractor.extract(
                segmentation=pred_seg,
                junction_heatmap=pred_junction,
                orientation=pred_orientation,
            )

            print(f"  Extracted: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

            # Visualize
            save_path = output_dir / f"{image_path.stem}_graph.png"
            visualize_graph(
                image=original_np,
                pred_seg=pred_seg,
                graph=graph,
                filename=image_path.name,
                save_path=save_path,
            )

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
