#!/usr/bin/env python3
"""
Visualize pixel head predictions on raw images (no ground truth required).

Usage:
    python scripts/visualize/pixel_head_predictions.py --checkpoint checkpoints/checkpoint_epoch_8.pt --image-dir data/output/scraped-images --num-samples 5
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


# Color map for segmentation classes
# BG=0 (white), M=1 (red), V=2 (blue), B=3 (black), U=4 (gray)
SEG_COLORS = np.array(
    [
        [255, 255, 255],  # Background - white
        [255, 0, 0],  # M - red
        [0, 0, 255],  # V - blue
        [0, 0, 0],  # B - black
        [128, 128, 128],  # U - gray
    ],
    dtype=np.uint8,
)


def seg_to_color(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB image."""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(5):
        mask = seg == c
        rgb[mask] = SEG_COLORS[c]
    return rgb


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


def visualize_sample(
    image: np.ndarray,
    pred_seg: np.ndarray,
    pred_junction: np.ndarray,
    pred_orientation: np.ndarray,
    filename: str,
    save_path: Path = None,
):
    """Visualize predictions for a single image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Input, Segmentation, Overlay
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_to_color(pred_seg))
    axes[0, 1].set_title("Predicted Segmentation")
    axes[0, 1].axis("off")

    # Create overlay
    h, w = pred_seg.shape
    # Resize input image to match prediction size
    img_resized = np.array(Image.fromarray(image).resize((w, h)))
    overlay = img_resized.copy()
    pred_color = seg_to_color(pred_seg)
    mask = pred_seg > 0
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * pred_color[mask]).astype(np.uint8)

    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("Segmentation Overlay")
    axes[0, 2].axis("off")

    # Row 2: Junctions, Orientation, Combined
    axes[1, 0].imshow(pred_junction, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Junctions")
    axes[1, 0].axis("off")

    # Orientation visualization - show as HSV color wheel
    angle = np.arctan2(pred_orientation[1], pred_orientation[0])
    magnitude = np.sqrt(pred_orientation[0] ** 2 + pred_orientation[1] ** 2)
    # Normalize angle to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = magnitude  # Saturation based on magnitude
    hsv[..., 2] = (pred_seg > 0).astype(np.float32)  # Value only where creases exist
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb

    orient_rgb = hsv_to_rgb(hsv)

    axes[1, 1].imshow(orient_rgb)
    axes[1, 1].set_title("Orientation Field")
    axes[1, 1].axis("off")

    # Combined view: overlay with junction peaks marked
    combined = overlay.copy()
    # Find junction peaks
    junction_thresh = pred_junction > 0.5
    ys, xs = np.where(junction_thresh)

    axes[1, 2].imshow(combined)
    if len(xs) > 0:
        axes[1, 2].scatter(xs, ys, c="lime", s=20, marker="o", alpha=0.8)
    axes[1, 2].set_title(f"Overlay + Junctions ({len(xs)} detected)")
    axes[1, 2].axis("off")

    plt.suptitle(f"Pixel Head Predictions: {filename}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize pixel head predictions on images")
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
        default="visualizations/pixel_head",
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

            # Visualize
            save_path = output_dir / f"{image_path.stem}_prediction.png"
            visualize_sample(
                image=original_np,
                pred_seg=pred_seg,
                pred_junction=pred_junction,
                pred_orientation=pred_orientation,
                filename=image_path.name,
                save_path=save_path,
            )

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
