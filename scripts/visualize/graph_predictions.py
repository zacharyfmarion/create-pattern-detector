#!/usr/bin/env python3
"""
Quick visualization of Graph Head predictions.
Generates a few sample images showing GT vs predictions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import CreasePatternDataset
from src.data.transforms import get_val_transform
from src.models.cp_detector import CreasePatternDetector
from src.models.graph.graph_head import GraphHead
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig
from src.data.graph_labels import generate_graph_labels

# Colors for edge types (FOLD assignments: 0=M, 1=V, 2=B, 3=U)
COLORS = {
    0: "#E74C3C",  # Mountain - Red
    1: "#3498DB",  # Valley - Blue
    2: "#2ECC71",  # Border - Green
    3: "#95A5A6",  # Unassigned - Gray
}
NAMES = ["Mountain", "Valley", "Border", "Unassigned"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel-checkpoint", type=str, required=True)
    parser.add_argument("--graph-checkpoint", type=str, default=None, help="Optional - skip to validate extraction only")
    parser.add_argument("--fold-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="visualizations/graph_predictions")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Edge existence threshold")
    args = parser.parse_args()

    device = torch.device("cpu")  # Use CPU for quick local eval
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")

    # Load pixel model
    pixel_ckpt = torch.load(args.pixel_checkpoint, map_location=device, weights_only=False)
    pixel_model = CreasePatternDetector(backbone="hrnet_w32", num_seg_classes=5)
    pixel_model.load_state_dict(pixel_ckpt["model_state_dict"])
    pixel_model.eval()

    # Load graph model (optional - for predictions)
    graph_head = None
    num_classes = 4  # Default
    if args.graph_checkpoint:
        graph_ckpt = torch.load(args.graph_checkpoint, map_location=device, weights_only=False)
        assignment_weight = graph_ckpt["graph_head_state_dict"]["assignment_head.head.3.weight"]
        num_classes = assignment_weight.shape[0]
        node_weight = graph_ckpt["graph_head_state_dict"]["node_extractor.mlp.0.weight"]
        backbone_channels = node_weight.shape[1] - 2 - num_classes - 2

        graph_head = GraphHead(
            backbone_channels=backbone_channels,
            node_dim=128,
            edge_dim=128,
            num_gnn_layers=4,
            num_classes=num_classes,
        )
        graph_head.load_state_dict(graph_ckpt["graph_head_state_dict"])
        graph_head.eval()

        print(f"Graph head: {num_classes} classes, backbone_channels={backbone_channels}")
        print(f"Trained for {graph_ckpt.get('epoch', '?')+1} epochs")
    else:
        print("No graph checkpoint - validating extraction pipeline only")

    # Load dataset
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=512,
        transform=get_val_transform(image_size=512),
        split="train",
    )

    graph_extractor = GraphExtractor(GraphExtractorConfig())

    print(f"\nGenerating {args.num_samples} visualizations...")

    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0)

        with torch.no_grad():
            pixel_out = pixel_model(image, return_features=True)

        seg_pred = pixel_out["segmentation"].argmax(dim=1)[0].cpu().numpy()
        junction_hm = pixel_out["junction"][0, 0].cpu().numpy()

        # Extract candidate graph
        cand = graph_extractor.extract(seg_pred, junction_hm)
        if cand is None or len(cand.vertices) < 2 or len(cand.edges) == 0:
            print(f"  Sample {idx}: No valid candidate graph")
            continue

        # Forward through graph head (if available)
        pred_exist = None
        pred_assign = None
        if graph_head is not None:
            vertices = torch.from_numpy(cand.vertices).float()
            edge_index = torch.from_numpy(cand.edges.T).long()
            seg_probs = torch.softmax(pixel_out["segmentation"], dim=1)
            with torch.no_grad():
                out = graph_head(
                    vertices=vertices,
                    edge_index=edge_index,
                    backbone_features=pixel_out["features"],
                    seg_probs=seg_probs,
                    image_size=512,
                )
            pred_exist = torch.sigmoid(out["edge_existence"]).numpy()
            pred_assign = out["edge_assignment"].numpy()

        # Get GT
        gt_graph = sample.get("graph")

        # Denormalize image
        img = image[0].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Create visualization - use white background for clarity
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img_shape = img.shape[:2]

        # 1. Ground Truth - clean white background
        ax = axes[0]
        ax.set_facecolor("white")
        ax.set_xlim(0, img_shape[1])
        ax.set_ylim(img_shape[0], 0)  # Flip y-axis

        gt_counts = {i: 0 for i in range(4)}
        if gt_graph is not None:
            gt_v = gt_graph["vertices"].numpy()
            gt_e = gt_graph["edges"].numpy()
            gt_a = gt_graph["assignments"].numpy()

            # Count GT edge types
            for a in gt_a:
                if 0 <= a < 4:
                    gt_counts[a] += 1
            count_str = f"M:{gt_counts[0]} V:{gt_counts[1]} B:{gt_counts[2]} U:{gt_counts[3]}"
            ax.set_title(f"Ground Truth ({len(gt_e)} edges)\n{count_str}", fontsize=9)

            for i, (s, d) in enumerate(gt_e):
                v1, v2 = gt_v[s], gt_v[d]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=COLORS[int(gt_a[i])], lw=2.5)
            ax.scatter(gt_v[:, 0], gt_v[:, 1], c="white", s=30, edgecolors="black", zorder=5, linewidths=1.5)
        else:
            ax.set_title("Ground Truth (N/A)")
        ax.set_aspect("equal")
        ax.axis("off")

        # 2. Candidate graph - show with extracted assignments (color-coded)
        ax = axes[1]
        ax.set_facecolor("white")
        ax.set_xlim(0, img_shape[1])
        ax.set_ylim(img_shape[0], 0)

        # Count candidate edge types
        cand_counts = {i: 0 for i in range(4)}
        for a in cand.assignments:
            if 0 <= a < 4:
                cand_counts[a] += 1
        count_str = f"M:{cand_counts[0]} V:{cand_counts[1]} B:{cand_counts[2]} U:{cand_counts[3]}"
        ax.set_title(f"Candidates ({len(cand.edges)} edges)\n{count_str}", fontsize=9)

        for i, (s, d) in enumerate(cand.edges):
            v1, v2 = cand.vertices[s], cand.vertices[d]
            edge_color = COLORS.get(int(cand.assignments[i]), "#888888")
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=edge_color, lw=1.5, alpha=0.7)

        # Show boundary vertices differently
        boundary_mask = cand.is_boundary if hasattr(cand, 'is_boundary') else np.zeros(len(cand.vertices), dtype=bool)
        interior = ~boundary_mask
        if interior.any():
            ax.scatter(cand.vertices[interior, 0], cand.vertices[interior, 1],
                      c="white", s=25, edgecolors="#666666", zorder=5, linewidths=1)
        if boundary_mask.any():
            ax.scatter(cand.vertices[boundary_mask, 0], cand.vertices[boundary_mask, 1],
                      c="yellow", s=35, edgecolors="orange", zorder=6, linewidths=1.5, marker='s')
        ax.set_aspect("equal")
        ax.axis("off")

        # 3. Predictions - white background (or N/A if no graph head)
        ax = axes[2]
        ax.set_facecolor("white")
        ax.set_xlim(0, img_shape[1])
        ax.set_ylim(img_shape[0], 0)

        num_pred = 0
        if pred_exist is not None:
            pred_mask = pred_exist > args.threshold
            pred_cls = pred_assign.argmax(axis=1)
            num_pred = pred_mask.sum()
            ax.set_title(f"Predicted ({num_pred} edges, thresh={args.threshold})")

            for i, (s, d) in enumerate(cand.edges):
                if pred_mask[i]:
                    v1, v2 = cand.vertices[s], cand.vertices[d]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=COLORS.get(pred_cls[i], "#888888"), lw=2.5)

            # Show used vertices
            used = np.zeros(len(cand.vertices), dtype=bool)
            for i, (s, d) in enumerate(cand.edges):
                if pred_mask[i]:
                    used[s] = used[d] = True
            if used.any():
                ax.scatter(cand.vertices[used, 0], cand.vertices[used, 1],
                          c="white", s=30, edgecolors="black", zorder=5, linewidths=1.5)
        else:
            ax.set_title("Predictions (N/A - no graph head)")
            ax.text(0.5, 0.5, "No graph checkpoint\nprovided", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')

        ax.set_aspect("equal")
        ax.axis("off")

        # Legend (uses FOLD assignment indices 0-3)
        patches = [mpatches.Patch(color=COLORS[i], label=NAMES[i]) for i in range(4)]
        fig.legend(handles=patches, loc="lower center", ncol=4)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = output_dir / f"sample_{idx}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

        # Print detailed diagnostic stats
        print(f"    GT edges: {len(gt_e) if gt_graph else 0} - M:{gt_counts[0]} V:{gt_counts[1]} B:{gt_counts[2]} U:{gt_counts[3]}")
        print(f"    Candidate edges: {len(cand.edges)} - M:{cand_counts[0]} V:{cand_counts[1]} B:{cand_counts[2]} U:{cand_counts[3]}")
        print(f"    Boundary vertices: {boundary_mask.sum()}/{len(cand.vertices)}")
        if pred_exist is not None:
            print(f"    Predicted edges: {num_pred}")

    print(f"\nDone! Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
