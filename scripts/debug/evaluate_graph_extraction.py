#!/usr/bin/env python3
"""
Evaluate graph extraction quality against original FOLD data.

Computes metrics comparing extracted graph vs GT FOLD graph:
- Vertex recall/precision: matched vertices within distance threshold
- Edge recall/precision: matched edges (both endpoints match)
- Assignment accuracy: correct M/V/B/U classification
- Topology metrics: degree distribution, connectivity

Usage:
    python scripts/debug/evaluate_graph_extraction.py --fold-dir data/output/scraped/raw --num-samples 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.data.dataset import CreasePatternDataset
from src.data.transforms import get_val_transform
from src.data.fold_parser import FOLDParser, transform_coords
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


def match_vertices(
    extracted_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    threshold: float = 5.0,
) -> dict:
    """
    Match extracted vertices to GT vertices.

    Returns:
        Dictionary with recall, precision, and matching info
    """
    if len(extracted_vertices) == 0 or len(gt_vertices) == 0:
        return {
            "recall": 0.0 if len(gt_vertices) > 0 else 1.0,
            "precision": 0.0 if len(extracted_vertices) > 0 else 1.0,
            "num_extracted": len(extracted_vertices),
            "num_gt": len(gt_vertices),
            "num_matched": 0,
            "gt_to_extracted": {},
            "extracted_to_gt": {},
        }

    # Compute pairwise distances
    distances = cdist(gt_vertices, extracted_vertices)

    # For each GT vertex, find closest extracted vertex
    gt_to_extracted = {}
    for gt_idx in range(len(gt_vertices)):
        min_dist = distances[gt_idx].min()
        if min_dist <= threshold:
            ext_idx = distances[gt_idx].argmin()
            gt_to_extracted[gt_idx] = (ext_idx, min_dist)

    # For each extracted vertex, find closest GT vertex
    extracted_to_gt = {}
    for ext_idx in range(len(extracted_vertices)):
        min_dist = distances[:, ext_idx].min()
        if min_dist <= threshold:
            gt_idx = distances[:, ext_idx].argmin()
            extracted_to_gt[ext_idx] = (gt_idx, min_dist)

    recall = len(gt_to_extracted) / len(gt_vertices)
    precision = len(extracted_to_gt) / len(extracted_vertices)

    return {
        "recall": recall,
        "precision": precision,
        "num_extracted": len(extracted_vertices),
        "num_gt": len(gt_vertices),
        "num_matched_gt": len(gt_to_extracted),
        "num_matched_ext": len(extracted_to_gt),
        "gt_to_extracted": gt_to_extracted,
        "extracted_to_gt": extracted_to_gt,
    }


def match_edges(
    extracted_graph,
    gt_vertices: np.ndarray,
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    vertex_matching: dict,
    threshold: float = 5.0,
) -> dict:
    """
    Match extracted edges to GT edges based on vertex matching.

    An edge matches if both endpoints match to the correct GT vertices.
    """
    if len(extracted_graph.edges) == 0 or len(gt_edges) == 0:
        return {
            "recall": 0.0 if len(gt_edges) > 0 else 1.0,
            "precision": 0.0 if len(extracted_graph.edges) > 0 else 1.0,
            "num_extracted": len(extracted_graph.edges),
            "num_gt": len(gt_edges),
            "num_matched": 0,
            "assignment_accuracy": 0.0,
        }

    gt_to_ext = vertex_matching["gt_to_extracted"]
    ext_to_gt = vertex_matching["extracted_to_gt"]

    # Build GT edge set (as frozensets for undirected matching)
    gt_edge_set = {}
    for i, (v1, v2) in enumerate(gt_edges):
        key = frozenset([int(v1), int(v2)])
        gt_edge_set[key] = i  # Store index for assignment lookup

    # Try to match each extracted edge
    matched_edges = []
    assignment_correct = 0

    for ext_edge_idx, (ext_v1, ext_v2) in enumerate(extracted_graph.edges):
        # Check if both extracted vertices map to GT vertices
        if ext_v1 not in ext_to_gt or ext_v2 not in ext_to_gt:
            continue

        gt_v1 = ext_to_gt[ext_v1][0]
        gt_v2 = ext_to_gt[ext_v2][0]

        key = frozenset([gt_v1, gt_v2])
        if key in gt_edge_set:
            gt_edge_idx = gt_edge_set[key]
            matched_edges.append((ext_edge_idx, gt_edge_idx))

            # Check assignment
            ext_assignment = extracted_graph.assignments[ext_edge_idx]
            gt_assignment = gt_assignments[gt_edge_idx]
            if ext_assignment == gt_assignment:
                assignment_correct += 1

    num_matched = len(matched_edges)
    recall = num_matched / len(gt_edges)
    precision = num_matched / len(extracted_graph.edges)
    assignment_accuracy = assignment_correct / num_matched if num_matched > 0 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "num_extracted": len(extracted_graph.edges),
        "num_gt": len(gt_edges),
        "num_matched": num_matched,
        "assignment_accuracy": assignment_accuracy,
        "assignment_correct": assignment_correct,
    }


def compute_topology_metrics(extracted_graph, gt_edges: np.ndarray, gt_vertices: np.ndarray) -> dict:
    """Compute topology-related metrics."""
    # Compute degree for each vertex
    ext_degrees = np.zeros(len(extracted_graph.vertices), dtype=int)
    for v1, v2 in extracted_graph.edges:
        ext_degrees[v1] += 1
        ext_degrees[v2] += 1

    gt_degrees = np.zeros(len(gt_vertices), dtype=int)
    for v1, v2 in gt_edges:
        gt_degrees[v1] += 1
        gt_degrees[v2] += 1

    return {
        "ext_avg_degree": float(ext_degrees.mean()) if len(ext_degrees) > 0 else 0,
        "gt_avg_degree": float(gt_degrees.mean()) if len(gt_degrees) > 0 else 0,
        "ext_max_degree": int(ext_degrees.max()) if len(ext_degrees) > 0 else 0,
        "gt_max_degree": int(gt_degrees.max()) if len(gt_degrees) > 0 else 0,
        "vertex_ratio": len(extracted_graph.vertices) / len(gt_vertices) if len(gt_vertices) > 0 else 0,
        "edge_ratio": len(extracted_graph.edges) / len(gt_edges) if len(gt_edges) > 0 else 0,
    }


def evaluate_sample(
    sample: dict,
    extractor: GraphExtractor,
    image_size: int,
    padding: int,
    vertex_threshold: float = 5.0,
) -> dict:
    """Evaluate graph extraction on a single sample."""
    # Get GT data
    gt_seg = sample["segmentation"].numpy()
    gt_junction = sample["junction_heatmap"].numpy()[0]
    gt_orientation = sample["orientation"].permute(1, 2, 0).numpy()

    # Extract graph from GT pixel data
    extracted = extractor.extract(
        segmentation=gt_seg,
        junction_heatmap=gt_junction,
        orientation=gt_orientation,
    )

    # Load original FOLD data for comparison
    fold_path = sample["meta"]["fold_path"]
    parser = FOLDParser()
    cp = parser.parse(fold_path)

    # Transform GT vertices to pixel coordinates
    gt_vertices_pixel, _ = transform_coords(
        cp.vertices,
        image_size=image_size,
        padding=padding,
    )

    # Match vertices
    vertex_metrics = match_vertices(
        extracted.vertices,
        gt_vertices_pixel,
        threshold=vertex_threshold,
    )

    # Match edges
    edge_metrics = match_edges(
        extracted,
        gt_vertices_pixel,
        cp.edges,
        cp.assignments,
        vertex_metrics,
        threshold=vertex_threshold,
    )

    # Topology metrics
    topology_metrics = compute_topology_metrics(
        extracted,
        cp.edges,
        gt_vertices_pixel,
    )

    return {
        "vertex": vertex_metrics,
        "edge": edge_metrics,
        "topology": topology_metrics,
        "filename": sample["meta"]["filename"],
    }


def visualize_comparison(
    sample: dict,
    extracted_graph,
    gt_vertices: np.ndarray,
    gt_edges: np.ndarray,
    gt_assignments: np.ndarray,
    metrics: dict,
    save_path: Path,
):
    """Visualize GT vs extracted graph side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    h, w = sample["segmentation"].shape

    assignment_colors = {0: 'red', 1: 'blue', 2: 'black', 3: 'gray'}

    # 1. Input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2. GT Graph
    axes[1].set_facecolor('white')
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)

    for i, (v1_idx, v2_idx) in enumerate(gt_edges):
        v1 = gt_vertices[v1_idx]
        v2 = gt_vertices[v2_idx]
        color = assignment_colors.get(int(gt_assignments[i]), 'gray')
        axes[1].plot([v1[0], v2[0]], [v1[1], v2[1]], color=color, linewidth=1.5)

    for v in gt_vertices:
        axes[1].plot(v[0], v[1], 'o', color='lime', markersize=3,
                    markeredgecolor='black', markeredgewidth=0.3)

    axes[1].set_title(f"GT Graph: {len(gt_vertices)} vertices, {len(gt_edges)} edges")
    axes[1].set_aspect('equal')
    axes[1].axis("off")

    # 3. Extracted Graph
    axes[2].set_facecolor('white')
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)

    for i, (v1_idx, v2_idx) in enumerate(extracted_graph.edges):
        v1 = extracted_graph.vertices[v1_idx]
        v2 = extracted_graph.vertices[v2_idx]
        color = assignment_colors.get(int(extracted_graph.assignments[i]), 'gray')
        axes[2].plot([v1[0], v2[0]], [v1[1], v2[1]], color=color, linewidth=1.5)

    for v in extracted_graph.vertices:
        axes[2].plot(v[0], v[1], 'o', color='orange', markersize=3,
                    markeredgecolor='black', markeredgewidth=0.3)

    v_rec = metrics["vertex"]["recall"]
    v_prec = metrics["vertex"]["precision"]
    e_rec = metrics["edge"]["recall"]
    e_prec = metrics["edge"]["precision"]

    axes[2].set_title(
        f"Extracted: {extracted_graph.num_vertices()} V, {extracted_graph.num_edges()} E\n"
        f"V-Rec:{v_rec:.2f} V-Prec:{v_prec:.2f} E-Rec:{e_rec:.2f} E-Prec:{e_prec:.2f}"
    )
    axes[2].set_aspect('equal')
    axes[2].axis("off")

    plt.suptitle(f"Graph Extraction Evaluation: {metrics['filename']}", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate graph extraction quality")
    parser.add_argument("--fold-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="visualizations/graph_extraction_eval")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--vertex-threshold", type=float, default=8.0,
                       help="Distance threshold for vertex matching (pixels)")
    parser.add_argument("--visualize", type=int, default=5)
    parser.add_argument("--split", type=str, default="val")

    # Extractor config overrides
    parser.add_argument("--junction-threshold", type=float, default=None)
    parser.add_argument("--junction-min-distance", type=float, default=None)
    parser.add_argument("--min-edge-length", type=float, default=None)
    parser.add_argument("--junction-radius", type=int, default=None)
    parser.add_argument("--bridge-gap-pixels", type=int, default=None)
    parser.add_argument("--junction-merge-distance", type=float, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset params
    padding = int(50 * args.image_size / 1024)
    line_width = max(1, int(2 * args.image_size / 1024))

    # Load dataset
    print(f"Loading dataset from {args.fold_dir}...")
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        padding=padding,
        line_width=line_width,
        transform=get_val_transform(args.image_size),
        split=args.split,
    )

    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples")
    print(f"Vertex matching threshold: {args.vertex_threshold} pixels")

    # Create graph extractor with optional overrides
    config = GraphExtractorConfig()
    if args.junction_threshold is not None:
        config.junction_threshold = args.junction_threshold
    if args.junction_min_distance is not None:
        config.junction_min_distance = args.junction_min_distance
    if args.min_edge_length is not None:
        config.min_edge_length = args.min_edge_length
    if args.junction_radius is not None:
        config.junction_radius = args.junction_radius
    if args.bridge_gap_pixels is not None:
        config.bridge_gap_pixels = args.bridge_gap_pixels
    if args.junction_merge_distance is not None:
        config.junction_merge_distance = args.junction_merge_distance

    print(f"Extractor config:")
    print(f"  junction_threshold={config.junction_threshold}")
    print(f"  junction_min_distance={config.junction_min_distance}")
    print(f"  junction_radius={config.junction_radius}")
    print(f"  min_edge_length={config.min_edge_length}")
    print(f"  bridge_gap_pixels={config.bridge_gap_pixels}")
    print(f"  junction_merge_distance={config.junction_merge_distance}")

    extractor = GraphExtractor(config)
    fold_parser = FOLDParser()

    # Collect metrics
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[i]

        metrics = evaluate_sample(
            sample,
            extractor,
            args.image_size,
            padding,
            args.vertex_threshold,
        )
        all_metrics.append(metrics)

        # Visualize first N samples
        if i < args.visualize:
            # Reload GT for visualization
            fold_path = sample["meta"]["fold_path"]
            cp = fold_parser.parse(fold_path)
            gt_vertices_pixel, _ = transform_coords(
                cp.vertices, image_size=args.image_size, padding=padding
            )

            # Re-extract for visualization
            gt_seg = sample["segmentation"].numpy()
            gt_junction = sample["junction_heatmap"].numpy()[0]
            gt_orientation = sample["orientation"].permute(1, 2, 0).numpy()
            extracted = extractor.extract(gt_seg, gt_junction, gt_orientation)

            save_path = output_dir / f"{metrics['filename']}_eval.png"
            visualize_comparison(
                sample, extracted, gt_vertices_pixel, cp.edges, cp.assignments,
                metrics, save_path
            )
            print(f"  Saved: {save_path}")

    # Aggregate metrics
    print("\n" + "=" * 70)
    print("GRAPH EXTRACTION QUALITY METRICS")
    print("=" * 70)

    vertex_recalls = [m["vertex"]["recall"] for m in all_metrics]
    vertex_precisions = [m["vertex"]["precision"] for m in all_metrics]
    edge_recalls = [m["edge"]["recall"] for m in all_metrics]
    edge_precisions = [m["edge"]["precision"] for m in all_metrics]
    assignment_accs = [m["edge"]["assignment_accuracy"] for m in all_metrics]

    print(f"\nVERTEX METRICS:")
    print(f"  Recall:    {np.mean(vertex_recalls):.4f} ± {np.std(vertex_recalls):.4f}")
    print(f"  Precision: {np.mean(vertex_precisions):.4f} ± {np.std(vertex_precisions):.4f}")

    print(f"\nEDGE METRICS:")
    print(f"  Recall:    {np.mean(edge_recalls):.4f} ± {np.std(edge_recalls):.4f}")
    print(f"  Precision: {np.mean(edge_precisions):.4f} ± {np.std(edge_precisions):.4f}")

    print(f"\nASSIGNMENT ACCURACY:")
    print(f"  Accuracy:  {np.mean(assignment_accs):.4f} ± {np.std(assignment_accs):.4f}")

    print(f"\nTOPOLOGY:")
    vertex_ratios = [m["topology"]["vertex_ratio"] for m in all_metrics]
    edge_ratios = [m["topology"]["edge_ratio"] for m in all_metrics]
    print(f"  Vertex ratio (ext/gt): {np.mean(vertex_ratios):.2f} ± {np.std(vertex_ratios):.2f}")
    print(f"  Edge ratio (ext/gt):   {np.mean(edge_ratios):.2f} ± {np.std(edge_ratios):.2f}")

    # Save metrics
    summary = {
        "config": {
            "junction_threshold": config.junction_threshold,
            "junction_min_distance": config.junction_min_distance,
            "min_edge_length": config.min_edge_length,
            "vertex_threshold": args.vertex_threshold,
        },
        "num_samples": num_samples,
        "vertex_recall": {"mean": float(np.mean(vertex_recalls)), "std": float(np.std(vertex_recalls))},
        "vertex_precision": {"mean": float(np.mean(vertex_precisions)), "std": float(np.std(vertex_precisions))},
        "edge_recall": {"mean": float(np.mean(edge_recalls)), "std": float(np.std(edge_recalls))},
        "edge_precision": {"mean": float(np.mean(edge_precisions)), "std": float(np.std(edge_precisions))},
        "assignment_accuracy": {"mean": float(np.mean(assignment_accs)), "std": float(np.std(assignment_accs))},
        "vertex_ratio": {"mean": float(np.mean(vertex_ratios)), "std": float(np.std(vertex_ratios))},
        "edge_ratio": {"mean": float(np.mean(edge_ratios)), "std": float(np.std(edge_ratios))},
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
