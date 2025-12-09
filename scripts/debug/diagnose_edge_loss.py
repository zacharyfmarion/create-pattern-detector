#!/usr/bin/env python3
"""
Diagnose why edges are being lost in graph extraction.

This script analyzes:
1. Are endpoints of missing edges detected as vertices?
2. Is there skeleton connectivity between the endpoints?
3. What causes edge tracing to fail?
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
import matplotlib.pyplot as plt

from src.data.dataset import CreasePatternDataset
from src.data.fold_parser import FOLDParser
from src.data.transforms import get_val_transform
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig
from src.postprocessing.skeletonize import skeletonize_segmentation


def analyze_missing_edges(
    sample,
    extractor: GraphExtractor,
    image_size: int,
    padding: int,
    vertex_threshold: float = 8.0,
):
    """Analyze why edges are missing from extraction."""
    # Get GT data
    segmentation = sample["segmentation"].numpy()
    junction_heatmap = sample["junction_heatmap"].numpy()
    orientation = sample["orientation"].permute(1, 2, 0).numpy()

    # Load original FOLD
    fold_path = sample["fold_path"]
    parser = FOLDParser()
    cp = parser.parse(fold_path)

    gt_vertices = cp.vertices
    gt_edges = cp.edges
    gt_assignments = cp.assignments

    # Scale GT to image coordinates
    scale = image_size - 2 * padding
    gt_vertices_img = gt_vertices * scale + padding

    # Extract graph
    extracted = extractor.extract(segmentation, junction_heatmap, orientation)

    # Build KDTree for vertex matching
    if len(extracted.vertices) > 0:
        extracted_tree = KDTree(extracted.vertices)

    # Match GT vertices to extracted vertices
    gt_to_extracted = {}
    for i, gt_v in enumerate(gt_vertices_img):
        if len(extracted.vertices) > 0:
            dist, idx = extracted_tree.query(gt_v, k=1)
            if dist < vertex_threshold:
                gt_to_extracted[i] = idx

    # Create set of extracted edges (as frozensets for order-independence)
    extracted_edge_set = set()
    for e in extracted.edges:
        extracted_edge_set.add(frozenset([e[0], e[1]]))

    # Analyze missing edges
    results = {
        "total_gt_edges": len(gt_edges),
        "matched_edges": 0,
        "missing_edges": 0,
        "missing_because_no_start_vertex": 0,
        "missing_because_no_end_vertex": 0,
        "missing_because_both_vertices_missing": 0,
        "missing_with_both_vertices_detected": 0,  # This is the interesting case
        "missing_edge_details": [],
    }

    # Get skeleton for connectivity analysis
    skeleton, _ = skeletonize_segmentation(segmentation, preserve_labels=True)

    for edge_idx, (v1, v2) in enumerate(gt_edges):
        # Check if this edge exists in extracted graph
        v1_matched = v1 in gt_to_extracted
        v2_matched = v2 in gt_to_extracted

        if v1_matched and v2_matched:
            ext_v1 = gt_to_extracted[v1]
            ext_v2 = gt_to_extracted[v2]
            ext_edge_key = frozenset([ext_v1, ext_v2])

            if ext_edge_key in extracted_edge_set:
                results["matched_edges"] += 1
                continue
            else:
                results["missing_with_both_vertices_detected"] += 1

                # Analyze why this edge is missing
                gt_v1_pos = gt_vertices_img[v1]
                gt_v2_pos = gt_vertices_img[v2]
                edge_length = np.linalg.norm(gt_v2_pos - gt_v1_pos)

                # Check skeleton connectivity
                has_skeleton_path = check_skeleton_path(
                    skeleton, gt_v1_pos, gt_v2_pos, max_dist=10
                )

                results["missing_edge_details"].append({
                    "gt_edge_idx": edge_idx,
                    "v1_pos": gt_v1_pos.tolist(),
                    "v2_pos": gt_v2_pos.tolist(),
                    "edge_length": float(edge_length),
                    "has_skeleton_path": has_skeleton_path,
                    "assignment": gt_assignments[edge_idx] if edge_idx < len(gt_assignments) else "U",
                })
        else:
            results["missing_edges"] += 1
            if not v1_matched and not v2_matched:
                results["missing_because_both_vertices_missing"] += 1
            elif not v1_matched:
                results["missing_because_no_start_vertex"] += 1
            else:
                results["missing_because_no_end_vertex"] += 1

    return results


def check_skeleton_path(skeleton, v1, v2, max_dist=10):
    """Check if there's skeleton connectivity between two points."""
    h, w = skeleton.shape

    # Check if skeleton exists near both endpoints
    x1, y1 = int(round(v1[0])), int(round(v1[1]))
    x2, y2 = int(round(v2[0])), int(round(v2[1]))

    def has_skeleton_nearby(x, y, radius):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    return True
        return False

    v1_has_skeleton = has_skeleton_nearby(x1, y1, max_dist)
    v2_has_skeleton = has_skeleton_nearby(x2, y2, max_dist)

    return v1_has_skeleton and v2_has_skeleton


def main():
    parser = argparse.ArgumentParser(description="Diagnose edge loss in graph extraction")
    parser.add_argument(
        "--fold-dir",
        type=str,
        default="data/output/scraped/raw",
        help="Directory containing FOLD files and images",
    )
    parser.add_argument(
        "--image-size", type=int, default=512, help="Image size"
    )
    parser.add_argument(
        "--padding", type=int, default=25, help="Padding around content"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10, help="Maximum samples to analyze"
    )
    parser.add_argument(
        "--vertex-threshold", type=float, default=8.0, help="Vertex matching threshold"
    )
    args = parser.parse_args()

    # Create dataset
    dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        padding=args.padding,
        transform=get_val_transform(args.image_size),
    )

    print(f"Loaded {len(dataset)} samples from {args.fold_dir}")

    # Create extractor
    config = GraphExtractorConfig()
    extractor = GraphExtractor(config)

    print(f"\nExtractor config:")
    print(f"  junction_threshold: {config.junction_threshold}")
    print(f"  junction_min_distance: {config.junction_min_distance}")
    print(f"  junction_radius: {config.junction_radius}")
    print(f"  min_edge_length: {config.min_edge_length}")
    print(f"  bridge_gap_pixels: {config.bridge_gap_pixels}")

    # Aggregate statistics
    total_stats = defaultdict(int)
    all_missing_details = []

    n_samples = min(args.max_samples, len(dataset))

    for i in range(n_samples):
        sample = dataset[i]
        fold_path = sample["fold_path"]
        print(f"\n[{i+1}/{n_samples}] Analyzing {Path(fold_path).stem}...")

        try:
            results = analyze_missing_edges(
                sample, extractor, args.image_size, args.padding, args.vertex_threshold
            )

            for key in ["total_gt_edges", "matched_edges", "missing_edges",
                       "missing_because_no_start_vertex", "missing_because_no_end_vertex",
                       "missing_because_both_vertices_missing", "missing_with_both_vertices_detected"]:
                total_stats[key] += results[key]

            all_missing_details.extend(results["missing_edge_details"])

            print(f"  GT edges: {results['total_gt_edges']}")
            print(f"  Matched: {results['matched_edges']} ({100*results['matched_edges']/max(1,results['total_gt_edges']):.1f}%)")
            print(f"  Missing (no vertex): {results['missing_edges']}")
            print(f"  Missing (both vertices ok): {results['missing_with_both_vertices_detected']}")

        except Exception as e:
            print(f"  Error: {e}")

    # Print summary
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)

    total_gt = total_stats["total_gt_edges"]
    print(f"\nTotal GT edges: {total_gt}")
    print(f"Matched edges: {total_stats['matched_edges']} ({100*total_stats['matched_edges']/max(1,total_gt):.1f}%)")
    print(f"\nMissing edges breakdown:")
    print(f"  - Missing start vertex: {total_stats['missing_because_no_start_vertex']}")
    print(f"  - Missing end vertex: {total_stats['missing_because_no_end_vertex']}")
    print(f"  - Missing both vertices: {total_stats['missing_because_both_vertices_missing']}")
    print(f"  - BOTH VERTICES OK but edge missing: {total_stats['missing_with_both_vertices_detected']}")

    # Analyze missing edges where both vertices were detected
    if all_missing_details:
        print(f"\n\nAnalyzing {len(all_missing_details)} edges where BOTH vertices were detected but edge was not traced:")

        lengths = [d["edge_length"] for d in all_missing_details]
        with_path = sum(1 for d in all_missing_details if d["has_skeleton_path"])
        without_path = len(all_missing_details) - with_path

        print(f"  Edge length stats: min={min(lengths):.1f}, max={max(lengths):.1f}, mean={np.mean(lengths):.1f}")
        print(f"  With skeleton path between endpoints: {with_path} ({100*with_path/len(all_missing_details):.1f}%)")
        print(f"  Without skeleton path: {without_path} ({100*without_path/len(all_missing_details):.1f}%)")

        # Breakdown by assignment
        by_assignment = defaultdict(int)
        for d in all_missing_details:
            by_assignment[d["assignment"]] += 1
        print(f"  By assignment: {dict(by_assignment)}")

        # Short edges
        short_edges = [d for d in all_missing_details if d["edge_length"] < 20]
        print(f"  Short edges (<20px): {len(short_edges)}")


if __name__ == "__main__":
    main()
