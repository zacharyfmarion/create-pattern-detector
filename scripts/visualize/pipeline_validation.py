#!/usr/bin/env python3
"""
Visualize pipeline validation results - compare GT graph to extracted graph.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

from src.data.fold_parser import FOLDParser
from src.data.annotations import GroundTruthGenerator
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


def visualize_comparison(
    fold_path: Path,
    gt_generator: GroundTruthGenerator,
    extractor: GraphExtractor,
    parser: FOLDParser,
    tolerance: float = 5.0,
    output_dir: Path = None,
):
    """Create visualization comparing GT and extracted graphs."""

    # Parse and generate
    cp = parser.parse(fold_path)
    gt = gt_generator.generate(cp)

    extracted = extractor.extract(
        segmentation=gt['segmentation'],
        junction_heatmap=gt['junction_heatmap'],
        orientation=gt['orientation'],
    )

    gt_verts = gt['vertices']
    gt_edges = gt['edges']
    gt_assignments = gt['assignments']

    ext_verts = extracted.vertices
    ext_edges = extracted.edges
    ext_assignments = extracted.assignments

    # Match vertices
    if len(ext_verts) > 0 and len(gt_verts) > 0:
        dists = cdist(ext_verts, gt_verts)
        ext_to_gt = {}
        gt_to_ext = {}
        matched_gt = set()

        for i in range(len(ext_verts)):
            min_j = np.argmin(dists[i])
            if dists[i, min_j] <= tolerance and min_j not in matched_gt:
                ext_to_gt[i] = min_j
                gt_to_ext[min_j] = i
                matched_gt.add(min_j)
    else:
        ext_to_gt = {}
        gt_to_ext = {}
        matched_gt = set()

    # Match edges
    gt_edge_set = {(min(v1, v2), max(v1, v2)): i for i, (v1, v2) in enumerate(gt_edges)}
    matched_ext_edges = set()
    matched_gt_edges = set()

    for i, (ext_v1, ext_v2) in enumerate(ext_edges):
        if ext_v1 in ext_to_gt and ext_v2 in ext_to_gt:
            gt_v1 = ext_to_gt[ext_v1]
            gt_v2 = ext_to_gt[ext_v2]
            key = (min(gt_v1, gt_v2), max(gt_v1, gt_v2))
            if key in gt_edge_set:
                matched_ext_edges.add(i)
                matched_gt_edges.add(gt_edge_set[key])

    # Color maps
    assignment_colors = {
        0: 'red',      # M
        1: 'blue',     # V
        2: 'black',    # B
        3: 'gray',     # U
    }

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. GT Graph
    ax = axes[0, 0]
    ax.imshow(gt['segmentation'], cmap='gray', alpha=0.3)

    for i, (v1, v2) in enumerate(gt_edges):
        p1, p2 = gt_verts[v1], gt_verts[v2]
        color = assignment_colors.get(gt_assignments[i], 'gray')
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, alpha=0.8)

    for i, v in enumerate(gt_verts):
        ax.plot(v[0], v[1], 'ko', markersize=4)

    ax.set_title(f'Ground Truth: {len(gt_verts)} vertices, {len(gt_edges)} edges')
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect('equal')

    # 2. Extracted Graph
    ax = axes[0, 1]
    ax.imshow(gt['segmentation'], cmap='gray', alpha=0.3)

    for i, (v1, v2) in enumerate(ext_edges):
        p1, p2 = ext_verts[v1], ext_verts[v2]
        color = assignment_colors.get(ext_assignments[i], 'gray')
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, alpha=0.8)

    for i, v in enumerate(ext_verts):
        color = 'green' if extracted.is_boundary[i] else 'black'
        ax.plot(v[0], v[1], 'o', color=color, markersize=4)

    ax.set_title(f'Extracted: {len(ext_verts)} vertices, {len(ext_edges)} edges')
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect('equal')

    # 3. Vertex Matching
    ax = axes[1, 0]
    ax.imshow(gt['junction_heatmap'], cmap='hot', alpha=0.5)

    # Draw GT vertices
    for i, v in enumerate(gt_verts):
        if i in matched_gt:
            ax.plot(v[0], v[1], 'go', markersize=8, label='Matched GT' if i == list(matched_gt)[0] else '')
        else:
            ax.plot(v[0], v[1], 'rx', markersize=10, markeredgewidth=2, label='Unmatched GT' if i == list(set(range(len(gt_verts))) - matched_gt)[0] else '')

    # Draw extracted vertices
    for i, v in enumerate(ext_verts):
        if i in ext_to_gt:
            ax.plot(v[0], v[1], 'g+', markersize=6)
        else:
            ax.plot(v[0], v[1], 'c^', markersize=6, label='Extra Extracted' if i == 0 else '')

    ax.set_title(f'Vertex Matching: {len(matched_gt)}/{len(gt_verts)} GT matched')
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect('equal')

    # 4. Edge Matching
    ax = axes[1, 1]
    ax.imshow(gt['segmentation'], cmap='gray', alpha=0.3)

    # Draw GT edges
    for i, (v1, v2) in enumerate(gt_edges):
        p1, p2 = gt_verts[v1], gt_verts[v2]
        if i in matched_gt_edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, alpha=0.7)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.7)

    # Draw extracted edges that don't match
    for i, (v1, v2) in enumerate(ext_edges):
        if i not in matched_ext_edges:
            p1, p2 = ext_verts[v1], ext_verts[v2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c--', linewidth=1.5, alpha=0.5)

    ax.set_title(f'Edge Matching: {len(matched_gt_edges)}/{len(gt_edges)} GT matched (green=matched, red=missing, cyan=extra)')
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect('equal')

    plt.suptitle(fold_path.stem, fontsize=14)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{fold_path.stem}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Print analysis
    print(f"\n{'='*60}")
    print(f"File: {fold_path.name}")
    print(f"{'='*60}")
    print(f"GT: {len(gt_verts)} vertices, {len(gt_edges)} edges")
    print(f"Extracted: {len(ext_verts)} vertices, {len(ext_edges)} edges")
    print(f"Matched vertices: {len(matched_gt)}/{len(gt_verts)} ({100*len(matched_gt)/len(gt_verts):.1f}%)")
    print(f"Matched edges: {len(matched_gt_edges)}/{len(gt_edges)} ({100*len(matched_gt_edges)/len(gt_edges):.1f}%)")

    # Analyze unmatched GT vertices
    unmatched_gt_verts = set(range(len(gt_verts))) - matched_gt
    if unmatched_gt_verts:
        print(f"\nUnmatched GT vertices ({len(unmatched_gt_verts)}):")

        # Categorize
        is_border = np.zeros(len(gt_verts), dtype=bool)
        crease_degrees = np.zeros(len(gt_verts), dtype=int)
        for edge_idx, (v1, v2) in enumerate(gt_edges):
            if gt_assignments[edge_idx] == 2:
                is_border[v1] = True
                is_border[v2] = True
            if gt_assignments[edge_idx] in (0, 1):
                crease_degrees[v1] += 1
                crease_degrees[v2] += 1

        border_missing = sum(1 for i in unmatched_gt_verts if is_border[i])
        interior_missing = len(unmatched_gt_verts) - border_missing

        print(f"  Border vertices missing: {border_missing}")
        print(f"  Interior vertices missing: {interior_missing}")

        # Show details for interior missing (most important)
        print(f"\n  Interior vertices missing (showing up to 10):")
        count = 0
        for i in sorted(unmatched_gt_verts):
            if not is_border[i] and count < 10:
                v = gt_verts[i]
                hm_val = gt['junction_heatmap'][int(v[1]), int(v[0])]
                print(f"    Vertex {i} at ({v[0]:.0f}, {v[1]:.0f}): heatmap={hm_val:.3f}, crease_degree={crease_degrees[i]}")
                count += 1


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline validation")
    parser.add_argument("--fold-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="visualizations/pipeline_validation")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)
    output_dir = Path(args.output_dir)
    fold_files = sorted(fold_dir.glob("*.fold"))[:args.max_files]

    print(f"Visualizing {len(fold_files)} files...")

    # Initialize
    fold_parser = FOLDParser()
    gt_generator = GroundTruthGenerator(
        image_size=args.image_size,
        padding=25,
        line_width=2,
        junction_sigma=3.0,
    )

    config = GraphExtractorConfig(
        junction_threshold=0.3,
        junction_min_distance=5,
        use_skeleton_junctions=False,
        junction_merge_distance=3.0,
        min_edge_length=5.0,
    )
    extractor = GraphExtractor(config)

    for fold_path in fold_files:
        visualize_comparison(
            fold_path,
            gt_generator,
            extractor,
            fold_parser,
            tolerance=args.tolerance,
            output_dir=output_dir,
        )

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
