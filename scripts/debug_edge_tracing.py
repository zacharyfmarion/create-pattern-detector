#!/usr/bin/env python3
"""Debug edge tracing to understand why edges are missing."""

import sys
sys.path.insert(0, '/Users/zacharymarion/src/cp-custom-detector')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.fold_parser import FOLDParser, transform_coords
from src.data.annotations import GroundTruthGenerator
from src.postprocessing.skeletonize import skeletonize_segmentation
from src.postprocessing.junctions import detect_junctions, add_boundary_vertices
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


def analyze_skeleton_connectivity(fold_path: Path, image_size: int = 512):
    """Analyze why edge tracing fails for a given FOLD file."""

    # Load FOLD and generate GT annotation
    fold_parser = FOLDParser()
    fold_data = fold_parser.parse(str(fold_path))

    gt_generator = GroundTruthGenerator(
        image_size=image_size,
        line_width=3,
        junction_sigma=5.0,
    )
    annotation = gt_generator.generate(fold_data)

    # Get GT segmentation and junction heatmap
    segmentation = annotation['segmentation']
    junction_heatmap = annotation['junction_heatmap']

    # Get GT vertices and edges from CreasePattern
    # Use the same transform as GroundTruthGenerator
    gt_vertices, _ = transform_coords(
        fold_data.vertices,
        image_size=image_size,
        padding=50,  # Default padding in GroundTruthGenerator
    )
    gt_edges = fold_data.edges

    # Skeletonize
    skeleton, skeleton_labels = skeletonize_segmentation(segmentation, preserve_labels=True)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. GT segmentation
    axes[0, 0].imshow(segmentation, cmap='tab10', vmin=0, vmax=4)
    axes[0, 0].set_title('GT Segmentation')

    # 2. Skeleton
    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title(f'Skeleton ({skeleton.sum()} pixels)')

    # 3. Skeleton with GT vertices overlaid
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].scatter(gt_vertices[:, 0], gt_vertices[:, 1], c='red', s=20, alpha=0.7)
    axes[0, 2].set_title('Skeleton + GT Vertices')

    # 4. Check skeleton coverage of GT edges
    # For each GT edge, check if there's a skeleton path between the vertices
    edge_has_skeleton = []
    for i, (v1, v2) in enumerate(gt_edges):
        p1 = gt_vertices[v1]
        p2 = gt_vertices[v2]

        # Sample points along the edge
        num_samples = int(np.linalg.norm(p2 - p1)) + 1
        t = np.linspace(0, 1, max(num_samples, 2))
        samples = p1 + np.outer(t, p2 - p1)

        # Check how many samples hit the skeleton
        hits = 0
        for sx, sy in samples:
            ix, iy = int(round(sx)), int(round(sy))
            if 0 <= iy < skeleton.shape[0] and 0 <= ix < skeleton.shape[1]:
                # Check skeleton within 2-pixel radius
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                            if skeleton[ny, nx]:
                                hits += 1
                                break
                    else:
                        continue
                    break

        coverage = hits / len(samples)
        edge_has_skeleton.append(coverage > 0.8)  # 80% coverage = good

    edge_has_skeleton = np.array(edge_has_skeleton)
    good_edges = edge_has_skeleton.sum()
    total_edges = len(gt_edges)

    # 5. Visualize which edges have skeleton coverage
    axes[1, 0].imshow(skeleton, cmap='gray')
    for i, (v1, v2) in enumerate(gt_edges):
        p1, p2 = gt_vertices[v1], gt_vertices[v2]
        color = 'green' if edge_has_skeleton[i] else 'red'
        axes[1, 0].plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1, alpha=0.7)
    axes[1, 0].set_title(f'Skeleton Coverage: {good_edges}/{total_edges} edges ({100*good_edges/total_edges:.1f}%)')

    # 6. Detected junctions vs GT vertices
    config = GraphExtractorConfig(
        junction_threshold=0.5,
        junction_min_distance=10,
        use_skeleton_junctions=False,
    )

    junctions = detect_junctions(
        junction_heatmap,
        skeleton=None,
        threshold=0.5,
        min_distance=10,
        use_skeleton_fallback=False,
        use_skeleton_refinement=False,
    )

    axes[1, 1].imshow(junction_heatmap, cmap='hot')
    axes[1, 1].scatter(gt_vertices[:, 0], gt_vertices[:, 1], c='blue', s=30, marker='x', label=f'GT ({len(gt_vertices)})')
    axes[1, 1].scatter(junctions[:, 0], junctions[:, 1], c='green', s=30, marker='o', label=f'Detected ({len(junctions)})')
    axes[1, 1].set_title('Junction Heatmap + Detection')
    axes[1, 1].legend()

    # 7. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Count close vertex pairs
    from scipy.spatial.distance import cdist
    dists = cdist(gt_vertices, gt_vertices)
    np.fill_diagonal(dists, np.inf)
    close_pairs_1px = (dists < 1.0).sum() // 2
    close_pairs_3px = (dists < 3.0).sum() // 2

    summary = f"""
    GT Vertices: {len(gt_vertices)}
    GT Edges: {len(gt_edges)}

    Skeleton pixels: {skeleton.sum()}

    Edges with skeleton coverage: {good_edges}/{total_edges} ({100*good_edges/total_edges:.1f}%)

    Detected junctions: {len(junctions)}

    Close vertex pairs (<1px): {close_pairs_1px}
    Close vertex pairs (<3px): {close_pairs_3px}
    """
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace', va='center')
    ax.set_title('Summary')

    plt.suptitle(fold_path.stem, fontsize=14)
    plt.tight_layout()

    # Save
    output_dir = Path('/Users/zacharymarion/src/cp-custom-detector/visualizations/edge_tracing_debug')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{fold_path.stem}_debug.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Edges with skeleton: {good_edges}/{total_edges} = {100*good_edges/total_edges:.1f}%")
    print(f"  Close pairs (<1px): {close_pairs_1px}")

    return good_edges / total_edges


if __name__ == '__main__':
    # Test on a few files
    data_dir = Path('/Users/zacharymarion/src/cp-custom-detector/data/training/full-training/fold')

    # Get files and sort by vertex count to pick simple/medium/complex examples
    fold_files = sorted(data_dir.glob('*.fold'))[:20]  # Check first 20

    # Pick 3 varied examples
    import json
    file_sizes = []
    for fpath in fold_files:
        with open(fpath) as f:
            data = json.load(f)
            n_verts = len(data.get('vertices_coords', []))
            file_sizes.append((n_verts, fpath))

    file_sizes.sort()

    # Pick smallest, medium, largest
    if len(file_sizes) >= 3:
        test_files = [
            file_sizes[0][1],  # Simple
            file_sizes[len(file_sizes)//2][1],  # Medium
            file_sizes[-1][1],  # Complex
        ]
    else:
        test_files = [f[1] for f in file_sizes]

    for fpath in test_files:
        print(f"\nAnalyzing {fpath.name}...")
        analyze_skeleton_connectivity(fpath)
