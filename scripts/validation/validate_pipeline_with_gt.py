#!/usr/bin/env python3
"""
Validate the skeletonization and graph extraction pipeline using ground truth.

This script tests the theoretical upper bound of the pipeline by:
1. Loading FOLD files and generating GT annotations
2. Running GraphExtractor on the GT annotations
3. Comparing extracted graph to the original GT graph
4. Computing metrics (vertex/edge precision/recall, assignment accuracy)

Usage:
    python scripts/validation/validate_pipeline_with_gt.py --fold-dir data/training/full-training/fold
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from src.data.fold_parser import FOLDParser
from src.data.annotations import GroundTruthGenerator
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


@dataclass
class MatchingResult:
    """Result of matching extracted graph to GT graph."""
    # Vertex matching
    gt_vertices: np.ndarray  # (N_gt, 2)
    extracted_vertices: np.ndarray  # (N_ext, 2)
    matched_gt_indices: List[int] = field(default_factory=list)  # GT vertices that were matched
    matched_ext_indices: List[int] = field(default_factory=list)  # Extracted vertices that matched
    gt_to_ext_map: Dict[int, int] = field(default_factory=dict)  # GT idx -> extracted idx
    ext_to_gt_map: Dict[int, int] = field(default_factory=dict)  # Extracted idx -> GT idx
    position_errors: List[float] = field(default_factory=list)  # Distance for matched pairs

    # Edge matching
    gt_edges: np.ndarray = None  # (E_gt, 2)
    extracted_edges: np.ndarray = None  # (E_ext, 2)
    matched_gt_edge_indices: List[int] = field(default_factory=list)
    matched_ext_edge_indices: List[int] = field(default_factory=list)

    # Assignment matching
    gt_assignments: np.ndarray = None
    extracted_assignments: np.ndarray = None
    assignment_correct: List[bool] = field(default_factory=list)


def match_vertices(
    gt_vertices: np.ndarray,
    extracted_vertices: np.ndarray,
    tolerance: float = 5.0,
) -> MatchingResult:
    """
    Match extracted vertices to GT vertices using greedy nearest-neighbor.

    Args:
        gt_vertices: (N_gt, 2) GT vertex positions
        extracted_vertices: (N_ext, 2) extracted vertex positions
        tolerance: Maximum distance for a match (pixels)

    Returns:
        MatchingResult with vertex matching info
    """
    result = MatchingResult(
        gt_vertices=gt_vertices,
        extracted_vertices=extracted_vertices,
    )

    if len(gt_vertices) == 0 or len(extracted_vertices) == 0:
        return result

    # Compute pairwise distances
    # gt_vertices: (N_gt, 2) -> (N_gt, 1, 2)
    # extracted_vertices: (N_ext, 2) -> (1, N_ext, 2)
    gt_exp = gt_vertices[:, np.newaxis, :]
    ext_exp = extracted_vertices[np.newaxis, :, :]
    distances = np.linalg.norm(gt_exp - ext_exp, axis=2)  # (N_gt, N_ext)

    # Greedy matching: repeatedly find closest unmatched pair
    matched_gt = set()
    matched_ext = set()

    while True:
        # Mask already matched
        mask = np.ones_like(distances) * np.inf
        for i in range(len(gt_vertices)):
            for j in range(len(extracted_vertices)):
                if i not in matched_gt and j not in matched_ext:
                    mask[i, j] = distances[i, j]

        # Find minimum
        min_idx = np.unravel_index(np.argmin(mask), mask.shape)
        min_dist = mask[min_idx]

        if min_dist > tolerance or min_dist == np.inf:
            break

        gt_idx, ext_idx = min_idx
        matched_gt.add(gt_idx)
        matched_ext.add(ext_idx)

        result.matched_gt_indices.append(gt_idx)
        result.matched_ext_indices.append(ext_idx)
        result.gt_to_ext_map[gt_idx] = ext_idx
        result.ext_to_gt_map[ext_idx] = gt_idx
        result.position_errors.append(min_dist)

    return result


def match_edges(
    result: MatchingResult,
    gt_edges: np.ndarray,
    extracted_edges: np.ndarray,
    gt_assignments: np.ndarray,
    extracted_assignments: np.ndarray,
) -> MatchingResult:
    """
    Match extracted edges to GT edges based on vertex matching.

    An edge is matched if both endpoints match corresponding GT vertices
    AND there exists a GT edge between those GT vertices.
    """
    result.gt_edges = gt_edges
    result.extracted_edges = extracted_edges
    result.gt_assignments = gt_assignments
    result.extracted_assignments = extracted_assignments

    if len(gt_edges) == 0 or len(extracted_edges) == 0:
        return result

    # Build GT edge lookup: (v1, v2) -> edge_idx (sorted order)
    gt_edge_map = {}
    for idx, (v1, v2) in enumerate(gt_edges):
        key = (min(v1, v2), max(v1, v2))
        gt_edge_map[key] = idx

    # Check each extracted edge
    for ext_edge_idx, (ext_v1, ext_v2) in enumerate(extracted_edges):
        # Check if both endpoints have GT matches
        if ext_v1 not in result.ext_to_gt_map or ext_v2 not in result.ext_to_gt_map:
            continue

        gt_v1 = result.ext_to_gt_map[ext_v1]
        gt_v2 = result.ext_to_gt_map[ext_v2]

        # Check if GT has this edge
        key = (min(gt_v1, gt_v2), max(gt_v1, gt_v2))
        if key in gt_edge_map:
            gt_edge_idx = gt_edge_map[key]

            result.matched_gt_edge_indices.append(gt_edge_idx)
            result.matched_ext_edge_indices.append(ext_edge_idx)

            # Check assignment match
            gt_assign = gt_assignments[gt_edge_idx]
            ext_assign = extracted_assignments[ext_edge_idx]
            result.assignment_correct.append(gt_assign == ext_assign)

    return result


@dataclass
class ValidationMetrics:
    """Aggregated metrics across all files."""
    # Counts
    num_files: int = 0

    # Vertex metrics
    total_gt_vertices: int = 0
    total_extracted_vertices: int = 0
    total_matched_vertices: int = 0
    position_errors: List[float] = field(default_factory=list)

    # Edge metrics
    total_gt_edges: int = 0
    total_extracted_edges: int = 0
    total_matched_edges: int = 0

    # Assignment metrics
    assignment_correct: int = 0
    assignment_total: int = 0
    per_class_correct: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    per_class_total: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})

    def vertex_precision(self) -> float:
        if self.total_extracted_vertices == 0:
            return 0.0
        return self.total_matched_vertices / self.total_extracted_vertices

    def vertex_recall(self) -> float:
        if self.total_gt_vertices == 0:
            return 0.0
        return self.total_matched_vertices / self.total_gt_vertices

    def edge_precision(self) -> float:
        if self.total_extracted_edges == 0:
            return 0.0
        return self.total_matched_edges / self.total_extracted_edges

    def edge_recall(self) -> float:
        if self.total_gt_edges == 0:
            return 0.0
        return self.total_matched_edges / self.total_gt_edges

    def mean_position_error(self) -> float:
        if len(self.position_errors) == 0:
            return 0.0
        return np.mean(self.position_errors)

    def assignment_accuracy(self) -> float:
        if self.assignment_total == 0:
            return 0.0
        return self.assignment_correct / self.assignment_total

    def per_class_accuracy(self) -> Dict[str, float]:
        class_names = {0: 'M', 1: 'V', 2: 'B', 3: 'U'}
        result = {}
        for cls_id, name in class_names.items():
            total = self.per_class_total[cls_id]
            if total > 0:
                result[name] = self.per_class_correct[cls_id] / total
            else:
                result[name] = 0.0
        return result


def validate_single_file(
    fold_path: Path,
    gt_generator: GroundTruthGenerator,
    extractor: GraphExtractor,
    parser: FOLDParser,
    tolerance: float = 5.0,
    verbose: bool = False,
) -> Optional[MatchingResult]:
    """
    Validate pipeline on a single FOLD file.

    Returns:
        MatchingResult or None if file couldn't be processed
    """
    try:
        # Parse FOLD file
        cp = parser.parse(fold_path)

        # Generate GT annotations
        gt = gt_generator.generate(cp)

        # Extract graph from GT annotations
        extracted = extractor.extract(
            segmentation=gt['segmentation'],
            junction_heatmap=gt['junction_heatmap'],
            orientation=gt['orientation'],
        )

        # Get GT graph in pixel coordinates
        gt_vertices = gt['vertices']  # Already in pixel coords from generator
        gt_edges = gt['edges']
        gt_assignments = gt['assignments']

        # Match vertices
        result = match_vertices(gt_vertices, extracted.vertices, tolerance)

        # Match edges
        result = match_edges(
            result,
            gt_edges,
            extracted.edges,
            gt_assignments,
            extracted.assignments,
        )

        if verbose:
            print(f"\n{fold_path.name}:")
            print(f"  GT: {len(gt_vertices)} vertices, {len(gt_edges)} edges")
            print(f"  Extracted: {len(extracted.vertices)} vertices, {len(extracted.edges)} edges")
            print(f"  Matched: {len(result.matched_gt_indices)} vertices, {len(result.matched_gt_edge_indices)} edges")

        return result

    except Exception as e:
        if verbose:
            print(f"Error processing {fold_path}: {e}")
        return None


def aggregate_metrics(results: List[MatchingResult]) -> ValidationMetrics:
    """Aggregate metrics across all validation results."""
    metrics = ValidationMetrics(num_files=len(results))

    for result in results:
        # Vertex metrics
        metrics.total_gt_vertices += len(result.gt_vertices)
        metrics.total_extracted_vertices += len(result.extracted_vertices)
        metrics.total_matched_vertices += len(result.matched_gt_indices)
        metrics.position_errors.extend(result.position_errors)

        # Edge metrics
        if result.gt_edges is not None:
            metrics.total_gt_edges += len(result.gt_edges)
        if result.extracted_edges is not None:
            metrics.total_extracted_edges += len(result.extracted_edges)
        metrics.total_matched_edges += len(result.matched_gt_edge_indices)

        # Assignment metrics
        for i, correct in enumerate(result.assignment_correct):
            metrics.assignment_total += 1
            if correct:
                metrics.assignment_correct += 1

            # Per-class tracking
            gt_edge_idx = result.matched_gt_edge_indices[i]
            gt_class = result.gt_assignments[gt_edge_idx]
            metrics.per_class_total[gt_class] += 1
            if correct:
                metrics.per_class_correct[gt_class] += 1

    return metrics


def print_metrics(metrics: ValidationMetrics):
    """Print formatted metrics summary."""
    print("\n" + "=" * 60)
    print("PIPELINE VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nFiles processed: {metrics.num_files}")

    print("\n--- Vertex Metrics ---")
    print(f"  GT vertices:        {metrics.total_gt_vertices}")
    print(f"  Extracted vertices: {metrics.total_extracted_vertices}")
    print(f"  Matched vertices:   {metrics.total_matched_vertices}")
    print(f"  Precision: {metrics.vertex_precision() * 100:.1f}%")
    print(f"  Recall:    {metrics.vertex_recall() * 100:.1f}%")
    print(f"  Mean position error: {metrics.mean_position_error():.2f} px")

    print("\n--- Edge Metrics ---")
    print(f"  GT edges:        {metrics.total_gt_edges}")
    print(f"  Extracted edges: {metrics.total_extracted_edges}")
    print(f"  Matched edges:   {metrics.total_matched_edges}")
    print(f"  Precision: {metrics.edge_precision() * 100:.1f}%")
    print(f"  Recall:    {metrics.edge_recall() * 100:.1f}%")

    print("\n--- Assignment Accuracy (on matched edges) ---")
    print(f"  Overall: {metrics.assignment_accuracy() * 100:.1f}%")
    per_class = metrics.per_class_accuracy()
    for cls_name in ['M', 'V', 'B', 'U']:
        total = metrics.per_class_total[{'M': 0, 'V': 1, 'B': 2, 'U': 3}[cls_name]]
        print(f"  {cls_name}: {per_class[cls_name] * 100:.1f}% (n={total})")

    print("\n" + "=" * 60)

    # Diagnose issues
    if metrics.vertex_recall() < 0.9:
        print("\n⚠️  Low vertex recall - pipeline is missing GT vertices")
        print("   Possible causes:")
        print("   - Junction heatmap threshold too high")
        print("   - Junction sigma too small in GT generation")

    if metrics.vertex_precision() < 0.9:
        print("\n⚠️  Low vertex precision - pipeline is creating extra vertices")
        print("   Possible causes:")
        print("   - Skeleton branch points creating spurious junctions")
        print("   - Junction heatmap threshold too low")

    if metrics.edge_recall() < 0.9:
        print("\n⚠️  Low edge recall - pipeline is missing GT edges")
        print("   Possible causes:")
        print("   - Edge tracing not reaching all junctions")
        print("   - Short edges being filtered out")

    if metrics.edge_precision() < 0.9:
        print("\n⚠️  Low edge precision - pipeline is creating extra edges")
        print("   Possible causes:")
        print("   - Skeleton has spurious branches")
        print("   - Edge cleanup not aggressive enough")

    if metrics.assignment_accuracy() < 0.9:
        print("\n⚠️  Low assignment accuracy - labels not transferring correctly")
        print("   Possible causes:")
        print("   - Edge path sampling issues")
        print("   - Segmentation label bleeding at junctions")


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline with GT")
    parser.add_argument(
        "--fold-dir",
        type=str,
        required=True,
        help="Directory containing .fold files",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size for rendering",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Vertex matching tolerance in pixels",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Find FOLD files
    fold_dir = Path(args.fold_dir)
    fold_files = sorted(fold_dir.glob("*.fold"))

    if len(fold_files) == 0:
        print(f"No .fold files found in {fold_dir}")
        return

    if args.max_files:
        fold_files = fold_files[:args.max_files]

    print(f"Found {len(fold_files)} FOLD files")

    # Initialize components
    fold_parser = FOLDParser()
    gt_generator = GroundTruthGenerator(
        image_size=args.image_size,
        padding=25,
        line_width=2,
        junction_sigma=3.0,  # Larger sigma for better detection
    )

    # Configure extractor for GT validation
    config = GraphExtractorConfig(
        junction_threshold=0.2,  # Very low threshold since GT heatmap is clean
        junction_min_distance=1,  # Minimal NMS to catch closely-spaced junctions
        use_skeleton_junctions=False,  # Don't use skeleton junctions with GT
        junction_merge_distance=1.0,  # Very tight merge distance
        min_edge_length=3.0,  # Lower min edge length
        merge_collinear=True,
        collinear_angle_threshold=10.0,
    )
    extractor = GraphExtractor(config)

    # Validate all files
    results = []
    for fold_path in tqdm(fold_files, desc="Validating"):
        result = validate_single_file(
            fold_path,
            gt_generator,
            extractor,
            fold_parser,
            tolerance=args.tolerance,
            verbose=args.verbose,
        )
        if result is not None:
            results.append(result)

    if len(results) == 0:
        print("No files could be processed!")
        return

    # Compute and print metrics
    metrics = aggregate_metrics(results)
    print_metrics(metrics)

    # Save results
    if args.output:
        output_data = {
            'num_files': metrics.num_files,
            'vertex': {
                'precision': metrics.vertex_precision(),
                'recall': metrics.vertex_recall(),
                'mean_error_px': metrics.mean_position_error(),
            },
            'edge': {
                'precision': metrics.edge_precision(),
                'recall': metrics.edge_recall(),
            },
            'assignment': {
                'accuracy': metrics.assignment_accuracy(),
                'per_class': metrics.per_class_accuracy(),
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
