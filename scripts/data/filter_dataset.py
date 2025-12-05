#!/usr/bin/env python3
"""
Filter crease pattern dataset to remove problematic files.

This script analyzes FOLD files and filters out those with:
1. Vertices that collapse to sub-pixel distances at target resolution
2. Too many vertices (overly complex patterns)
3. Other quality issues

Usage:
    python scripts/data/filter_dataset.py --fold-dir data/training/full-training/fold --output-dir data/training/filtered
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.data.fold_parser import FOLDParser, transform_coords


@dataclass
class FilterResult:
    """Result of filtering analysis for a single file."""
    path: Path
    num_vertices: int
    num_edges: int
    min_vertex_distance: float  # Minimum distance between any two vertices (pixels)
    num_close_pairs: int  # Number of vertex pairs within threshold
    passed: bool
    rejection_reason: Optional[str] = None


def analyze_fold_file(
    fold_path: Path,
    parser: FOLDParser,
    image_size: int = 512,
    padding: int = 50,
    min_distance_threshold: float = 3.0,
    max_vertices: int = 500,
    max_close_pairs: int = 5,
) -> FilterResult:
    """
    Analyze a FOLD file for quality issues.

    Args:
        fold_path: Path to FOLD file
        parser: FOLDParser instance
        image_size: Target image size for rendering
        padding: Padding around the pattern
        min_distance_threshold: Minimum allowed distance between vertices (pixels)
        max_vertices: Maximum number of vertices allowed
        max_close_pairs: Maximum number of close vertex pairs allowed

    Returns:
        FilterResult with analysis
    """
    try:
        cp = parser.parse(str(fold_path))
    except Exception as e:
        return FilterResult(
            path=fold_path,
            num_vertices=0,
            num_edges=0,
            min_vertex_distance=0.0,
            num_close_pairs=0,
            passed=False,
            rejection_reason=f"Parse error: {e}",
        )

    num_vertices = cp.num_vertices
    num_edges = cp.num_edges

    # Transform vertices to pixel coordinates at target resolution
    pixel_vertices, _ = transform_coords(
        cp.vertices,
        image_size=image_size,
        padding=padding,
    )

    # Compute pairwise distances
    if len(pixel_vertices) > 1:
        distances = cdist(pixel_vertices, pixel_vertices)
        # Set diagonal to infinity so we don't count self-distances
        np.fill_diagonal(distances, np.inf)

        min_distance = distances.min()
        # Count pairs below threshold (divide by 2 since matrix is symmetric)
        num_close_pairs = int((distances < min_distance_threshold).sum() // 2)
    else:
        min_distance = np.inf
        num_close_pairs = 0

    # Determine if file passes filters
    passed = True
    rejection_reason = None

    if num_vertices > max_vertices:
        passed = False
        rejection_reason = f"Too many vertices: {num_vertices} > {max_vertices}"
    elif num_close_pairs > max_close_pairs:
        passed = False
        rejection_reason = f"Too many close vertex pairs: {num_close_pairs} > {max_close_pairs} (min dist: {min_distance:.2f}px)"
    elif min_distance < 1.0 and num_vertices > 10:
        # Very close vertices in non-trivial patterns
        passed = False
        rejection_reason = f"Sub-pixel vertex distance: {min_distance:.2f}px"

    return FilterResult(
        path=fold_path,
        num_vertices=num_vertices,
        num_edges=num_edges,
        min_vertex_distance=float(min_distance),
        num_close_pairs=num_close_pairs,
        passed=passed,
        rejection_reason=rejection_reason,
    )


def filter_dataset(
    fold_dir: Path,
    image_size: int = 512,
    padding: int = 50,
    min_distance_threshold: float = 3.0,
    max_vertices: int = 500,
    max_close_pairs: int = 5,
) -> Tuple[List[FilterResult], List[FilterResult]]:
    """
    Filter all FOLD files in a directory.

    Returns:
        Tuple of (passed_files, rejected_files)
    """
    parser = FOLDParser()
    fold_files = sorted(fold_dir.glob("*.fold"))

    passed = []
    rejected = []

    for fold_path in tqdm(fold_files, desc="Analyzing files"):
        result = analyze_fold_file(
            fold_path,
            parser,
            image_size=image_size,
            padding=padding,
            min_distance_threshold=min_distance_threshold,
            max_vertices=max_vertices,
            max_close_pairs=max_close_pairs,
        )

        if result.passed:
            passed.append(result)
        else:
            rejected.append(result)

    return passed, rejected


def copy_filtered_files(
    passed: List[FilterResult],
    source_fold_dir: Path,
    output_dir: Path,
    source_image_dir: Optional[Path] = None,
):
    """Copy passed files to output directory."""
    output_fold_dir = output_dir / "fold"
    output_fold_dir.mkdir(parents=True, exist_ok=True)

    if source_image_dir:
        output_image_dir = output_dir / "images"
        output_image_dir.mkdir(parents=True, exist_ok=True)

    for result in tqdm(passed, desc="Copying files"):
        # Copy FOLD file
        src_fold = result.path
        dst_fold = output_fold_dir / src_fold.name
        shutil.copy2(src_fold, dst_fold)

        # Copy corresponding image if it exists
        if source_image_dir:
            image_name = src_fold.stem + ".png"
            src_image = source_image_dir / image_name
            if src_image.exists():
                dst_image = output_image_dir / image_name
                shutil.copy2(src_image, dst_image)


def print_summary(passed: List[FilterResult], rejected: List[FilterResult]):
    """Print filtering summary."""
    total = len(passed) + len(rejected)

    print("\n" + "=" * 60)
    print("DATASET FILTERING SUMMARY")
    print("=" * 60)

    print(f"\nTotal files:    {total}")
    print(f"Passed:         {len(passed)} ({100*len(passed)/total:.1f}%)")
    print(f"Rejected:       {len(rejected)} ({100*len(rejected)/total:.1f}%)")

    if passed:
        vertices = [r.num_vertices for r in passed]
        edges = [r.num_edges for r in passed]
        min_dists = [r.min_vertex_distance for r in passed if np.isfinite(r.min_vertex_distance)]

        print("\n--- Passed Files Statistics ---")
        print(f"  Vertices: min={min(vertices)}, max={max(vertices)}, mean={np.mean(vertices):.1f}")
        print(f"  Edges:    min={min(edges)}, max={max(edges)}, mean={np.mean(edges):.1f}")
        if min_dists:
            print(f"  Min vertex dist: min={min(min_dists):.2f}px, mean={np.mean(min_dists):.2f}px")

    if rejected:
        print("\n--- Rejection Reasons ---")
        reasons = {}
        for r in rejected:
            reason_type = r.rejection_reason.split(":")[0] if r.rejection_reason else "Unknown"
            reasons[reason_type] = reasons.get(reason_type, 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        # Show some examples
        print("\n--- Example Rejections ---")
        for r in rejected[:5]:
            print(f"  {r.path.name}: {r.rejection_reason}")


def main():
    parser = argparse.ArgumentParser(description="Filter crease pattern dataset")
    parser.add_argument("--fold-dir", type=Path, required=True, help="Input FOLD directory")
    parser.add_argument("--image-dir", type=Path, help="Input image directory (optional)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for filtered files")
    parser.add_argument("--image-size", type=int, default=512, help="Target image size")
    parser.add_argument("--min-distance", type=float, default=3.0,
                        help="Minimum vertex distance threshold (pixels)")
    parser.add_argument("--max-vertices", type=int, default=500,
                        help="Maximum number of vertices allowed")
    parser.add_argument("--max-close-pairs", type=int, default=5,
                        help="Maximum number of close vertex pairs allowed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't copy files")
    parser.add_argument("--save-report", type=Path, help="Save detailed report to JSON")

    args = parser.parse_args()

    if not args.fold_dir.exists():
        print(f"Error: FOLD directory not found: {args.fold_dir}")
        sys.exit(1)

    # Run filtering
    passed, rejected = filter_dataset(
        args.fold_dir,
        image_size=args.image_size,
        min_distance_threshold=args.min_distance,
        max_vertices=args.max_vertices,
        max_close_pairs=args.max_close_pairs,
    )

    # Print summary
    print_summary(passed, rejected)

    # Save report if requested
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "settings": {
                "image_size": args.image_size,
                "min_distance": args.min_distance,
                "max_vertices": args.max_vertices,
                "max_close_pairs": args.max_close_pairs,
            },
            "summary": {
                "total": len(passed) + len(rejected),
                "passed": len(passed),
                "rejected": len(rejected),
            },
            "passed": [
                {
                    "name": r.path.name,
                    "vertices": r.num_vertices,
                    "edges": r.num_edges,
                    "min_distance": r.min_vertex_distance,
                }
                for r in passed
            ],
            "rejected": [
                {
                    "name": r.path.name,
                    "vertices": r.num_vertices,
                    "edges": r.num_edges,
                    "min_distance": r.min_vertex_distance,
                    "reason": r.rejection_reason,
                }
                for r in rejected
            ],
        }

        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.save_report}")

    # Copy files if not dry run and output dir specified
    if args.output_dir and not args.dry_run:
        print(f"\nCopying {len(passed)} files to {args.output_dir}...")
        copy_filtered_files(
            passed,
            args.fold_dir,
            args.output_dir,
            args.image_dir,
        )
        print("Done!")
    elif args.output_dir and args.dry_run:
        print(f"\n[Dry run] Would copy {len(passed)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
