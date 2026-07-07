"""High-recall crop proposal generation for VertexRefinerV1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ModuleNotFoundError:  # pragma: no cover - scikit-image is in project requirements.
    skeletonize = None

from src.data.vertex_refiner_targets import SquareFrame
from src.models.vertex_refiner_contract import CROP_SIZE_PX


@dataclass(frozen=True)
class VertexProposal:
    x: float
    y: float
    score: float
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ProposalConfig:
    crop_size: int = CROP_SIZE_PX
    merge_radius_px: float = 12.0
    junction_threshold: float = 0.15
    source_ink_threshold: float = 0.25
    boundary_band_px: int = 3
    min_boundary_run_px: int = 1
    hough_threshold: int = 24
    hough_min_line_length_px: int = 16
    hough_max_line_gap_px: int = 4
    intersection_support_radius_px: int = 2
    max_hough_segments: int = 80


PROVENANCE_PRIORITY: dict[str, float] = {
    "gt_training_anchor": 10.0,
    "cpline_junction_peak": 1.0,
    "square_frame_corner": 0.95,
    "source_line_arrangement_intersection": 0.8,
    "source_skeleton_branchpoint": 0.6,
    "boundary_contact_top": 0.5,
    "boundary_contact_right": 0.5,
    "boundary_contact_bottom": 0.5,
    "boundary_contact_left": 0.5,
    "source_skeleton_endpoint": 0.2,
}


def crop_origin_for_center(
    center_xy: tuple[float, float],
    *,
    crop_size: int = CROP_SIZE_PX,
) -> tuple[int, int]:
    """Return the integer full-image origin for a crop centered near ``center_xy``."""
    return (
        int(round(float(center_xy[0]) - crop_size / 2.0)),
        int(round(float(center_xy[1]) - crop_size / 2.0)),
    )


def generate_vertex_refiner_proposals(
    *,
    source_ink_probability: np.ndarray,
    junction_probability: np.ndarray | None = None,
    junction_offset: np.ndarray | None = None,
    square_frame: SquareFrame | None = None,
    gt_vertices: np.ndarray | None = None,
    include_gt_training_anchors: bool = False,
    config: ProposalConfig | None = None,
) -> list[VertexProposal]:
    """Generate merged high-recall crop proposals from image and junction evidence."""
    cfg = config or ProposalConfig()
    proposals: list[VertexProposal] = []
    if junction_probability is not None:
        proposals.extend(
            junction_peak_proposals(
                junction_probability,
                junction_offset=junction_offset,
                threshold=cfg.junction_threshold,
            )
        )
    proposals.extend(
        skeleton_node_proposals(
            source_ink_probability,
            threshold=cfg.source_ink_threshold,
        )
    )
    proposals.extend(line_arrangement_intersection_proposals(source_ink_probability, config=cfg))
    if square_frame is not None:
        proposals.extend(square_frame_corner_proposals(square_frame))
        proposals.extend(
            boundary_contact_proposals(
                source_ink_probability,
                square_frame=square_frame,
                config=cfg,
            )
        )
    if include_gt_training_anchors and gt_vertices is not None:
        proposals.extend(
            VertexProposal(
                x=float(vertex[0]),
                y=float(vertex[1]),
                score=1.0,
                provenance=("gt_training_anchor",),
            )
            for vertex in np.asarray(gt_vertices, dtype=np.float32)
        )
    return merge_proposals(proposals, merge_radius_px=cfg.merge_radius_px)


def select_vertex_refiner_proposals(
    proposals: Iterable[VertexProposal],
    *,
    max_count: int | None,
    crop_size: int = CROP_SIZE_PX,
    image_shape: tuple[int, int] | None = None,
) -> list[VertexProposal]:
    """Select a deterministic, high-coverage subset of merged proposals.

    Proposal generation intentionally over-produces candidates. A simple
    ``proposals[:N]`` cap is brittle because score ties are common on clean
    renders and can collapse to top-left spatial order. This selector keeps
    quality as the primary signal while greedily preferring crops that add new
    coarse image coverage.
    """
    ranked = rank_proposals(proposals)
    if max_count is None or len(ranked) <= int(max_count):
        return ranked
    if max_count <= 0:
        return []

    quality_values = np.asarray([proposal_quality(proposal) for proposal in ranked], dtype=np.float32)
    quality_min = float(np.min(quality_values))
    quality_range = max(float(np.max(quality_values)) - quality_min, 1e-6)
    remaining = list(range(len(ranked)))
    selected_indices: list[int] = []
    selected_xy: list[tuple[float, float]] = []
    coverage_grid: np.ndarray | None = None
    cell_size = max(4, int(round(crop_size / 12.0)))
    if image_shape is not None:
        height, width = int(image_shape[0]), int(image_shape[1])
        coverage_grid = np.zeros(
            (max(1, int(np.ceil(height / cell_size))), max(1, int(np.ceil(width / cell_size)))),
            dtype=bool,
        )

    while remaining and len(selected_indices) < int(max_count):
        best_position = 0
        best_value: tuple[float, float, tuple[float, float, float, tuple[str, ...]]] | None = None
        for position, proposal_index in enumerate(remaining):
            proposal = ranked[proposal_index]
            quality_component = (float(quality_values[proposal_index]) - quality_min) / quality_range
            coverage_component = _new_coverage_fraction(
                proposal,
                coverage_grid=coverage_grid,
                crop_size=crop_size,
                cell_size=cell_size,
            )
            diversity_component = _diversity_fraction(proposal, selected_xy, crop_size=crop_size)
            value = (
                0.45 * quality_component + 0.45 * coverage_component + 0.10 * diversity_component,
                quality_component,
                proposal_sort_key(proposal),
            )
            if best_value is None or value > best_value:
                best_value = value
                best_position = position
        proposal_index = remaining.pop(best_position)
        proposal = ranked[proposal_index]
        selected_indices.append(proposal_index)
        selected_xy.append((proposal.x, proposal.y))
        _mark_covered(
            proposal,
            coverage_grid=coverage_grid,
            crop_size=crop_size,
            cell_size=cell_size,
        )
    return [ranked[index] for index in selected_indices]


def rank_proposals(proposals: Iterable[VertexProposal]) -> list[VertexProposal]:
    """Return proposals in deterministic quality order."""
    return sorted(proposals, key=proposal_sort_key, reverse=True)


def proposal_quality(proposal: VertexProposal) -> float:
    """Score a proposal by source confidence plus provenance reliability."""
    provenance_bonus = max((PROVENANCE_PRIORITY.get(source, 0.0) for source in proposal.provenance), default=0.0)
    return float(proposal.score) + provenance_bonus


def proposal_sort_key(proposal: VertexProposal) -> tuple[float, float, float, tuple[str, ...]]:
    """Quality sort key with stable spatial tiebreakers."""
    return (proposal_quality(proposal), -float(proposal.y), -float(proposal.x), proposal.provenance)


def junction_peak_proposals(
    junction_probability: np.ndarray,
    *,
    junction_offset: np.ndarray | None = None,
    threshold: float = 0.15,
    offset_scale_px: float = 1.0,
) -> list[VertexProposal]:
    """Create proposals from low-threshold local maxima in a junction heatmap."""
    heatmap = np.asarray(junction_probability, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("junction_probability must have shape HxW")
    dilated = cv2.dilate(heatmap, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.where((heatmap >= float(threshold)) & (heatmap >= dilated - 1e-6))
    offsets = _offset_hwc(junction_offset, heatmap.shape)
    proposals: list[VertexProposal] = []
    for row, col in zip(ys.tolist(), xs.tolist()):
        dx = 0.0 if offsets is None else float(offsets[row, col, 0]) * float(offset_scale_px)
        dy = 0.0 if offsets is None else float(offsets[row, col, 1]) * float(offset_scale_px)
        proposals.append(
            VertexProposal(
                x=float(col) + dx,
                y=float(row) + dy,
                score=float(heatmap[row, col]),
                provenance=("cpline_junction_peak",),
            )
        )
    return proposals


def skeleton_node_proposals(
    source_ink_probability: np.ndarray,
    *,
    threshold: float = 0.25,
) -> list[VertexProposal]:
    """Create proposals at source-image skeleton endpoints and branchpoints."""
    ink = np.asarray(source_ink_probability, dtype=np.float32) >= float(threshold)
    if not np.any(ink):
        return []
    if skeletonize is not None:
        skeleton = skeletonize(ink).astype(bool)
    else:  # pragma: no cover - scikit-image is available in normal dev/test envs.
        skeleton = ink
    neighbors = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    neighbor_count = neighbors.astype(np.int16) - skeleton.astype(np.int16)
    endpoint = skeleton & (neighbor_count == 1)
    branch = skeleton & (neighbor_count >= 3)
    proposals = _connected_component_centers(endpoint, "source_skeleton_endpoint", score=0.65)
    proposals.extend(_connected_component_centers(branch, "source_skeleton_branchpoint", score=0.75))
    return proposals


def line_arrangement_intersection_proposals(
    source_ink_probability: np.ndarray,
    *,
    config: ProposalConfig | None = None,
) -> list[VertexProposal]:
    """Create proposals from Hough segment intersections with local ink support."""
    cfg = config or ProposalConfig()
    prob = np.asarray(source_ink_probability, dtype=np.float32)
    binary = (prob >= cfg.source_ink_threshold).astype(np.uint8) * 255
    if not np.any(binary):
        return []
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length_px,
        maxLineGap=cfg.hough_max_line_gap_px,
    )
    if lines is None:
        return []
    segments = [tuple(int(v) for v in line[0]) for line in lines[: cfg.max_hough_segments]]
    proposals: list[VertexProposal] = []
    height, width = prob.shape
    for index, a in enumerate(segments):
        for b in segments[index + 1 :]:
            point = _segment_intersection(a, b)
            if point is None:
                continue
            x, y = point
            if not (0.0 <= x < width and 0.0 <= y < height):
                continue
            row = int(round(y))
            col = int(round(x))
            radius = int(cfg.intersection_support_radius_px)
            patch = prob[
                max(0, row - radius) : min(height, row + radius + 1),
                max(0, col - radius) : min(width, col + radius + 1),
            ]
            if patch.size == 0 or float(patch.max()) < cfg.source_ink_threshold:
                continue
            proposals.append(
                VertexProposal(
                    x=float(x),
                    y=float(y),
                    score=float(patch.max()),
                    provenance=("source_line_arrangement_intersection",),
                )
            )
    return proposals


def boundary_contact_proposals(
    source_ink_probability: np.ndarray,
    *,
    square_frame: SquareFrame,
    config: ProposalConfig | None = None,
) -> list[VertexProposal]:
    """Create proposals where source ink intersects the rendered square frame."""
    cfg = config or ProposalConfig()
    prob = np.asarray(source_ink_probability, dtype=np.float32)
    height, width = prob.shape
    proposals: list[VertexProposal] = []
    sides = (
        ("top", "x", square_frame.y_min, square_frame.x_min, square_frame.x_max),
        ("bottom", "x", square_frame.y_max, square_frame.x_min, square_frame.x_max),
        ("left", "y", square_frame.x_min, square_frame.y_min, square_frame.y_max),
        ("right", "y", square_frame.x_max, square_frame.y_min, square_frame.y_max),
    )
    for side, axis, fixed, start, end in sides:
        fixed_i = int(round(float(fixed)))
        start_i = int(round(float(start)))
        end_i = int(round(float(end)))
        runs: list[tuple[int, int, float]] = []
        if axis == "x" and 0 <= fixed_i < height:
            lo = max(0, fixed_i - cfg.boundary_band_px)
            hi = min(height, fixed_i + cfg.boundary_band_px + 1)
            values = prob[lo:hi, max(0, start_i) : min(width, end_i + 1)].max(axis=0)
            runs = _runs(values >= cfg.source_ink_threshold, start_offset=max(0, start_i))
            for run_start, run_end, _ in runs:
                if run_end - run_start + 1 < cfg.min_boundary_run_px:
                    continue
                col = 0.5 * (run_start + run_end)
                score = float(values[run_start - max(0, start_i) : run_end - max(0, start_i) + 1].max())
                proposals.append(
                    VertexProposal(col, float(fixed_i), score, (f"boundary_contact_{side}",))
                )
        elif axis == "y" and 0 <= fixed_i < width:
            lo = max(0, fixed_i - cfg.boundary_band_px)
            hi = min(width, fixed_i + cfg.boundary_band_px + 1)
            values = prob[max(0, start_i) : min(height, end_i + 1), lo:hi].max(axis=1)
            runs = _runs(values >= cfg.source_ink_threshold, start_offset=max(0, start_i))
            for run_start, run_end, _ in runs:
                if run_end - run_start + 1 < cfg.min_boundary_run_px:
                    continue
                row = 0.5 * (run_start + run_end)
                score = float(values[run_start - max(0, start_i) : run_end - max(0, start_i) + 1].max())
                proposals.append(
                    VertexProposal(float(fixed_i), row, score, (f"boundary_contact_{side}",))
                )
    return proposals


def square_frame_corner_proposals(square_frame: SquareFrame) -> list[VertexProposal]:
    """Create proposals at the four known corners of the rectified square frame."""
    corners = (
        (square_frame.x_min, square_frame.y_min),
        (square_frame.x_max, square_frame.y_min),
        (square_frame.x_max, square_frame.y_max),
        (square_frame.x_min, square_frame.y_max),
    )
    return [
        VertexProposal(
            x=float(x),
            y=float(y),
            score=1.0,
            provenance=("square_frame_corner",),
        )
        for x, y in corners
    ]


def merge_proposals(
    proposals: Iterable[VertexProposal],
    *,
    merge_radius_px: float = 12.0,
) -> list[VertexProposal]:
    """Merge nearby proposal centers while preserving provenance labels."""
    merged: list[VertexProposal] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (-float(item.score), float(item.y), float(item.x), item.provenance),
    ):
        match_index: int | None = None
        for index, existing in enumerate(merged):
            if np.hypot(existing.x - proposal.x, existing.y - proposal.y) <= merge_radius_px:
                match_index = index
                break
        if match_index is None:
            merged.append(proposal)
            continue
        existing = merged[match_index]
        existing_weight = max(existing.score, 1e-3)
        proposal_weight = max(proposal.score, 1e-3)
        total = existing_weight + proposal_weight
        provenance = tuple(sorted({*existing.provenance, *proposal.provenance}))
        merged[match_index] = VertexProposal(
            x=(existing.x * existing_weight + proposal.x * proposal_weight) / total,
            y=(existing.y * existing_weight + proposal.y * proposal_weight) / total,
            score=max(existing.score, proposal.score),
            provenance=provenance,
        )
    return rank_proposals(merged)


def _new_coverage_fraction(
    proposal: VertexProposal,
    *,
    coverage_grid: np.ndarray | None,
    crop_size: int,
    cell_size: int,
) -> float:
    if coverage_grid is None:
        return 1.0
    row_min, row_max, col_min, col_max = _proposal_grid_bounds(
        proposal,
        coverage_grid=coverage_grid,
        crop_size=crop_size,
        cell_size=cell_size,
    )
    crop = coverage_grid[row_min:row_max, col_min:col_max]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(~crop)) / float(crop.size)


def _mark_covered(
    proposal: VertexProposal,
    *,
    coverage_grid: np.ndarray | None,
    crop_size: int,
    cell_size: int,
) -> None:
    if coverage_grid is None:
        return
    row_min, row_max, col_min, col_max = _proposal_grid_bounds(
        proposal,
        coverage_grid=coverage_grid,
        crop_size=crop_size,
        cell_size=cell_size,
    )
    coverage_grid[row_min:row_max, col_min:col_max] = True


def _proposal_grid_bounds(
    proposal: VertexProposal,
    *,
    coverage_grid: np.ndarray,
    crop_size: int,
    cell_size: int,
) -> tuple[int, int, int, int]:
    origin_x, origin_y = crop_origin_for_center((proposal.x, proposal.y), crop_size=crop_size)
    col_min = max(0, int(np.floor(origin_x / cell_size)))
    row_min = max(0, int(np.floor(origin_y / cell_size)))
    col_max = min(coverage_grid.shape[1], int(np.ceil((origin_x + crop_size) / cell_size)))
    row_max = min(coverage_grid.shape[0], int(np.ceil((origin_y + crop_size) / cell_size)))
    return row_min, max(row_min, row_max), col_min, max(col_min, col_max)


def _diversity_fraction(
    proposal: VertexProposal,
    selected_xy: list[tuple[float, float]],
    *,
    crop_size: int,
) -> float:
    if not selected_xy:
        return 1.0
    min_distance = min(float(np.hypot(proposal.x - x, proposal.y - y)) for x, y in selected_xy)
    return float(np.clip(min_distance / max(float(crop_size) * 2.0, 1.0), 0.0, 1.0))


def _connected_component_centers(mask: np.ndarray, provenance: str, *, score: float) -> list[VertexProposal]:
    components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    proposals: list[VertexProposal] = []
    for label in range(1, components):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        x, y = centroids[label]
        proposals.append(VertexProposal(float(x), float(y), float(score), (provenance,)))
    return proposals


def _runs(mask: np.ndarray, *, start_offset: int) -> list[tuple[int, int, float]]:
    runs: list[tuple[int, int, float]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start + start_offset, index - 1 + start_offset, 1.0))
            start = None
    if start is not None:
        runs.append((start + start_offset, len(mask) - 1 + start_offset, 1.0))
    return runs


def _segment_intersection(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if not (_within_segment(px, py, a) and _within_segment(px, py, b)):
        return None
    return float(px), float(py)


def _within_segment(x: float, y: float, segment: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = segment
    tolerance = 1e-3
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _offset_hwc(offset: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if offset is None:
        return None
    array = np.asarray(offset, dtype=np.float32)
    if array.shape == (shape[0], shape[1], 2):
        return array
    if array.shape == (2, shape[0], shape[1]):
        return np.transpose(array, (1, 2, 0))
    raise ValueError("junction_offset must have shape HxWx2 or 2xHxW")
