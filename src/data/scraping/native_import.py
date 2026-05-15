"""Native origami CP asset preservation and best-effort conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from src.data.fold_parser import FOLDParser

from .manifest import sanitize_slug, sha256_file


@dataclass
class NativeImportResult:
    source_path: str
    preserved_path: str | None
    converted_fold_path: str | None
    status: str
    reason: str | None = None
    segment_count: int = 0
    vertex_count: int = 0
    edge_count: int = 0
    content_sha256: str | None = None
    candidate_pair_count: int = 0
    timings: dict[str, float] | None = None


@dataclass(frozen=True)
class _Segment:
    p0: tuple[float, float]
    p1: tuple[float, float]


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_cp_segments(text: str) -> list[_Segment]:
    """Parse a common line-segment `.cp` text file.

    CPoogle contains many files where each non-comment line is either
    `assignment x1 y1 x2 y2` or `x1 y1 x2 y2`. Assignment values are not
    standardized enough here, so conversion preserves geometry and marks all
    crease assignments as `U`.
    """
    segments: list[_Segment] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [float(v) for v in _NUMBER_RE.findall(line)]
        if len(values) >= 5:
            x1, y1, x2, y2 = values[-4:]
        elif len(values) == 4:
            x1, y1, x2, y2 = values
        else:
            continue
        if math.hypot(x2 - x1, y2 - y1) > 1e-9:
            segments.append(_Segment((x1, y1), (x2, y2)))
    return segments


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    eps: float = 1e-9,
) -> tuple[float, float] | None:
    """Return the point intersection for non-parallel segments, if any."""
    p = np.array(a, dtype=np.float64)
    r = np.array(b, dtype=np.float64) - p
    q = np.array(c, dtype=np.float64)
    s = np.array(d, dtype=np.float64) - q
    denom = _cross(r, s)
    if abs(denom) < eps:
        return None
    qp = q - p
    t = _cross(qp, s) / denom
    u = _cross(qp, r) / denom
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        pt = p + np.clip(t, 0.0, 1.0) * r
        return (float(pt[0]), float(pt[1]))
    return None


def _candidate_segment_pairs(segments: list[_Segment]) -> set[tuple[int, int]]:
    """Find segment pairs that can intersect using a coarse bounding-box grid."""
    n_segments = len(segments)
    if n_segments < 2:
        return set()

    coords = np.array([point for segment in segments for point in (segment.p0, segment.p1)], dtype=np.float64)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    grid_dim = max(16, min(256, int(math.sqrt(n_segments) * 2)))
    cell = span / grid_dim

    cells: dict[tuple[int, int], list[int]] = {}
    for idx, segment in enumerate(segments):
        x0 = min(segment.p0[0], segment.p1[0])
        x1 = max(segment.p0[0], segment.p1[0])
        y0 = min(segment.p0[1], segment.p1[1])
        y1 = max(segment.p0[1], segment.p1[1])
        gx0 = int(np.clip(math.floor((x0 - min_xy[0]) / cell[0]), 0, grid_dim - 1))
        gx1 = int(np.clip(math.floor((x1 - min_xy[0]) / cell[0]), 0, grid_dim - 1))
        gy0 = int(np.clip(math.floor((y0 - min_xy[1]) / cell[1]), 0, grid_dim - 1))
        gy1 = int(np.clip(math.floor((y1 - min_xy[1]) / cell[1]), 0, grid_dim - 1))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                cells.setdefault((gx, gy), []).append(idx)

    pairs: set[tuple[int, int]] = set()
    for bucket in cells.values():
        if len(bucket) < 2:
            continue
        for offset, i in enumerate(bucket):
            for j in bucket[offset + 1 :]:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def _bboxes_overlap(a: _Segment, b: _Segment, eps: float = 1e-9) -> bool:
    return (
        max(min(a.p0[0], a.p1[0]), min(b.p0[0], b.p1[0]))
        <= min(max(a.p0[0], a.p1[0]), max(b.p0[0], b.p1[0])) + eps
        and max(min(a.p0[1], a.p1[1]), min(b.p0[1], b.p1[1]))
        <= min(max(a.p0[1], a.p1[1]), max(b.p0[1], b.p1[1])) + eps
    )


def _param_on_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p: tuple[float, float],
) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return 0.0
    return ((p[0] - p0[0]) * dx + (p[1] - p0[1]) * dy) / denom


def _normalize_points(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
    arr = np.array(points, dtype=np.float64)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    width, height = max_xy - min_xy
    span = max(float(width), float(height), 1e-9)
    centered_offset = np.array([(span - width) / 2.0, (span - height) / 2.0])
    normalized = ((arr - min_xy + centered_offset) / span).clip(0.0, 1.0)
    aspect = float(width / height) if height > 1e-9 else float("inf")
    return [(float(x), float(y)) for x, y in normalized], aspect


def cp_segments_to_fold(
    segments: list[_Segment],
    source_name: str,
    add_square_border: bool = True,
) -> tuple[dict[str, Any], int]:
    """Convert CP line segments into a geometry-only FOLD dictionary."""
    if not segments:
        raise ValueError("No line segments found")

    split_points: list[list[tuple[float, float]]] = [[s.p0, s.p1] for s in segments]
    candidate_pairs = _candidate_segment_pairs(segments)
    for i, j in candidate_pairs:
        a = segments[i]
        b = segments[j]
        if not _bboxes_overlap(a, b):
            continue
        pt = _segment_intersection(a.p0, a.p1, b.p0, b.p1)
        if pt is None:
            continue
        split_points[i].append(pt)
        split_points[j].append(pt)

    raw_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for segment, points in zip(segments, split_points):
        unique = list({(round(p[0], 10), round(p[1], 10)): p for p in points}.values())
        unique.sort(key=lambda p: _param_on_segment(segment.p0, segment.p1, p))
        for p0, p1 in zip(unique, unique[1:]):
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 1e-8:
                raw_edges.append((p0, p1))

    all_points = [p for edge in raw_edges for p in edge]
    normalized_points, aspect = _normalize_points(all_points)
    normalized_edges = list(zip(normalized_points[0::2], normalized_points[1::2]))

    snap_tol = 1e-6
    vertices: list[tuple[float, float]] = []
    vertex_lookup: dict[tuple[int, int], int] = {}

    def vertex_index(point: tuple[float, float]) -> int:
        key = (int(round(point[0] / snap_tol)), int(round(point[1] / snap_tol)))
        existing = vertex_lookup.get(key)
        if existing is not None:
            return existing
        vertices.append(point)
        vertex_lookup[key] = len(vertices) - 1
        return len(vertices) - 1

    edge_set: set[tuple[int, int]] = set()
    edges: list[list[int]] = []
    assignments: list[str] = []

    def add_edge(p0: tuple[float, float], p1: tuple[float, float], assignment: str) -> None:
        i = vertex_index(p0)
        j = vertex_index(p1)
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key in edge_set:
            return
        edge_set.add(key)
        edges.append([i, j])
        assignments.append(assignment)

    for p0, p1 in normalized_edges:
        add_edge(p0, p1, "U")

    if add_square_border and 0.90 <= aspect <= 1.10:
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for p0, p1 in zip(corners, corners[1:] + corners[:1]):
            add_edge(p0, p1, "B")

    return {
        "file_spec": 1.1,
        "file_creator": "cp-detector real scraper",
        "file_title": source_name,
        "file_description": "Geometry-only conversion from native .cp line segments; assignments set to U.",
        "vertices_coords": [[round(x, 8), round(y, 8)] for x, y in vertices],
        "edges_vertices": edges,
        "edges_assignment": assignments,
        "frame_classes": ["creasePattern"],
        "frame_attributes": ["2D"],
    }, len(candidate_pairs)


def convert_cp_file(input_path: str | Path, output_path: str | Path) -> NativeImportResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    start = perf_counter()
    text = input_path.read_text(encoding="utf-8", errors="replace")
    after_read = perf_counter()
    segments = parse_cp_segments(text)
    after_parse = perf_counter()
    fold, candidate_pair_count = cp_segments_to_fold(segments, input_path.stem)
    after_convert = perf_counter()

    # Validate that the project's parser can consume the emitted structure.
    cp = FOLDParser().parse_dict(fold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fold, indent=2), encoding="utf-8")
    after_write = perf_counter()
    return NativeImportResult(
        source_path=input_path.as_posix(),
        preserved_path=None,
        converted_fold_path=output_path.as_posix(),
        status="converted",
        segment_count=len(segments),
        vertex_count=cp.num_vertices,
        edge_count=cp.num_edges,
        content_sha256=sha256_file(input_path),
        candidate_pair_count=candidate_pair_count,
        timings={
            "read_seconds": after_read - start,
            "parse_seconds": after_parse - after_read,
            "convert_seconds": after_convert - after_parse,
            "write_seconds": after_write - after_convert,
            "total_seconds": after_write - start,
        },
    )


def import_native_asset(
    input_path: str | Path,
    native_root: str | Path,
    source: str,
    asset_id: str,
) -> NativeImportResult:
    """Preserve a native asset and convert `.cp` or validate `.fold` when possible."""
    input_path = Path(input_path)
    native_root = Path(native_root)
    safe_source = sanitize_slug(source)
    safe_id = sanitize_slug(asset_id.replace(":", "-"))
    preserved_dir = native_root / "raw" / safe_source
    preserved_dir.mkdir(parents=True, exist_ok=True)
    preserved_path = preserved_dir / f"{safe_id}-{sanitize_slug(input_path.name)}"
    shutil.copy2(input_path, preserved_path)

    suffix = input_path.suffix.lower()
    base_result = NativeImportResult(
        source_path=input_path.as_posix(),
        preserved_path=preserved_path.as_posix(),
        converted_fold_path=None,
        status="preserved",
        content_sha256=sha256_file(input_path),
    )

    if suffix == ".cp":
        converted_path = (
            native_root
            / "converted_fold"
            / safe_source
            / f"{safe_id}-{sanitize_slug(input_path.stem)}.fold"
        )
        try:
            result = convert_cp_file(input_path, converted_path)
            result.preserved_path = preserved_path.as_posix()
            return result
        except Exception as exc:  # noqa: BLE001 - conversion diagnostics belong in manifests
            base_result.status = "rejected"
            base_result.reason = f"cp_conversion_failed: {exc}"
            return base_result

    if suffix == ".fold":
        try:
            cp = FOLDParser().parse(input_path)
            base_result.status = "validated_fold"
            base_result.vertex_count = cp.num_vertices
            base_result.edge_count = cp.num_edges
            return base_result
        except Exception as exc:  # noqa: BLE001
            base_result.status = "rejected"
            base_result.reason = f"fold_parse_failed: {exc}"
            return base_result

    base_result.reason = "preserved_unconverted"
    return base_result
