#!/usr/bin/env python3
"""Rank native FOLD crease patterns by box-pleat-like geometry.

The first-pass signal is intentionally geometric: find a rotated orthogonal
frame that explains most non-boundary crease length, then check whether those
axis-aligned creases reuse row/column coordinates like a grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fold_parser import FOLDParser, transform_coords  # noqa: E402

ASSIGNMENT_NAMES = {0: "M", 1: "V", 2: "B", 3: "U"}
ALGORITHM_ID = "box_pleat_geometry_v1"
CANDIDATE_TIERS = ("strong", "review", "review-name-prior")
DEFAULT_FOLD_ROOT = Path("data/output/scraped/native/converted_fold")


@dataclass(frozen=True)
class BoxPleatFeatures:
    id: str
    path: str
    relative_path: str
    content_sha256: str
    canonical_fold_sha256: str
    vertices: int
    edges: int
    non_border_edges: int
    assignment_counts: dict[str, int]
    candidate_tier: str
    bp_score: float
    orthogonal_length_ratio: float
    orthogonal_edge_ratio: float
    axis_balance: float
    best_frame_deg: float
    diagonal_length_ratio: float
    repeated_horizontal_lines: int
    repeated_vertical_lines: int
    repeated_coord_count: int
    repeated_axis_segment_fraction: float
    grid_coord_score: float
    name_prior: float
    warnings: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json_lines(rows: Iterable[dict | str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if isinstance(row, str):
            line = row
        else:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_number(value: float) -> float:
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0.0 else rounded


def canonical_crease_pattern_payload(cp) -> dict:
    return {
        "vertices_coords": [
            [canonical_number(coord) for coord in vertex] for vertex in cp.vertices.tolist()
        ],
        "edges_vertices": [[int(a), int(b)] for a, b in cp.edges.tolist()],
        "edges_assignment": [FOLDParser.ASSIGNMENT_LABELS[int(value)] for value in cp.assignments],
    }


def canonical_crease_pattern_sha256(cp) -> str:
    return digest_json_lines([canonical_crease_pattern_payload(cp)])


def angle_distance_deg(angles: np.ndarray, targets: np.ndarray | float) -> np.ndarray:
    diff = np.abs(angles - targets)
    return np.minimum(diff, 180.0 - diff)


def edge_geometry(cp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = cp.vertices.astype(np.float64)
    edge_vertices = vertices[cp.edges]
    deltas = edge_vertices[:, 1, :] - edge_vertices[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 180.0
    return edge_vertices, lengths, angles


def find_best_orthogonal_frame(
    angles: np.ndarray,
    lengths: np.ndarray,
    tolerance_deg: float,
    frame_step_deg: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if len(angles) == 0:
        empty = np.zeros(0, dtype=bool)
        return 0.0, empty, empty, empty

    thetas = np.arange(0.0, 90.0, frame_step_deg, dtype=np.float64)
    if len(thetas) == 0:
        thetas = np.array([0.0], dtype=np.float64)

    dist0 = angle_distance_deg(angles[:, None], thetas[None, :])
    dist1 = angle_distance_deg(angles[:, None], (thetas[None, :] + 90.0) % 180.0)
    in0 = dist0 <= tolerance_deg
    in1 = dist1 <= tolerance_deg
    in_axis = in0 | in1
    weighted = (lengths[:, None] * in_axis).sum(axis=0)
    mean_axis_dist = np.divide(
        (lengths[:, None] * np.minimum(dist0, dist1) * in_axis).sum(axis=0),
        np.maximum(weighted, 1e-9),
    )
    axis0_weighted = (lengths[:, None] * in0).sum(axis=0)
    axis1_weighted = (lengths[:, None] * in1).sum(axis=0)
    balance = np.minimum(axis0_weighted, axis1_weighted) / np.maximum(
        np.maximum(axis0_weighted, axis1_weighted),
        1e-9,
    )
    # Prefer the frame that explains the most length, then the one with both
    # directions represented. The tiny term is only a deterministic tie-breaker.
    ranking = weighted + 1e-6 * balance - 1e-6 * mean_axis_dist - 1e-9 * thetas
    best_idx = int(np.argmax(ranking))
    best_theta = float(thetas[best_idx])
    return best_theta, in_axis[:, best_idx], in0[:, best_idx], in1[:, best_idx]


def rotate_points(points: np.ndarray, theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    return points @ rotation.T


def repeated_axis_lines(
    edge_vertices: np.ndarray,
    axis0_mask: np.ndarray,
    axis1_mask: np.ndarray,
    theta_deg: float,
    coord_tolerance_frac: float,
    min_segments_per_line: int,
) -> tuple[int, int, float]:
    if len(edge_vertices) == 0:
        return 0, 0, 0.0

    flat_vertices = edge_vertices.reshape(-1, 2)
    rotated_flat = rotate_points(flat_vertices, theta_deg)
    span_x = float(np.ptp(rotated_flat[:, 0]))
    span_y = float(np.ptp(rotated_flat[:, 1]))
    scale = max(span_x, span_y, 1.0)
    bin_size = max(scale * coord_tolerance_frac, 1e-9)
    rotated_edges = rotated_flat.reshape(edge_vertices.shape)

    horizontal_bins = np.rint(rotated_edges[axis0_mask, :, 1].mean(axis=1) / bin_size).astype(int)
    vertical_bins = np.rint(rotated_edges[axis1_mask, :, 0].mean(axis=1) / bin_size).astype(int)

    horizontal_counts = Counter(int(value) for value in horizontal_bins)
    vertical_counts = Counter(int(value) for value in vertical_bins)
    repeated_horizontal = sum(
        1 for count in horizontal_counts.values() if count >= min_segments_per_line
    )
    repeated_vertical = sum(
        1 for count in vertical_counts.values() if count >= min_segments_per_line
    )

    repeated_segments = sum(
        count for count in horizontal_counts.values() if count >= min_segments_per_line
    ) + sum(count for count in vertical_counts.values() if count >= min_segments_per_line)
    axis_segments = int(axis0_mask.sum() + axis1_mask.sum())
    repeated_fraction = repeated_segments / max(axis_segments, 1)
    return repeated_horizontal, repeated_vertical, float(repeated_fraction)


def name_prior(path: Path) -> float:
    stem = path.stem.lower()
    strong_patterns = [
        r"(^|[-_\s])bp($|[-_\s])",
        r"box[-_\s]?pleat",
        r"box[-_\s]?pleated",
    ]
    if any(re.search(pattern, stem) for pattern in strong_patterns):
        return 1.0
    if re.search(r"(^|[-_\s])grid($|[-_\s])", stem):
        return 0.35
    return 0.0


def classify_candidate(
    score: float,
    orthogonal_length_ratio: float,
    axis_balance: float,
    repeated_coord_count: int,
    non_border_edges: int,
    prior: float,
) -> str:
    if (
        non_border_edges >= 25
        and score >= 0.72
        and orthogonal_length_ratio >= 0.72
        and axis_balance >= 0.18
        and repeated_coord_count >= 4
    ):
        return "strong"
    if (
        non_border_edges >= 18
        and score >= 0.58
        and orthogonal_length_ratio >= 0.58
        and axis_balance >= 0.10
    ):
        return "review"
    if prior >= 1.0 and non_border_edges >= 12 and orthogonal_length_ratio >= 0.45:
        return "review-name-prior"
    return "weak"


def score_crease_pattern(
    cp,
    path: Path,
    fold_root: Path,
    tolerance_deg: float = 10.0,
    frame_step_deg: float = 0.5,
    coord_tolerance_frac: float = 0.003,
    min_segments_per_line: int = 2,
) -> BoxPleatFeatures:
    edge_vertices, lengths, angles = edge_geometry(cp)
    valid_length = lengths > 1e-9
    non_border = cp.assignments != 2
    crease_mask = valid_length & non_border
    warnings: list[str] = []

    if int(non_border.sum()) != int(crease_mask.sum()):
        warnings.append("ignored_zero_length_non_border_edges")

    crease_lengths = lengths[crease_mask]
    crease_angles = angles[crease_mask]
    crease_edge_vertices = edge_vertices[crease_mask]
    total_length = float(crease_lengths.sum())

    best_theta, in_axis, axis0_mask, axis1_mask = find_best_orthogonal_frame(
        crease_angles,
        crease_lengths,
        tolerance_deg=tolerance_deg,
        frame_step_deg=frame_step_deg,
    )

    axis_length = float(crease_lengths[in_axis].sum()) if len(crease_lengths) else 0.0
    axis0_length = float(crease_lengths[axis0_mask].sum()) if len(crease_lengths) else 0.0
    axis1_length = float(crease_lengths[axis1_mask].sum()) if len(crease_lengths) else 0.0
    orthogonal_length_ratio = axis_length / max(total_length, 1e-9)
    orthogonal_edge_ratio = float(in_axis.mean()) if len(in_axis) else 0.0
    axis_balance = min(axis0_length, axis1_length) / max(max(axis0_length, axis1_length), 1e-9)
    diagonal_length_ratio = 1.0 - orthogonal_length_ratio if total_length > 0 else 1.0

    repeated_h, repeated_v, repeated_fraction = repeated_axis_lines(
        crease_edge_vertices,
        axis0_mask,
        axis1_mask,
        theta_deg=best_theta,
        coord_tolerance_frac=coord_tolerance_frac,
        min_segments_per_line=min_segments_per_line,
    )
    repeated_coord_count = repeated_h + repeated_v
    grid_coord_score = min(repeated_coord_count / 12.0, 1.0) * 0.55 + repeated_fraction * 0.45
    density_score = min(int(crease_mask.sum()) / 80.0, 1.0)
    prior = name_prior(path)
    bp_score = (
        0.58 * orthogonal_length_ratio
        + 0.17 * axis_balance
        + 0.18 * grid_coord_score
        + 0.05 * density_score
        + 0.02 * prior
    )
    bp_score = max(0.0, min(1.0, bp_score))

    counts = Counter(ASSIGNMENT_NAMES[int(value)] for value in cp.assignments)
    rel_path = (
        path.relative_to(fold_root).as_posix()
        if path.is_relative_to(fold_root)
        else path.as_posix()
    )
    tier = classify_candidate(
        score=bp_score,
        orthogonal_length_ratio=orthogonal_length_ratio,
        axis_balance=axis_balance,
        repeated_coord_count=repeated_coord_count,
        non_border_edges=int(crease_mask.sum()),
        prior=prior,
    )

    return BoxPleatFeatures(
        id=path.stem,
        path=path.as_posix(),
        relative_path=rel_path,
        content_sha256=sha256_file(path),
        canonical_fold_sha256=canonical_crease_pattern_sha256(cp),
        vertices=int(cp.num_vertices),
        edges=int(cp.num_edges),
        non_border_edges=int(crease_mask.sum()),
        assignment_counts=dict(sorted(counts.items())),
        candidate_tier=tier,
        bp_score=round(float(bp_score), 6),
        orthogonal_length_ratio=round(float(orthogonal_length_ratio), 6),
        orthogonal_edge_ratio=round(float(orthogonal_edge_ratio), 6),
        axis_balance=round(float(axis_balance), 6),
        best_frame_deg=round(float(best_theta), 3),
        diagonal_length_ratio=round(float(diagonal_length_ratio), 6),
        repeated_horizontal_lines=int(repeated_h),
        repeated_vertical_lines=int(repeated_v),
        repeated_coord_count=int(repeated_coord_count),
        repeated_axis_segment_fraction=round(float(repeated_fraction), 6),
        grid_coord_score=round(float(grid_coord_score), 6),
        name_prior=round(float(prior), 3),
        warnings=warnings,
    )


def score_fold_files(
    fold_paths: Iterable[Path],
    fold_root: Path,
    tolerance_deg: float,
    frame_step_deg: float,
    coord_tolerance_frac: float,
    min_segments_per_line: int,
) -> tuple[list[BoxPleatFeatures], list[dict[str, str]]]:
    parser = FOLDParser()
    records: list[BoxPleatFeatures] = []
    errors: list[dict[str, str]] = []
    for path in sorted(fold_paths):
        try:
            cp = parser.parse(path)
            records.append(
                score_crease_pattern(
                    cp,
                    path=path,
                    fold_root=fold_root,
                    tolerance_deg=tolerance_deg,
                    frame_step_deg=frame_step_deg,
                    coord_tolerance_frac=coord_tolerance_frac,
                    min_segments_per_line=min_segments_per_line,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by real corpus preflight
            errors.append({"path": path.as_posix(), "error": str(exc)})
    records.sort(
        key=lambda record: (
            -record.bp_score,
            -record.orthogonal_length_ratio,
            -record.axis_balance,
            record.id,
        )
    )
    return records, errors


def write_jsonl(path: Path, records: Iterable[BoxPleatFeatures]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_csv(path: Path, records: list[BoxPleatFeatures]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    rows = [asdict(record) for record in records]
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["assignment_counts"] = json.dumps(row["assignment_counts"], sort_keys=True)
            row["warnings"] = json.dumps(row["warnings"], sort_keys=True)
            writer.writerow(row)


def pathless_record_fingerprint(record: BoxPleatFeatures) -> dict:
    payload = asdict(record)
    for key in ("id", "path", "relative_path", "content_sha256", "warnings"):
        payload.pop(key, None)
    return payload


def build_fingerprints(
    records: list[BoxPleatFeatures],
    errors: list[dict[str, str]],
    candidate_records: list[BoxPleatFeatures],
    candidate_tiers: tuple[str, ...],
) -> dict:
    tiers = Counter(record.candidate_tier for record in records)
    return {
        "algorithm": ALGORITHM_ID,
        "parsed_count": len(records),
        "error_count": len(errors),
        "tier_counts": dict(sorted(tiers.items())),
        "candidate_tiers": list(candidate_tiers),
        "candidate_count": len(candidate_records),
        "all_inputs_canonical_sha256": digest_json_lines(
            sorted(record.canonical_fold_sha256 for record in records)
        ),
        "selected_canonical_sha256": digest_json_lines(
            sorted(record.canonical_fold_sha256 for record in candidate_records)
        ),
        "selected_ranked_features_sha256": digest_json_lines(
            pathless_record_fingerprint(record) for record in candidate_records
        ),
    }


def compare_expected_fingerprints(fingerprints: dict, expected: dict) -> dict:
    keys = sorted(expected)
    return {
        key: {
            "expected": expected[key],
            "actual": fingerprints.get(key),
            "matches": fingerprints.get(key) == expected[key],
        }
        for key in keys
    }


def build_summary(
    records: list[BoxPleatFeatures],
    errors: list[dict[str, str]],
    fold_root: Path,
    args: argparse.Namespace,
    fingerprints: dict,
    verification: dict | None,
) -> dict:
    tiers = Counter(record.candidate_tier for record in records)
    return {
        "generated_by": "scripts/data/find_box_pleat_candidates.py",
        "algorithm": ALGORITHM_ID,
        "fold_root": fold_root.as_posix(),
        "fold_count": len(records) + len(errors),
        "parsed_count": len(records),
        "error_count": len(errors),
        "tier_counts": dict(sorted(tiers.items())),
        "parameters": {
            "angle_tolerance_deg": args.angle_tolerance_deg,
            "frame_step_deg": args.frame_step_deg,
            "coord_tolerance_frac": args.coord_tolerance_frac,
            "min_segments_per_line": args.min_segments_per_line,
            "review_count": args.review_count,
            "candidate_tiers": list(args.candidate_tiers),
        },
        "fingerprints": fingerprints,
        "verification": verification,
        "top_candidates": [asdict(record) for record in records[: min(20, len(records))]],
        "errors": errors,
    }


def render_fold_preview(
    cp,
    features: BoxPleatFeatures,
    image_size: int = 384,
    padding: int = 20,
) -> Image.Image:
    pixel_vertices, _ = transform_coords(cp.vertices, image_size=image_size, padding=padding)
    edge_vertices, lengths, angles = edge_geometry(cp)
    valid_length = lengths > 1e-9
    non_border = cp.assignments != 2
    crease_mask = valid_length & non_border
    _, in_axis, axis0_mask, axis1_mask = find_best_orthogonal_frame(
        angles[crease_mask],
        lengths[crease_mask],
        tolerance_deg=10.0,
        frame_step_deg=0.5,
    )

    crease_indices = np.where(crease_mask)[0]
    axis0_indices = set(int(idx) for idx in crease_indices[axis0_mask])
    axis1_indices = set(int(idx) for idx in crease_indices[axis1_mask])
    axis_indices = set(int(idx) for idx in crease_indices[in_axis])

    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    for edge_idx, (v1_idx, v2_idx) in enumerate(cp.edges):
        assignment = int(cp.assignments[edge_idx])
        if assignment == 2:
            color = (30, 30, 30)
            width = 2
        elif edge_idx in axis0_indices:
            color = (0, 133, 144)
            width = 2
        elif edge_idx in axis1_indices:
            color = (37, 88, 190)
            width = 2
        elif edge_idx in axis_indices:
            color = (75, 145, 110)
            width = 2
        else:
            color = (205, 92, 72)
            width = 1
        v1 = pixel_vertices[v1_idx]
        v2 = pixel_vertices[v2_idx]
        draw.line((float(v1[0]), float(v1[1]), float(v2[0]), float(v2[1])), fill=color, width=width)
    return image


def resize_to_tile(image: Image.Image, tile_size: int) -> Image.Image:
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    canvas.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
    return canvas


def font(size: int = 11) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_contact_sheet(
    records: list[BoxPleatFeatures],
    output_path: Path,
    columns: int,
    tile_size: int,
) -> None:
    if not records:
        return
    parser = FOLDParser()
    label_height = 76
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = font(11)
    for index, record in enumerate(records):
        cp = parser.parse(record.path)
        tile = resize_to_tile(render_fold_preview(cp, record), tile_size)
        col = index % columns
        row = index // columns
        x = col * tile_size
        y = row * (tile_size + label_height)
        sheet.paste(tile, (x, y))
        draw.rectangle(
            (x, y + tile_size, x + tile_size, y + tile_size + label_height), fill=(247, 247, 247)
        )
        label = (
            f"{record.candidate_tier} score={record.bp_score:.3f} "
            f"orth={record.orthogonal_length_ratio:.2f} bal={record.axis_balance:.2f}\n"
            f"grid={record.repeated_coord_count} theta={record.best_frame_deg:.1f} "
            f"E={record.non_border_edges}\n"
            f"{record.id[:46]}"
        )
        draw.multiline_text(
            (x + 6, y + tile_size + 5), label, fill=(20, 20, 20), font=label_font, spacing=2
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def resolve_fold_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_eval_spec(path: Path | None) -> dict:
    if path is None:
        return {}
    spec_path = path if path.is_absolute() else REPO_ROOT / path
    return json.loads(spec_path.read_text(encoding="utf-8"))


def eval_spec_defaults(spec: dict) -> dict:
    finder = spec.get("finder", {})
    params = finder.get("parameters", {})
    selection = finder.get("selection", {})
    input_spec = finder.get("input", {})
    return {
        "pattern": input_spec.get("glob", "**/*.fold"),
        "angle_tolerance_deg": params.get("angle_tolerance_deg", 10.0),
        "frame_step_deg": params.get("frame_step_deg", 0.5),
        "coord_tolerance_frac": params.get("coord_tolerance_frac", 0.003),
        "min_segments_per_line": params.get("min_segments_per_line", 2),
        "candidate_tiers": tuple(selection.get("candidate_tiers", CANDIDATE_TIERS)),
    }


def parse_candidate_tiers(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    tiers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tiers:
        raise ValueError("--candidate-tiers must include at least one tier")
    return tiers


def main(argv: list[str] | None = None) -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--eval-spec", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    eval_spec = load_eval_spec(pre_args.eval_spec)
    defaults = eval_spec_defaults(eval_spec)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-spec",
        type=Path,
        default=pre_args.eval_spec,
        help="Path to a tracked box-pleat eval spec JSON file.",
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=DEFAULT_FOLD_ROOT,
        help="Directory containing native converted .fold files.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/box_pleat_eval"),
        help="Directory for ranked candidate outputs.",
    )
    parser.add_argument(
        "--pattern", default=defaults["pattern"], help="Glob pattern under --fold-root."
    )
    parser.add_argument(
        "--angle-tolerance-deg", type=float, default=defaults["angle_tolerance_deg"]
    )
    parser.add_argument("--frame-step-deg", type=float, default=defaults["frame_step_deg"])
    parser.add_argument(
        "--coord-tolerance-frac", type=float, default=defaults["coord_tolerance_frac"]
    )
    parser.add_argument(
        "--min-segments-per-line", type=int, default=defaults["min_segments_per_line"]
    )
    parser.add_argument(
        "--candidate-tiers",
        default=",".join(defaults["candidate_tiers"]),
        help="Comma-separated candidate tiers to include in review_candidates.json.",
    )
    parser.add_argument("--review-count", type=int, default=80)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--contact-sheet-tile-size", type=int, default=230)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument(
        "--verify-spec",
        action="store_true",
        help="Compare generated path-independent fingerprints with --eval-spec expected values.",
    )
    args = parser.parse_args(argv)
    args.candidate_tiers = parse_candidate_tiers(args.candidate_tiers)

    fold_root = resolve_fold_root(args.fold_root)
    artifact_dir = (
        args.artifact_dir if args.artifact_dir.is_absolute() else REPO_ROOT / args.artifact_dir
    )
    if not fold_root.exists():
        raise SystemExit(
            f"FOLD root not found: {fold_root}\n"
            "Use --fold-root or run scripts/data/link_shared_scraped_data.sh."
        )

    fold_paths = sorted(fold_root.glob(args.pattern))
    if not fold_paths:
        raise SystemExit(f"No .fold files matched {args.pattern!r} under {fold_root}")

    records, errors = score_fold_files(
        fold_paths,
        fold_root=fold_root,
        tolerance_deg=args.angle_tolerance_deg,
        frame_step_deg=args.frame_step_deg,
        coord_tolerance_frac=args.coord_tolerance_frac,
        min_segments_per_line=args.min_segments_per_line,
    )

    candidate_records = [
        record for record in records if record.candidate_tier in args.candidate_tiers
    ]
    contact_sheet_records = candidate_records[: args.review_count]
    fingerprints = build_fingerprints(
        records,
        errors=errors,
        candidate_records=candidate_records,
        candidate_tiers=args.candidate_tiers,
    )
    verification = None
    if eval_spec.get("expected"):
        verification = compare_expected_fingerprints(fingerprints, eval_spec["expected"])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(artifact_dir / "ranked_candidates.jsonl", records)
    write_csv(artifact_dir / "ranked_candidates.csv", records)
    (artifact_dir / "review_candidates.json").write_text(
        json.dumps([asdict(record) for record in candidate_records], indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2) + "\n",
        encoding="utf-8",
    )
    if verification is not None:
        (artifact_dir / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = build_summary(
        records,
        errors,
        fold_root=fold_root,
        args=args,
        fingerprints=fingerprints,
        verification=verification,
    )
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_contact_sheet:
        save_contact_sheet(
            contact_sheet_records,
            artifact_dir / "top_candidates_contact_sheet.png",
            columns=args.contact_sheet_columns,
            tile_size=args.contact_sheet_tile_size,
        )

    print(
        json.dumps(
            {
                "fold_root": fold_root.as_posix(),
                "parsed": len(records),
                "errors": len(errors),
                "strong": sum(1 for record in records if record.candidate_tier == "strong"),
                "review": sum(1 for record in records if record.candidate_tier == "review"),
                "review_name_prior": sum(
                    1 for record in records if record.candidate_tier == "review-name-prior"
                ),
                "review_candidates": len(candidate_records),
                "fingerprints": fingerprints,
                "verification": verification,
                "artifact_dir": artifact_dir.as_posix(),
            },
            indent=2,
        )
    )
    if args.verify_spec:
        if not args.eval_spec:
            raise SystemExit("--verify-spec requires --eval-spec")
        if verification is None:
            raise SystemExit(f"No expected fingerprints found in {args.eval_spec}")
        mismatches = [key for key, result in verification.items() if not result["matches"]]
        if mismatches:
            raise SystemExit(f"Eval spec verification failed for: {', '.join(mismatches)}")


if __name__ == "__main__":
    main()
