#!/usr/bin/env python3
"""Build deterministic Phase 2 real-FOLD fixture manifests and preflight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fold_parser import FOLDParser, transform_coords  # noqa: E402


ASSIGNMENT_NAMES = {0: "M", 1: "V", 2: "B", 3: "U"}
SEG_COLORS = np.array(
    [
        [255, 255, 255],
        [220, 40, 40],
        [40, 80, 220],
        [10, 10, 10],
        [140, 140, 140],
    ],
    dtype=np.uint8,
)

BUCKETS = (
    ("tiny", 0, 99),
    ("small", 100, 249),
    ("medium", 250, 599),
    ("large", 600, 1499),
    ("stress", 1500, math.inf),
)

SMOKE_TARGETS = {
    "tiny": 2,
    "small": 2,
    "medium": 3,
    "large": 3,
    "stress": 2,
}

CURATED_TARGETS = {
    "tiny": 8,
    "small": 12,
    "medium": 16,
    "large": 16,
    "stress": 8,
}


@dataclass(frozen=True)
class FoldRecord:
    id: str
    path: str
    content_sha256: str
    vertices: int
    edges: int
    assignments: dict[str, int]
    bucket: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_for_edges(edge_count: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= edge_count <= hi:
            return name
    raise ValueError(f"No bucket for edge count {edge_count}")


def stable_id(path: Path) -> str:
    return path.stem


def load_records(fold_root: Path, repo_root: Path) -> list[FoldRecord]:
    parser = FOLDParser()
    records: list[FoldRecord] = []
    for path in sorted(fold_root.glob("*.fold")):
        cp = parser.parse(path)
        counts = Counter(ASSIGNMENT_NAMES[int(value)] for value in cp.assignments)
        rel_path = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()
        edge_count = int(cp.num_edges)
        records.append(
            FoldRecord(
                id=stable_id(path),
                path=rel_path,
                content_sha256=sha256_file(path),
                vertices=int(cp.num_vertices),
                edges=edge_count,
                assignments=dict(sorted(counts.items())),
                bucket=bucket_for_edges(edge_count),
            )
        )
    return records


def select_evenly(records: list[FoldRecord], targets: dict[str, int]) -> list[FoldRecord]:
    by_bucket: dict[str, list[FoldRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.bucket].append(record)

    selected: list[FoldRecord] = []
    for bucket, target in targets.items():
        bucket_records = sorted(
            by_bucket.get(bucket, []),
            key=lambda item: (item.edges, item.content_sha256, item.id),
        )
        if not bucket_records:
            continue
        if len(bucket_records) <= target:
            selected.extend(bucket_records)
            continue
        indices = np.linspace(0, len(bucket_records) - 1, target)
        selected.extend(bucket_records[int(round(idx))] for idx in indices)

    return sorted(selected, key=lambda item: (bucket_sort_key(item.bucket), item.edges, item.id))


def take_evenly(records: list[FoldRecord], count: int) -> list[FoldRecord]:
    if count <= 0 or len(records) <= count:
        return records
    indices = np.linspace(0, len(records) - 1, count)
    return [records[int(round(index))] for index in indices]


def bucket_sort_key(bucket: str) -> int:
    return [name for name, _, _ in BUCKETS].index(bucket)


def write_manifest(path: Path, name: str, records: list[FoldRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": "Phase 2 deterministic vectorizer real-FOLD fixture manifest.",
        "generated_by": "scripts/vectorization/build_phase2_real_fixtures.py",
        "generated_from": "data/output/scraped/final/native_files/fold",
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_stats(records: list[FoldRecord]) -> dict:
    edge_values = [record.edges for record in records]
    vertex_values = [record.vertices for record in records]
    assignments = Counter()
    buckets = Counter()
    for record in records:
        assignments.update(record.assignments)
        buckets[record.bucket] += 1

    def percentile(values: list[int], pct: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.array(values), pct))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "edge_count": {
            "min": min(edge_values) if edge_values else 0,
            "p50": percentile(edge_values, 50),
            "p90": percentile(edge_values, 90),
            "max": max(edge_values) if edge_values else 0,
            "mean": float(np.mean(edge_values)) if edge_values else 0.0,
        },
        "vertex_count": {
            "min": min(vertex_values) if vertex_values else 0,
            "p50": percentile(vertex_values, 50),
            "p90": percentile(vertex_values, 90),
            "max": max(vertex_values) if vertex_values else 0,
            "mean": float(np.mean(vertex_values)) if vertex_values else 0.0,
        },
        "bucket_counts": dict(sorted(buckets.items(), key=lambda item: bucket_sort_key(item[0]))),
        "assignment_counts": dict(sorted(assignments.items())),
    }


def save_histogram(records: list[FoldRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_values = [record.edges for record in records]

    width, height = 1200, 720
    margin_left, margin_right = 90, 35
    margin_top, margin_bottom = 70, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    text_font = font()

    if not edge_values:
        draw.text((margin_left, margin_top), "No records", fill=(20, 20, 20), font=text_font)
        image.save(output_path)
        return

    counts, bin_edges = np.histogram(edge_values, bins=40)
    max_count = max(int(counts.max()), 1)

    draw.text(
        (margin_left, 25),
        "Phase 2 Real FOLD Complexity Distribution",
        fill=(20, 20, 20),
        font=text_font,
    )
    draw.line(
        (margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h),
        fill=(30, 30, 30),
        width=2,
    )
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(30, 30, 30), width=2)

    bar_w = plot_w / len(counts)
    for index, count in enumerate(counts):
        x0 = margin_left + index * bar_w
        x1 = margin_left + (index + 1) * bar_w - 1
        y1 = margin_top + plot_h
        y0 = y1 - (int(count) / max_count) * plot_h
        draw.rectangle((x0, y0, x1, y1), fill=(65, 105, 168), outline=(255, 255, 255))

    for tick in np.linspace(0, max_count, 5):
        y = margin_top + plot_h - (tick / max_count) * plot_h
        draw.line((margin_left - 5, y, margin_left + plot_w, y), fill=(225, 225, 225))
        draw.text((12, y - 7), str(int(round(tick))), fill=(45, 45, 45), font=text_font)

    for tick in np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 6):
        x = margin_left + ((tick - bin_edges[0]) / max(float(bin_edges[-1] - bin_edges[0]), 1.0)) * plot_w
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 5), fill=(30, 30, 30))
        draw.text((x - 20, margin_top + plot_h + 12), str(int(round(tick))), fill=(45, 45, 45), font=text_font)

    draw.text((margin_left + plot_w // 2 - 55, height - 45), "Edges per FOLD file", fill=(20, 20, 20), font=text_font)
    draw.text((12, margin_top - 30), "File count", fill=(20, 20, 20), font=text_font)
    image.save(output_path)


def render_fold_labels(cp, image_size: int, padding: int) -> Image.Image:
    """Render a simple GT-style assignment image without OpenCV."""
    pixel_vertices, _ = transform_coords(cp.vertices, image_size=image_size, padding=padding)
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    for edge_idx, (v1_idx, v2_idx) in enumerate(cp.edges):
        assignment = int(cp.assignments[edge_idx])
        color = tuple(int(value) for value in SEG_COLORS[assignment + 1])
        v1 = pixel_vertices[v1_idx]
        v2 = pixel_vertices[v2_idx]
        draw.line((float(v1[0]), float(v1[1]), float(v2[0]), float(v2[1])), fill=color, width=2)

    for vertex in pixel_vertices:
        x, y = float(vertex[0]), float(vertex[1])
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 180, 0))
    return image


def resize_to_tile(image: Image.Image, size: int) -> Image.Image:
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def save_contact_sheet(
    records: Iterable[FoldRecord],
    repo_root: Path,
    output_path: Path,
    image_size: int = 384,
    tile_size: int = 220,
    columns: int = 4,
) -> None:
    records = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No records to render")

    parser = FOLDParser()
    padding = max(12, image_size // 32)

    label_height = 48
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    text_font = font()

    for index, record in enumerate(records):
        fold_path = resolve_record_path(record, repo_root)
        cp = parser.parse(fold_path)
        tile = resize_to_tile(render_fold_labels(cp, image_size=image_size, padding=padding), tile_size)

        col = index % columns
        row = index // columns
        x = col * tile_size
        y = row * (tile_size + label_height)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y + tile_size, x + tile_size, y + tile_size + label_height), fill=(248, 248, 248))
        label = f"{record.bucket} | V {record.vertices} E {record.edges}\n{record.id[:34]}"
        draw.multiline_text((x + 6, y + tile_size + 5), label, fill=(20, 20, 20), font=text_font, spacing=2)

    sheet.save(output_path)


def resolve_record_path(record: FoldRecord, repo_root: Path) -> Path:
    path = Path(record.path)
    if path.is_absolute():
        return path
    return repo_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("data/output/scraped/final/native_files/fold"),
        help="Directory containing final real .fold files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("fixtures/phase2_real_folds"),
        help="Tracked output directory for small deterministic manifests.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("visualizations/phase2_vectorizer/preflight"),
        help="Ignored output directory for preflight artifacts.",
    )
    parser.add_argument(
        "--contact-sheet-count",
        type=int,
        default=32,
        help="Number of curated fixtures to render in the preflight contact sheet.",
    )
    args = parser.parse_args()

    fold_root = args.fold_root if args.fold_root.is_absolute() else REPO_ROOT / args.fold_root
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else REPO_ROOT / args.manifest_dir
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else REPO_ROOT / args.artifact_dir

    if not fold_root.exists():
        raise SystemExit(
            f"FOLD root not found: {fold_root}\n"
            "Run scripts/data/link_shared_scraped_data.sh or set CP_SCRAPED_DATASET first."
        )

    records = load_records(fold_root, REPO_ROOT)
    if not records:
        raise SystemExit(f"No .fold files found in {fold_root}")

    smoke = select_evenly(records, SMOKE_TARGETS)
    curated_gate = select_evenly(records, CURATED_TARGETS)
    full_stress = sorted(records, key=lambda item: (bucket_sort_key(item.bucket), item.edges, item.id))

    write_manifest(manifest_dir / "smoke.json", "smoke", smoke)
    write_manifest(manifest_dir / "curated_gate.json", "curated_gate", curated_gate)
    write_manifest(manifest_dir / "full_stress.json", "full_stress", full_stress)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    stats = build_stats(records)
    (artifact_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    save_histogram(records, artifact_dir / "complexity_histogram.png")
    save_contact_sheet(
        take_evenly(curated_gate, args.contact_sheet_count),
        REPO_ROOT,
        artifact_dir / "curated_gate_contact_sheet.png",
    )

    print(json.dumps({
        "records": len(records),
        "smoke": len(smoke),
        "curated_gate": len(curated_gate),
        "full_stress": len(full_stress),
        "manifest_dir": manifest_dir.as_posix(),
        "artifact_dir": artifact_dir.as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
