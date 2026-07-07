#!/usr/bin/env python3
"""Render V3 VertexRefiner visual QA sheets before training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES  # noqa: E402
from src.data.cpline_dataset import load_manifest_records, resolve_fold_path  # noqa: E402
from src.data.fold_parser import FOLDParser  # noqa: E402
from src.data.vertex_refiner_dataset import (  # noqa: E402
    crop_array,
    extract_vertex_refiner_crop,
    render_vertex_refiner_sample,
)
from src.data.vertex_refiner_proposals import VertexProposal  # noqa: E402
from src.data.vertex_refiner_targets import (  # noqa: E402
    distance_to_ink_map,
    grayscale_image,
    source_ink_probability,
)
from src.models.vertex_refiner_contract import CROP_SIZE_PX, V3_INPUT_CHANNELS  # noqa: E402

DEFAULT_PROFILES = (
    "clean",
    "line-style-light",
    "print-light",
    "print-medium-lite",
    "faint-light",
    "vertex-light-rendered",
)
DEFAULT_ACCEPTED_IMAGES = Path(
    "/Users/zacharymarion/Documents/datasets/create-pattern-detector/scraped/final/accepted_images"
)
CHANNEL_DISPLAY_RANGES = {
    "source_orientation_cos2": (-1.0, 1.0),
    "source_orientation_sin2": (-1.0, 1.0),
    "signed_distance_to_frame": (-1.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated synthetic profiles for the V3 channel sheet.",
    )
    parser.add_argument("--accepted-images-dir", type=Path, default=DEFAULT_ACCEPTED_IMAGES)
    parser.add_argument("--accepted-count", type=int, default=8)
    parser.add_argument("--accepted-crops-per-image", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/vertex_refiner_v3_qa"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _parse_profiles(args.profiles)

    channel_sheet, channel_sidecar = write_input_channel_sheet(
        manifest=manifest,
        split=args.split,
        max_edges=args.max_edges,
        image_size=args.image_size,
        seed=args.seed,
        profiles=profiles,
        output_dir=output_dir,
    )
    accepted_sheet, accepted_sidecar = write_accepted_image_crop_sheet(
        accepted_images_dir=args.accepted_images_dir,
        output_dir=output_dir,
        count=args.accepted_count,
        crops_per_image=args.accepted_crops_per_image,
    )
    print(
        json.dumps(
            {
                "input_channel_sheet": channel_sheet.as_posix(),
                "input_channel_sidecar": channel_sidecar.as_posix(),
                "accepted_image_crop_sheet": (
                    None if accepted_sheet is None else accepted_sheet.as_posix()
                ),
                "accepted_image_crop_sidecar": (
                    None if accepted_sidecar is None else accepted_sidecar.as_posix()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def write_input_channel_sheet(
    *,
    manifest: Path,
    split: str,
    max_edges: int,
    image_size: int,
    seed: int,
    profiles: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    record = _select_record(manifest, split=split, max_edges=max_edges)
    cp = FOLDParser().parse(resolve_fold_path(record, manifest))
    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        sample = render_vertex_refiner_sample(
            cp,
            image_size=image_size,
            padding=_padding(image_size),
            line_width=_line_width(image_size),
            augment_profile=profile,
            seed=seed + profile_index * 1009,
        )
        proposals = (
            ("interior", _centered_vertex_proposal(sample.pixel_vertices, sample.edges)),
            (
                "boundary",
                _boundary_vertex_proposal(
                    sample.pixel_vertices,
                    sample.edges,
                    sample.square_frame,
                ),
            ),
        )
        for crop_kind, proposal in proposals:
            crop = extract_vertex_refiner_crop(sample, proposal, input_version="v3")
            rows.append(
                {
                    "profile": profile,
                    "selected_profile": sample.metadata["render"]["selected_profile"],
                    "crop_kind": crop_kind,
                    "seed": seed + profile_index * 1009,
                    "proposal": {
                        "x": proposal.x,
                        "y": proposal.y,
                        "provenance": list(proposal.provenance),
                    },
                    "input": crop["input"],
                }
            )

    out = output_dir / "input_channels.png"
    _write_channel_grid(rows, out)
    sidecar = output_dir / "input_channels.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "create-pattern-detector/vertex-refiner-v3-input-channel-qa/v1",
                "manifest": manifest.as_posix(),
                "record_id": str(record["id"]),
                "split": split,
                "image_size": image_size,
                "channels": [channel.name for channel in V3_INPUT_CHANNELS],
                "rows": [
                    {
                        "profile": row["profile"],
                        "selected_profile": row["selected_profile"],
                        "crop_kind": row["crop_kind"],
                        "seed": row["seed"],
                        "proposal": row["proposal"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out, sidecar


def write_accepted_image_crop_sheet(
    *,
    accepted_images_dir: Path,
    output_dir: Path,
    count: int,
    crops_per_image: int,
) -> tuple[Path | None, Path | None]:
    image_paths = _accepted_image_paths(accepted_images_dir, count=count)
    if not image_paths:
        sidecar = output_dir / "accepted_image_crops.skipped.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema": "create-pattern-detector/vertex-refiner-v3-accepted-crop-qa/v1",
                    "accepted_images_dir": accepted_images_dir.as_posix(),
                    "skipped": "no_images_found",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return None, sidecar

    rows: list[dict[str, Any]] = []
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        gray = grayscale_image(image)
        ink = source_ink_probability(image)
        distance = distance_to_ink_map(ink, normalize_by_px=CROP_SIZE_PX)
        origins = _select_ink_rich_crop_origins(ink, count=crops_per_image)
        for crop_index, origin in enumerate(origins):
            rows.append(
                {
                    "path": path,
                    "crop_index": crop_index,
                    "origin": origin,
                    "raw": _crop_rgb(image, origin, CROP_SIZE_PX),
                    "gray": crop_array(gray, origin, CROP_SIZE_PX, pad_value=1.0),
                    "ink": crop_array(ink, origin, CROP_SIZE_PX, pad_value=0.0),
                    "distance": crop_array(distance, origin, CROP_SIZE_PX, pad_value=1.0),
                }
            )

    out = output_dir / "accepted_image_crops.png"
    _write_accepted_grid(rows, out)
    sidecar = output_dir / "accepted_image_crops.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "create-pattern-detector/vertex-refiner-v3-accepted-crop-qa/v1",
                "accepted_images_dir": accepted_images_dir.as_posix(),
                "crop_size": CROP_SIZE_PX,
                "rows": [
                    {
                        "path": row["path"].as_posix(),
                        "crop_index": row["crop_index"],
                        "origin": list(row["origin"]),
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out, sidecar


def _write_channel_grid(rows: list[dict[str, Any]], out: Path) -> None:
    channel_names = [channel.name for channel in V3_INPUT_CHANNELS]
    scale = 2
    tile = CROP_SIZE_PX * scale
    gutter = 6
    label_h = 22
    row_label_w = 150
    width = row_label_w + len(channel_names) * tile + (len(channel_names) - 1) * gutter
    height = label_h + len(rows) * tile + max(0, len(rows) - 1) * gutter
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for col, name in enumerate(channel_names):
        x = row_label_w + col * (tile + gutter)
        draw.text((x + 2, 3), _short_channel_name(name), fill=(30, 30, 30))
    for row_index, row in enumerate(rows):
        y = label_h + row_index * (tile + gutter)
        row_label = f"{row['profile']}\n{row['crop_kind']}"
        if row["selected_profile"] != row["profile"]:
            row_label = f"{row_label} -> {row['selected_profile']}"
        draw.multiline_text((4, y + 4), row_label, fill=(30, 30, 30), spacing=2)
        tensor = np.asarray(row["input"], dtype=np.float32)
        for col, name in enumerate(channel_names):
            x = row_label_w + col * (tile + gutter)
            panel = _channel_to_image(tensor[col], name).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(panel, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _write_accepted_grid(rows: list[dict[str, Any]], out: Path) -> None:
    columns = ("raw", "gray", "ink", "distance")
    scale = 2
    tile = CROP_SIZE_PX * scale
    gutter = 6
    label_h = 20
    row_label_w = 190
    width = row_label_w + len(columns) * tile + (len(columns) - 1) * gutter
    height = label_h + len(rows) * tile + max(0, len(rows) - 1) * gutter
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for col, name in enumerate(columns):
        x = row_label_w + col * (tile + gutter)
        draw.text((x + 2, 3), name, fill=(30, 30, 30))
    for row_index, row in enumerate(rows):
        y = label_h + row_index * (tile + gutter)
        label = f"{row['path'].name}\n{row['origin'][0]},{row['origin'][1]}"
        draw.multiline_text((4, y + 4), label[:72], fill=(30, 30, 30), spacing=2)
        for col, name in enumerate(columns):
            x = row_label_w + col * (tile + gutter)
            if name == "raw":
                panel = Image.fromarray(row[name]).resize((tile, tile), Image.Resampling.NEAREST)
            else:
                panel = _channel_to_image(row[name], name).resize(
                    (tile, tile),
                    Image.Resampling.NEAREST,
                )
            sheet.paste(panel, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _channel_to_image(channel: np.ndarray, name: str) -> Image.Image:
    lo, hi = CHANNEL_DISPLAY_RANGES.get(name, (0.0, 1.0))
    values = (np.asarray(channel, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    gray = (np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")


def _crop_rgb(image: np.ndarray, origin: tuple[int, int], crop_size: int) -> np.ndarray:
    x0, y0 = origin
    output = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
    src_x0 = max(0, int(x0))
    src_y0 = max(0, int(y0))
    src_x1 = min(image.shape[1], int(x0) + crop_size)
    src_y1 = min(image.shape[0], int(y0) + crop_size)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output
    dst_x0 = src_x0 - int(x0)
    dst_y0 = src_y0 - int(y0)
    output[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return output


def _select_ink_rich_crop_origins(ink: np.ndarray, *, count: int) -> list[tuple[int, int]]:
    height, width = ink.shape
    crop_size = CROP_SIZE_PX
    if height < crop_size or width < crop_size:
        return [(int((width - crop_size) / 2), int((height - crop_size) / 2))]
    stride = max(16, crop_size // 2)
    scored: list[tuple[float, tuple[int, int]]] = []
    for y in range(0, height - crop_size + 1, stride):
        for x in range(0, width - crop_size + 1, stride):
            crop = ink[y : y + crop_size, x : x + crop_size]
            score = float(crop.mean())
            if score > 0.01:
                scored.append((score, (x, y)))
    if not scored:
        return [(int((width - crop_size) / 2), int((height - crop_size) / 2))]
    selected: list[tuple[int, int]] = []
    for _, origin in sorted(scored, key=lambda item: item[0], reverse=True):
        if all(abs(origin[0] - x) >= crop_size or abs(origin[1] - y) >= crop_size for x, y in selected):
            selected.append(origin)
        if len(selected) >= max(1, int(count)):
            break
    return selected or [scored[0][1]]


def _centered_vertex_proposal(vertices: np.ndarray, edges: np.ndarray) -> VertexProposal:
    vertices = np.asarray(vertices, dtype=np.float32)
    degrees = np.zeros((len(vertices),), dtype=np.int32)
    for v0, v1 in np.asarray(edges, dtype=np.int64):
        degrees[int(v0)] += 1
        degrees[int(v1)] += 1
    center = np.asarray([float(vertices[:, 0].mean()), float(vertices[:, 1].mean())], dtype=np.float32)
    candidates = np.where(degrees >= 2)[0]
    if candidates.size == 0:
        candidates = np.arange(len(vertices))
    vertex_index = int(candidates[np.argmin(np.linalg.norm(vertices[candidates] - center[None, :], axis=1))])
    x, y = vertices[vertex_index]
    return VertexProposal(float(x), float(y), 1.0, ("visual_qa_gt_vertex",))


def _boundary_vertex_proposal(vertices: np.ndarray, edges: np.ndarray, square_frame: Any) -> VertexProposal:
    vertices = np.asarray(vertices, dtype=np.float32)
    degrees = np.zeros((len(vertices),), dtype=np.int32)
    for v0, v1 in np.asarray(edges, dtype=np.int64):
        degrees[int(v0)] += 1
        degrees[int(v1)] += 1
    frame_distances = np.minimum.reduce(
        [
            np.abs(vertices[:, 0] - float(square_frame.x_min)),
            np.abs(vertices[:, 0] - float(square_frame.x_max)),
            np.abs(vertices[:, 1] - float(square_frame.y_min)),
            np.abs(vertices[:, 1] - float(square_frame.y_max)),
        ]
    )
    corner_distances = np.minimum.reduce(
        [
            np.hypot(vertices[:, 0] - float(square_frame.x_min), vertices[:, 1] - float(square_frame.y_min)),
            np.hypot(vertices[:, 0] - float(square_frame.x_max), vertices[:, 1] - float(square_frame.y_min)),
            np.hypot(vertices[:, 0] - float(square_frame.x_max), vertices[:, 1] - float(square_frame.y_max)),
            np.hypot(vertices[:, 0] - float(square_frame.x_min), vertices[:, 1] - float(square_frame.y_max)),
        ]
    )
    candidates = np.where((degrees >= 2) & (frame_distances <= 2.0) & (corner_distances > 3.0))[0]
    if candidates.size == 0:
        candidates = np.where((degrees >= 1) & (frame_distances <= 2.0))[0]
    if candidates.size == 0:
        candidates = np.arange(len(vertices))
    center = np.asarray([float(vertices[:, 0].mean()), float(vertices[:, 1].mean())], dtype=np.float32)
    vertex_index = int(candidates[np.argmin(np.linalg.norm(vertices[candidates] - center[None, :], axis=1))])
    x, y = vertices[vertex_index]
    return VertexProposal(float(x), float(y), 1.0, ("visual_qa_boundary_vertex",))


def _accepted_image_paths(directory: Path, *, count: int) -> list[Path]:
    if not directory.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)[
        : max(0, int(count))
    ]


def _select_record(manifest: Path, *, split: str, max_edges: int) -> dict[str, Any]:
    for record in load_manifest_records(manifest):
        if record.get("split") == split and int(record["edges"]) <= int(max_edges):
            return record
    raise SystemExit(f"No records found in {manifest} for split={split}")


def _parse_profiles(value: str) -> list[str]:
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    unsupported = [profile for profile in profiles if profile not in AUGMENT_PROFILES]
    if unsupported:
        expected = ", ".join(AUGMENT_PROFILES)
        raise SystemExit(f"Unsupported profiles: {', '.join(unsupported)}. Expected one of: {expected}")
    return profiles


def _short_channel_name(name: str) -> str:
    return {
        "source_ink_probability": "ink",
        "source_distance_to_ink": "dist",
        "source_orientation_cos2": "cos2",
        "source_orientation_sin2": "sin2",
        "signed_distance_to_frame": "frame dist",
        "frame_edge_mask": "frame",
        "inside_paper_mask": "inside",
        "boundary_contact_prior": "bd prior",
        "crop_x_normalized": "crop x",
        "crop_y_normalized": "crop y",
    }.get(name, name.removeprefix("image_"))


def _padding(image_size: int) -> int:
    return max(8, int(32 * image_size / 1024))


def _line_width(image_size: int) -> int:
    return max(1, int(2 * image_size / 768))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
