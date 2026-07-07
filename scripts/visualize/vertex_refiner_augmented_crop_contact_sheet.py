#!/usr/bin/env python3
"""Render large contact sheets of augmented VertexRefiner crops."""

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
    extract_vertex_refiner_crop,
    render_vertex_refiner_sample,
)
from src.data.vertex_refiner_proposals import (  # noqa: E402
    ProposalConfig,
    VertexProposal,
    generate_vertex_refiner_proposals,
    select_vertex_refiner_proposals,
)
from src.models.vertex_refiner_contract import CROP_SIZE_PX  # noqa: E402

V3_LIGHT_PROFILES = (
    "clean",
    "square-symmetry",
    "line-style-light",
    "print-light",
    "print-medium-lite",
    "faint-light",
    "vertex-light-rendered",
)
BASE_RENDER_PROFILES = (
    "clean",
    "square-symmetry",
    "line-style",
    "line-style-light",
    "dark-mode",
    "print-light",
    "print-medium-lite",
    "print-medium",
    "faint-light",
    "photo-light",
    "photo-dark",
    "vertex-light-rendered",
)
CROP_KIND_COLORS = {
    "interior_gt": (45, 130, 255),
    "boundary_gt": (20, 170, 95),
    "proposal": (245, 150, 45),
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
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--variants-per-profile", type=int, default=12)
    parser.add_argument("--crops-per-render", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--profiles",
        default="v3-light",
        help=(
            "Comma-separated profiles, or preset: v3-light, base-render, all. "
            "Default is v3-light."
        ),
    )
    parser.add_argument(
        "--draw-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw target local vertices on each crop.",
    )
    parser.add_argument("--tile-scale", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("visualizations/vertex_refiner_v3_qa/augmented_crop_contact_sheet.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _resolve(args.manifest)
    out = _resolve(args.out)
    profiles = _parse_profiles(args.profiles)
    records = _select_records(
        manifest,
        split=args.split,
        count=args.records,
        max_edges=args.max_edges,
    )
    rows = build_rows(
        manifest=manifest,
        records=records,
        profiles=profiles,
        image_size=args.image_size,
        variants_per_profile=args.variants_per_profile,
        crops_per_render=args.crops_per_render,
        seed=args.seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_contact_sheet(
        rows,
        records=records,
        out=out,
        draw_gt=args.draw_gt,
        tile_scale=max(1, int(args.tile_scale)),
    )
    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "create-pattern-detector/vertex-refiner-augmented-crop-contact-sheet/v1",
                "manifest": manifest.as_posix(),
                "split": args.split,
                "max_edges": args.max_edges,
                "image_size": args.image_size,
                "crop_size": CROP_SIZE_PX,
                "profiles": profiles,
                "records": [str(record["id"]) for record in records],
                "variants_per_profile": args.variants_per_profile,
                "crops_per_render": args.crops_per_render,
                "seed": args.seed,
                "draw_gt": args.draw_gt,
                "rows": [_row_sidecar(row) for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sheet": out.as_posix(), "sidecar": sidecar.as_posix()}, indent=2))
    return 0


def build_rows(
    *,
    manifest: Path,
    records: list[dict[str, Any]],
    profiles: list[str],
    image_size: int,
    variants_per_profile: int,
    crops_per_render: int,
    seed: int,
) -> list[dict[str, Any]]:
    parser = FOLDParser()
    cps = [(record, parser.parse(resolve_fold_path(record, manifest))) for record in records]
    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        variant_count = 1 if profile == "clean" else max(1, int(variants_per_profile))
        for variant_index in range(variant_count):
            row = {
                "profile": profile,
                "variant_index": variant_index,
                "tiles": [],
            }
            for record_index, (record, cp) in enumerate(cps):
                render_seed = seed + profile_index * 100_000 + variant_index * 1_000 + record_index
                sample = render_vertex_refiner_sample(
                    cp,
                    image_size=image_size,
                    padding=_padding(image_size),
                    line_width=_line_width(image_size),
                    augment_profile=profile,
                    seed=render_seed,
                )
                proposals = _crop_proposals(sample, crops_per_render=crops_per_render)
                for crop_kind, proposal in proposals:
                    crop = extract_vertex_refiner_crop(sample, proposal, input_version="v3")
                    row["tiles"].append(
                        {
                            "record_id": str(record["id"]),
                            "render_seed": render_seed,
                            "profile": profile,
                            "selected_profile": sample.metadata["render"]["selected_profile"],
                            "crop_kind": crop_kind,
                            "proposal": proposal,
                            "crop_origin_xy": crop["crop_origin_xy"],
                            "target": crop["metadata"]["target"],
                            "render_params": sample.metadata["render"]["params"],
                            "image": _crop_rgb(sample.image, crop["crop_origin_xy"], CROP_SIZE_PX),
                            "local_vertices": crop["targets"].local_vertices,
                        }
                    )
            rows.append(row)
    return rows


def write_contact_sheet(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    out: Path,
    draw_gt: bool,
    tile_scale: int,
) -> None:
    max_tiles = max(len(row["tiles"]) for row in rows)
    tile = CROP_SIZE_PX * tile_scale
    gutter = 4
    row_label_w = 190
    header_h = 34
    width = row_label_w + max_tiles * tile + max(0, max_tiles - 1) * gutter
    height = header_h + len(rows) * tile + max(0, len(rows) - 1) * gutter
    sheet = Image.new("RGB", (width, height), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    _draw_column_headers(
        draw,
        first_row=rows[0],
        tile=tile,
        gutter=gutter,
        row_label_w=row_label_w,
    )
    for row_index, row in enumerate(rows):
        y = header_h + row_index * (tile + gutter)
        draw.multiline_text((6, y + 6), _row_label(row), fill=(25, 25, 25), spacing=2)
        for col_index, tile_data in enumerate(row["tiles"]):
            x = row_label_w + col_index * (tile + gutter)
            panel = Image.fromarray(tile_data["image"])
            if draw_gt:
                panel = _draw_gt_vertices(panel, tile_data["local_vertices"], tile_data["crop_kind"])
            if tile_scale != 1:
                panel = panel.resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(panel, (x, y))
    sheet.save(out)


def _draw_column_headers(
    draw: ImageDraw.ImageDraw,
    *,
    first_row: dict[str, Any],
    tile: int,
    gutter: int,
    row_label_w: int,
) -> None:
    draw.text((6, 4), "profile / sampled params", fill=(25, 25, 25))
    for col_index, tile_data in enumerate(first_row["tiles"]):
        x = row_label_w + col_index * (tile + gutter)
        label = f"{str(tile_data['record_id'])[:12]}\n{tile_data['crop_kind']}"
        draw.multiline_text((x + 2, 2), label, fill=(25, 25, 25), spacing=1)


def _row_label(row: dict[str, Any]) -> str:
    selected = [str(tile["selected_profile"]) for tile in row["tiles"]]
    profile = str(row["profile"])
    label = profile
    if selected and len(set(selected)) == 1 and selected[0] != profile:
        label = f"{profile} -> {selected[0]}"
    elif selected and len(set(selected)) > 1:
        label = f"{profile} -> mixed"
    if int(row["variant_index"]) > 0:
        label = f"{label}\nvariant {int(row['variant_index']) + 1}"
    return label


def _draw_gt_vertices(
    panel: Image.Image,
    local_vertices: np.ndarray,
    crop_kind: str,
) -> Image.Image:
    result = panel.copy()
    draw = ImageDraw.Draw(result)
    color = CROP_KIND_COLORS.get(crop_kind, (245, 150, 45))
    for x, y in np.asarray(local_vertices, dtype=np.float32).reshape(-1, 2):
        if not (0 <= x < CROP_SIZE_PX and 0 <= y < CROP_SIZE_PX):
            continue
        radius = 2
        draw.ellipse(
            (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
            outline=color,
            width=1,
        )
        draw.line((float(x) - 3, float(y), float(x) + 3, float(y)), fill=color, width=1)
        draw.line((float(x), float(y) - 3, float(x), float(y) + 3), fill=color, width=1)
    return result


def _crop_proposals(sample: Any, *, crops_per_render: int) -> list[tuple[str, VertexProposal]]:
    requested = max(1, int(crops_per_render))
    proposals: list[tuple[str, VertexProposal]] = [
        ("interior_gt", _centered_vertex_proposal(sample.pixel_vertices, sample.edges)),
    ]
    if requested >= 2:
        proposals.append(
            (
                "boundary_gt",
                _boundary_vertex_proposal(sample.pixel_vertices, sample.edges, sample.square_frame),
            )
        )
    if len(proposals) >= requested:
        return proposals[:requested]
    image_proposals = generate_vertex_refiner_proposals(
        source_ink_probability=sample.source_ink_probability,
        square_frame=sample.square_frame,
        config=ProposalConfig(crop_size=CROP_SIZE_PX),
    )
    selected = select_vertex_refiner_proposals(
        image_proposals,
        max_count=requested * 3,
        crop_size=CROP_SIZE_PX,
        image_shape=sample.source_ink_probability.shape,
    )
    for proposal in selected:
        if any(_distance(proposal, existing) < 8.0 for _, existing in proposals):
            continue
        proposals.append(("proposal", proposal))
        if len(proposals) >= requested:
            break
    return proposals


def _centered_vertex_proposal(vertices: np.ndarray, edges: np.ndarray) -> VertexProposal:
    vertices = np.asarray(vertices, dtype=np.float32)
    degrees = _vertex_degrees(vertices, edges)
    center = np.asarray([float(vertices[:, 0].mean()), float(vertices[:, 1].mean())], dtype=np.float32)
    candidates = np.where(degrees >= 2)[0]
    if candidates.size == 0:
        candidates = np.arange(len(vertices))
    vertex_index = int(candidates[np.argmin(np.linalg.norm(vertices[candidates] - center[None, :], axis=1))])
    x, y = vertices[vertex_index]
    return VertexProposal(float(x), float(y), 1.0, ("visual_qa_interior_vertex",))


def _boundary_vertex_proposal(vertices: np.ndarray, edges: np.ndarray, square_frame: Any) -> VertexProposal:
    vertices = np.asarray(vertices, dtype=np.float32)
    degrees = _vertex_degrees(vertices, edges)
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
            np.hypot(
                vertices[:, 0] - float(square_frame.x_min),
                vertices[:, 1] - float(square_frame.y_min),
            ),
            np.hypot(
                vertices[:, 0] - float(square_frame.x_max),
                vertices[:, 1] - float(square_frame.y_min),
            ),
            np.hypot(
                vertices[:, 0] - float(square_frame.x_max),
                vertices[:, 1] - float(square_frame.y_max),
            ),
            np.hypot(
                vertices[:, 0] - float(square_frame.x_min),
                vertices[:, 1] - float(square_frame.y_max),
            ),
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


def _vertex_degrees(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    degrees = np.zeros((len(vertices),), dtype=np.int32)
    for v0, v1 in np.asarray(edges, dtype=np.int64):
        degrees[int(v0)] += 1
        degrees[int(v1)] += 1
    return degrees


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


def _row_sidecar(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": row["profile"],
        "variant_index": row["variant_index"],
        "tiles": [
            {
                "record_id": tile["record_id"],
                "render_seed": tile["render_seed"],
                "profile": tile["profile"],
                "selected_profile": tile["selected_profile"],
                "crop_kind": tile["crop_kind"],
                "proposal": {
                    "x": tile["proposal"].x,
                    "y": tile["proposal"].y,
                    "score": tile["proposal"].score,
                    "provenance": list(tile["proposal"].provenance),
                },
                "crop_origin_xy": list(tile["crop_origin_xy"]),
                "target": tile["target"],
                "render_params": tile["render_params"],
            }
            for tile in row["tiles"]
        ],
    }


def _parse_profiles(value: str) -> list[str]:
    stripped = value.strip()
    if stripped == "v3-light":
        return list(V3_LIGHT_PROFILES)
    if stripped == "base-render":
        return list(BASE_RENDER_PROFILES)
    if stripped == "all":
        return list(AUGMENT_PROFILES)
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    unsupported = [profile for profile in profiles if profile not in AUGMENT_PROFILES]
    if unsupported:
        expected = ", ".join(AUGMENT_PROFILES)
        raise SystemExit(f"Unsupported profiles: {', '.join(unsupported)}. Expected one of: {expected}")
    return profiles


def _select_records(
    manifest: Path,
    *,
    split: str,
    count: int,
    max_edges: int | None,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in load_manifest_records(manifest)
        if record.get("split") == split and (max_edges is None or int(record["edges"]) <= max_edges)
    ]
    if not records:
        raise SystemExit(f"No records found in {manifest} for split={split}")
    return records[: max(1, int(count))]


def _distance(left: VertexProposal, right: VertexProposal) -> float:
    return float(np.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y)))


def _padding(image_size: int) -> int:
    return max(8, int(32 * image_size / 1024))


def _line_width(image_size: int) -> int:
    return max(1, int(2 * image_size / 768))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
