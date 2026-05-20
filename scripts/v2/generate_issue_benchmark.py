#!/usr/bin/env python3
"""Generate synthetic V2 issue examples for visual benchmark review."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.cpline_dataset import render_cpline_sample
from src.data.fold_parser import CreasePattern
from src.data.v2_labels import build_v2_label_sidecar, draw_v2_label_overlay

ASSIGNMENT_COLORS = {
    0: (220, 40, 40),
    1: (40, 80, 220),
    2: (0, 0, 0),
    3: (120, 120, 120),
}
MASK_COLOR = (255, 0, 180)


@dataclass(frozen=True)
class IssueSpec:
    code: str
    title: str
    description: str
    pattern_factory: Callable[[int], CreasePattern]
    renderer: Callable[[CreasePattern, int, int, np.random.Generator], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/v2_issue_benchmark"),
        help="Directory for generated issue images and manifest.",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--examples-per-issue", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument(
        "--contact-sheet-width",
        type=int,
        default=360,
        help="Displayed cell width in the overview contact sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    contact_items: list[dict[str, Any]] = []
    issue_specs = build_issue_specs()
    rng = np.random.default_rng(args.seed)

    for issue_index, spec in enumerate(issue_specs):
        issue_dir = examples_dir / spec.code
        issue_dir.mkdir(parents=True, exist_ok=True)
        for example_index in range(args.examples_per_issue):
            example_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            example_rng = np.random.default_rng(example_seed)
            cp = spec.pattern_factory(example_index)
            rendered = spec.renderer(cp, args.image_size, example_index, example_rng)
            sample_id = f"{spec.code}_{example_index:02d}"

            clean_path = issue_dir / f"{sample_id}_clean.png"
            issue_path = issue_dir / f"{sample_id}_issue.png"
            target_path = issue_dir / f"{sample_id}_target_overlay.png"
            mask_path = issue_dir / f"{sample_id}_oracle_mask.png"
            sidecar_path = issue_dir / f"{sample_id}.json"
            label_path = issue_dir / f"{sample_id}_v2_labels.json"
            label_overlay_path = issue_dir / f"{sample_id}_v2_labels_overlay.png"

            write_rgb(clean_path, rendered["clean"])
            write_rgb(issue_path, rendered["issue"])
            write_rgb(target_path, rendered["target_overlay"])
            if rendered.get("oracle_mask") is not None:
                write_rgb(mask_path, mask_to_rgb(rendered["oracle_mask"]))
                mask_rel: str | None = str(mask_path.relative_to(output_dir))
            else:
                mask_rel = None
            mask_kind = oracle_mask_kind(spec.code, mask_rel)
            label_sidecar = build_v2_label_sidecar(
                sample_id=sample_id,
                issue=spec.code,
                image_size=args.image_size,
                vertices=rendered["vertices"],
                edges=rendered["edges"],
                assignments=rendered["assignments"],
                metadata=rendered.get("metadata", {}),
                oracle_mask=mask_rel,
                oracle_mask_kind=mask_kind,
            )
            label_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")
            write_rgb(label_overlay_path, draw_v2_label_overlay(rendered["issue"], label_sidecar))
            sidecar = {
                "id": sample_id,
                "issue": spec.code,
                "title": spec.title,
                "description": spec.description,
                "seed": example_seed,
                "image_size": args.image_size,
                "vertices": rendered["vertices"].round(3).tolist(),
                "edges": rendered["edges"].astype(int).tolist(),
                "assignments": rendered["assignments"].astype(int).tolist(),
                "v2_label_sidecar": str(label_path.relative_to(output_dir)),
                "metadata": rendered.get("metadata", {}),
            }
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

            row = {
                "id": sample_id,
                "issue": spec.code,
                "title": spec.title,
                "description": spec.description,
                "seed": example_seed,
                "image_size": args.image_size,
                "clean_image": str(clean_path.relative_to(output_dir)),
                "issue_image": str(issue_path.relative_to(output_dir)),
                "target_overlay": str(target_path.relative_to(output_dir)),
                "oracle_mask": mask_rel,
                "oracle_mask_kind": mask_kind,
                "sidecar": str(sidecar_path.relative_to(output_dir)),
                "v2_label_sidecar": str(label_path.relative_to(output_dir)),
                "v2_label_overlay": str(label_overlay_path.relative_to(output_dir)),
                "metadata": rendered.get("metadata", {}),
            }
            rows.append(row)
            contact_items.append(
                {
                    **row,
                    "clean_path": clean_path,
                    "issue_path": issue_path,
                    "target_path": target_path,
                    "mask_path": mask_path if mask_rel is not None else None,
                    "label_path": label_overlay_path,
                }
            )

    manifest = {
        "schema": "cp-detector/v2-issue-benchmark/v1",
        "note": (
            "Synthetic issue benchmark for V2.0. No real-world labels are included; "
            "all issue modes are generated from known square CP graphs. "
            "Cropped-away borders and symmetry completion are excluded because they "
            "would reward hallucinating unrecoverable structure."
        ),
        "image_size": args.image_size,
        "examples_per_issue": args.examples_per_issue,
        "seed": args.seed,
        "issues": [
            {"code": spec.code, "title": spec.title, "description": spec.description}
            for spec in issue_specs
        ],
        "examples": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "v2_training_manifest.jsonl").write_text(
        "".join(json.dumps(training_manifest_row(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_readme(output_dir, manifest)
    write_contact_sheet(
        contact_items,
        output_dir / "contact_sheet.png",
        cell_width=args.contact_sheet_width,
    )
    print(json.dumps({"output_dir": str(output_dir), "examples": len(rows)}, indent=2))


def build_issue_specs() -> list[IssueSpec]:
    return [
        IssueSpec(
            code="text_false_positive",
            title="Text False Positive",
            description="Text labels are drawn as dark line-like strokes over a valid CP.",
            pattern_factory=symmetric_cp,
            renderer=render_text_issue,
        ),
        IssueSpec(
            code="watermark_false_positive",
            title="Watermark False Positive",
            description="A translucent diagonal watermark overlaps real creases.",
            pattern_factory=symmetric_cp,
            renderer=render_watermark_issue,
        ),
        IssueSpec(
            code="guide_grid_false_positive",
            title="Guide Grid False Positive",
            description="A background guide grid should be suppressed as non-crease evidence.",
            pattern_factory=symmetric_cp,
            renderer=render_grid_issue,
        ),
        IssueSpec(
            code="dashed_line_support",
            title="Dashed Line Support",
            description="Valid crease carriers are rendered as dashed/gapped line evidence.",
            pattern_factory=symmetric_cp,
            renderer=render_dashed_issue,
        ),
        IssueSpec(
            code="faint_low_contrast",
            title="Faint Low Contrast",
            description="Creases are faint and close to the background color.",
            pattern_factory=symmetric_cp,
            renderer=render_faint_issue,
        ),
        IssueSpec(
            code="ambiguous_mv",
            title="Ambiguous M/V",
            description="Geometry is visible but M/V color information is intentionally removed.",
            pattern_factory=symmetric_cp,
            renderer=render_ambiguous_mv_issue,
        ),
    ]


def oracle_mask_kind(issue: str, mask_rel: str | None) -> str | None:
    if mask_rel is None:
        return None
    if issue in {"text_false_positive", "watermark_false_positive", "guide_grid_false_positive"}:
        return "non_crease_artifact"
    if issue == "dashed_line_support":
        return "dashed_target_support"
    return "target_line_support"


def training_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "cp-detector/v2-training-record/v1",
        "id": row["id"],
        "issue": row["issue"],
        "image": row["issue_image"],
        "clean_reference": row["clean_image"],
        "target_overlay": row["target_overlay"],
        "fold_graph_sidecar": row["sidecar"],
        "v2_label_sidecar": row["v2_label_sidecar"],
        "artifact_mask": row["oracle_mask"] if row.get("oracle_mask_kind") == "non_crease_artifact" else None,
        "target_support_mask": row["oracle_mask"] if row.get("oracle_mask_kind") != "non_crease_artifact" else None,
        "oracle_mask_kind": row.get("oracle_mask_kind"),
        "image_size": row["image_size"],
        "split": "visual_qa",
    }


def symmetric_cp(_: int = 0) -> CreasePattern:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
            [0.25, 0.0],
            [0.75, 0.0],
            [1.0, 0.25],
            [1.0, 0.75],
            [0.75, 1.0],
            [0.25, 1.0],
            [0.0, 0.75],
            [0.0, 0.25],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 9],
            [9, 5],
            [5, 10],
            [10, 1],
            [1, 11],
            [11, 6],
            [6, 12],
            [12, 2],
            [2, 13],
            [13, 7],
            [7, 14],
            [14, 3],
            [3, 15],
            [15, 8],
            [8, 16],
            [16, 0],
            [5, 4],
            [4, 7],
            [8, 4],
            [4, 6],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
            [9, 4],
            [10, 4],
            [11, 4],
            [12, 4],
            [13, 4],
            [14, 4],
            [15, 4],
            [16, 4],
        ],
        dtype=np.int64,
    )
    assignments = np.array(
        [2] * 16
        + [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            3,
            3,
            0,
            0,
            1,
            1,
            3,
            3,
        ],
        dtype=np.int8,
    )
    return CreasePattern(vertices=vertices, edges=edges, assignments=assignments)


def base_render(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
    *,
    profile: str = "clean",
) -> dict[str, Any]:
    sample = render_cpline_sample(
        cp,
        image_size=image_size,
        padding=max(8, int(32 * image_size / 1024)),
        line_width=max(1, int(2 * image_size / 768)),
        augment_profile=profile,
        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        square_symmetry="identity",
    )
    clean = sample.image.copy()
    return {
        "clean": clean,
        "issue": clean.copy(),
        "target_overlay": draw_target_overlay(clean, sample.pixel_vertices, sample.edges, sample.assignments),
        "oracle_mask": None,
        "vertices": sample.pixel_vertices,
        "edges": sample.edges,
        "assignments": sample.assignments,
        "metadata": {
            "base_profile": profile,
            "example_index": example_index,
            "render_metadata": sample.metadata,
        },
    }


def render_text_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="clean")
    issue = result["issue"]
    mask = np.zeros(issue.shape[:2], dtype=np.uint8)
    variants = [
        (["CREASE", "PATTERN", f"REF {example_index + 1}"], 0.10, 0.20, 1.0),
        (["MOUNTAIN", "VALLEY", "BASE"], 0.47, 0.18, 0.82),
        (["ORIGAMI", "DIAGRAM", "NO. 4"], 0.17, 0.63, 0.72),
        ([f"CP-{example_index + 7:02d}", "GRIDLESS"], 0.53, 0.58, 0.92),
    ]
    lines, x_frac, y_frac, scale_mult = variants[example_index % len(variants)]
    font_scale = image_size / 420.0 * scale_mult
    thickness = max(1, int(round(image_size / 240 * scale_mult)))
    x = int(image_size * x_frac)
    y = int(image_size * y_frac)
    for offset, text in enumerate(lines):
        org = (x, y + offset * int(image_size * 0.095))
        cv2.putText(
            issue,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(mask, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness + 2, cv2.LINE_AA)
    result["issue"] = issue
    result["oracle_mask"] = mask
    result["target_overlay"] = draw_target_overlay(issue, result["vertices"], result["edges"], result["assignments"])
    result["metadata"]["issue_transform"] = "dark text overlay"
    result["metadata"]["artifact_kind"] = "text"
    return result


def render_watermark_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="print-light")
    overlay = result["issue"].copy()
    mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
    watermark_variants = [
        ("CP-REFERENCE", -28.0, 0.42, image_size / 300.0),
        ("ORIGAMI ARCHIVE", 21.0, 0.32, image_size / 350.0),
        ("folding.example", -43.0, 0.38, image_size / 390.0),
        ("CREASING GUIDE", 8.0, 0.28, image_size / 360.0),
    ]
    text, rotation_degrees, alpha, font_scale = watermark_variants[example_index % len(watermark_variants)]
    thickness = max(2, image_size // 190)
    center = (image_size // 2, image_size // 2)
    text_layer = np.zeros_like(overlay)
    mask_layer = np.zeros_like(mask)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    org = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    cv2.putText(text_layer, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (40, 40, 40), thickness, cv2.LINE_AA)
    cv2.putText(mask_layer, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness + 2, cv2.LINE_AA)
    rotation = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    text_layer = cv2.warpAffine(text_layer, rotation, (image_size, image_size))
    mask = cv2.warpAffine(mask_layer, rotation, (image_size, image_size))
    issue = cv2.addWeighted(overlay, 1.0, text_layer, alpha, 0.0)
    result["issue"] = issue.astype(np.uint8)
    result["oracle_mask"] = mask
    result["target_overlay"] = draw_target_overlay(issue, result["vertices"], result["edges"], result["assignments"])
    result["metadata"]["issue_transform"] = f"translucent watermark {text!r} rotated {rotation_degrees:.1f}deg"
    result["metadata"]["artifact_kind"] = "watermark"
    return result


def render_grid_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="clean")
    issue = result["issue"].copy()
    mask = np.zeros(issue.shape[:2], dtype=np.uint8)
    spacing = max(16, int(round(image_size / (12 + 4 * (example_index % 4)))))
    color_value = int(165 + 18 * min(example_index % 3, 2))
    color = (color_value, color_value, color_value)
    for value in range(spacing // 2, image_size, spacing):
        cv2.line(issue, (value, 0), (value, image_size - 1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(issue, (0, value), (image_size - 1, value), color, 1, lineType=cv2.LINE_AA)
        cv2.line(mask, (value, 0), (value, image_size - 1), 255, 2, lineType=cv2.LINE_AA)
        cv2.line(mask, (0, value), (image_size - 1, value), 255, 2, lineType=cv2.LINE_AA)
    if example_index % 2 == 1:
        major_spacing = spacing * 4
        major_color = tuple(max(80, c - 45) for c in color)
        for value in range(major_spacing // 2, image_size, major_spacing):
            cv2.line(issue, (value, 0), (value, image_size - 1), major_color, 2, lineType=cv2.LINE_AA)
            cv2.line(issue, (0, value), (image_size - 1, value), major_color, 2, lineType=cv2.LINE_AA)
            cv2.line(mask, (value, 0), (value, image_size - 1), 255, 3, lineType=cv2.LINE_AA)
            cv2.line(mask, (0, value), (image_size - 1, value), 255, 3, lineType=cv2.LINE_AA)
    result["issue"] = issue
    result["oracle_mask"] = mask
    result["target_overlay"] = draw_target_overlay(issue, result["vertices"], result["edges"], result["assignments"])
    result["metadata"]["issue_transform"] = f"background guide grid spacing {spacing}px"
    result["metadata"]["artifact_kind"] = "guide_grid"
    return result


def render_dashed_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="clean")
    vertices = result["vertices"]
    edges = result["edges"]
    assignments = result["assignments"]
    line_width = max(1, int(2 * image_size / 768))
    dash_px = max(5.0, image_size * (0.018 + 0.006 * (example_index % 3)))
    gap_px = dash_px * (0.9 + 0.35 * (example_index % 2))
    issue = draw_cp_image(
        vertices,
        edges,
        assignments,
        image_size=image_size,
        line_width=line_width,
        dashed_non_border=True,
        dash_px=dash_px,
        gap_px=gap_px,
    )
    mask = draw_line_mask(vertices, edges, assignments, image_size=image_size, line_width=line_width + 2, non_border_only=True)
    result["issue"] = issue
    result["oracle_mask"] = mask
    result["target_overlay"] = draw_target_overlay(issue, vertices, edges, assignments)
    result["metadata"]["issue_transform"] = f"non-border creases rendered dashed dash={dash_px:.1f}px gap={gap_px:.1f}px"
    result["metadata"]["line_style"] = "dashed"
    return result


def render_faint_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="clean")
    clean = result["issue"].astype(np.float32)
    background = np.full_like(clean, 252.0)
    alpha_values = [0.16, 0.10, 0.065, 0.045]
    alpha = alpha_values[min(example_index, len(alpha_values) - 1)]
    issue = (background * (1.0 - alpha) + clean * alpha).clip(0, 255).astype(np.uint8)
    noise = rng.normal(0.0, 1.2 + 0.5 * example_index, issue.shape).astype(np.float32)
    issue = np.clip(issue.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    issue = cv2.GaussianBlur(issue, (3, 3), 0)
    result["issue"] = issue
    result["oracle_mask"] = draw_line_mask(result["vertices"], result["edges"], result["assignments"], image_size=image_size, line_width=3)
    result["target_overlay"] = draw_target_overlay(issue, result["vertices"], result["edges"], result["assignments"])
    result["metadata"]["issue_transform"] = f"crease alpha {alpha:.2f} on near-white background"
    result["metadata"]["line_style"] = "faint_solid"
    return result


def render_ambiguous_mv_issue(
    cp: CreasePattern,
    image_size: int,
    example_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = base_render(cp, image_size, example_index, rng, profile="clean")
    palette = {
        0: (90, 90, 90),
        1: (90, 90, 90),
        2: (0, 0, 0),
        3: (90, 90, 90),
    }
    issue = draw_cp_image(
        result["vertices"],
        result["edges"],
        result["assignments"],
        image_size=image_size,
        line_width=max(1, int(2 * image_size / 768)),
        palette=palette,
    )
    result["issue"] = issue
    result["oracle_mask"] = None
    result["target_overlay"] = draw_target_overlay(issue, result["vertices"], result["edges"], result["assignments"])
    result["metadata"]["issue_transform"] = "M/V colors collapsed to monochrome gray"
    result["metadata"]["expected_assignment_behavior"] = "geometry can be recovered; M/V should be U or ambiguous unless inferred"
    result["metadata"]["assignment_observation"] = "ambiguous_non_border"
    return result


def draw_cp_image(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    *,
    image_size: int,
    line_width: int,
    palette: dict[int, tuple[int, int, int]] | None = None,
    dashed_non_border: bool = False,
    dash_px: float | None = None,
    gap_px: float | None = None,
) -> np.ndarray:
    palette = ASSIGNMENT_COLORS if palette is None else palette
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    for edge, assignment_value in zip(edges, assignments):
        assignment = int(assignment_value)
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        color = palette.get(assignment, palette[3])
        if dashed_non_border and assignment != 2:
            draw_dashed_line(image, p0, p1, color, line_width, dash_px=dash_px, gap_px=gap_px)
        else:
            cv2.line(image, point(p0), point(p1), color, line_width, cv2.LINE_AA)
    return image


def draw_dashed_line(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
    dash_px: float | None = None,
    gap_px: float | None = None,
) -> None:
    start = np.asarray(p0, dtype=np.float32)
    end = np.asarray(p1, dtype=np.float32)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return
    direction = vector / length
    dash = max(5.0, float(dash_px) if dash_px is not None else length * 0.075)
    gap = max(4.0, float(gap_px) if gap_px is not None else dash * 0.75)
    t = 0.0
    while t < length:
        t_end = min(length, t + dash)
        q0 = start + direction * t
        q1 = start + direction * t_end
        cv2.line(image, point(q0), point(q1), color, line_width, cv2.LINE_AA)
        t += dash + gap


def draw_line_mask(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    *,
    image_size: int,
    line_width: int,
    non_border_only: bool = False,
) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for edge, assignment_value in zip(edges, assignments):
        if non_border_only and int(assignment_value) == 2:
            continue
        cv2.line(mask, point(vertices[int(edge[0])]), point(vertices[int(edge[1])]), 255, line_width, cv2.LINE_AA)
    return mask


def draw_target_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    for edge, assignment_value in zip(edges, assignments):
        color = ASSIGNMENT_COLORS.get(int(assignment_value), ASSIGNMENT_COLORS[3])
        cv2.line(
            overlay,
            point(vertices[int(edge[0])]),
            point(vertices[int(edge[1])]),
            color,
            2,
            cv2.LINE_AA,
        )
    for vertex in vertices:
        cv2.circle(overlay, point(vertex), 3, (30, 160, 30), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    rgb[mask > 0] = MASK_COLOR
    return rgb


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def point(point_array: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point_array[0]))), int(round(float(point_array[1]))))


def write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V2 Synthetic Issue Benchmark",
        "",
        "This directory is generated by `scripts/v2/generate_issue_benchmark.py`.",
        "",
        "It contains synthetic issue examples only. The examples are deliberately",
        "generated from known square CP graphs because no labeled real-world issue",
        "set exists yet.",
        "",
        "Cropped-away borders and symmetry completion are intentionally excluded",
        "from this benchmark. Those cases should be reported as unsupported or",
        "reserved for a future explicit recovery mode, not scored as V2 recovery",
        "targets.",
        "",
        "Files:",
        "",
        "- `manifest.json`: full benchmark manifest.",
        "- `manifest.jsonl`: one issue example per row.",
        "- `v2_training_manifest.jsonl`: dry-run training records pointing to",
        "  issue images and V2 label sidecars.",
        "- `contact_sheet.png`: visual overview for manual approval.",
        "- `examples/<issue>/`: clean image, issue image, oracle mask when present,",
        "  target graph overlay, V2 label overlay, graph sidecar, and V2 label sidecar.",
        "",
        "Issue slices:",
        "",
    ]
    for issue in manifest["issues"]:
        lines.append(f"- `{issue['code']}`: {issue['description']}")
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(
    items: list[dict[str, Any]],
    save_path: Path,
    *,
    cell_width: int,
) -> None:
    if not items:
        return
    columns = [
        ("clean_path", "Clean reference"),
        ("issue_path", "Issue image"),
        ("target_path", "Target overlay"),
        ("label_path", "V2 labels"),
        ("mask_path", "Oracle mask"),
    ]
    label_height = 72
    cell_gap = 12
    footer_height = 24
    cell_height = cell_width + label_height + footer_height
    sheet_width = cell_gap + len(columns) * (cell_width + cell_gap)
    sheet_height = cell_gap + len(items) * (cell_height + cell_gap)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_idx, item in enumerate(items):
        y = cell_gap + row_idx * (cell_height + cell_gap)
        title = f"{item['issue']} / {item['id']}"
        draw.text((cell_gap, y), title, fill=(20, 20, 20), font=font)
        for col_idx, (key, label) in enumerate(columns):
            x = cell_gap + col_idx * (cell_width + cell_gap)
            if item.get(key) is None:
                image = Image.new("RGB", (cell_width, cell_width), (248, 248, 248))
                image_draw = ImageDraw.Draw(image)
                image_draw.text((12, 12), "none", fill=(120, 120, 120), font=font)
            else:
                image = Image.open(item[key]).convert("RGB")
            image.thumbnail((cell_width, cell_width), Image.Resampling.LANCZOS)
            top = y + label_height
            sheet.paste(image, (x, top))
            draw.text((x, top + image.height + 3), label, fill=(80, 80, 80), font=font)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(save_path)


if __name__ == "__main__":
    main()
