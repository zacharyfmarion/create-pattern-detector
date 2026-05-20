"""V2 issue augmentations and auxiliary CPLineNet targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

V2_LIGHT_AUGMENT_PROFILES = (
    "v2-text",
    "v2-watermark",
    "v2-guide-grid",
    "v2-dashed",
    "v2-faint",
    "v2-ambiguous-mv",
    "v2-combined",
)
V2_DARK_AUGMENT_PROFILES = tuple(f"v2-dark-{profile.removeprefix('v2-')}" for profile in V2_LIGHT_AUGMENT_PROFILES)
V2_AUGMENT_PROFILES = (*V2_LIGHT_AUGMENT_PROFILES, *V2_DARK_AUGMENT_PROFILES)
V2_LINE_STYLE_IDS = {
    "solid": 0,
    "dashed": 1,
    "faint": 2,
    "monochrome": 3,
}
V2_LINE_STYLE_NAMES = {value: key for key, value in V2_LINE_STYLE_IDS.items()}


@dataclass(frozen=True)
class V2AugmentationResult:
    image: np.ndarray
    non_crease_mask: np.ndarray
    target_line_mask: np.ndarray
    line_style: np.ndarray
    observed_assignment: np.ndarray
    metadata: dict[str, Any]


def default_v2_targets(
    *,
    line_prob: np.ndarray,
    assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return no-op V2 targets for non-V2 render profiles."""
    target_line_mask = (np.asarray(line_prob, dtype=np.float32) > 0.05).astype(np.float32)
    non_crease_mask = np.zeros_like(target_line_mask, dtype=np.float32)
    line_style = np.full(target_line_mask.shape, -100, dtype=np.int64)
    line_style[target_line_mask > 0.0] = V2_LINE_STYLE_IDS["solid"]
    return non_crease_mask, target_line_mask, line_style, np.asarray(assignment, dtype=np.int64).copy()


def apply_v2_augmentation(
    *,
    profile: str,
    image: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    line_prob: np.ndarray,
    assignment_target: np.ndarray,
    image_size: int,
    line_width: int,
    background: tuple[int, int, int],
    palette: dict[int, tuple[int, int, int]],
    rng: np.random.Generator,
) -> V2AugmentationResult:
    """Apply a recoverable V2 visual issue and emit matching auxiliary targets."""
    if profile not in V2_AUGMENT_PROFILES:
        raise ValueError(f"Unsupported V2 augmentation profile: {profile}")

    issue_profile = v2_issue_profile(profile)
    dark_mode = is_v2_dark_profile(profile) or _is_dark_background(background)
    result = np.asarray(image, dtype=np.uint8).copy()
    non_crease_mask, target_line_mask, line_style, observed_assignment = default_v2_targets(
        line_prob=line_prob,
        assignment=assignment_target,
    )
    modes: list[str] = []

    if issue_profile in {"v2-dashed", "v2-combined"}:
        result = _render_cp_lines(
            vertices,
            edges,
            assignments,
            image_size=image_size,
            line_width=line_width,
            background=background,
            palette=palette,
            rng=rng,
            dashed_non_border=True,
            monochrome=issue_profile == "v2-combined",
            dark_mode=dark_mode,
        )
        non_border = _line_mask(
            vertices,
            edges,
            assignments,
            image_size=image_size,
            line_width=line_width + 2,
            non_border_only=True,
        )
        line_style[(non_border > 0) & (target_line_mask > 0)] = V2_LINE_STYLE_IDS["dashed"]
        modes.append("dashed")

    if issue_profile in {"v2-ambiguous-mv", "v2-combined"}:
        if issue_profile != "v2-combined":
            result = _render_cp_lines(
                vertices,
                edges,
                assignments,
                image_size=image_size,
                line_width=line_width,
                background=background,
                palette=palette,
                rng=rng,
                dashed_non_border=False,
                monochrome=True,
                dark_mode=dark_mode,
            )
        non_border = _line_mask(
            vertices,
            edges,
            assignments,
            image_size=image_size,
            line_width=line_width + 2,
            non_border_only=True,
        )
        active = (non_border > 0) & (target_line_mask > 0)
        if issue_profile == "v2-combined":
            active &= line_style != V2_LINE_STYLE_IDS["dashed"]
        line_style[active] = V2_LINE_STYLE_IDS["monochrome"]
        observed_assignment[(observed_assignment == 0) | (observed_assignment == 1)] = 3
        modes.append("ambiguous_mv")

    if issue_profile in {"v2-faint", "v2-combined"}:
        result = _fade_lines(
            result,
            background=background,
            rng=rng,
            combined=issue_profile == "v2-combined",
            dark_mode=dark_mode,
        )
        non_border = _line_mask(
            vertices,
            edges,
            assignments,
            image_size=image_size,
            line_width=line_width + 2,
            non_border_only=True,
        )
        faint_style = V2_LINE_STYLE_IDS["faint"]
        if issue_profile == "v2-faint":
            line_style[(non_border > 0) & (target_line_mask > 0)] = faint_style
        modes.append("faint")

    if issue_profile in {"v2-guide-grid", "v2-combined"}:
        result, mask = _add_guide_grid(
            result,
            rng=rng,
            dark_mode=dark_mode,
            combined=issue_profile == "v2-combined",
        )
        non_crease_mask = np.maximum(non_crease_mask, mask.astype(np.float32))
        modes.append("guide_grid")

    if issue_profile in {"v2-watermark", "v2-combined"}:
        result, mask = _add_watermark(result, rng=rng, dark_mode=dark_mode)
        non_crease_mask = np.maximum(non_crease_mask, mask.astype(np.float32))
        modes.append("watermark")

    if issue_profile in {"v2-text", "v2-combined"}:
        result, mask = _add_text_labels(result, rng=rng, combined=issue_profile == "v2-combined", dark_mode=dark_mode)
        non_crease_mask = np.maximum(non_crease_mask, mask.astype(np.float32))
        modes.append("text")

    # Avoid contradictory positive targets where a synthetic artifact lands
    # directly on top of a real crease carrier.
    non_crease_mask[target_line_mask > 0] = 0.0

    metadata = {
        "profile": profile,
        "issue_profile": issue_profile,
        "dark_mode": dark_mode,
        "modes": modes,
        "non_crease_pixels": int(np.count_nonzero(non_crease_mask > 0)),
        "target_line_pixels": int(np.count_nonzero(target_line_mask > 0)),
        "line_style_counts": _style_counts(line_style),
        "observed_assignment_counts": _assignment_counts(observed_assignment),
    }
    return V2AugmentationResult(
        image=result,
        non_crease_mask=non_crease_mask.astype(np.float32),
        target_line_mask=target_line_mask.astype(np.float32),
        line_style=line_style.astype(np.int64),
        observed_assignment=observed_assignment.astype(np.int64),
        metadata=metadata,
    )


def v2_issue_profile(profile: str) -> str:
    if profile.startswith("v2-dark-"):
        return f"v2-{profile.removeprefix('v2-dark-')}"
    return profile


def is_v2_dark_profile(profile: str) -> bool:
    return profile.startswith("v2-dark-")


def _render_cp_lines(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    *,
    image_size: int,
    line_width: int,
    background: tuple[int, int, int],
    palette: dict[int, tuple[int, int, int]],
    rng: np.random.Generator,
    dashed_non_border: bool,
    monochrome: bool,
    dark_mode: bool,
) -> np.ndarray:
    image = np.full((image_size, image_size, 3), background, dtype=np.uint8)
    dash_px = max(6.0, float(image_size) * float(rng.uniform(0.015, 0.026)))
    gap_px = dash_px * float(rng.uniform(0.75, 1.25))
    mono_shade = int(rng.integers(135, 190) if dark_mode else rng.integers(45, 115))
    for edge_idx, edge in enumerate(edges):
        assignment = int(assignments[edge_idx])
        color = palette.get(assignment, palette.get(3, (120, 120, 120)))
        if monochrome and assignment != 2:
            color = (mono_shade, mono_shade, mono_shade)
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        if dashed_non_border and assignment != 2:
            _draw_dashed_line(image, p0, p1, color, line_width, dash_px=dash_px, gap_px=gap_px)
        else:
            cv2.line(image, _point(p0), _point(p1), color, line_width, cv2.LINE_AA)
    return image


def _draw_dashed_line(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
    *,
    dash_px: float,
    gap_px: float,
) -> None:
    start = np.asarray(p0, dtype=np.float32)
    end = np.asarray(p1, dtype=np.float32)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return
    direction = vector / length
    cursor = 0.0
    while cursor < length:
        segment_end = min(length, cursor + dash_px)
        q0 = start + direction * cursor
        q1 = start + direction * segment_end
        cv2.line(image, _point(q0), _point(q1), color, line_width, cv2.LINE_AA)
        cursor += dash_px + gap_px


def _fade_lines(
    image: np.ndarray,
    *,
    background: tuple[int, int, int],
    rng: np.random.Generator,
    combined: bool,
    dark_mode: bool,
) -> np.ndarray:
    bg = np.full_like(image, background, dtype=np.uint8).astype(np.float32)
    if dark_mode and combined:
        alpha = float(rng.uniform(0.58, 0.74))
    elif dark_mode:
        alpha = float(rng.uniform(0.42, 0.58))
    else:
        alpha = float(rng.uniform(0.18, 0.36) if combined else rng.uniform(0.08, 0.18))
    faded = bg * (1.0 - alpha) + image.astype(np.float32) * alpha
    noise = rng.normal(0.0, 1.2 if combined else 1.8, faded.shape).astype(np.float32)
    faded = np.clip(faded + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        faded = cv2.GaussianBlur(faded, (3, 3), 0)
    return faded


def _add_text_labels(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    combined: bool,
    dark_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    result = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    labels = ["CP", "fold", "source", "grid", "ref", "mountain", "valley"]
    count = int(rng.integers(1, 3 if not combined else 4))
    for _ in range(count):
        text = str(rng.choice(labels))
        scale = float(rng.uniform(0.45, 0.9 if combined else 1.15)) * h / 384.0
        thickness = max(1, int(round(rng.uniform(1.0, 2.2) * h / 384.0)))
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = int(rng.integers(max(1, w // 20), max(w // 20 + 1, w - size[0] - w // 20)))
        y = int(rng.integers(max(size[1] + 4, h // 8), max(size[1] + 5, h - h // 10)))
        if dark_mode:
            color = tuple(int(v) for v in rng.integers(170, 235, size=3))
        else:
            color = tuple(int(v) for v in rng.integers(35, 100, size=3))
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(mask, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness + 2, cv2.LINE_AA)
    return result, (mask > 0).astype(np.float32)


def _add_watermark(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    dark_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    text = str(rng.choice(["ORIGAMI", "CREASING", "DIAGRAM", "CP"]))
    scale = float(rng.uniform(1.2, 2.0)) * h / 384.0
    thickness = max(1, int(round(rng.uniform(2.0, 3.5) * h / 384.0)))
    mask_layer = np.zeros((h, w), dtype=np.uint8)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((w - size[0]) // 2, (h + size[1]) // 2)
    cv2.putText(mask_layer, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness + 3, cv2.LINE_AA)
    angle = float(rng.uniform(-36.0, -18.0) if rng.random() < 0.5 else rng.uniform(18.0, 36.0))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    mask_layer = cv2.warpAffine(mask_layer, matrix, (w, h))
    alpha = (mask_layer.astype(np.float32) / 255.0) * float(rng.uniform(0.18, 0.34))
    color = np.array((210, 210, 210) if dark_mode else (35, 35, 35), dtype=np.float32)
    result = image.astype(np.float32)
    result = result * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return result.astype(np.uint8), (mask_layer > 0).astype(np.float32)


def _add_guide_grid(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    dark_mode: bool,
    combined: bool,
) -> tuple[np.ndarray, np.ndarray]:
    result = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    spacing = int(rng.choice([24, 32, 40, 48, 64]) * h / 384.0)
    spacing = max(12, spacing)
    offset_x = int(rng.integers(0, spacing))
    offset_y = int(rng.integers(0, spacing))
    if dark_mode and combined:
        color = tuple(int(v) for v in rng.integers(68, 108, size=3))
    elif dark_mode:
        color = tuple(int(v) for v in rng.integers(85, 135, size=3))
    else:
        color = tuple(int(v) for v in rng.integers(165, 218, size=3))
    for x in range(offset_x, w, spacing):
        cv2.line(result, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
        cv2.line(mask, (x, 0), (x, h - 1), 255, 2, cv2.LINE_AA)
    for y in range(offset_y, h, spacing):
        cv2.line(result, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
        cv2.line(mask, (0, y), (w - 1, y), 255, 2, cv2.LINE_AA)
    if rng.random() < 0.35:
        major = spacing * int(rng.choice([2, 3]))
        if dark_mode and combined:
            major_color = (118, 118, 118)
        else:
            major_color = (150, 150, 150) if dark_mode else (145, 145, 145)
        for x in range(offset_x, w, major):
            cv2.line(result, (x, 0), (x, h - 1), major_color, 1, cv2.LINE_AA)
            cv2.line(mask, (x, 0), (x, h - 1), 255, 3, cv2.LINE_AA)
        for y in range(offset_y, h, major):
            cv2.line(result, (0, y), (w - 1, y), major_color, 1, cv2.LINE_AA)
            cv2.line(mask, (0, y), (w - 1, y), 255, 3, cv2.LINE_AA)
    return result, (mask > 0).astype(np.float32)


def _line_mask(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    *,
    image_size: int,
    line_width: int,
    non_border_only: bool,
) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for edge_idx, edge in enumerate(edges):
        if non_border_only and int(assignments[edge_idx]) == 2:
            continue
        cv2.line(mask, _point(vertices[int(edge[0])]), _point(vertices[int(edge[1])]), 255, line_width, cv2.LINE_AA)
    return (mask > 0).astype(np.float32)


def _style_counts(line_style: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for style_id, name in V2_LINE_STYLE_NAMES.items():
        value = int(np.count_nonzero(line_style == style_id))
        if value:
            counts[name] = value
    return counts


def _assignment_counts(assignment: np.ndarray) -> dict[str, int]:
    names = {0: "M", 1: "V", 2: "B", 3: "U", -100: "ignore"}
    return {
        name: int(np.count_nonzero(assignment == label))
        for label, name in names.items()
        if np.count_nonzero(assignment == label)
    }


def _point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _is_dark_background(background: tuple[int, int, int]) -> bool:
    return float(np.mean(np.asarray(background, dtype=np.float32))) < 96.0
