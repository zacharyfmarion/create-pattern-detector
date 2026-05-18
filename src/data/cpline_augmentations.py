"""CPLineNet-specific render-time augmentations.

These augmentations are intentionally vector-first: any geometric perturbation
is applied to pixel-space graph vertices before dense CPLineNet labels are
rendered. Photometric effects touch only the input image.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import cv2
import numpy as np

from src.data.fold_parser import CreasePattern, transform_coords
from src.vectorization.evidence import render_vectorizer_evidence_from_pixels


BASE_AUGMENT_PROFILES = (
    "clean",
    "square-symmetry",
    "line-style",
    "dark-mode",
    "print-light",
    "print-medium",
    "photo-light",
    "photo-dark",
)
AUGMENT_MIXES: dict[str, tuple[tuple[str, float, str | None], ...]] = {
    "stage-base": (
        ("clean", 0.35, None),
        ("square-symmetry", 0.25, None),
        ("line-style", 0.40, None),
    ),
    "stage-balanced": (
        ("clean", 0.10, None),
        ("square-symmetry", 0.10, None),
        ("line-style", 0.15, None),
        ("print-light", 0.15, None),
        ("print-medium", 0.15, None),
        ("photo-light", 0.10, None),
        ("dark-mode", 0.15, None),
        ("photo-dark", 0.10, None),
    ),
}
MIXED_PROFILE_ENTRIES = AUGMENT_MIXES["stage-balanced"]
MIXED_AUGMENT_PROFILES = tuple(AUGMENT_MIXES) + ("mixed",)
AUGMENT_PROFILES = (
    *BASE_AUGMENT_PROFILES,
    *MIXED_AUGMENT_PROFILES,
)
SQUARE_SYMMETRIES = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip-horizontal",
    "flip-vertical",
    "transpose",
    "anti-transpose",
)
NON_IDENTITY_SQUARE_SYMMETRIES = tuple(symmetry for symmetry in SQUARE_SYMMETRIES if symmetry != "identity")
DARK_MODE_STYLE_VARIANTS = (
    "dark-default",
    "dark-muted",
    "dark-gray",
    "dark-bright",
)

ASSIGNMENT_RGB = {
    0: (220, 40, 40),
    1: (40, 80, 220),
    2: (0, 0, 0),
    3: (120, 120, 120),
}


@dataclass(frozen=True)
class AugmentedCplineSample:
    image: np.ndarray
    line_prob: np.ndarray
    angle: np.ndarray
    junction_heatmap: np.ndarray
    junction_offset: np.ndarray
    junction_mask: np.ndarray
    assignment: np.ndarray
    pixel_vertices: np.ndarray
    edges: np.ndarray
    assignments: np.ndarray
    metadata: dict[str, Any]


def normalize_augment_profile(
    augment_profile: str | None = None,
    *,
    render_noise: str | None = None,
) -> str:
    """Resolve the new augmentation profile while accepting the old flag."""
    profile = augment_profile or "clean"
    if render_noise and render_noise != "clean":
        mapped = "print-light" if render_noise == "mild" else render_noise
        if profile == "clean":
            warnings.warn(
                "--render-noise is deprecated for CPLineNet; use --augment-profile instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            profile = mapped
    if profile not in AUGMENT_PROFILES:
        raise ValueError(f"Unsupported augment_profile: {profile}")
    return profile


def render_augmented_cpline_sample(
    cp: CreasePattern,
    *,
    image_size: int,
    padding: int,
    line_width: int,
    augment_profile: str = "clean",
    render_noise: str | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    style_variant: str | None = None,
    square_symmetry: str | None = None,
    base_pixel_vertices: np.ndarray | None = None,
) -> AugmentedCplineSample:
    """Render an augmented CPLineNet input and matching dense labels."""
    profile = normalize_augment_profile(augment_profile, render_noise=render_noise)
    local_rng = np.random.default_rng(seed) if rng is None else rng
    start = perf_counter()

    selected_profile, selected_style_variant = _select_profile(profile, local_rng)
    if style_variant is not None:
        selected_style_variant = style_variant
    params = _sample_render_params(
        selected_profile,
        image_size=image_size,
        line_width=line_width,
        rng=local_rng,
        style_variant=selected_style_variant,
        square_symmetry=square_symmetry,
    )
    if base_pixel_vertices is None:
        pixel_vertices, _ = transform_coords(cp.vertices, image_size=image_size, padding=padding)
    else:
        pixel_vertices = np.asarray(base_pixel_vertices, dtype=np.float32)
    transformed_vertices = _apply_geometry(pixel_vertices, params["homography"])
    transformed_vertices = _apply_square_symmetry(
        transformed_vertices,
        image_size=image_size,
        symmetry=params["square_symmetry"],
    )
    target_assignments = _target_assignments(cp.assignments, params)

    target_line_width = int(params["target_line_width"])
    rendered = render_vectorizer_evidence_from_pixels(
        pixel_vertices=transformed_vertices,
        edges=cp.edges,
        assignments=target_assignments,
        image_size=image_size,
        line_width=target_line_width,
        junction_sigma=max(1.0, 2.5 * image_size / 768),
    )
    image = _render_input_image_from_pixels(
        transformed_vertices,
        cp.edges,
        cp.assignments,
        image_size=image_size,
        params=params,
    )
    image = _apply_photometric_effects(image, params, local_rng)

    junction_offset, junction_mask = _junction_offsets(rendered.pixel_vertices, rendered.edges, image_size)
    assignment = rendered.evidence.assignment_labels.astype(np.int64) - 1
    assignment[rendered.evidence.assignment_labels == 0] = -100
    clipped_vertices = int(
        np.sum(
            (transformed_vertices[:, 0] < 0)
            | (transformed_vertices[:, 0] >= image_size)
            | (transformed_vertices[:, 1] < 0)
            | (transformed_vertices[:, 1] >= image_size)
        )
    )
    metadata = {
        "profile": profile,
        "selected_profile": selected_profile,
        "style_variant": params.get("style_variant", selected_style_variant),
        "square_symmetry": params["square_symmetry"],
        "seed": seed,
        "render_ms": (perf_counter() - start) * 1000.0,
        "line_width": int(params["line_width"]),
        "target_line_width": target_line_width,
        "grid_enabled": bool(params["grid_enabled"]),
        "geometry_applied": bool(params["geometry_applied"]),
        "clipped_vertices": clipped_vertices,
        "edge_count": int(len(rendered.edges)),
        "line_pixels": int(np.count_nonzero(rendered.evidence.line_prob > 0.05)),
        "junction_pixels": int(np.count_nonzero(rendered.evidence.junction_heatmap > 0.05)),
        "params": _metadata_params(params),
    }
    return AugmentedCplineSample(
        image=image,
        line_prob=rendered.evidence.line_prob.astype(np.float32),
        angle=rendered.evidence.angle.astype(np.float32),
        junction_heatmap=rendered.evidence.junction_heatmap.astype(np.float32),
        junction_offset=junction_offset,
        junction_mask=junction_mask,
        assignment=assignment,
        pixel_vertices=rendered.pixel_vertices.astype(np.float32),
        edges=rendered.edges.astype(np.int64),
        assignments=rendered.assignments.astype(np.int8),
        metadata=metadata,
    )


def _select_profile(profile: str, rng: np.random.Generator) -> tuple[str, str | None]:
    if profile == "mixed":
        return _sample_mix_entry(MIXED_PROFILE_ENTRIES, rng)
    if profile in AUGMENT_MIXES:
        return _sample_mix_entry(AUGMENT_MIXES[profile], rng)
    return profile, None


def _sample_mix_entry(
    entries: tuple[tuple[str, float, str | None], ...],
    rng: np.random.Generator,
) -> tuple[str, str | None]:
    weights = np.array([entry[1] for entry in entries], dtype=np.float64)
    index = int(rng.choice(len(entries), p=weights / weights.sum()))
    selected_profile, _, style_variant = entries[index]
    return selected_profile, style_variant


def _sample_render_params(
    profile: str,
    *,
    image_size: int,
    line_width: int,
    rng: np.random.Generator,
    style_variant: str | None,
    square_symmetry: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "profile": profile,
        "background": (255, 255, 255),
        "palette": dict(ASSIGNMENT_RGB),
        "line_width": int(line_width),
        "target_line_width": int(line_width),
        "line_alpha": 1.0,
        "grid_enabled": False,
        "brightness": 0.0,
        "contrast": 1.0,
        "noise_std": 0.0,
        "blur_kernel": 0,
        "jpeg_quality": 0,
        "lighting_gradient": False,
        "geometry_applied": False,
        "homography": np.eye(3, dtype=np.float32),
        "assignment_target_mode": "original",
        "square_symmetry": _select_square_symmetry(profile, rng, square_symmetry),
    }
    if profile == "clean":
        return params
    if params["square_symmetry"] != "identity":
        params["geometry_applied"] = True
    if profile == "square-symmetry":
        return params
    if profile == "line-style":
        _apply_line_style_params(params, rng, line_width=line_width)
    elif profile == "dark-mode":
        _apply_dark_mode_params(params, rng, image_size=image_size, line_width=line_width, style_variant=style_variant)
    elif profile == "print-light":
        _apply_line_style_params(params, rng, line_width=line_width, mild=True)
        params.update(
            {
                "background": tuple(int(v) for v in rng.integers(240, 256, size=3)),
                "brightness": float(rng.uniform(-5, 5)),
                "contrast": float(rng.uniform(0.94, 1.06)),
                "noise_std": float(rng.uniform(0.0, 3.5)),
                "blur_kernel": int(rng.choice([0, 0, 3])),
                "jpeg_quality": int(rng.integers(82, 98)),
            }
        )
    elif profile == "print-medium":
        _apply_line_style_params(params, rng, line_width=line_width)
        params.update(
            {
                "background": tuple(int(v) for v in rng.integers(222, 252, size=3)),
                "brightness": float(rng.uniform(-12, 10)),
                "contrast": float(rng.uniform(0.86, 1.14)),
                "noise_std": float(rng.uniform(1.0, 7.0)),
                "blur_kernel": int(rng.choice([0, 3, 3, 5])),
                "jpeg_quality": int(rng.integers(68, 94)),
                "lighting_gradient": bool(rng.random() < 0.45),
            }
        )
    elif profile == "photo-light":
        _apply_line_style_params(params, rng, line_width=line_width, mild=True)
        params.update(
            {
                "background": tuple(int(v) for v in rng.integers(230, 256, size=3)),
                "brightness": float(rng.uniform(-10, 8)),
                "contrast": float(rng.uniform(0.88, 1.12)),
                "noise_std": float(rng.uniform(0.5, 5.5)),
                "blur_kernel": int(rng.choice([0, 3, 3, 5])),
                "jpeg_quality": int(rng.integers(72, 96)),
                "lighting_gradient": bool(rng.random() < 0.65),
                "geometry_applied": True,
                "homography": _sample_mild_homography(image_size, rng),
            }
        )
    elif profile == "photo-dark":
        _apply_dark_mode_params(
            params,
            rng,
            image_size=image_size,
            line_width=line_width,
            style_variant=style_variant,
        )
        max_width = max(line_width, int(round(line_width * 1.7)) + 1)
        width = int(rng.integers(max(1, line_width), max_width + 1))
        params.update(
            {
                "line_width": width,
                "target_line_width": width,
                "line_alpha": float(rng.uniform(0.78, 1.0)),
                "brightness": float(rng.uniform(-8, 8)),
                "contrast": float(rng.uniform(0.88, 1.14)),
                "noise_std": float(rng.uniform(0.5, 5.5)),
                "blur_kernel": int(rng.choice([0, 3, 3, 5])),
                "jpeg_quality": int(rng.integers(72, 96)),
                "lighting_gradient": bool(rng.random() < 0.65),
                "geometry_applied": True,
                "homography": _sample_mild_homography(image_size, rng),
            }
        )
    else:
        raise ValueError(f"Unsupported selected profile: {profile}")
    return params


def _apply_line_style_params(
    params: dict[str, Any],
    rng: np.random.Generator,
    *,
    line_width: int,
    mild: bool = False,
) -> None:
    max_width = max(line_width, int(round(line_width * (1.5 if mild else 2.0))) + 1)
    width = int(rng.integers(max(1, line_width), max_width + 1))
    palette_kind = str(rng.choice(["assignment", "assignment", "monochrome", "muted"]))
    if palette_kind == "monochrome":
        shade = int(rng.integers(20, 95))
        unassigned = int(rng.integers(80, 170))
        palette = {0: (shade, shade, shade), 1: (shade, shade, shade), 2: (0, 0, 0), 3: (unassigned, unassigned, unassigned)}
        params["assignment_target_mode"] = "mv_to_unassigned"
    elif palette_kind == "muted":
        palette = {
            0: _jitter_color((190, 55, 55), rng, 22),
            1: _jitter_color((55, 95, 190), rng, 22),
            2: _jitter_color((20, 20, 20), rng, 12),
            3: _jitter_color((120, 120, 120), rng, 25),
        }
    else:
        palette = {
            0: _jitter_color(ASSIGNMENT_RGB[0], rng, 28),
            1: _jitter_color(ASSIGNMENT_RGB[1], rng, 28),
            2: _jitter_color(ASSIGNMENT_RGB[2], rng, 8),
            3: _jitter_color(ASSIGNMENT_RGB[3], rng, 22),
        }
    params["palette"] = palette
    params["line_width"] = width
    params["target_line_width"] = width
    params["line_alpha"] = float(rng.uniform(0.78 if mild else 0.68, 1.0))
    params["palette_kind"] = palette_kind


def _target_assignments(assignments: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    result = np.asarray(assignments, dtype=np.int8).copy()
    if params["assignment_target_mode"] == "mv_to_unassigned":
        result[(result == 0) | (result == 1)] = 3
    return result


def _select_square_symmetry(
    profile: str,
    rng: np.random.Generator,
    requested: str | None,
) -> str:
    if requested is not None:
        if requested not in SQUARE_SYMMETRIES:
            raise ValueError(f"Unsupported square_symmetry: {requested}")
        return requested
    if profile != "square-symmetry":
        return "identity"
    return str(rng.choice(NON_IDENTITY_SQUARE_SYMMETRIES))


def _apply_dark_mode_params(
    params: dict[str, Any],
    rng: np.random.Generator,
    *,
    image_size: int,
    line_width: int,
    style_variant: str | None,
) -> None:
    variant = style_variant or str(rng.choice(DARK_MODE_STYLE_VARIANTS))
    if variant not in DARK_MODE_STYLE_VARIANTS:
        raise ValueError(f"Unsupported dark-mode style_variant: {variant}")
    bg = int(rng.integers(18, 34))
    if variant == "dark-muted":
        bg = int(rng.integers(28, 48))
    params["background"] = (bg, bg, min(56, bg + int(rng.integers(0, 10))))
    params["line_width"] = int(line_width)
    params["target_line_width"] = int(line_width)
    params["line_alpha"] = float(rng.uniform(0.84, 1.0))
    if variant == "dark-muted":
        red_base = (205, 82, 92)
        blue_base = (42, 118, 210)
    else:
        red_base = (250, 92, 104) if variant != "dark-bright" else (255, 115, 126)
        blue_base = (24, 145, 255) if variant != "dark-bright" else (68, 175, 255)
    border_base = (176, 176, 176) if variant != "dark-gray" else (118, 118, 118)
    unassigned_base = (140, 140, 140) if variant != "dark-gray" else (104, 104, 104)
    params["palette"] = {
        0: _jitter_color(red_base, rng, 15),
        1: _jitter_color(blue_base, rng, 15),
        2: _jitter_color(border_base, rng, 12),
        3: _jitter_color(unassigned_base, rng, 16),
    }
    params["style_variant"] = variant


def _sample_mild_homography(image_size: int, rng: np.random.Generator) -> np.ndarray:
    max_jitter = image_size * float(rng.uniform(0.006, 0.024))
    src = np.array(
        [[0, 0], [image_size - 1, 0], [image_size - 1, image_size - 1], [0, image_size - 1]],
        dtype=np.float32,
    )
    dst = src + rng.uniform(-max_jitter, max_jitter, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float32)


def _apply_geometry(vertices: np.ndarray, homography: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    if np.allclose(homography, np.eye(3, dtype=np.float32)):
        return vertices.copy()
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ homography.T
    denom = np.maximum(np.abs(transformed[:, 2:3]), 1e-6)
    return (transformed[:, :2] / denom).astype(np.float32)


def _apply_square_symmetry(
    vertices: np.ndarray,
    *,
    image_size: int,
    symmetry: str,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    x = vertices[:, 0]
    y = vertices[:, 1]
    max_coord = float(image_size - 1)
    if symmetry == "identity":
        transformed = np.stack([x, y], axis=1)
    elif symmetry == "rotate90":
        transformed = np.stack([max_coord - y, x], axis=1)
    elif symmetry == "rotate180":
        transformed = np.stack([max_coord - x, max_coord - y], axis=1)
    elif symmetry == "rotate270":
        transformed = np.stack([y, max_coord - x], axis=1)
    elif symmetry == "flip-horizontal":
        transformed = np.stack([max_coord - x, y], axis=1)
    elif symmetry == "flip-vertical":
        transformed = np.stack([x, max_coord - y], axis=1)
    elif symmetry == "transpose":
        transformed = np.stack([y, x], axis=1)
    elif symmetry == "anti-transpose":
        transformed = np.stack([max_coord - y, max_coord - x], axis=1)
    else:
        raise ValueError(f"Unsupported square_symmetry: {symmetry}")
    return transformed.astype(np.float32)


def _render_input_image_from_pixels(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    *,
    image_size: int,
    params: dict[str, Any],
) -> np.ndarray:
    image = np.full((image_size, image_size, 3), params["background"], dtype=np.uint8)
    overlay = image.copy()
    line_width = int(params["line_width"])
    palette = params["palette"]
    for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
        p0 = vertices[int(v1_idx)]
        p1 = vertices[int(v2_idx)]
        assignment = int(assignments[edge_idx])
        cv2.line(
            overlay,
            _point(p0),
            _point(p1),
            palette.get(assignment, palette[3]),
            line_width,
            lineType=cv2.LINE_AA,
        )
    alpha = float(params["line_alpha"])
    if alpha >= 0.999:
        return overlay
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0).astype(np.uint8)

def _apply_photometric_effects(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    result = image.astype(np.float32)
    if params["lighting_gradient"]:
        h, w = result.shape[:2]
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        gradient = np.cos(angle) * (xx - 0.5) + np.sin(angle) * (yy - 0.5)
        strength = float(rng.uniform(0.06, 0.16))
        result *= 1.0 + strength * gradient[..., None]
    result = result * float(params["contrast"]) + float(params["brightness"])
    noise_std = float(params["noise_std"])
    if noise_std > 0:
        result += rng.normal(0.0, noise_std, result.shape).astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)

    blur_kernel = int(params["blur_kernel"])
    if blur_kernel >= 3:
        result = cv2.GaussianBlur(result, (blur_kernel, blur_kernel), 0)

    jpeg_quality = int(params["jpeg_quality"])
    if jpeg_quality > 0:
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return result


def _junction_offsets(
    vertices: np.ndarray,
    edges: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    offset = np.zeros((image_size, image_size, 2), dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=bool)
    degrees = np.zeros(len(vertices), dtype=np.int32)
    for v1, v2 in edges:
        degrees[int(v1)] += 1
        degrees[int(v2)] += 1
    for vertex_idx, (x, y) in enumerate(vertices):
        if degrees[vertex_idx] < 1:
            continue
        col = int(round(float(x)))
        row = int(round(float(y)))
        if 0 <= row < image_size and 0 <= col < image_size:
            offset[row, col, 0] = float(x) - col
            offset[row, col, 1] = float(y) - row
            mask[row, col] = True
    return offset, mask


def _point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _jitter_color(
    color: tuple[int, int, int],
    rng: np.random.Generator,
    amount: int,
) -> tuple[int, int, int]:
    base = np.array(color, dtype=np.int16)
    jittered = base + rng.integers(-amount, amount + 1, size=3, dtype=np.int16)
    return tuple(int(v) for v in np.clip(jittered, 0, 255))


def _metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    ignored = {"homography"}
    result: dict[str, Any] = {}
    for key, value in params.items():
        if key in ignored:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = {str(k): list(v) if isinstance(v, tuple) else v for k, v in value.items()}
        elif isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    result["homography"] = np.asarray(params["homography"]).round(6).tolist()
    return result
