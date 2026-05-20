"""Square-domain preprocessing seam for Phase 5 inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

AlphaMattePolicy = Literal["auto", "white", "black"]


@dataclass(frozen=True)
class RectificationResult:
    """Canonical square image plus metadata about the preprocessing transform."""

    rectified_rgb: np.ndarray
    input_rgb: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    homography_image_to_square: np.ndarray
    rectification_confidence: float
    alpha_matte_policy: str
    alpha_matte_rgb: tuple[int, int, int] | None
    padding_rgb: tuple[int, int, int]
    warnings: list[dict[str, object]] = field(default_factory=list)
    transform: dict[str, object] = field(default_factory=dict)

    def metadata(self) -> dict[str, object]:
        return {
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "homography_image_to_square": self.homography_image_to_square.tolist(),
            "rectification_confidence": self.rectification_confidence,
            "alpha_matte_policy": self.alpha_matte_policy,
            "alpha_matte_rgb": (
                None if self.alpha_matte_rgb is None else list(self.alpha_matte_rgb)
            ),
            "padding_rgb": list(self.padding_rgb),
            "warnings": list(self.warnings),
            "transform": dict(self.transform),
        }


@dataclass(frozen=True)
class _PanelCandidate:
    quad: np.ndarray
    confidence: float
    method: str
    metrics: dict[str, object]


class SquareRectifier:
    """Phase 5 square-domain rectifier.

    This accepts a readable CP image supplied with `--rectified`, detects a
    visible crease-pattern panel when possible, and normalizes that panel to the
    model square. Fully automatic real-photo/document rectification remains a
    Phase 6 concern.
    """

    def __init__(
        self,
        *,
        image_size: int = 1024,
        alpha_matte: AlphaMattePolicy = "auto",
        analysis_max_side: int = 1600,
        min_panel_confidence: float = 0.72,
        border_margin_ratio: float = 32.0 / 1024.0,
        source_crop_padding_ratio: float = 0.025,
    ) -> None:
        if alpha_matte not in {"auto", "white", "black"}:
            raise ValueError(f"Unsupported alpha matte policy: {alpha_matte}")
        self.image_size = int(image_size)
        self.alpha_matte = alpha_matte
        self.analysis_max_side = int(analysis_max_side)
        self.min_panel_confidence = float(min_panel_confidence)
        self.border_margin_ratio = float(border_margin_ratio)
        self.source_crop_padding_ratio = float(source_crop_padding_ratio)

    def rectify(self, image_path: str | Path, *, rectified: bool) -> RectificationResult:
        if not rectified:
            raise NotImplementedError(
                "Automatic square/photo rectification is a Phase 6 feature. "
                "Use --rectified for already-square readable CP images."
            )

        path = Path(image_path)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            original_size = tuple(int(value) for value in image.size)
            input_rgb, matte_rgb, matte_policy, matte_warnings = _image_to_rgb(
                image,
                policy=self.alpha_matte,
            )

        warnings = list(matte_warnings)
        height, width = input_rgb.shape[:2]
        if width != height:
            warnings.append(
                {
                    "code": "rectified_input_not_square",
                    "message": (
                        "--rectified input is not square; it was resized with "
                        "letterbox padding for Phase 5 inference."
                    ),
                    "severity": "warning",
                    "details": {"width": int(width), "height": int(height)},
                }
            )

        panel = _detect_cp_panel(
            input_rgb,
            analysis_max_side=self.analysis_max_side,
            min_confidence=self.min_panel_confidence,
        )
        if panel is not None:
            if _is_full_frame_panel(input_rgb, panel.quad):
                rectified_rgb, transform, homography, padding_rgb = _resize_and_pad(
                    input_rgb,
                    image_size=self.image_size,
                    fallback_padding_rgb=matte_rgb,
                )
                transform = {
                    **transform,
                    "mode": "full_frame_resize",
                    "method": panel.method,
                    "full_frame_panel": True,
                    "detected_source_quad": [
                        [float(value) for value in point]
                        for point in _order_quad_points(
                            _clip_quad(panel.quad, width=width, height=height)
                        ).tolist()
                    ],
                    "confidence": float(panel.confidence),
                    "metrics": panel.metrics,
                }
                rectification_confidence = max(1.0 if width == height else 0.85, panel.confidence)
            else:
                rectified_rgb, transform, homography, padding_rgb = _warp_panel_to_square(
                    input_rgb,
                    panel=panel,
                    image_size=self.image_size,
                    border_margin_ratio=self.border_margin_ratio,
                    source_crop_padding_ratio=self.source_crop_padding_ratio,
                    fallback_padding_rgb=matte_rgb,
                )
                rectification_confidence = panel.confidence
        else:
            rectified_rgb, transform, homography, padding_rgb = _resize_and_pad(
                input_rgb,
                image_size=self.image_size,
                fallback_padding_rgb=matte_rgb,
            )
            rectification_confidence = 1.0 if width == height else 0.85

        density_warning = _dense_input_warning(rectified_rgb)
        if density_warning is not None:
            warnings.append(density_warning)

        return RectificationResult(
            rectified_rgb=rectified_rgb,
            input_rgb=input_rgb,
            original_size=original_size,
            processed_size=(int(width), int(height)),
            homography_image_to_square=homography,
            rectification_confidence=rectification_confidence,
            alpha_matte_policy=matte_policy,
            alpha_matte_rgb=matte_rgb,
            padding_rgb=padding_rgb,
            warnings=warnings,
            transform=transform,
        )


def _image_to_rgb(
    image: Image.Image,
    *,
    policy: AlphaMattePolicy,
) -> tuple[np.ndarray, tuple[int, int, int] | None, str, list[dict[str, object]]]:
    warnings: list[dict[str, object]] = []
    if "A" not in image.getbands() and "transparency" not in image.info:
        return np.asarray(image.convert("RGB"), dtype=np.uint8), None, "none", warnings

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    matte, matte_source, matte_warnings = _choose_alpha_matte(rgba, policy=policy)
    warnings.extend(matte_warnings)

    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = rgba[:, :, :3].astype(np.float32)
    matte_array = np.asarray(matte, dtype=np.float32).reshape(1, 1, 3)
    composited = foreground * alpha + matte_array * (1.0 - alpha)
    rgb = np.rint(composited).clip(0, 255).astype(np.uint8)
    return rgb, matte, matte_source, warnings


def _detect_cp_panel(
    rgb: np.ndarray,
    *,
    analysis_max_side: int,
    min_confidence: float,
) -> _PanelCandidate | None:
    small, scale = _analysis_image(rgb, analysis_max_side=analysis_max_side)
    edges = _edge_map(small)
    candidates = _quad_panel_candidates(small, edges, scale=scale)
    density = _density_panel_candidate(small, edges, scale=scale)
    if density is not None:
        candidates.append(density)
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    best = candidates[0]
    if best.confidence < min_confidence:
        return None
    return best


def _is_full_frame_panel(rgb: np.ndarray, quad: np.ndarray) -> bool:
    """Return true when a detected CP panel already spans the square image."""
    height, width = rgb.shape[:2]
    if width <= 0 or height <= 0:
        return False
    if abs(width - height) > max(2, int(round(max(width, height) * 0.02))):
        return False

    ordered = _order_quad_points(_clip_quad(quad, width=width, height=height))
    frame_area = float(max(1, (width - 1) * (height - 1)))
    area_ratio = abs(float(cv2.contourArea(ordered))) / frame_area
    edge_tolerance = max(6.0, min(width, height) * 0.025)
    x_values = ordered[:, 0]
    y_values = ordered[:, 1]
    touches_frame = (
        float(np.min(x_values)) <= edge_tolerance
        and float(np.max(x_values)) >= float(width - 1) - edge_tolerance
        and float(np.min(y_values)) <= edge_tolerance
        and float(np.max(y_values)) >= float(height - 1) - edge_tolerance
    )
    return area_ratio >= 0.94 and touches_frame


def _analysis_image(rgb: np.ndarray, *, analysis_max_side: int) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    max_side = max(width, height)
    if max_side <= analysis_max_side:
        return rgb.copy(), 1.0
    scale = float(analysis_max_side) / float(max_side)
    resized = cv2.resize(
        rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray))
    lower = max(0, int(0.66 * median))
    upper = min(255, int(1.33 * median))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3)
    if int(np.count_nonzero(edges)) < max(64, int(edges.size * 0.001)):
        edges = cv2.Canny(gray, 40, 140, apertureSize=3)
    return edges


def _quad_panel_candidates(
    small_rgb: np.ndarray,
    edges: np.ndarray,
    *,
    scale: float,
) -> list[_PanelCandidate]:
    height, width = edges.shape[:2]
    image_area = float(max(1, width * height))
    kernel_size = max(3, int(round(min(width, height) * 0.003)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[_PanelCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:800]:
        area = float(cv2.contourArea(contour))
        if area < max(256.0, image_area * 0.018):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        for epsilon_ratio in (0.01, 0.015, 0.02, 0.03, 0.045):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad_small = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
            key = tuple(int(round(value / 4.0)) for value in quad_small.reshape(-1))
            if key in seen:
                break
            seen.add(key)
            confidence, metrics = _score_panel_quad(small_rgb, quad_small)
            if confidence >= 0.42:
                candidates.append(
                    _PanelCandidate(
                        quad=(quad_small / scale).astype(np.float32),
                        confidence=confidence,
                        method="border_quad",
                        metrics=metrics,
                    )
                )
            break
    return candidates


def _density_panel_candidate(
    small_rgb: np.ndarray,
    edges: np.ndarray,
    *,
    scale: float,
) -> _PanelCandidate | None:
    height, width = edges.shape[:2]
    min_side = min(width, height)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(18, int(min_side * 0.028)),
        minLineLength=max(18, int(min_side * 0.055)),
        maxLineGap=max(4, int(min_side * 0.012)),
    )
    if lines is None or len(lines) < 8:
        return None

    line_map = np.zeros_like(edges)
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        if float(np.hypot(x2 - x1, y2 - y1)) < max(18, min_side * 0.055):
            continue
        cv2.line(line_map, (x1, y1), (x2, y2), 255, 2, lineType=cv2.LINE_AA)

    kernel_size = max(5, int(round(min_side * 0.025)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    connected = cv2.dilate(line_map, kernel, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    candidates: list[_PanelCandidate] = []
    for component in range(1, component_count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < max(400, int(width * height * 0.015)) or min(w, h) < max(24, min_side * 0.14):
            continue
        pad = max(2, int(round(min(w, h) * 0.01)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width - 1, x + w + pad)
        y1 = min(height - 1, y + h + pad)
        quad_small = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float32,
        )
        confidence, metrics = _score_panel_quad(small_rgb, quad_small)
        if float(metrics.get("square_score", 0.0)) < 0.62:
            continue
        confidence = min(0.86, confidence + 0.08)
        metrics = {**metrics, "density_component_area": int(area)}
        if confidence >= 0.50:
            candidates.append(
                _PanelCandidate(
                    quad=(quad_small / scale).astype(np.float32),
                    confidence=confidence,
                    method="density_bbox",
                    metrics=metrics,
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.confidence)


def _score_panel_quad(
    small_rgb: np.ndarray,
    quad: np.ndarray,
    *,
    sample_size: int = 256,
) -> tuple[float, dict[str, object]]:
    ordered = _order_quad_points(quad)
    height, width = small_rgb.shape[:2]
    area = abs(float(cv2.contourArea(ordered)))
    area_ratio = area / float(max(1, width * height))
    if area_ratio <= 0.0:
        return 0.0, {"area_ratio": 0.0}

    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    mean_width = max(1e-6, (top + bottom) * 0.5)
    mean_height = max(1e-6, (left + right) * 0.5)
    aspect = mean_width / mean_height
    square_score = _clamp01(1.0 - abs(math.log(max(aspect, 1e-6))) / math.log(1.8))
    size_score = _clamp01(area_ratio / 0.16)
    if area_ratio >= 0.96:
        size_score *= 0.85

    dst = np.array(
        [[0, 0], [sample_size - 1, 0], [sample_size - 1, sample_size - 1], [0, sample_size - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        small_rgb,
        homography,
        (sample_size, sample_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_edges = _edge_map(warped)
    band = max(3, sample_size // 64)
    border_mask = np.zeros_like(warped_edges, dtype=bool)
    border_mask[:band, :] = True
    border_mask[-band:, :] = True
    border_mask[:, :band] = True
    border_mask[:, -band:] = True
    border_score = _clamp01(float(np.count_nonzero(warped_edges[border_mask])) / (4.0 * sample_size * 0.55))

    interior_pad = band * 3
    interior = warped_edges[interior_pad:-interior_pad, interior_pad:-interior_pad]
    interior_area = float(max(1, interior.size))
    edge_density = float(np.count_nonzero(interior)) / interior_area
    density_score = _clamp01(edge_density / 0.045)

    lines = cv2.HoughLinesP(
        interior,
        rho=1,
        theta=np.pi / 180.0,
        threshold=18,
        minLineLength=18,
        maxLineGap=4,
    )
    line_count = 0
    angle_bins: set[int] = set()
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 18.0:
                continue
            line_count += 1
            angle = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
            angle_bins.add(int(angle // 15.0))
    line_score = _clamp01(line_count / 28.0)
    diversity_score = _clamp01(len(angle_bins) / 5.0)

    occupied_cells = 0
    cells = 8
    for cell_y in range(cells):
        y0 = cell_y * interior.shape[0] // cells
        y1 = (cell_y + 1) * interior.shape[0] // cells
        for cell_x in range(cells):
            x0 = cell_x * interior.shape[1] // cells
            x1 = (cell_x + 1) * interior.shape[1] // cells
            cell = interior[y0:y1, x0:x1]
            if int(np.count_nonzero(cell)) > int(cell.size * 0.005):
                occupied_cells += 1
    coverage_score = float(occupied_cells) / float(cells * cells)

    confidence = _clamp01(
        0.18 * border_score
        + 0.30 * square_score
        + 0.18 * line_score
        + 0.13 * coverage_score
        + 0.08 * density_score
        + 0.06 * diversity_score
        + 0.07 * size_score
    )
    metrics: dict[str, object] = {
        "area_ratio": float(area_ratio),
        "aspect": float(aspect),
        "square_score": float(square_score),
        "border_score": float(border_score),
        "edge_density": float(edge_density),
        "density_score": float(density_score),
        "line_count": int(line_count),
        "angle_bin_count": int(len(angle_bins)),
        "line_score": float(line_score),
        "coverage_score": float(coverage_score),
        "size_score": float(size_score),
    }
    return confidence, metrics


def _warp_panel_to_square(
    rgb: np.ndarray,
    *,
    panel: _PanelCandidate,
    image_size: int,
    border_margin_ratio: float,
    source_crop_padding_ratio: float,
    fallback_padding_rgb: tuple[int, int, int] | None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, tuple[int, int, int]]:
    height, width = rgb.shape[:2]
    detected_quad = _clip_quad(panel.quad, width=width, height=height)
    ordered_detected = _order_quad_points(detected_quad)
    padding_px = _source_crop_padding_px(
        ordered_detected,
        padding_ratio=source_crop_padding_ratio,
    )
    padded_quad = _expand_quad(ordered_detected, padding_px=padding_px)
    ordered = _order_quad_points(_clip_quad(padded_quad, width=width, height=height))
    border_margin_px = max(2, int(round(float(image_size) * float(border_margin_ratio))))
    border_margin_px = min(border_margin_px, max(0, (image_size - 2) // 2))
    low = float(border_margin_px)
    high = float(image_size - 1 - border_margin_px)
    dst = np.array(
        [
            [low, low],
            [high, low],
            [high, high],
            [low, high],
        ],
        dtype=np.float32,
    )
    target_padding_px = _target_crop_padding_px(
        source_quad=ordered_detected,
        target_quad=dst,
        source_padding_px=padding_px,
    )
    padded_dst = _clip_quad(
        _expand_quad(dst, padding_px=target_padding_px),
        width=image_size,
        height=image_size,
    )
    homography = cv2.getPerspectiveTransform(ordered.astype(np.float32), padded_dst)
    padding_rgb = fallback_padding_rgb or _infer_padding_rgb(rgb)
    warped = cv2.warpPerspective(
        rgb,
        homography,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in padding_rgb),
    )
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(padded_dst).astype(np.int32), 255)
    warped[mask == 0] = np.asarray(padding_rgb, dtype=np.uint8)
    transform = {
        "mode": "detect_quad_warp" if panel.method == "border_quad" else "detect_density_crop",
        "method": panel.method,
        "border_margin_px": int(border_margin_px),
        "border_margin_ratio": float(border_margin_ratio),
        "source_crop_padding_px": float(padding_px),
        "source_crop_padding_ratio": float(source_crop_padding_ratio),
        "target_crop_padding_px": float(target_padding_px),
        "detected_source_quad": [
            [float(value) for value in point] for point in ordered_detected.tolist()
        ],
        "source_quad": [[float(value) for value in point] for point in ordered.tolist()],
        "target_quad": [[float(value) for value in point] for point in dst.tolist()],
        "padded_target_quad": [[float(value) for value in point] for point in padded_dst.tolist()],
        "confidence": float(panel.confidence),
        "metrics": panel.metrics,
        "resized_size": [int(image_size), int(image_size)],
    }
    return warped.astype(np.uint8), transform, homography.astype(np.float32), padding_rgb


def _clip_quad(quad: np.ndarray, *, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(quad, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(max(0, width - 1)))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(max(0, height - 1)))
    return clipped


def _source_crop_padding_px(quad: np.ndarray, *, padding_ratio: float) -> float:
    ordered = _order_quad_points(quad)
    side_lengths = [
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
        float(np.linalg.norm(ordered[3] - ordered[0])),
    ]
    min_side = max(1.0, min(side_lengths))
    return max(2.0, min(48.0, min_side * max(0.0, padding_ratio)))


def _target_crop_padding_px(
    *,
    source_quad: np.ndarray,
    target_quad: np.ndarray,
    source_padding_px: float,
) -> float:
    source_side = max(1.0, _mean_side_length(source_quad))
    target_side = max(1.0, _mean_side_length(target_quad))
    return max(1.0, min(64.0, source_padding_px * target_side / source_side))


def _mean_side_length(quad: np.ndarray) -> float:
    ordered = _order_quad_points(quad)
    return float(
        (
            np.linalg.norm(ordered[1] - ordered[0])
            + np.linalg.norm(ordered[2] - ordered[1])
            + np.linalg.norm(ordered[2] - ordered[3])
            + np.linalg.norm(ordered[3] - ordered[0])
        )
        / 4.0
    )


def _expand_quad(quad: np.ndarray, *, padding_px: float) -> np.ndarray:
    ordered = _order_quad_points(quad)
    center = np.mean(ordered, axis=0)
    expanded = ordered.copy()
    for idx, point in enumerate(ordered):
        vector = point - center
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-6:
            continue
        expanded[idx] = point + vector / distance * float(padding_px) * math.sqrt(2.0)
    return expanded.astype(np.float32)


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmin(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmax(diffs))],
        ],
        dtype=np.float32,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _choose_alpha_matte(
    rgba: np.ndarray,
    *,
    policy: AlphaMattePolicy,
) -> tuple[tuple[int, int, int], str, list[dict[str, object]]]:
    if policy == "white":
        return (255, 255, 255), "white", []
    if policy == "black":
        return (0, 0, 0), "black", []

    transparent = rgba[:, :, 3] <= 8
    if int(np.count_nonzero(transparent)) >= 64:
        colors = rgba[:, :, :3][transparent]
        spread = np.percentile(colors, 90, axis=0) - np.percentile(colors, 10, axis=0)
        if float(np.mean(spread)) <= 16.0:
            return _median_rgb(colors), "auto-transparent-rgb", []

    opaque_border = _border_pixels(rgba[:, :, :3], mask=rgba[:, :, 3] >= 245)
    if len(opaque_border) >= 64:
        spread = np.percentile(opaque_border, 90, axis=0) - np.percentile(opaque_border, 10, axis=0)
        if float(np.mean(spread)) <= 48.0:
            return _median_rgb(opaque_border), "auto-opaque-border", []

    return (
        (255, 255, 255),
        "auto-fallback-white",
        [
            {
                "code": "alpha_matte_fallback_white",
                "message": (
                    "Transparent input did not contain a reliable stored matte; "
                    "alpha was composited onto white."
                ),
                "severity": "warning",
            }
        ],
    )


def _resize_and_pad(
    rgb: np.ndarray,
    *,
    image_size: int,
    fallback_padding_rgb: tuple[int, int, int] | None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, tuple[int, int, int]]:
    height, width = rgb.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Input image has zero width or height")

    scale = min(image_size / float(width), image_size / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    offset_x = int((image_size - resized_width) // 2)
    offset_y = int((image_size - resized_height) // 2)

    padding_rgb = fallback_padding_rgb or _infer_padding_rgb(rgb)
    canvas = Image.new("RGB", (image_size, image_size), padding_rgb)
    resized = Image.fromarray(rgb, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas.paste(resized, (offset_x, offset_y))

    scale_x = (resized_width - 1) / max(width - 1, 1)
    scale_y = (resized_height - 1) / max(height - 1, 1)
    homography = np.array(
        [
            [scale_x, 0.0, float(offset_x)],
            [0.0, scale_y, float(offset_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transform = {
        "mode": "resize" if width == height else "resize_pad",
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "resized_size": [int(resized_width), int(resized_height)],
    }
    return np.asarray(canvas, dtype=np.uint8), transform, homography, padding_rgb


def _dense_input_warning(rgb: np.ndarray) -> dict[str, object] | None:
    height, width = rgb.shape[:2]
    if width <= 0 or height <= 0:
        return None
    edges = _edge_map(rgb)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
    min_side = min(width, height)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(24, int(min_side * 0.035)),
        minLineLength=max(16, int(min_side * 0.045)),
        maxLineGap=max(4, int(min_side * 0.010)),
    )
    line_count = 0
    line_length_density = 0.0
    if lines is not None:
        lengths = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length >= max(16.0, min_side * 0.045):
                lengths.append(length)
        line_count = len(lengths)
        line_length_density = float(np.sum(lengths)) / float(max(1, width * height))

    dense = (
        edge_density >= 0.11
        or line_count >= 260
        or (line_count >= 140 and line_length_density >= 0.30)
    )
    if not dense:
        return None
    return {
        "code": "dense_input_evidence",
        "message": (
            "The rectified input contains dense line evidence that may be outside "
            "the Phase 3 V1 1024px readable-geometry envelope."
        ),
        "severity": "warning",
        "details": {
            "edge_density": edge_density,
            "hough_line_count": int(line_count),
            "line_length_density": line_length_density,
            "image_size": [int(width), int(height)],
        },
    }


def _infer_padding_rgb(rgb: np.ndarray) -> tuple[int, int, int]:
    border = _border_pixels(rgb)
    if len(border) == 0:
        return (255, 255, 255)
    return _median_rgb(border)


def _border_pixels(rgb: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    height, width = rgb.shape[:2]
    thickness = max(1, int(round(min(width, height) * 0.04)))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:thickness, :] = True
    border_mask[-thickness:, :] = True
    border_mask[:, :thickness] = True
    border_mask[:, -thickness:] = True
    if mask is not None:
        border_mask &= mask
    return rgb[border_mask]


def _median_rgb(values: np.ndarray) -> tuple[int, int, int]:
    median = np.median(values.astype(np.float32), axis=0)
    return tuple(int(value) for value in np.rint(median).clip(0, 255))
