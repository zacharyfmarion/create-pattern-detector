"""Detect, crop, and score clean crease-pattern regions from scraped assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps

from .manifest import sha256_file, sanitize_slug


class PDFRenderingUnavailable(RuntimeError):
    """Raised when a PDF asset cannot be rendered because no renderer is installed."""


@dataclass
class CropDetectionResult:
    asset_id: str
    source_path: str
    crop_path: str | None
    status: str
    needs_review: bool
    bbox: tuple[int, int, int, int] | None
    page_index: int | None
    cp_score: float
    clean_digital_score: float
    hand_drawn_score: float
    photo_text_score: float
    perceptual_hash: str | None
    content_sha256: str | None
    reasons: list[str]
    metrics: dict[str, float]
    gemini: dict[str, object] | None = None


@dataclass
class _Analysis:
    bbox: tuple[int, int, int, int] | None
    cp_score: float
    clean_digital_score: float
    hand_drawn_score: float
    photo_text_score: float
    reasons: list[str]
    metrics: dict[str, float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _foreground_metrics(crop_rgb: np.ndarray) -> dict[str, float]:
    """Estimate whether a crop is sparse line art or a filled/photo object."""
    if crop_rgb.size == 0:
        return {
            "foreground_fraction": 1.0,
            "largest_foreground_component_ratio": 1.0,
            "foreground_saturation_mean": 0.0,
            "background_distance_mean": 0.0,
            "photo_object_score": 1.0,
            "line_art_score": 0.0,
        }

    height, width = crop_rgb.shape[:2]
    patch = max(4, min(height, width) // 12)
    samples = np.concatenate(
        [
            crop_rgb[:patch, :patch].reshape(-1, 3),
            crop_rgb[:patch, -patch:].reshape(-1, 3),
            crop_rgb[-patch:, :patch].reshape(-1, 3),
            crop_rgb[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    background = np.median(samples, axis=0)
    rgb_float = crop_rgb.astype(np.float32)
    distance = np.linalg.norm(rgb_float - background, axis=2)
    threshold = max(28.0, float(np.std(samples)) * 2.2)
    foreground = distance > threshold

    # Small close gaps between shaded object pixels should remain filled, while
    # thin CP lines stay sparse.
    kernel_size = max(3, int(min(height, width) * 0.006))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    foreground_fraction = float(np.count_nonzero(filled_foreground)) / float(max(1, height * width))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(filled_foreground, connectivity=8)
    if component_count > 1:
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
    else:
        largest_component = 0
    largest_component_ratio = float(largest_component) / float(max(1, height * width))

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    foreground_saturation_mean = float(saturation[foreground].mean()) if np.any(foreground) else 0.0
    background_distance_mean = float(distance.mean())

    photo_object_score = _clamp01(
        0.45 * _clamp01(foreground_fraction / 0.42)
        + 0.35 * _clamp01(largest_component_ratio / 0.30)
        + 0.20 * _clamp01(foreground_saturation_mean / 95.0)
    )
    line_art_score = _clamp01(
        0.45 * (1.0 - _clamp01(foreground_fraction / 0.36))
        + 0.35 * (1.0 - _clamp01(largest_component_ratio / 0.24))
        + 0.20 * (1.0 - _clamp01(background_distance_mean / 80.0))
    )
    return {
        "foreground_fraction": float(foreground_fraction),
        "largest_foreground_component_ratio": float(largest_component_ratio),
        "foreground_saturation_mean": float(foreground_saturation_mean),
        "background_distance_mean": float(background_distance_mean),
        "photo_object_score": float(photo_object_score),
        "line_art_score": float(line_art_score),
    }


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    return np.array(image)


def average_hash(image: Image.Image, size: int = 8) -> str:
    """Simple perceptual hash for duplicate grouping."""
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(gray, dtype=np.float32)
    bits = arr > arr.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def _render_pdf_with_pypdfium(path: Path, dpi: int, max_pages: int | None) -> Iterator[tuple[int, Image.Image]]:
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    pdf = pdfium.PdfDocument(path)
    scale = dpi / 72.0
    page_count = len(pdf)
    if max_pages is not None:
        page_count = min(page_count, max_pages)
    for page_index in range(page_count):
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        yield page_index, bitmap.to_pil()


def iter_asset_images(
    path: str | Path,
    pdf_dpi: int = 220,
    max_pdf_pages: int | None = 8,
) -> Iterator[tuple[int | None, Image.Image]]:
    """Yield PIL images for image assets or rendered PDF pages."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            yield from _render_pdf_with_pypdfium(path, pdf_dpi, max_pdf_pages)
            return
        except ImportError as exc:
            raise PDFRenderingUnavailable(
                "PDF rendering requires pypdfium2; install project requirements to enable it"
            ) from exc
    with Image.open(path) as image:
        yield None, image.copy()


class CPCropDetector:
    """Heuristic detector for non-hand-drawn CP line-art crops."""

    def __init__(
        self,
        auto_accept_threshold: float = 0.75,
        review_threshold: float = 0.45,
        min_auto_side: int = 512,
        min_review_side: int = 256,
        analysis_max_side: int = 1600,
        crop_padding_ratio: float = 0.025,
    ) -> None:
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold
        self.min_auto_side = min_auto_side
        self.min_review_side = min_review_side
        self.analysis_max_side = analysis_max_side
        self.crop_padding_ratio = crop_padding_ratio

    def analyze(self, image: Image.Image) -> _Analysis:
        rgb = _to_rgb_array(image)
        height, width = rgb.shape[:2]
        scale = min(1.0, self.analysis_max_side / max(width, height))
        if scale < 1.0:
            small = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = rgb

        small_h, small_w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        min_line_length = max(24, int(min(small_w, small_h) * 0.045))
        hough_threshold = max(24, int(min(small_w, small_h) * 0.035))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max(6, int(min(small_w, small_h) * 0.01)),
        )

        line_map = np.zeros_like(edges)
        total_length = 0.0
        angle_bins: set[int] = set()
        line_count = 0
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in line]
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < min_line_length:
                    continue
                cv2.line(line_map, (x1, y1), (x2, y2), 255, 2, lineType=cv2.LINE_AA)
                total_length += length
                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
                angle_bins.add(int(angle // 15))
                line_count += 1

        mask_for_bbox = line_map
        if line_count < 4:
            median = float(np.median(gray))
            if median >= 128:
                _, threshold_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, threshold_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_for_bbox = cv2.bitwise_or(mask_for_bbox, threshold_mask)

        dilate_size = max(3, int(min(small_w, small_h) * 0.008))
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        bbox_mask = cv2.dilate(mask_for_bbox, kernel, iterations=1)
        ys, xs = np.nonzero(bbox_mask)

        reasons: list[str] = []
        bbox: tuple[int, int, int, int] | None
        if len(xs) == 0 or len(ys) == 0:
            bbox = None
            reasons.append("no_line_region")
        else:
            pad = int(min(small_w, small_h) * self.crop_padding_ratio)
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(small_w, int(xs.max()) + 1 + pad)
            y1 = min(small_h, int(ys.max()) + 1 + pad)
            bbox = (
                int(round(x0 / scale)),
                int(round(y0 / scale)),
                int(round(x1 / scale)),
                int(round(y1 / scale)),
            )

        crop_area = float(max(1, small_w * small_h))
        if bbox is not None:
            sx0, sy0, sx1, sy1 = [int(round(v * scale)) for v in bbox]
            sx0, sy0 = max(0, sx0), max(0, sy0)
            sx1, sy1 = min(small_w, sx1), min(small_h, sy1)
            crop_area = float(max(1, (sx1 - sx0) * (sy1 - sy0)))
            crop_edges = edges[sy0:sy1, sx0:sx1]
            crop_lines = line_map[sy0:sy1, sx0:sx1]
            crop_rgb = small[sy0:sy1, sx0:sx1]
        else:
            crop_edges = edges
            crop_lines = line_map
            crop_rgb = small

        line_pixels = float(np.count_nonzero(crop_lines))
        edge_pixels = float(np.count_nonzero(crop_edges))
        line_density = line_pixels / crop_area
        edge_density = edge_pixels / crop_area
        long_line_ratio = line_pixels / max(edge_pixels, 1.0)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            crop_w, crop_h = max(1, x1 - x0), max(1, y1 - y0)
        else:
            crop_w, crop_h = width, height
        aspect = crop_w / max(crop_h, 1)
        aspect_score = _clamp01(1.0 - abs(np.log(max(aspect, 1e-6))) / np.log(1.65))

        color_std = float(np.mean(np.std(crop_rgb.astype(np.float32), axis=(0, 1)))) if crop_rgb.size else 0.0
        foreground = _foreground_metrics(crop_rgb)
        line_count_score = _clamp01(line_count / 28.0)
        length_score = _clamp01(total_length / max(min(small_w, small_h) * 6.0, 1.0))
        density_score = _clamp01(line_density / 0.035)
        diversity_score = _clamp01(len(angle_bins) / 5.0)

        texture_without_lines = _clamp01((edge_density / 0.16) * (1.0 - _clamp01(long_line_ratio / 0.55)))
        photo_text_score = _clamp01(
            0.45 * texture_without_lines
            + 0.25 * _clamp01(color_std / 85.0)
            + 0.30 * foreground["photo_object_score"]
        )
        clean_digital_score = _clamp01(
            0.35 * _clamp01(long_line_ratio / 0.55)
            + 0.30 * line_count_score
            + 0.20 * density_score
            + 0.15 * foreground["line_art_score"]
        )
        hand_drawn_score = _clamp01(
            0.55 * (1.0 - _clamp01(long_line_ratio / 0.45)) + 0.25 * texture_without_lines + 0.20 * (1.0 - clean_digital_score)
        )
        cp_score = _clamp01(
            0.28 * line_count_score
            + 0.22 * length_score
            + 0.18 * density_score
            + 0.12 * diversity_score
            + 0.12 * aspect_score
            + 0.18 * clean_digital_score
            - 0.35 * photo_text_score
            - 0.30 * foreground["photo_object_score"]
            - 0.10 * hand_drawn_score
        )

        if line_count < 8:
            reasons.append("few_long_lines")
        if min(crop_w, crop_h) < self.min_auto_side:
            reasons.append("small_crop")
        if photo_text_score >= 0.55:
            reasons.append("high_photo_text_score")
        if hand_drawn_score >= 0.60:
            reasons.append("likely_hand_drawn_or_noisy")
        if aspect_score < 0.65:
            reasons.append("non_square_region")
        if foreground["photo_object_score"] >= 0.55:
            reasons.append("likely_photo_or_folded_model")
        if foreground["foreground_fraction"] >= 0.36:
            reasons.append("filled_foreground_not_line_art")

        metrics = {
            "width": float(width),
            "height": float(height),
            "crop_width": float(crop_w),
            "crop_height": float(crop_h),
            "line_count": float(line_count),
            "total_line_length": float(total_length / max(scale, 1e-9)),
            "line_density": float(line_density),
            "edge_density": float(edge_density),
            "long_line_ratio": float(long_line_ratio),
            "angle_bin_count": float(len(angle_bins)),
            "aspect_score": float(aspect_score),
            "color_std": float(color_std),
            **foreground,
        }
        return _Analysis(
            bbox=bbox,
            cp_score=float(cp_score),
            clean_digital_score=float(clean_digital_score),
            hand_drawn_score=float(hand_drawn_score),
            photo_text_score=float(photo_text_score),
            reasons=reasons,
            metrics=metrics,
        )

    def classify(self, analysis: _Analysis) -> tuple[str, bool]:
        bbox = analysis.bbox
        if bbox is None:
            return "rejected", False
        x0, y0, x1, y1 = bbox
        min_side = min(x1 - x0, y1 - y0)
        dense_lines = analysis.metrics.get("line_count", 0.0) >= 12 and analysis.metrics.get("line_density", 0.0) >= 0.002
        aspect_score = analysis.metrics.get("aspect_score", 0.0)
        photo_object_score = analysis.metrics.get("photo_object_score", 1.0)
        line_art_score = analysis.metrics.get("line_art_score", 0.0)
        foreground_fraction = analysis.metrics.get("foreground_fraction", 1.0)
        largest_component_ratio = analysis.metrics.get("largest_foreground_component_ratio", 1.0)
        if (
            photo_object_score >= 0.70
            or foreground_fraction >= 0.48
            or largest_component_ratio >= 0.42
        ):
            return "rejected", False
        if (
            analysis.cp_score >= self.auto_accept_threshold
            and min_side >= self.min_auto_side
            and dense_lines
            and analysis.photo_text_score < 0.42
            and photo_object_score < 0.45
            and foreground_fraction < 0.32
            and largest_component_ratio < 0.26
            and line_art_score >= 0.45
            and aspect_score >= 0.65
        ):
            return "review", True
        if (
            analysis.cp_score >= self.review_threshold
            and min_side >= self.min_review_side
            and photo_object_score < 0.60
            and foreground_fraction < 0.42
        ):
            return "review", True
        return "rejected", False

    def crop_image(self, image: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
        if bbox is None:
            return ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.exif_transpose(image).convert("RGB").crop(bbox)

    def detect_and_save(
        self,
        image: Image.Image,
        source_path: str | Path,
        asset_id: str,
        crop_dir: str | Path,
        basename: str,
        page_index: int | None = None,
    ) -> CropDetectionResult:
        analysis = self.analyze(image)
        status, needs_review = self.classify(analysis)
        crop_path: Path | None = None
        digest: str | None = None
        phash: str | None = None

        if status in {"accepted", "review"}:
            crop = self.crop_image(image, analysis.bbox)
            phash = average_hash(crop)
            crop_dir = Path(crop_dir) / status
            crop_dir.mkdir(parents=True, exist_ok=True)
            page_suffix = "" if page_index is None else f"-p{page_index:03d}"
            crop_path = crop_dir / f"{sanitize_slug(basename)}{page_suffix}-{phash}.png"
            crop.save(crop_path)
            digest = sha256_file(crop_path)

        return CropDetectionResult(
            asset_id=asset_id,
            source_path=Path(source_path).as_posix(),
            crop_path=crop_path.as_posix() if crop_path else None,
            status=status,
            needs_review=needs_review,
            bbox=analysis.bbox,
            page_index=page_index,
            cp_score=analysis.cp_score,
            clean_digital_score=analysis.clean_digital_score,
            hand_drawn_score=analysis.hand_drawn_score,
            photo_text_score=analysis.photo_text_score,
            perceptual_hash=phash,
            content_sha256=digest,
            reasons=analysis.reasons,
            metrics=analysis.metrics,
        )


def process_asset_for_crops(
    asset_path: str | Path,
    asset_id: str,
    crop_root: str | Path,
    detector: CPCropDetector | None = None,
    pdf_dpi: int = 220,
    max_pdf_pages: int | None = 8,
) -> list[CropDetectionResult]:
    """Process an image/PDF asset into zero or more crop results."""
    detector = detector or CPCropDetector()
    asset_path = Path(asset_path)
    results: list[CropDetectionResult] = []
    basename = sanitize_slug(f"{asset_id}-{asset_path.stem}")
    for page_index, image in iter_asset_images(asset_path, pdf_dpi=pdf_dpi, max_pdf_pages=max_pdf_pages):
        results.append(
            detector.detect_and_save(
                image=image,
                source_path=asset_path,
                asset_id=asset_id,
                crop_dir=crop_root,
                basename=basename,
                page_index=page_index,
            )
        )
    return results


def write_contact_sheet(
    crop_records: list[CropDetectionResult],
    output_path: str | Path,
    max_items: int = 80,
    thumb_size: int = 180,
) -> Path | None:
    """Write a simple visual review contact sheet for accepted/review crops."""
    records = [r for r in crop_records if r.crop_path][:max_items]
    if not records:
        return None
    cols = 5
    rows = int(np.ceil(len(records) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + 28)), "white")
    for idx, record in enumerate(records):
        with Image.open(record.crop_path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (thumb_size, thumb_size))
        x = (idx % cols) * thumb_size
        y = (idx // cols) * (thumb_size + 28)
        sheet.paste(thumb, (x, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
