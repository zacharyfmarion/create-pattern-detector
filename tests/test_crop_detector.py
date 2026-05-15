import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.data.scraping.crop_detector import CPCropDetector, process_asset_for_crops


def make_clean_cp(size: int = 768) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for i in range(0, size + 1, size // 8):
        draw.line((i, 0, i, size), fill="black", width=2)
        draw.line((0, i, size, i), fill="black", width=2)
    draw.line((0, 0, size, size), fill="red", width=2)
    draw.line((0, size, size, 0), fill="blue", width=2)
    return image


def test_crop_detector_accepts_clean_light_mode_cp(tmp_path):
    image_path = tmp_path / "cp.png"
    make_clean_cp().save(image_path)

    results = process_asset_for_crops(image_path, "asset-1", tmp_path / "crops")

    assert len(results) == 1
    assert results[0].status == "review"
    assert results[0].needs_review is True
    assert results[0].cp_score >= 0.75
    assert results[0].crop_path is not None


def test_crop_detector_handles_dark_mode_cp(tmp_path):
    image = make_clean_cp()
    arr = 255 - np.array(image)
    image_path = tmp_path / "dark.png"
    Image.fromarray(arr).save(image_path)

    results = process_asset_for_crops(image_path, "asset-2", tmp_path / "crops")

    assert results[0].status == "review"
    assert results[0].crop_path is not None


def test_crop_detector_accepts_colored_line_art_cp(tmp_path):
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    for i in range(0, 768 + 1, 96):
        draw.line((i, 0, i, 768), fill=(220, 0, 0), width=2)
        draw.line((0, i, 768, i), fill=(220, 0, 0), width=2)
    draw.line((0, 0, 768, 768), fill=(40, 40, 220), width=2)
    draw.line((0, 768, 768, 0), fill=(40, 40, 220), width=2)
    image_path = tmp_path / "red_cp.png"
    image.save(image_path)

    result = process_asset_for_crops(image_path, "asset-red", tmp_path / "crops")[0]

    assert result.status == "review"


def test_crop_detector_rejects_photo_like_noise(tmp_path):
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, size=(768, 768, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (11, 11), 0)
    image_path = tmp_path / "photo.jpg"
    Image.fromarray(noise).save(image_path)

    detector = CPCropDetector()
    result = process_asset_for_crops(image_path, "asset-3", tmp_path / "crops", detector=detector)[0]

    assert result.status == "rejected"
