from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.inference.cli import main as cli_main
from src.inference.pipeline import (
    InferenceConfig,
    InferenceResult,
    build_report_payload,
    write_inference_outputs,
)
from src.inference.rectifier import RectificationResult, SquareRectifier
from src.vectorization import (
    AttributedPlanarGraph,
    PlanarGraphResult,
    RepairResult,
    build_quality_report,
    graph_to_fold_dict,
)


def test_rectifier_preserves_opaque_dark_mode_pixels(tmp_path: Path) -> None:
    path = tmp_path / "dark.png"
    image = np.full((16, 16, 3), 12, dtype=np.uint8)
    image[4:12, 7:9] = [230, 230, 230]
    Image.fromarray(image).save(path)

    result = SquareRectifier(image_size=32).rectify(path, rectified=True)

    assert result.rectified_rgb.shape == (32, 32, 3)
    assert result.alpha_matte_policy == "none"
    assert float(result.rectified_rgb.mean()) < 70.0
    assert result.warnings == []


def test_rectifier_auto_alpha_uses_transparent_stored_dark_matte(tmp_path: Path) -> None:
    path = tmp_path / "transparent_dark.png"
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[:, :, :3] = [0, 0, 0]
    rgba[6:10, :, :3] = [240, 240, 240]
    rgba[6:10, :, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)

    result = SquareRectifier(image_size=32, alpha_matte="auto").rectify(path, rectified=True)

    assert result.alpha_matte_policy == "auto-transparent-rgb"
    assert result.alpha_matte_rgb == (0, 0, 0)
    assert float(result.rectified_rgb[0, 0].mean()) < 5.0


def test_rectifier_warns_and_pads_non_square_rectified_input(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    image = np.full((8, 16, 3), 24, dtype=np.uint8)
    Image.fromarray(image).save(path)

    result = SquareRectifier(image_size=32).rectify(path, rectified=True)

    assert result.rectified_rgb.shape == (32, 32, 3)
    assert result.transform["mode"] == "resize_pad"
    assert result.padding_rgb == (24, 24, 24)
    assert [warning["code"] for warning in result.warnings] == ["rectified_input_not_square"]


def test_rectifier_crops_axis_aligned_cp_panel_from_page(tmp_path: Path) -> None:
    panel = _synthetic_cp_panel(size=240)
    page = Image.new("RGB", (520, 360), "white")
    draw = ImageDraw.Draw(page)
    draw.text((315, 34), "Carp v1.1", fill=(0, 0, 0))
    draw.rectangle((335, 130, 470, 285), fill=(185, 185, 185), outline=(110, 110, 110))
    page.paste(panel, (34, 58))
    path = tmp_path / "page.png"
    page.save(path)

    result = SquareRectifier(image_size=128).rectify(path, rectified=True)

    assert result.transform["mode"] == "detect_quad_warp"
    source_quad = np.asarray(result.transform["source_quad"], dtype=np.float32)
    expected = np.asarray([[34, 58], [273, 58], [273, 297], [34, 297]], dtype=np.float32)
    assert float(np.max(np.abs(source_quad - expected))) <= 4.0
    interior = _rectified_panel_region(result.rectified_rgb, result.transform)
    assert float(np.mean(np.abs(interior.astype(np.float32) - _resize_np(panel, interior.shape[0])))) < 18.0
    margin = int(result.transform["border_margin_px"])
    assert float(result.rectified_rgb[0, :, :].mean()) > 245.0
    assert float(result.rectified_rgb[:, 0, :].mean()) > 245.0
    assert float(result.rectified_rgb[margin, margin:-margin, :].mean()) < 90.0
    assert float(result.rectified_rgb[margin:-margin, margin, :].mean()) < 90.0


def test_rectifier_perspective_warps_skewed_cp_panel(tmp_path: Path) -> None:
    panel = _synthetic_cp_panel(size=260)
    source = np.asarray([[0, 0], [259, 0], [259, 259], [0, 259]], dtype=np.float32)
    dest = np.asarray([[84, 64], [488, 92], [458, 424], [52, 390]], dtype=np.float32)
    page = np.full((480, 560, 3), 255, dtype=np.uint8)
    cv2.putText(page, "designer notes", (330, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.rectangle(page, (400, 300), (535, 430), (210, 210, 210), -1)
    homography = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(np.asarray(panel), homography, (560, 480), borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full((260, 260), 255, dtype=np.uint8), homography, (560, 480))
    page[mask > 0] = warped[mask > 0]
    path = tmp_path / "skewed.png"
    Image.fromarray(page).save(path)

    result = SquareRectifier(image_size=128).rectify(path, rectified=True)

    assert result.transform["mode"] == "detect_quad_warp"
    source_quad = np.asarray(result.transform["source_quad"], dtype=np.float32)
    nearest = _nearest_quad_distances(source_quad, dest)
    assert float(np.max(nearest)) <= 12.0
    interior = _rectified_panel_region(result.rectified_rgb, result.transform)
    assert float(np.mean(np.abs(interior.astype(np.float32) - _resize_np(panel, interior.shape[0])))) < 34.0


def test_rectifier_crops_dark_mode_cp_panel_without_white_matte(tmp_path: Path) -> None:
    panel = _synthetic_cp_panel(size=220, dark=True)
    page = Image.new("RGB", (360, 430), (42, 42, 42))
    draw = ImageDraw.Draw(page)
    draw.text((35, 24), "dark mode CP", fill=(220, 220, 220))
    draw.rectangle((240, 62, 325, 130), fill=(86, 86, 86))
    page.paste(panel, (58, 150))
    path = tmp_path / "dark_page.png"
    page.save(path)

    result = SquareRectifier(image_size=128).rectify(path, rectified=True)

    assert result.transform["mode"] == "detect_quad_warp"
    assert float(result.rectified_rgb.mean()) < 95.0
    assert result.padding_rgb == (42, 42, 42)


def test_fold_writer_allows_phase5_metadata_schema() -> None:
    graph = _simple_graph()
    report = build_quality_report(graph)

    fold = graph_to_fold_dict(
        graph,
        report=report,
        file_creator="cp-detector cp-detect",
        metadata_schema="cp-detector/cp-detect/v1",
        extra_metadata={"checkpoint_id": "phase3-v1"},
    )

    assert fold["file_creator"] == "cp-detector cp-detect"
    assert fold["cp_detector"]["schema"] == "cp-detector/cp-detect/v1"
    assert fold["cp_detector"]["checkpoint_id"] == "phase3-v1"


def test_failed_inference_writes_report_but_no_fold(tmp_path: Path) -> None:
    graph = _empty_graph()
    report = build_quality_report(graph)
    rectification = RectificationResult(
        rectified_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        input_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        original_size=(16, 16),
        processed_size=(16, 16),
        homography_image_to_square=np.eye(3, dtype=np.float32),
        rectification_confidence=1.0,
        alpha_matte_policy="none",
        alpha_matte_rgb=None,
        padding_rgb=(0, 0, 0),
    )
    result = InferenceResult(
        input_path=tmp_path / "input.png",
        rectification=rectification,
        graph=graph,
        repair=RepairResult(graph=graph, actions=[]),
        quality_report=report,
        line_prob=np.zeros((16, 16), dtype=np.float32),
        junction_heatmap=np.zeros((16, 16), dtype=np.float32),
        output_fold=tmp_path / "out.fold",
        report_path=tmp_path / "out.report.json",
        debug_dir=None,
    )

    write_inference_outputs(
        result,
        config=InferenceConfig(rectified=True, include_debug=False, verify_checkpoint=False),
        manifest=_manifest(),
    )

    assert report.status == "failed"
    assert not (tmp_path / "out.fold").exists()
    payload = json.loads((tmp_path / "out.report.json").read_text(encoding="utf-8"))
    assert payload["outputs"]["fold_written"] is False
    assert payload["status"] == "failed"


def test_report_payload_includes_rectification_warnings(tmp_path: Path) -> None:
    graph = _simple_graph()
    report = build_quality_report(graph)
    rectification = RectificationResult(
        rectified_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        input_rgb=np.zeros((8, 16, 3), dtype=np.uint8),
        original_size=(16, 8),
        processed_size=(16, 8),
        homography_image_to_square=np.eye(3, dtype=np.float32),
        rectification_confidence=0.85,
        alpha_matte_policy="none",
        alpha_matte_rgb=None,
        padding_rgb=(0, 0, 0),
        warnings=[{"code": "rectified_input_not_square", "severity": "warning"}],
    )
    result = InferenceResult(
        input_path=tmp_path / "wide.png",
        rectification=rectification,
        graph=graph,
        repair=RepairResult(graph=graph, actions=[]),
        quality_report=report,
        line_prob=np.zeros((16, 16), dtype=np.float32),
        junction_heatmap=np.zeros((16, 16), dtype=np.float32),
    )

    payload = build_report_payload(
        result,
        config=InferenceConfig(rectified=True, verify_checkpoint=False),
        manifest=_manifest(),
        fold_written=True,
    )

    assert payload["warnings"]["rectification"][0]["code"] == "rectified_input_not_square"


def test_cli_requires_rectified_flag(tmp_path: Path, capsys) -> None:
    path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)

    exit_code = cli_main([str(path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Phase 6" in captured.err


def _simple_graph() -> AttributedPlanarGraph:
    result = PlanarGraphResult(
        vertices_coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        pixel_vertices=np.array([[0.0, 0.0], [15.0, 0.0]], dtype=np.float32),
        edges_vertices=np.array([[0, 1]], dtype=np.int64),
        edges_assignment=np.array([3], dtype=np.int8),
        edge_support=np.array([1.0], dtype=np.float32),
        vertex_support=np.ones(2, dtype=np.float32),
    )
    return AttributedPlanarGraph.from_planar_result(result)


def _empty_graph() -> AttributedPlanarGraph:
    return AttributedPlanarGraph(
        vertices_coords=np.empty((0, 2), dtype=np.float32),
        edges_vertices=np.empty((0, 2), dtype=np.int64),
        edges_assignment=np.empty(0, dtype=np.int8),
        edge_support=np.empty(0, dtype=np.float32),
        vertex_support=np.empty(0, dtype=np.float32),
        pixel_vertices=np.empty((0, 2), dtype=np.float32),
        assignment_confidence=np.empty(0, dtype=np.float32),
        assignment_margin=np.empty(0, dtype=np.float32),
        assignment_source=[],
    )


def _manifest() -> dict:
    return {
        "id": "phase3-v1-cpline-hrnet-w18-stage-balanced",
        "checkpoint": {"sha256": "abc", "sizeBytes": 1},
        "model": {"class": "CPLineNet", "imageSize": 1024},
        "inference": {"vectorizerThreshold": 0.65, "batchnormMode": "batch-stats"},
    }


def _synthetic_cp_panel(*, size: int, dark: bool = False) -> Image.Image:
    background = (18, 18, 18) if dark else (255, 255, 255)
    border = (232, 232, 232) if dark else (0, 0, 0)
    valley = (96, 155, 255) if dark else (30, 80, 230)
    mountain = (255, 105, 105) if dark else (220, 25, 25)
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=max(2, size // 120))
    step = max(24, size // 8)
    for offset in range(step, size, step):
        color = mountain if (offset // step) % 2 else valley
        draw.line((offset, 0, offset, size - 1), fill=color, width=2)
        draw.line((0, offset, size - 1, offset), fill=color, width=2)
    draw.line((0, 0, size - 1, size - 1), fill=mountain, width=2)
    draw.line((0, size - 1, size - 1, 0), fill=valley, width=2)
    center = size // 2
    draw.line((center, 0, size - 1, center), fill=mountain, width=2)
    draw.line((center, size - 1, size - 1, center), fill=valley, width=2)
    return image


def _resize_np(image: Image.Image, size: int) -> np.ndarray:
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def _rectified_panel_region(rgb: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    margin = int(transform["border_margin_px"])
    return rgb[margin : rgb.shape[0] - margin, margin : rgb.shape[1] - margin]


def _nearest_quad_distances(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.min(np.linalg.norm(expected - actual_point, axis=1))) for actual_point in actual],
        dtype=np.float32,
    )
