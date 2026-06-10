"""Phase 5 production inference pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.inference.rectifier import AlphaMattePolicy, RectificationResult, SquareRectifier
from src.models import CPLineNet
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode
from src.vectorization import (
    AttributedPlanarGraph,
    EdgeAssignmentConfig,
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    QualityReport,
    QualityReportConfig,
    RepairConfig,
    RepairResult,
    SquareTopologyDecoder,
    SquareTopologyDecoderConfig,
    attribute_graph_from_logits,
    build_quality_report,
    conservative_repair,
    cpline_outputs_to_evidence,
    save_fold,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_MANIFEST = REPO_ROOT / "artifacts/checkpoints/phase3-v1-cpline.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt"
MAIN_REPO_CHECKPOINT = Path(
    "/Users/zacharymarion/Documents/code/create-pattern-detector/"
    "checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt"
)
FOLD_METADATA_SCHEMA = "cp-detector/cp-detect/v1"
REPORT_SCHEMA = "cp-detector/cp-detect-report/v1"
STATUS_ORDER = ("valid", "repaired", "ambiguous", "outside_v1_envelope", "failed")


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    checkpoint_manifest: Path = DEFAULT_CHECKPOINT_MANIFEST
    device: str = "auto"
    image_size: int = 1024
    threshold: float | None = None
    batchnorm_mode: str = "batch-stats"
    rectified: bool = False
    alpha_matte: AlphaMattePolicy = "auto"
    infer_assignments: bool = False
    repair_near_endpoint_crossings: bool = False
    include_debug: bool = True
    verify_checkpoint: bool = True


@dataclass
class InferenceResult:
    input_path: Path
    rectification: RectificationResult
    graph: AttributedPlanarGraph
    repair: RepairResult
    quality_report: QualityReport
    line_prob: np.ndarray
    junction_heatmap: np.ndarray
    output_fold: Path | None = None
    report_path: Path | None = None
    debug_dir: Path | None = None
    report_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.quality_report.status


class CPDetectPipeline:
    """Load CPLineNet once and run the Phase 5 image-to-FOLD pipeline."""

    def __init__(self, config: InferenceConfig) -> None:
        if config.batchnorm_mode not in BATCHNORM_MODES:
            raise ValueError(f"Unsupported batchnorm_mode={config.batchnorm_mode!r}")
        self.config = config
        self.manifest = load_checkpoint_manifest(config.checkpoint_manifest)
        self.threshold = _manifest_threshold(self.manifest, config.threshold)
        self.device = select_device(config.device)
        self._verify_checkpoint_if_requested()
        self.model = load_cpline_model(config.checkpoint, self.device)
        self.rectifier = SquareRectifier(
            image_size=config.image_size,
            alpha_matte=config.alpha_matte,
        )
        self.builder = build_stage4_builder(
            config.image_size,
            self.threshold,
            repair_near_endpoint_crossings=config.repair_near_endpoint_crossings,
        )
        self.assignment_config = EdgeAssignmentConfig()
        self.repair_config = RepairConfig(image_size=config.image_size)
        self.report_config = QualityReportConfig(image_size=config.image_size)

    def detect(
        self,
        input_path: str | Path,
        *,
        output_fold: str | Path | None = None,
        report_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
    ) -> InferenceResult:
        path = Path(input_path)
        rectification = self.rectifier.rectify(path, rectified=self.config.rectified)
        image_tensor = _image_tensor(rectification.rectified_rgb, self.device)

        with torch.no_grad(), model_eval_with_batchnorm_mode(
            self.model,
            batchnorm_mode=self.config.batchnorm_mode,
        ):
            outputs = self.model(image_tensor)

        evidence = cpline_outputs_to_evidence(
            outputs,
            batch_index=0,
            line_threshold=self.threshold,
        )
        line_prob = evidence.line_prob
        junction_heatmap = evidence.junction_heatmap
        graph_result = self.builder.build(evidence)
        attributed = attribute_graph_from_logits(
            graph_result,
            outputs["assignment_logits"][0].detach().cpu(),
            line_prob=line_prob,
            config=self.assignment_config,
        )
        repair = conservative_repair(
            attributed,
            line_prob=line_prob,
            config=self.repair_config,
            infer_assignments=self.config.infer_assignments,
        )
        quality_report = build_quality_report(
            repair.graph,
            repair_actions=repair.actions,
            config=self.report_config,
        )
        apply_rectification_warnings_to_report(quality_report, rectification)

        result = InferenceResult(
            input_path=path,
            rectification=rectification,
            graph=repair.graph,
            repair=repair,
            quality_report=quality_report,
            line_prob=line_prob.astype(np.float32),
            junction_heatmap=junction_heatmap.astype(np.float32),
            output_fold=None if output_fold is None else Path(output_fold),
            report_path=None if report_path is None else Path(report_path),
            debug_dir=None if debug_dir is None else Path(debug_dir),
        )
        write_inference_outputs(result, config=self.config, manifest=self.manifest)
        return result

    def _verify_checkpoint_if_requested(self) -> None:
        checkpoint = _resolve_path(self.config.checkpoint)
        if not checkpoint.exists():
            hint = ""
            if MAIN_REPO_CHECKPOINT.exists():
                hint = (
                    f" Recover it with: mkdir -p {checkpoint.parent} && "
                    f"cp {MAIN_REPO_CHECKPOINT} {checkpoint}"
                )
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}.{hint}")
        if self.config.verify_checkpoint:
            verify_checkpoint_file(checkpoint, self.manifest)


def build_stage4_builder(
    image_size: int,
    threshold: float,
    *,
    repair_near_endpoint_crossings: bool = False,
) -> SquareTopologyDecoder:
    return SquareTopologyDecoder(
        SquareTopologyDecoderConfig(
            image_size=image_size,
            line_threshold=threshold,
            hough_threshold=10,
            hough_min_line_length=6,
            hough_max_line_gap=4,
            min_edge_support=0.45,
            junction_threshold=0.20,
            junction_nms_radius=2,
            vertex_merge_px=max(1.0, 1.5 * image_size / 768),
            line_vertex_distance_px=max(2.0, 4.0 * image_size / 768),
        )
    )


def build_legacy_stage4_builder(
    image_size: int,
    threshold: float,
    *,
    repair_near_endpoint_crossings: bool = False,
) -> PlanarGraphBuilder:
    return PlanarGraphBuilder(
        PlanarGraphBuilderConfig(
            image_size=image_size,
            line_threshold=threshold,
            hough_threshold=10,
            hough_min_line_length=6,
            hough_max_line_gap=4,
            min_edge_support=0.45,
            junction_threshold=0.20,
            junction_nms_radius=2,
            vertex_merge_px=max(1.0, 1.5 * image_size / 768),
            line_vertex_distance_px=max(2.0, 4.0 * image_size / 768),
            direct_edge_max_vertices=256,
            direct_edge_short_max_vertices=512,
            planar_cleanup_max_edges=2500,
            repair_near_endpoint_crossings=repair_near_endpoint_crossings,
        )
    )


def load_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _resolve_path(Path(path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_checkpoint_file(path: str | Path, manifest: dict[str, Any]) -> None:
    checkpoint = Path(path)
    expected = manifest.get("checkpoint", {})
    expected_size = expected.get("sizeBytes")
    if expected_size is not None and checkpoint.stat().st_size != int(expected_size):
        raise ValueError(
            f"Checkpoint size mismatch for {checkpoint}: "
            f"expected {expected_size}, got {checkpoint.stat().st_size}"
        )
    expected_sha = expected.get("sha256")
    if expected_sha:
        actual_sha = _sha256_file(checkpoint)
        if actual_sha != str(expected_sha):
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {checkpoint}: "
                f"expected {expected_sha}, got {actual_sha}"
            )


def load_cpline_model(checkpoint: str | Path, device: torch.device) -> CPLineNet:
    loaded = torch.load(_resolve_path(Path(checkpoint)), map_location=device, weights_only=False)
    config = loaded.get("config", {})
    model = CPLineNet(
        backbone=config.get("backbone", "hrnet_w18"),
        pretrained=False,
        hidden_channels=int(config.get("hidden_channels", 128)),
        v2_heads=bool(config.get("v2_heads", False)),
    ).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but torch.backends.mps.is_available() is false")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    return device


def write_inference_outputs(
    result: InferenceResult,
    *,
    config: InferenceConfig,
    manifest: dict[str, Any],
) -> None:
    fold_written = False
    if result.output_fold is not None and result.status != "failed":
        result.output_fold.parent.mkdir(parents=True, exist_ok=True)
        save_fold(
            result.graph,
            result.output_fold,
            report=result.quality_report,
            repair_actions=result.repair.actions,
            file_creator="cp-detector cp-detect",
            metadata_schema=FOLD_METADATA_SCHEMA,
            extra_metadata={
                "checkpoint_id": manifest.get("id"),
                "rectification": result.rectification.metadata(),
            },
        )
        fold_written = True

    report_payload = build_report_payload(
        result,
        config=config,
        manifest=manifest,
        fold_written=fold_written,
    )
    result.report_payload = report_payload
    if result.report_path is not None:
        result.report_path.parent.mkdir(parents=True, exist_ok=True)
        result.report_path.write_text(
            json.dumps(_json_safe(report_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if result.debug_dir is not None and config.include_debug:
        write_debug_artifacts(result, result.debug_dir)


def build_report_payload(
    result: InferenceResult,
    *,
    config: InferenceConfig,
    manifest: dict[str, Any],
    fold_written: bool,
) -> dict[str, Any]:
    quality = result.quality_report.to_dict()
    rectification = result.rectification.metadata()
    return {
        "schema": REPORT_SCHEMA,
        "input": {
            "path": str(result.input_path),
            "rectified": bool(config.rectified),
        },
        "status": result.status,
        "warnings": {
            "rectification": rectification["warnings"],
            "quality": quality["warnings"],
        },
        "quality_report": quality,
        "rectification": rectification,
        "checkpoint": {
            "path": str(config.checkpoint),
            "manifest": str(config.checkpoint_manifest),
            "id": manifest.get("id"),
            "sha256": (manifest.get("checkpoint") or {}).get("sha256"),
        },
        "model": manifest.get("model", {}),
        "config": {
            "image_size": int(config.image_size),
            "threshold": _manifest_threshold(manifest, config.threshold),
            "batchnorm_mode": config.batchnorm_mode,
            "device": config.device,
            "alpha_matte": config.alpha_matte,
            "infer_assignments": bool(config.infer_assignments),
            "repair_near_endpoint_crossings": bool(config.repair_near_endpoint_crossings),
        },
        "graph": graph_metadata(result.graph),
        "outputs": {
            "fold": None if result.output_fold is None else str(result.output_fold),
            "fold_written": fold_written,
            "report": None if result.report_path is None else str(result.report_path),
            "debug_dir": None if result.debug_dir is None else str(result.debug_dir),
        },
    }


def graph_metadata(graph: AttributedPlanarGraph) -> dict[str, Any]:
    return {
        "vertices": int(graph.num_vertices),
        "edges": int(graph.num_edges),
        "edge_support_mean": float(np.mean(graph.edge_support)) if graph.num_edges else 0.0,
        "assignment_confidence_mean": (
            float(np.mean(graph.assignment_confidence)) if graph.num_edges else 0.0
        ),
        "assignment_source_counts": {
            "observed": int(sum(source == "observed" for source in graph.assignment_source)),
            "unknown": int(sum(source == "unknown" for source in graph.assignment_source)),
            "inferred": int(sum(source == "inferred" for source in graph.assignment_source)),
        },
    }


def apply_rectification_warnings_to_report(
    report: QualityReport,
    rectification: RectificationResult,
) -> None:
    """Promote input-level envelope warnings into the overall Phase 5 status."""
    codes = {str(warning.get("code", "")) for warning in rectification.warnings}
    if "dense_input_evidence" not in codes:
        return
    if STATUS_ORDER.index(report.status) < STATUS_ORDER.index("outside_v1_envelope"):
        report.status = "outside_v1_envelope"


def write_debug_artifacts(result: InferenceResult, debug_dir: str | Path) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.rectification.input_rgb).save(path / "input.png")
    Image.fromarray(result.rectification.rectified_rgb).save(path / "rectified.png")
    Image.fromarray(_heatmap_image(result.line_prob)).save(path / "line_prob.png")
    Image.fromarray(_heatmap_image(result.junction_heatmap)).save(path / "junction_heatmap.png")
    Image.fromarray(_graph_overlay(result, assignment_colors=False)).save(path / "graph_overlay.png")
    Image.fromarray(_graph_overlay(result, assignment_colors=True)).save(
        path / "assignment_overlay.png"
    )
    (path / "graph.json").write_text(
        json.dumps(_json_safe(_graph_payload(result.graph)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.report_payload:
        (path / "report.json").write_text(
            json.dumps(_json_safe(result.report_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_batch_summary(
    results: list[InferenceResult],
    output_path: str | Path,
    *,
    config: InferenceConfig,
) -> None:
    rows = [
        {
            "input": str(result.input_path),
            "status": result.status,
            "fold": None if result.output_fold is None else str(result.output_fold),
            "fold_written": bool(
                result.report_payload.get("outputs", {}).get("fold_written", False)
            ),
            "report": None if result.report_path is None else str(result.report_path),
            "debug_dir": None if result.debug_dir is None else str(result.debug_dir),
        }
        for result in results
    ]
    status_counts = {status: sum(row["status"] == status for row in rows) for status in STATUS_ORDER}
    payload = {
        "schema": "cp-detector/cp-detect-batch/v1",
        "count": len(rows),
        "status_counts": status_counts,
        "config": {
            "image_size": config.image_size,
            "threshold": config.threshold,
            "batchnorm_mode": config.batchnorm_mode,
            "alpha_matte": config.alpha_matte,
        },
        "results": rows,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )


def _heatmap_image(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _graph_overlay(result: InferenceResult, *, assignment_colors: bool) -> np.ndarray:
    image = Image.fromarray(result.rectification.rectified_rgb.copy())
    draw = ImageDraw.Draw(image)
    vertices = result.graph.pixel_vertices
    edges = result.graph.edges_vertices
    assignments = result.graph.edges_assignment
    for edge_idx, (v1, v2) in enumerate(edges):
        p0 = tuple(float(value) for value in vertices[int(v1)])
        p1 = tuple(float(value) for value in vertices[int(v2)])
        color = _assignment_color(int(assignments[edge_idx])) if assignment_colors else (255, 214, 10)
        width = 3 if int(assignments[edge_idx]) == 2 else 2
        draw.line([p0, p1], fill=color, width=width)
    for x, y in vertices:
        draw.ellipse((float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2), fill=(20, 20, 20))
    return np.asarray(image, dtype=np.uint8)


def _assignment_color(assignment: int) -> tuple[int, int, int]:
    return {
        0: (225, 29, 72),
        1: (37, 99, 235),
        2: (17, 24, 39),
        3: (107, 114, 128),
    }.get(assignment, (107, 114, 128))


def _graph_payload(graph: AttributedPlanarGraph) -> dict[str, Any]:
    return {
        "vertices_coords": graph.vertices_coords,
        "pixel_vertices": graph.pixel_vertices,
        "edges_vertices": graph.edges_vertices,
        "edges_assignment": graph.edges_assignment,
        "edge_support": graph.edge_support,
        "assignment_confidence": graph.assignment_confidence,
        "assignment_margin": graph.assignment_margin,
        "assignment_source": graph.assignment_source,
    }


def _manifest_threshold(manifest: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    inference = manifest.get("inference") or {}
    return float(inference.get("vectorizerThreshold", 0.65))


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
