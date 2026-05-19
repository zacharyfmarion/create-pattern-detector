#!/usr/bin/env python3
"""Local API/static server for the Vite Stage Inspector app."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_dataset import CplineFoldDataset, cpline_collate  # noqa: E402
from src.models import CPLineNet  # noqa: E402
from src.models.batchnorm import model_eval_with_batchnorm_mode  # noqa: E402
from src.vectorization import (  # noqa: E402
    EdgeAssignmentConfig,
    PlanarGraphBuilder,
    PlanarGraphBuilderConfig,
    QualityReportConfig,
    RepairConfig,
    VectorizerEvidence,
    attribute_graph_from_logits,
    build_quality_report,
    build_stage4_diagnostic_payload,
    conservative_repair,
    evaluate_graph,
)

DEFAULT_EVAL_DIR = Path("visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24x6")
DEFAULT_CHECKPOINT = Path("checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt")
DEFAULT_MANIFEST = Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl")
DEFAULT_DIST = Path("web/stage-inspector/dist")
DEFAULT_CACHE = Path("visualizations/stage4_inspector/cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frontend-dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = StageInspectorService(
        eval_dir=_resolve(args.eval_dir),
        checkpoint=_resolve(args.checkpoint),
        manifest=_resolve(args.manifest),
        cache_dir=_resolve(args.cache_dir),
        device_request=args.device,
    )
    dist_dir = _resolve(args.frontend_dist)

    class Handler(StageInspectorHandler):
        inspector = service
        frontend_dist = dist_dir

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Stage Inspector server listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Stage Inspector server.", flush=True)
    finally:
        server.server_close()


class StageInspectorService:
    def __init__(
        self,
        *,
        eval_dir: Path,
        checkpoint: Path,
        manifest: Path,
        cache_dir: Path,
        device_request: str,
    ) -> None:
        self.eval_dir = eval_dir
        self.checkpoint = checkpoint
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_request = device_request
        self._summary: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] | None = None
        self._model: CPLineNet | None = None
        self._device: torch.device | None = None

    def stages(self) -> dict[str, Any]:
        return {
            "stages": [
                {
                    "id": "stage1",
                    "label": "Stage 1",
                    "title": "Synthetic Data",
                    "status": "scaffolded",
                },
                {
                    "id": "stage2",
                    "label": "Stage 2",
                    "title": "Deterministic Vectorizer",
                    "status": "scaffolded",
                },
                {
                    "id": "stage3",
                    "label": "Stage 3",
                    "title": "CPLineNet Evidence",
                    "status": "scaffolded",
                },
                {
                    "id": "stage4",
                    "label": "Stage 4",
                    "title": "Assignments, Repair, Reports",
                    "status": "implemented",
                },
            ]
        }

    def examples_index(self) -> dict[str, Any]:
        rows = [self._row_for_client(row) for row in self.rows()]
        profiles = sorted({row["profile"] for row in rows})
        families = sorted({row["family"] for row in rows})
        statuses = sorted({row["status"] for row in rows})
        warnings = sorted({warning for row in rows for warning in row["warnings"]})
        repairs = sorted({repair for row in rows for repair in row["repairs"]})
        return {
            "summary": self.summary(),
            "rows": rows,
            "filters": {
                "profiles": profiles,
                "families": families,
                "statuses": statuses,
                "warnings": warnings,
                "repairs": repairs,
            },
            "counts": {
                "status": dict(Counter(row["status"] for row in rows)),
                "warnings": dict(Counter(warning for row in rows for warning in row["warnings"])),
                "profiles": dict(Counter(row["profile"] for row in rows)),
                "families": dict(Counter(row["family"] for row in rows)),
            },
            "metricRanges": {
                "edgeRecall": _range(row["edgeRecall"] for row in rows),
                "edgePrecision": _range(row["edgePrecision"] for row in rows),
                "assignmentAccuracy": _range(row["assignmentAccuracy"] for row in rows),
            },
        }

    def get_example(self, example_key: str) -> dict[str, Any]:
        row = self._find_row(example_key)
        return self._diagnostic_for_row(row, params={}, force=False)

    def recompute(self, body: dict[str, Any]) -> dict[str, Any]:
        key = str(body.get("exampleKey") or body.get("key") or "")
        if not key:
            raise HttpError(HTTPStatus.BAD_REQUEST, "POST body must include exampleKey")
        row = self._find_row(key)
        params = {
            "threshold": body.get("threshold"),
            "inferAssignments": body.get("inferAssignments"),
            "repair": body.get("repair") or {},
            "report": body.get("report") or {},
        }
        return self._diagnostic_for_row(row, params=params, force=True)

    def summary(self) -> dict[str, Any]:
        if self._summary is None:
            path = self.eval_dir / "summary.json"
            if not path.exists():
                raise HttpError(
                    HTTPStatus.NOT_FOUND,
                    f"Stage 4 summary not found at {path.relative_to(REPO_ROOT)}",
                )
            self._summary = json.loads(path.read_text(encoding="utf-8"))
        return self._summary

    def rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            path = self.eval_dir / "per_sample_metrics.jsonl"
            if not path.exists():
                raise HttpError(
                    HTTPStatus.NOT_FOUND,
                    f"Stage 4 metrics not found at {path.relative_to(REPO_ROOT)}",
                )
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            self._rows = rows
        return self._rows

    def _diagnostic_for_row(
        self,
        row: dict[str, Any],
        *,
        params: dict[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        cache_key = _cache_key(row, params)
        diagnostic_path = self.cache_dir / "diagnostics" / f"{cache_key}.json"
        if diagnostic_path.exists() and not force:
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            diagnostic["cache"] = {"hit": True, "key": cache_key}
            return diagnostic

        diagnostic = self._compute_diagnostic(row, params=params, cache_key=cache_key)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
        diagnostic["cache"] = {"hit": False, "key": cache_key}
        return diagnostic

    def _compute_diagnostic(
        self,
        row: dict[str, Any],
        *,
        params: dict[str, Any],
        cache_key: str,
    ) -> dict[str, Any]:
        self._assert_inputs_available()
        summary = self.summary()
        threshold = _float_param(params.get("threshold"), float(summary.get("threshold", 0.65)))
        infer_assignments = bool(params.get("inferAssignments") or summary.get("infer_assignments", False))
        repair_config = RepairConfig(
            image_size=int(summary.get("image_size", 1024)),
            **_known_config_overrides(params.get("repair") or {}, RepairConfig),
        )
        report_config = QualityReportConfig(
            image_size=int(summary.get("image_size", 1024)),
            **_known_config_overrides(params.get("report") or {}, QualityReportConfig),
        )
        assignment_config = EdgeAssignmentConfig()
        builder = make_builder(int(summary.get("image_size", 1024)), threshold)
        batch = self._batch_for_row(row)
        model, device = self._load_model()
        with torch.no_grad(), model_eval_with_batchnorm_mode(
            model,
            batchnorm_mode=str(summary.get("batchnorm_mode", "batch-stats")),
        ):
            outputs = model(batch["image"].to(device))
            line_prob = torch.sigmoid(outputs["line_logits"][0, 0]).detach().cpu().numpy()
            angle = outputs["angle"][0].detach().cpu().permute(1, 2, 0).numpy()
            junction_heatmap = torch.sigmoid(outputs["junction_logits"][0, 0]).detach().cpu().numpy()
            evidence = VectorizerEvidence(
                line_prob=line_prob.astype(np.float32),
                angle=angle.astype(np.float32),
                junction_heatmap=junction_heatmap.astype(np.float32),
                assignment_labels=None,
            )
            graph_result = builder.build(evidence)
            attributed = attribute_graph_from_logits(
                graph_result,
                outputs["assignment_logits"][0].detach().cpu(),
                line_prob=line_prob,
                config=assignment_config,
            )
            repair = conservative_repair(
                attributed,
                line_prob=line_prob,
                config=repair_config,
                infer_assignments=infer_assignments,
            )
            report = build_quality_report(
                repair.graph,
                repair_actions=repair.actions,
                config=report_config,
            )

        gt_graph = batch["graph"][0]
        image_size = int(summary.get("image_size", 1024))
        vertex_tolerance_px = max(3.0, 5.0 * image_size / 768)
        metrics = evaluate_graph(
            repair.graph.to_planar_result(),
            gt_vertices=gt_graph["vertices"].numpy(),
            gt_edges=gt_graph["edges"].numpy(),
            gt_assignments=gt_graph["assignments"].numpy(),
            vertex_tolerance_px=vertex_tolerance_px,
        ).to_dict()
        image = (batch["image"][0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        asset_path = self._write_input_asset(cache_key, image)
        report_dict = report.to_dict()
        row_payload = {
            **row,
            "status": report.status,
            "warnings": [warning["code"] for warning in report_dict["warnings"]],
            "repairs": [action["code"] for action in report_dict["repair_actions"]],
            "threshold": threshold,
            "inferAssignments": infer_assignments,
            **metrics,
        }
        diagnostic = build_stage4_diagnostic_payload(
            row=row_payload,
            image_url=self._asset_url(asset_path),
            image_size=image_size,
            gt_vertices=gt_graph["vertices"].numpy(),
            gt_edges=gt_graph["edges"].numpy(),
            gt_assignments=gt_graph["assignments"].numpy(),
            pred_vertices=repair.graph.pixel_vertices,
            pred_edges=repair.graph.edges_vertices,
            pred_assignments=repair.graph.edges_assignment,
            pred_edge_support=repair.graph.edge_support,
            pred_assignment_confidence=repair.graph.assignment_confidence,
            pred_assignment_margin=repair.graph.assignment_margin,
            pred_assignment_source=repair.graph.assignment_source,
            report=report_dict,
            metrics=metrics,
            vertex_tolerance_px=vertex_tolerance_px,
        )
        diagnostic["recomputeParams"] = {
            "threshold": threshold,
            "inferAssignments": infer_assignments,
            "repair": asdict(repair_config),
            "report": asdict(report_config),
        }
        return diagnostic

    def _batch_for_row(self, row: dict[str, Any]) -> dict[str, Any]:
        summary = self.summary()
        profile = str(row["profile"])
        image_size = int(summary.get("image_size", 1024))
        max_edges = int(summary.get("max_edges", 300))
        split = str(summary.get("split", "val"))
        seed = int(summary.get("seed", 19))
        family_sampling = str(summary.get("family_sampling", "balanced"))
        limit = max(int(summary.get("samples_per_profile", 24)), int(row.get("sample_index", 0)) + 1)
        dataset = CplineFoldDataset(
            self.manifest,
            split=split,
            limit=limit,
            max_edges=max_edges,
            image_size=image_size,
            augment_profile=profile,
            seed=seed,
            family_sampling=family_sampling,
        )
        item = None
        # CplineFoldDataset advances a profile RNG for each rendered sample. Replay
        # earlier rows so recompute matches the original sequential validation run.
        for index in range(int(row["sample_index"]) + 1):
            item = dataset[index]
        assert item is not None
        if item["meta"]["id"] != row["id"]:
            raise HttpError(
                HTTPStatus.CONFLICT,
                f"Dataset selection mismatch: expected {row['id']} but got {item['meta']['id']}",
            )
        return cpline_collate([item])

    def _load_model(self) -> tuple[CPLineNet, torch.device]:
        if self._model is not None and self._device is not None:
            return self._model, self._device
        device = select_device(self.device_request)
        loaded = torch.load(self.checkpoint, map_location=device, weights_only=False)
        config = loaded.get("config", {})
        model = CPLineNet(
            backbone=config.get("backbone", "hrnet_w18"),
            pretrained=False,
            hidden_channels=int(config.get("hidden_channels", 128)),
        ).to(device)
        model.load_state_dict(loaded["model_state_dict"])
        model.eval()
        self._model = model
        self._device = device
        return model, device

    def _assert_inputs_available(self) -> None:
        missing = []
        for label, path in [("checkpoint", self.checkpoint), ("manifest", self.manifest)]:
            if not path.exists():
                missing.append(f"{label}: {path.relative_to(REPO_ROOT)}")
        if missing:
            raise HttpError(
                HTTPStatus.NOT_FOUND,
                "Missing required Stage 4 input(s): " + ", ".join(missing),
            )

    def _find_row(self, example_key: str) -> dict[str, Any]:
        for row in self.rows():
            if _example_key(row) == example_key:
                return row
        raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown Stage 4 example: {example_key}")

    def _row_for_client(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": _example_key(row),
            "id": row["id"],
            "sampleIndex": int(row["sample_index"]),
            "family": row["family"],
            "bucket": row["bucket"],
            "profile": row["profile"],
            "status": row["status"],
            "warnings": list(row.get("warnings", [])),
            "repairs": list(row.get("repairs", [])),
            "edgePrecision": float(row["edge_precision"]),
            "edgeRecall": float(row["edge_recall"]),
            "vertexPrecision": float(row["vertex_precision"]),
            "vertexRecall": float(row["vertex_recall"]),
            "assignmentAccuracy": float(row["assignment_accuracy"]),
            "structuralValid": bool((row.get("structural_validity") or {}).get("valid", False)),
            "predEdges": int(row["pred_edges"]),
            "gtEdges": int(row["gt_edges"]),
            "unknownEdges": int(row.get("unknown_edges", 0)),
            "observedEdges": int(row.get("observed_edges", 0)),
        }

    def _write_input_asset(self, cache_key: str, image: np.ndarray) -> Path:
        path = self.cache_dir / "assets" / cache_key / "input.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        return path

    @staticmethod
    def _asset_url(path: Path) -> str:
        return "/api/assets/" + path.relative_to(REPO_ROOT).as_posix()


class StageInspectorHandler(SimpleHTTPRequestHandler):
    inspector: StageInspectorService
    frontend_dist: Path

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/api/stages":
                return self._send_json(self.inspector.stages())
            if path == "/api/stage4/examples":
                return self._send_json(self.inspector.examples_index())
            if path.startswith("/api/stage4/examples/"):
                key = path.removeprefix("/api/stage4/examples/")
                return self._send_json(self.inspector.get_example(key))
            if path.startswith("/api/assets/"):
                asset = path.removeprefix("/api/assets/")
                return self._send_asset(asset)
            if path.startswith("/api/"):
                raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")
            return self._send_frontend(path)
        except HttpError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception as exc:  # noqa: BLE001 - local dev server should explain failures
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            body = self._read_json_body()
            if path == "/api/stage4/recompute":
                return self._send_json(self.inspector.recompute(body))
            raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")
        except HttpError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        encoded = json.dumps({"error": message, "status": int(status)}).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_asset(self, asset: str) -> None:
        path = (REPO_ROOT / asset).resolve()
        allowed_roots = [
            self.inspector.cache_dir.resolve(),
            self.inspector.eval_dir.resolve(),
        ]
        if not any(_is_relative_to(path, root) for root in allowed_roots):
            raise HttpError(HTTPStatus.FORBIDDEN, "Asset path is outside inspector asset roots")
        if not path.exists() or not path.is_file():
            raise HttpError(HTTPStatus.NOT_FOUND, f"Asset not found: {asset}")
        self.path = path.as_posix()
        with path.open("rb") as handle:
            data = handle.read()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", self.guess_type(path.as_posix()))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_frontend(self, path: str) -> None:
        if not self.frontend_dist.exists():
            payload = {
                "message": "Stage Inspector frontend has not been built yet.",
                "dev": "Run `cd web/stage-inspector && npm run dev` for Vite dev mode.",
            }
            return self._send_json(payload)
        requested = (self.frontend_dist / path.lstrip("/")).resolve()
        if _is_relative_to(requested, self.frontend_dist.resolve()) and requested.is_file():
            target = requested
        else:
            target = self.frontend_dist / "index.html"
        with target.open("rb") as handle:
            data = handle.read()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", self.guess_type(target.as_posix()))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[stage-inspector] " + format % args + "\n")


class HttpError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_builder(image_size: int, threshold: float) -> PlanarGraphBuilder:
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
        )
    )


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise HttpError(HTTPStatus.BAD_REQUEST, "MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HttpError(HTTPStatus.BAD_REQUEST, "CUDA requested but unavailable")
    return device


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _example_key(row: dict[str, Any]) -> str:
    raw_id = str(row.get("id", "example"))
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_id)[:90]
    return f"{row.get('profile', 'profile')}__{int(row.get('sample_index', 0)):03d}__{safe}"


def _cache_key(row: dict[str, Any], params: dict[str, Any]) -> str:
    base = _example_key(row)
    normalized = json.dumps({"version": 2, "params": params}, sort_keys=True, default=str)
    suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{suffix}"


def _range(values: Any) -> dict[str, float]:
    items = [float(value) for value in values]
    return {"min": min(items), "max": max(items)} if items else {"min": 0.0, "max": 0.0}


def _float_param(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _known_config_overrides(values: dict[str, Any], config_type: type[Any]) -> dict[str, Any]:
    allowed = set(getattr(config_type, "__dataclass_fields__", {}))
    return {key: value for key, value in values.items() if key in allowed and key != "image_size"}


def _is_relative_to(path: Path, root: Path) -> bool:
    with contextlib.suppress(ValueError):
        path.relative_to(root)
        return True
    return False


if __name__ == "__main__":
    main()
