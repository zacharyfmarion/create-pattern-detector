from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verify_vertex_refiner_run_config_accepts_expected_values(tmp_path: Path) -> None:
    run_config = tmp_path / "run_config.json"
    run_config.write_text(
        json.dumps(
            {
                "device": "cuda",
                "auxiliary_mode": "zero",
                "image_size": 1024,
                "include_gt_training_anchors": True,
                "include_val_gt_anchors": False,
                "lr": 0.0003,
                "model_version": "v2",
                "abort_loss_threshold": 1000.0,
                "boundary_gt_anchor_repeats": 3,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/training/verify_vertex_refiner_run_config.py",
            "--run-config",
            str(run_config),
            "--expect-str",
            "device=cuda",
            "--expect-str",
            "auxiliary_mode=zero",
            "--expect-int",
            "image_size=1024",
            "--expect-bool",
            "include_gt_training_anchors=true",
            "--expect-bool",
            "include_val_gt_anchors=false",
            "--expect-float",
            "lr=0.0003",
            "--expect-str",
            "model_version=v2",
            "--expect-float",
            "abort_loss_threshold=1000",
            "--expect-int",
            "boundary_gt_anchor_repeats=3",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_verify_vertex_refiner_run_config_rejects_mismatch(tmp_path: Path) -> None:
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({"auxiliary_mode": "rendered-labels"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/training/verify_vertex_refiner_run_config.py",
            "--run-config",
            str(run_config),
            "--expect-str",
            "auxiliary_mode=zero",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "auxiliary_mode" in result.stdout
