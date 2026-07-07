from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_PATH = REPO_ROOT / "artifacts/evaluations/vertex-refiner-v1-phase0-baselines.json"


def test_phase0_baselines_record_clean15_gt_ceiling() -> None:
    snapshot = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    baselines = snapshot["baselines"]

    model = baselines["clean15_source_line_model_junctions"]["strictTopology"]
    gt = baselines["clean15_source_line_gt_junctions"]["strictTopology"]
    gt_merge1 = baselines["clean15_source_line_gt_junctions_merge1"]["strictTopology"]

    assert model["totalSamples"] == 15
    assert gt["totalSamples"] == 15
    assert gt_merge1["totalSamples"] == 15
    assert model["exactTopologySamples"] == 3
    assert gt["exactTopologySamples"] == 5
    assert gt_merge1["exactTopologySamples"] == 5
    assert model["vertexF1"] < gt["vertexF1"] < gt_merge1["vertexF1"]
    assert model["edgeF1"] < gt["edgeF1"] < gt_merge1["edgeF1"]
    assert gt_merge1["unmatchedGtVertices"] <= snapshot["promotionGates"][
        "clean15UnmatchedGtVerticesMax"
    ]


def test_phase0_baselines_record_box_pleat_and_candidate_coverage() -> None:
    snapshot = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))

    box_pleat = snapshot["baselines"]["box_pleat_native_source_line_model_junctions"]
    assert box_pleat["sampleCount"] == 179
    assert box_pleat["strictTopology"]["exactTopologySamples"] == 0
    assert box_pleat["strictTopology"]["vertexF1"] < 0.7

    coverage = snapshot["candidateCoverage"]
    model = coverage["clean15_source_line_model_junctions"]
    cv = coverage["clean15_source_line_cv_junctions"]
    assert model["gtEdgesEvaluated"] == cv["gtEdgesEvaluated"] == 2249
    assert model["adapterEndpointAvailable"] == 2209
    assert cv["adapterEndpointAvailable"] == 747
    assert model["candidateOracleRecall"] > cv["candidateOracleRecall"]
    assert model["selectedRecall"] > cv["selectedRecall"]
