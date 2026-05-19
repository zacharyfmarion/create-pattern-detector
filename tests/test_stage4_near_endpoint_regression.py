import os
from pathlib import Path

import pytest

from scripts.inspector.stage_inspector_server import (
    DEFAULT_CACHE,
    DEFAULT_CHECKPOINT,
    DEFAULT_EVAL_DIR,
    DEFAULT_MANIFEST,
    StageInspectorService,
)


@pytest.mark.skipif(
    os.environ.get("CP_RUN_CHECKPOINT_TESTS") != "1",
    reason="set CP_RUN_CHECKPOINT_TESTS=1 to run checkpoint-backed Stage 4 regression tests",
)
def test_rabbit_ear_5wk08_000155_recovers_known_near_border_edges() -> None:
    required = [DEFAULT_EVAL_DIR, DEFAULT_CHECKPOINT, DEFAULT_MANIFEST]
    if not all(Path(path).exists() for path in required):
        pytest.skip("checkpoint eval artifacts, checkpoint, or manifest are unavailable")

    service = StageInspectorService(
        eval_dir=Path(DEFAULT_EVAL_DIR).resolve(),
        checkpoint=Path(DEFAULT_CHECKPOINT).resolve(),
        manifest=Path(DEFAULT_MANIFEST).resolve(),
        cache_dir=Path(DEFAULT_CACHE).resolve(),
        device_request="cpu",
    )
    row = next(
        item
        for item in service.rows()
        if item["profile"] == "clean" and item["id"] == "rabbit_ear_fold_program_v1-5wk08-000155"
    )

    diagnostic = service._compute_diagnostic(
        row,
        params={"repairNearEndpointCrossings": True},
        cache_key="test_near_endpoint_regression",
    )
    gt_edges = diagnostic["graph"]["groundTruth"]["edges"]
    edge_state = {edge["id"]: edge["match"]["state"] for edge in gt_edges}

    assert edge_state[53] == "matched"
    assert edge_state[68] == "matched"
