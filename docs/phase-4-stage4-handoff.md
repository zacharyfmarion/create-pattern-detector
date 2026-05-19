# Phase 4 Stage 4 Handoff

Status: Stage 4 is implemented as the conservative "honesty layer" on top of
the blessed Phase 3 CPLineNet checkpoint. It targets readable, rectified
1024px crease-pattern inputs. It does not retrain the model, productionize the
`cp-detect` CLI, or solve full photo rectification.

## What Stage 4 Does

Given a model prediction and a vectorized graph, Stage 4:

1. Samples CPLineNet `assignment_logits` along each `PlanarGraphResult` edge.
2. Emits edge assignments, confidence, margin, source, and support.
3. Applies conservative repairs that avoid inventing crease semantics.
4. Builds a quality report with one status:
   `valid`, `repaired`, `ambiguous`, `outside_v1_envelope`, or `failed`.
5. Exports FOLD with namespaced `cp_detector` metadata.
6. Produces UI-ready diagnostics for the local Stage Inspector app.

The implementation is intentionally cautious. Low-confidence visual evidence
becomes `U` rather than a guessed mountain or valley. Dense/tiny Rabbit Ear
cases are treated as V2 envelope warnings, not as a reason to silently output a
confident bad graph.

## Component Map

- `src/vectorization/edge_assignment.py`
  - `EdgeAssignmentConfig`
  - `assign_edges_from_logits`
  - `AttributedPlanarGraph`
- `src/vectorization/constraint_repair.py`
  - `RepairConfig`
  - `conservative_repair`
  - Optional `infer_assignments` local M/V completion.
- `src/vectorization/quality_report.py`
  - `QualityReportConfig`
  - `build_quality_report`
  - Origami-aware warning/status precedence.
- `src/vectorization/fold_writer.py`
  - `graph_to_fold_dict`
  - `save_fold`
- `src/vectorization/diagnostics.py`
  - `build_stage4_diagnostic_payload`
  - Missing/extra/matched edge classification for the inspector UI.
- `scripts/evals/eval_stage4_assignment.py`
  - Small deterministic assignment/report smoke eval.
- `scripts/evals/eval_stage4_checkpoint.py`
  - Checkpoint-backed Stage 4 benchmark with visual examples.
- `scripts/evals/compare_stage4_runs.py`
  - Metrics-gated comparison of two Stage 4 eval directories.
- `scripts/inspector/stage_inspector_server.py`
  - Local Python API server for interactive diagnostics.
- `web/stage-inspector/`
  - Vite + React + TypeScript Stage Inspector app.

## Setup From A Fresh Worktree

Use the shared Python environment and shared synthetic dataset. Do not create a
new per-worktree virtualenv or copy the dataset into the repo.

```bash
scripts/setup_python_env.sh
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
.venv/bin/python scripts/data/smoke_shared_synthetic_data.py \
  --root data/generated/synthetic/cp_training_mix_v1
```

The blessed checkpoint is registered in:

```text
artifacts/checkpoints/phase3-v1-cpline.json
```

The expected ignored local weight path is:

```text
checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
```

If the checkpoint is missing, do not retrain by default. First check previous
worktrees and the main repo checkout:

```bash
find /Users/zacharymarion/.codex/worktrees \
  -path '*/checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt' \
  -print
find /Users/zacharymarion/Documents/code \
  -path '*/checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt' \
  -print
```

Then copy the recovered file back to the expected ignored path and verify it
against the manifest:

```bash
mkdir -p checkpoints/runpod_phase3_curriculum/stage-balanced
shasum -a 256 checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
stat -f '%z bytes' checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
```

Expected manifest values:

- SHA-256:
  `a2a3d31d2ff80d3cf76952e463d965d03e6c46358a9802c2640a39b57d7732d8`
- Size: `138787408` bytes

## Reproduce The Current Stage 4 Eval

Assignment/report smoke:

```bash
.venv/bin/python scripts/evals/eval_stage4_assignment.py \
  --output-dir visualizations/stage4_assignment_smoke
```

Checkpoint-backed benchmark used for the current inspector data:

```bash
.venv/bin/python scripts/evals/eval_stage4_checkpoint.py \
  --checkpoint checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --split val \
  --profiles clean,line-style,print-light,dark-mode,photo-light,photo-dark \
  --samples-per-profile 24 \
  --family-sampling balanced \
  --threshold 0.65 \
  --batchnorm-mode batch-stats \
  --image-size 1024 \
  --max-edges 300 \
  --seed 19 \
  --output-dir visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24x6
```

The latest local baseline at that path had:

- 144 total files.
- Edge precision/recall: `0.9170 / 0.8965`.
- Vertex precision/recall: `0.9827 / 0.9348`.
- Assignment accuracy: `0.9881`.
- Border F1: `0.8181`.
- Structural validity rate: `0.9931`.
- Status counts: `valid=1`, `ambiguous=72`, `outside_v1_envelope=70`,
  `failed=1`.

Those status counts are stricter than graph recovery alone because Stage 4 also
warns on envelope, assignment, and origami diagnostics.

## Run The Stage Inspector

Backend:

```bash
.venv/bin/python scripts/inspector/stage_inspector_server.py --port 8765
```

Frontend:

```bash
cd web/stage-inspector
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

The Python server can also serve the built app:

```bash
cd web/stage-inspector
npm run build
cd ../..
.venv/bin/python scripts/inspector/stage_inspector_server.py --port 8765
```

The inspector reads `summary.json` and `per_sample_metrics.jsonl` from the eval
directory, then recomputes per-example diagnostics when needed.

## Near-Endpoint Crossing Repair Status

A gated `PlanarGraphBuilderConfig.repair_near_endpoint_crossings` option exists
for the specific failure mode where a high-support crease endpoint crosses an
unsplit border edge just before cleanup deletes the crease. It is off by
default.

Enable it in eval:

```bash
.venv/bin/python scripts/evals/eval_stage4_checkpoint.py \
  --checkpoint checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --split val \
  --profiles clean,line-style,print-light,dark-mode,photo-light,photo-dark \
  --samples-per-profile 24 \
  --family-sampling balanced \
  --threshold 0.65 \
  --batchnorm-mode batch-stats \
  --image-size 1024 \
  --max-edges 300 \
  --seed 19 \
  --repair-near-endpoint-crossings \
  --output-dir visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24_candidate
```

Compare baseline and candidate:

```bash
.venv/bin/python scripts/evals/compare_stage4_runs.py \
  --baseline-dir visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24x6 \
  --candidate-dir visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24_candidate \
  --output-dir visualizations/stage4_checkpoint_eval/phase3_stage4_1024_n24_comparison
```

The most recent local gated comparison improved the target sample
`rabbit_ear_fold_program_v1-5wk08-000155` by matching two additional GT edges,
but the aggregate balanced score delta was only `+0.00043`, below the
pre-registered `+0.002` promotion threshold. Because of that, the repair remains
disabled by default until a larger benchmark or better variant clears the gate.

## Tests

Fast Stage 4 tests:

```bash
.venv/bin/python -m pytest \
  tests/test_stage4_vectorization.py \
  tests/test_stage4_diagnostics.py \
  tests/test_planar_graph_builder.py
```

Checkpoint-backed target regression, gated because it needs local weights and
dataset artifacts:

```bash
CP_RUN_CHECKPOINT_TESTS=1 .venv/bin/python -m pytest tests/test_stage4_near_endpoint_regression.py
```

Frontend build smoke:

```bash
cd web/stage-inspector
npm run build
```

## Rectifier Phase Placement

The square rectifier is split across Phase 5 and Phase 6 in the roadmap.

Phase 5 should add the production inference shell:

- `cp-detect`
- `cp-detect --rectified` first
- batch mode
- debug artifacts
- JSON reports

In Phase 5, the rectifier can exist as an interface or lightweight preprocessor,
but the first supported path should still trust already-square inputs.

Phase 6 is where the real square/photo rectifier becomes a benchmarked feature:

- Detect or choose the paper/CP square.
- Perspective-warp to the canonical square input.
- Add rectification tests.
- Build a small real-image benchmark.
- Fine-tune and validate without regressing synthetic performance.

So the practical answer is:

- Add the rectifier API seam in Phase 5.
- Make `--rectified` the first production mode.
- Make automatic square rectification a Phase 6 robustness deliverable.

## Non-Goals For Stage 4

- No new model training.
- No checkpoint replacement.
- No production `cp-detect` CLI yet.
- No automatic symmetry enforcement.
- No full real-photo rectification.
- No GNN graph correctness layer.
- No default inferred M/V semantics.
