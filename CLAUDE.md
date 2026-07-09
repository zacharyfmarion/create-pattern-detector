# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning model for detecting and classifying crease patterns in origami
diagrams. The system extracts valid origami graphs from images while maintaining
mathematical constraints (Kawasaki/Maekawa theorems, 2-colorability).

The production system is **vector-first**: a learned line/junction/assignment
model (`CPLineNet`) produces dense visual evidence, and a deterministic
geometry stage builds and repairs the planar FOLD graph. The older mask-first
prototype (HRNet pixel head → skeletonize → GNN graph head) has been removed;
see `ROADMAP.md` for the architecture decision.

## Commands

### Setup
```bash
scripts/setup_python_env.sh
```
Reuses a shared dependency virtualenv across git worktrees and links it into the
current worktree as `.venv`. See `AGENTS.md` for `--adopt-local` and reset flags.

### Detect a CP image
```bash
cp-detect --rectified input.png \
  --output output.fold \
  --report output.report.json \
  --debug-dir debug/
```
Entry point: `src/inference/cli.py` → `src/inference/pipeline.py`. See
`docs/phase-5-inference-cli.md` for checkpoint recovery, output layout, and
debug artifacts.

### Train the model
CPLineNet training is the roadmap-native path in
`scripts/training/train_cpline_smoke.py`, wrapped by the RunPod curriculum
scripts (`scripts/training/run_cpline_runpod_*.sh`). **Before training,
exporting, or promoting a checkpoint, read `docs/model-training-history.md`** —
it is the source of truth for the current model, checkpoint registry, and ONNX
export provenance. Resolve the promoted checkpoint through
`artifacts/checkpoints/current-browser-model.json`, not a hard-coded path.

Run-config verification: `scripts/training/verify_cpline_run_config.py`.

### Code Quality
```bash
black src/ scripts/          # Format (line-length=100)
ruff check src/ scripts/     # Lint
mypy src/                    # Type check
pytest tests/                # Test suite
```

### Data Generation (TypeScript/Bun)
```bash
cd data/ts-generation
bun run src/generate-dataset.ts --count 1000 --method mixed
bun run src/validate-scraped.ts
```

## Architecture

**Inference pipeline** (`src/inference/pipeline.py`):
```
input image
  -> SquareRectifier            (src/inference/rectifier.py)
  -> CPLineNet                  (src/models/cpline_net.py; HRNet backbone)
  -> cpline_outputs_to_evidence (src/vectorization/cpline_adapter.py)
  -> PlanarGraphBuilder /
     SquareTopologyDecoder      (src/vectorization/)
  -> EdgeAssignment             (src/vectorization/edge_assignment.py)
  -> OrigamiConstraintRepair    (src/vectorization/constraint_repair.py)
  -> QualityReport + FOLDWriter (src/vectorization/quality_report.py, fold_writer.py)
```
The learned model detects visual evidence for lines, junctions, and
assignments. The final graph is built by deterministic geometry and validated as
a planar FOLD graph. Vertex refinement lives in `src/models/vertex_refiner.py`.

## Key Source Directories

- `src/models/cpline_net.py` - CPLineNet: dense line/junction/assignment fields
- `src/models/backbone/hrnet.py` - HRNet feature extractor (shared, used by CPLineNet)
- `src/models/vertex_refiner.py` - Vertex refinement heads (V1/V2/V3)
- `src/models/losses/` - CPLineNet loss functions
- `src/vectorization/` - Deterministic pixel-evidence → planar FOLD graph:
  - `cpline_adapter.py` - CPLineNet outputs → vectorizer evidence
  - `planar_graph_builder.py` / `square_topology_decoder.py` - graph construction
  - `edge_assignment.py` - M/V/B/U assignment from logits
  - `constraint_repair.py` - conservative origami-constraint repair
  - `quality_report.py` / `diagnostics.py` - validation + honest failure reports
  - `fold_writer.py` - FOLD output
- `src/inference/` - CLI, pipeline orchestration, rectifier
- `src/data/` - datasets (`cpline_dataset.py`), FOLD parsing, augmentations
- `scripts/training/` - CPLineNet + vertex-refiner training and RunPod wrappers

## Origami Domain Constraints

- **Edge Assignments:** M (mountain), V (valley), B (border), U (unassigned). F and C map to U.
- **Kawasaki theorem:** Alternating angle sum around interior vertex = π
- **Maekawa theorem:** |#Mountain - #Valley| = 2 at interior vertices
- **Even degree:** Interior vertices must have even degree
- **Square domain:** Patterns normalized to [0,1]×[0,1]

## Data Tiers

Located in `data/output/synthetic/raw/` and `data/output/scraped/`:
- **Tier S:** Passes all validation (Maekawa, Kawasaki, 2-colorability, flat-foldability)
- **Tier A:** Passes local validation only (primary training data)
- **Rejected:** Fails validation

## Planning Docs

Implementation plans live in `implementation-plan/` and are **git-ignored local
working notes** (not tracked), to avoid stale plans causing confusion. The
tracked source of truth for direction is `ROADMAP.md`; for training/model state
it is `docs/model-training-history.md`.
