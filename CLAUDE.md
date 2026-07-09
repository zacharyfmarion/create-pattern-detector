# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is the **data + training + evaluation** side of an origami
crease-pattern detection system. It trains a dense line/junction/assignment model
(`CPLineNet`) and a vertex-refiner, and exports them to ONNX.

It does **not** contain a production inference path. Everything downstream of the
model — decoding the dense ONNX signals into a FOLD graph, exact-solve topology
reconstruction, flat-fold verification, the browser/desktop app, and the
production correctness benchmarks — lives in the Rust monorepo
**`~/Documents/code/tree-maker-rust`** ("Ori Studio"). The Rust decode path is the
production source of truth. When the model changes, the handoff is a re-exported
ONNX file, tracked by pointers in that repo's `scripts/cp-detect/` directory
(`current-model.json`, `current-vertex-refiner.json`, `export-cpline-onnx.py`).

The older mask-first prototype (HRNet pixel head → skeletonize → GNN graph head)
and the earlier Python inference CLI have both been removed.

## Commands

### Setup
```bash
scripts/setup_python_env.sh
```
Reuses a shared dependency virtualenv across git worktrees and links it into the
current worktree as `.venv`. See `AGENTS.md` for `--adopt-local` and reset flags.

### Train
CPLineNet: `scripts/training/train_cpline_smoke.py` (roadmap-native path),
wrapped by the RunPod curriculum scripts (`scripts/training/run_cpline_runpod_*.sh`).
Vertex-refiner: `scripts/training/train_vertex_refiner.py`.
**Before training, exporting, or promoting a checkpoint, read
`docs/model-training-history.md`** — it is the source of truth for the current
model, checkpoint registry, and ONNX export provenance. Resolve the promoted
checkpoint through `artifacts/checkpoints/current-browser-model.json`, not a
hard-coded path. Run-config verification: `scripts/training/verify_cpline_run_config.py`.

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

CPLineNet predicts dense evidence fields (line probability, orientation,
junction heatmap/offset, M/V/B/U assignment logits, boundary/style heads). During
training and evaluation these fields are decoded through the in-repo deterministic
vectorizer to compute graph-quality metrics:

```
CPLineNet dense fields
  -> cpline_outputs_to_evidence   (src/vectorization/cpline_adapter.py)
  -> PlanarGraphBuilder /
     SquareTopologyDecoder        (src/vectorization/)
  -> edge assignment              (src/vectorization/edge_assignment.py)
  -> conservative repair          (src/vectorization/constraint_repair.py)
  -> metrics / quality report     (src/vectorization/metrics.py, quality_report.py)
```

**`src/vectorization/` is a training/eval instrument, not the production
decoder.** The shipped decode path (junction-first candidate graph, exact-solve
topology, flat-fold verification, browser runtime) is the Rust implementation in
`tree-maker-rust`. Keep the Python vectorizer honest as an eval baseline, but do
not treat it as production behavior.

## Key Source Directories

- `src/models/cpline_net.py` - CPLineNet: dense line/junction/assignment fields
- `src/models/backbone/hrnet.py` - HRNet feature extractor (shared, used by CPLineNet)
- `src/models/vertex_refiner.py` - Vertex refinement heads (V1/V2/V3)
- `src/models/losses/` - CPLineNet loss functions
- `src/vectorization/` - deterministic pixel-evidence → planar FOLD graph, used to
  compute training/eval metrics (not production)
- `src/evaluation/` - vertex-refiner evaluation (recall diagnostics, global merge)
- `src/data/` - datasets (`cpline_dataset.py`), FOLD parsing, augmentations, scraping
- `scripts/training/` - CPLineNet + vertex-refiner training and RunPod wrappers
- `scripts/evals/`, `scripts/vectorization/` - evaluation and vectorizer-baseline scripts

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
working notes** (not tracked), to avoid stale plans causing confusion. The tracked
source of truth for training/model state is `docs/model-training-history.md`.
Direction for the downstream decoder/app/benchmark lives in the `tree-maker-rust`
repo, not here.
