# Crease Pattern Detector

Training and data pipeline for the origami crease-pattern detection model: a
dense line/junction/assignment model (`CPLineNet`) plus a vertex-refiner, trained
here and exported to ONNX.

## Scope

This repository owns **data generation, training, and evaluation** only. It
produces the ONNX models and the deterministic ML eval spec.

Everything downstream of the model — decoding the dense ONNX signals into a FOLD
graph, exact-solve topology reconstruction, flat-fold verification, the
browser/desktop app, and the production correctness benchmarks — lives in the
Rust monorepo **`~/Documents/code/tree-maker-rust`** ("Ori Studio"). Once a model
is exported to ONNX, all post-processing happens there; the Rust decode path is
the production source of truth. There is no Python inference/CLI in this repo.

Model/vertex-refiner pointers and the ONNX export scripts live in that repo under
`scripts/cp-detect/` (`current-model.json`, `current-vertex-refiner.json`,
`export-cpline-onnx.py`).

## Installation

```bash
scripts/setup_python_env.sh
```

The setup script reuses a shared dependency virtualenv across git worktrees and
links it into the current worktree as `.venv`. See `AGENTS.md` for flags.

## Model Source Of Truth

Before training, exporting, or promoting a checkpoint, read
`docs/model-training-history.md`. It records the current downstream/browser model,
the checkpoint registry entry, ONNX export provenance, and why older close-pair
or diagnostic runs were not promoted. Resolve the current promoted checkpoint
through `artifacts/checkpoints/current-browser-model.json` instead of copying a
checkpoint path into docs.

## Train Model

CPLineNet training is the roadmap-native path in
`scripts/training/train_cpline_smoke.py`, wrapped by the RunPod curriculum
scripts (`scripts/training/run_cpline_runpod_*.sh`). Vertex-refiner training is
`scripts/training/train_vertex_refiner.py`. Read `docs/model-training-history.md`
and `docs/runpod-quickstart.md` first.

## Evaluate

Training evaluates predicted dense fields through the in-repo deterministic
vectorizer (`src/vectorization/`) to produce graph-quality metrics. Note this
vectorizer is a **training/eval instrument**, not the production decoder — the
shipped decode path is the Rust implementation in `tree-maker-rust`.

The box-pleat / grid-line-suppression eval is defined by
`docs/evals/box-pleat-native-v1.md` and `eval_specs/box_pleat_native_v1.json`,
verified by path-independent canonical FOLD fingerprints. The Rust repo consumes
that spec to build its product-side correctness pack.
