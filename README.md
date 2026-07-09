# Crease Pattern Detector

A deep learning model for detecting and classifying crease patterns in origami diagrams.

## Installation

```bash
scripts/setup_python_env.sh
```

The setup script reuses a shared dependency virtualenv across git worktrees and
links it into the current worktree as `.venv`.

## Model Source Of Truth

Before using, exporting, or retraining CPLineNet, read
`docs/model-training-history.md`. It records the current downstream/browser model,
the checkpoint registry entry, ONNX export provenance, and why older close-pair
or diagnostic runs were not promoted. Resolve the current promoted checkpoint
through `artifacts/checkpoints/current-browser-model.json` instead of copying a
checkpoint path into docs.

## Usage

### Detect A CP Image

```bash
cp-detect --rectified input.png \
  --output output.fold \
  --report output.report.json \
  --debug-dir debug/
```

Phase 5 supports readable CP images and page/screenshot images with a visible
crease-pattern border. The rectifier crops the CP panel, perspective-warps it to
the canonical square, preserves the detected border inside a small clean margin,
and falls back to resize/pad only when the panel cannot be detected confidently.
Full arbitrary photo/document rectification remains Phase 6. Transparent inputs
use `--alpha-matte auto` by default so dark-mode CPs are not flattened onto white
unless the matte is truly ambiguous.

See `docs/phase-5-inference-cli.md` for checkpoint recovery, output layout, and
debug artifacts.

### Box-Pleat Native Eval

Use `docs/evals/box-pleat-native-v1.md` and
`eval_specs/box_pleat_native_v1.json` to regenerate the deterministic
box-pleat candidate set from the native converted-FOLD corpus. The eval is
verified by path-independent FOLD content fingerprints instead of tracked local
paths.

### Train Model

CPLineNet training is the roadmap-native path in
`scripts/training/train_cpline_smoke.py`, wrapped by the RunPod curriculum
scripts (`scripts/training/run_cpline_runpod_*.sh`). Read
`docs/model-training-history.md` before training, exporting, or promoting a
checkpoint.
