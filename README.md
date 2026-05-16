# Crease Pattern Detector

A deep learning model for detecting and classifying crease patterns in origami diagrams.

## Installation

```bash
scripts/setup_python_env.sh
```

The setup script reuses a shared dependency virtualenv across git worktrees and
links it into the current worktree as `.venv`.

## Usage

### Validate Pipeline

```bash
python scripts/validation/validate_pipeline.py --fold-dir data/output/synthetic/raw/tier-a
```

### Train Model

```bash
python scripts/training/train_pixel_head.py --fold-dir data/output/synthetic/raw/tier-a
```
