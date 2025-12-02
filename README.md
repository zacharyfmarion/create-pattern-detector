# Crease Pattern Detector

A deep learning model for detecting and classifying crease patterns in origami diagrams.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Validate Pipeline

```bash
python scripts/validate_pipeline.py --fold-dir data/output/synthetic/raw/tier-a
```

### Train Model

```bash
python scripts/train.py --fold-dir data/output/synthetic/raw/tier-a
```
