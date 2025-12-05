# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning model for detecting and classifying crease patterns in origami diagrams. The system extracts valid origami graphs from images while maintaining mathematical constraints (Kawasaki/Maekawa theorems, 2-colorability).

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Training
```bash
python scripts/training/train_pixel_head.py --fold-dir full_train --epochs 5 --save-every 1 --batch-size 8 --image-size 512 --wandb --wandb-project cp-detector --resume checkpoints/<checkpoint>.pt

python scripts/training/train_graph_head.py \
   --pixel-checkpoint checkpoints/checkpoint_epoch_8.pt \
   --data-dir data/training/full-training \
   --epochs 30 \
   --batch-size 4 \
   --lr 5e-4 \
   --checkpoint-dir checkpoints/graph
```

### Validation/Testing
```bash
python scripts/validation/validate_pipeline.py --fold-dir data/output/synthetic/raw/tier-a
python scripts/debug/test_graph_extraction.py
python scripts/validation/sanity_check.py
```

### Data Generation (TypeScript/Bun)
```bash
cd data/ts-generation
bun run src/generate-dataset.ts --count 1000 --method mixed
bun run src/validate-scraped.ts
```

### Code Quality
```bash
black src/ scripts/          # Format (line-length=100)
ruff check src/ scripts/     # Lint
mypy src/                    # Type check
```

## Architecture

**Three-Phase Training Pipeline:**

1. **Phase 1: Pixel Head** (50 epochs) - Train backbone + pixel head
   - Input: Rendered crease pattern image (1024×1024)
   - Model: HRNet-W32 backbone → Pixel Head (3 branches)
   - Outputs: Segmentation mask (5 classes), orientation field (cos/sin θ), junction heatmap

2. **Post-processing: Graph Extraction** (non-learned)
   - Skeletonize segmentation → 1px wide lines
   - Detect vertices from junction heatmap peaks + skeleton junctions
   - Trace edges along skeleton between vertices
   - Output: Over-complete candidate graph (biased toward recall)

3. **Phase 2: Graph Head** (30 epochs) - Freeze pixel head, train GNN
   - Input: Candidate graph + backbone features
   - Model: 4-layer GAT with edge/node updates
   - Outputs: Edge existence (keep/drop), edge assignment (M/V/B/U), vertex refinement (±5px)

4. **Phase 3: End-to-End** (20 epochs) - Fine-tune all components together

**Data Flow:**
```
FOLD files → Rendered images → Pixel Head → Post-processing → Candidate graph → Graph Head → Final pattern
```

## Key Source Directories

- `src/models/cp_detector.py` - Main detector combining backbone + heads
- `src/models/backbone/hrnet.py` - HRNet feature extractor (stride 4)
- `src/models/heads/pixel_head.py` - Multi-task prediction (seg, orientation, junction)
- `src/models/graph/` - Phase 2 GNN components (graph_head.py, layers.py, features.py)
- `src/postprocessing/` - Pixel → Graph conversion:
  - `skeletonize.py` - Morphological thinning
  - `junctions.py` - Vertex detection from heatmap + skeleton
  - `edge_tracing.py` - Follow skeleton to form edges
  - `cleanup.py` - Merge collinear edges, remove noise
  - `graph_extraction.py` - Main orchestrator
- `src/data/` - Dataset, FOLD parser, transforms, annotations
- `src/training/trainer.py` - Training loop with AMP, scheduling, checkpointing

## Origami Domain Constraints

- **Edge Assignments:** M (mountain), V (valley), B (border), U (unassigned). F and C map to U.
- **Kawasaki theorem:** Alternating angle sum around interior vertex = π
- **Maekawa theorem:** |#Mountain - #Valley| = 2 at interior vertices
- **Even degree:** Interior vertices must have even degree
- **Square domain:** Patterns normalized to [0,1]×[0,1]

## Data Tiers

Located in `data/output/synthetic/raw/` and `data/output/scraped/`:
- **Tier S:** Passes all validation (Maekawa, Kawasaki, 2-colorability, flat-foldability)
- **Tier A:** Passes local validation only (primary training data, ~89K synthetic + ~155 scraped)
- **Rejected:** Fails validation

## Configuration

Training config in `configs/base.yaml`:
- Image size: 1024×1024, stride 4
- 5 segmentation classes: BG=0, M=1, V=2, B=3, U=4
- Loss weights: seg=1.0, orient=0.5, junction=1.0
- Data split: 85/10/5 (train/val/test)

## Graph Extraction Philosophy

The post-processing stage creates an **over-complete candidate graph** biased toward recall:
- If a real crease is missing from candidates, the Graph Head cannot recover it
- Extra spurious edges can be filtered by the Graph Head via edge existence prediction
- Key parameters tuned for high recall: low junction threshold (0.55), small NMS distance (3px), bridge 2px gaps
