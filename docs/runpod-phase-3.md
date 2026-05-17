# Phase 3 RunPod Handoff

This is the recommended first paid GPU path after the local Phase 3 augmentation
gates. The goal is not to jump straight to a giant `mixed` run. Run the same
curriculum that passed locally, with `hrnet_w18` at 1024px and checkpoint
initialization between stages.

## Local Status

Local gates used the Phase 2 real scraped FOLD fixture before the synthetic
raw-manifest release was available. CPLine training now reads fold-only
`raw-manifest.jsonl` datasets directly, with `foldPath` relative to the manifest
root and `split` selecting train/val/test rows. Use the linked
`cp_training_mix_v1` root for the next run so both TreeMaker and Rabbit Ear rows
are available.

| Stage | Init | Clean edge F1 | Aug edge F1 | Structural validity |
| --- | --- | ---: | ---: | ---: |
| `stage-light` | scratch | 0.963 | 0.823 | 100% / 100% |
| `stage-print` | `stage-light` | 0.977 | 0.790 | 100% / 100% |
| `stage-dark` | `stage-print` | 0.987 | 0.712 | 100% / 100% |
| `stage-dark-grid` | `stage-dark` | 0.983 | 0.637 | 100% / 100% |
| `mixed` short check | `stage-dark-grid` | 0.971 | 0.826 | 100% / 100% |

The grid-inclusive dark-mode slice is the main known hard case. Faint grid
rendering fixed the catastrophic overproduction case, but this slice should be
monitored separately on RunPod.

A 1024px `hrnet_w18` preflight ran locally for two steps on MPS with batch size
1. That proves the full-size command path and tensor shapes, not model quality.

## RunPod Setup

From a fresh pod:

```bash
git checkout codex/phase3-real-folds
scripts/setup_python_env.sh
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
.venv/bin/python -m pytest tests/test_cpline_phase3.py -q
```

If the synthetic dataset is mounted somewhere other than the default shared path,
set one of these first:

```bash
export CP_SHARED_DATA_ROOT=/path/to/create-pattern-detector-datasets
export CP_SYNTHETIC_DATASET=/path/to/create-pattern-detector-datasets/synthetic/cp_training_mix_v1
```

## First Run

For a 24GB GPU, start conservatively:

```bash
OUTPUT_ROOT=checkpoints/runpod_phase3_curriculum \
MANIFEST=data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
TRAIN_COUNT=512 \
VAL_COUNT=64 \
BATCH_SIZE=1 \
NUM_WORKERS=4 \
IMAGE_SIZE=1024 \
BACKBONE=hrnet_w18 \
RUN_MIXED=0 \
scripts/training/run_cpline_runpod_curriculum.sh
```

The script runs:

1. `stage-light` from scratch.
2. `stage-print` initialized from `stage-light`.
3. `stage-dark` initialized from `stage-print`.
4. `stage-dark-grid` initialized from `stage-dark`.

Set `RUN_MIXED=1` only after reviewing the `stage-dark-grid` summary.

Each stage writes `summary.json`, `latest.pt`, `run_config.json`, and prediction
cache artifacts under its output directory.

## Review Gates

After each stage, inspect:

- `val_graph_sweep.best_threshold`
- clean and augmented edge F1
- clean and augmented structural validity
- predicted edge count versus ground-truth edge count
- dark-grid examples specifically once `stage-dark-grid` starts

Stop or lower grid probability if augmented validation shows a large predicted
edge explosion on grid examples.

## Suggested Follow-Up

If the first RunPod curriculum is stable, increase `TRAIN_COUNT` and `VAL_COUNT`
before increasing augmentation aggressiveness. Use full `mixed` only after the
staged summaries are healthy.
