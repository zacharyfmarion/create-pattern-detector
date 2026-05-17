# Phase 3 RunPod Handoff

This is the recommended first paid GPU path after the local Phase 3 augmentation
gates. The goal is not to jump straight to a giant `mixed` run. Run the same
curriculum that passed locally, with `hrnet_w18` at 1024px and checkpoint
initialization between stages.

## Local Status

Local gates now use the 14k synthetic mixed raw manifest:
`data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl`.
CPLine training reads fold-only `raw-manifest.jsonl` datasets directly, with
`foldPath` relative to the manifest root and `split` selecting train/val/test
rows.

The latest local MPS pass was an architecture gate, not a quality target:
384px, tiny backbone, 64 train / 16 val, `max_edges=1000`, and
`--graph-eval-count 4` to keep dense graph vectorization bounded.

| Stage | Init | Train Loss | Clean Val Loss | Aug Val Loss | Clean Edge F1 | Aug Edge F1 | Structural Validity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage-light` | scratch | 5.661 -> 1.710 | 1.569 | 1.441 | 0.084 | 0.074 | 100% / 100% |
| `stage-print` | `stage-light` | 1.607 -> 1.249 | 1.318 | 1.154 | 0.086 | 0.051 | 100% / 100% |
| `stage-dark` | `stage-print` | 1.292 -> 1.086 | 1.216 | 2.315 | 0.081 | 0.040 | 100% / 100% |
| `stage-dark-grid` | `stage-dark` | 1.125 -> 0.976 | 1.094 | 3.119 | 0.079 | 0.048 | 100% / 100% |

The local tiny model still overproduces edges heavily on dense mixed samples,
and dark/dark-grid augmented validation remains the hardest slice. That is
expected for the short local gate; RunPod should monitor predicted edge count
versus ground truth and dark-grid examples separately.

A 1024px `hrnet_w18` preflight ran locally for two MPS steps with batch size 1
against the mixed manifest. Loss moved `3.694 -> 2.543`; graph quality was not
meaningful after two steps, but the full-size command path, tensor shapes, and
MPS memory path are proven.

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

For the very first paid shakedown, optionally set `GRAPH_EVAL_COUNT=32` so each
stage vectorizes a bounded validation subset. Leave it unset when you want full
graph-eval summaries.

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
