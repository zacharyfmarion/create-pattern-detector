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

Earlier local MPS passes were architecture gates, not quality targets. The
current curriculum supersedes the old sequential light/print/dark staging:

1. `stage-base`: short geometry and assignment warmup with clean, line-style,
   and square-symmetry samples.
2. `stage-balanced`: main training mix with light, print/photo-light, dark, and
   photo-dark samples present together.
3. Optional targeted continuation only after deterministic eval identifies a
   specific weakness.

The local tiny model still overproduces edges heavily on dense mixed samples.
That is expected for short local gates; RunPod should monitor predicted edge
count versus ground truth separately for clean, dark-mode, photo-light, and
photo-dark examples.

A 1024px `hrnet_w18` preflight ran locally for two MPS steps with batch size 1
against the mixed manifest. Loss moved `3.694 -> 2.543`; graph quality was not
meaningful after two steps, but the full-size command path, tensor shapes, and
MPS memory path are proven.

## RunPod Setup

### Local API key setup

Create a restricted RunPod API key and keep it out of git:

- `api.runpod.io/graphql`: `Read / Write`
- `api.runpod.ai`: `None` unless you also plan to manage Serverless endpoints

This Phase 3 flow uses `runpodctl` for Pods, GPU availability, account info, and
file transfer; the CLI defaults to `https://api.runpod.io/graphql`. `Read only`
is not enough because we need to create/start/stop Pods. Avoid `All` unless the
restricted key proves insufficient.

Store the key locally:

```bash
cp configs/runpod.env.example configs/runpod.env
# Paste RUNPOD_API_KEY into configs/runpod.env. Do not paste it into chat.
set -a; source configs/runpod.env; set +a
runpodctl config --apiKey "$RUNPOD_API_KEY"
runpodctl user
```

Alternatively, `runpodctl doctor` can prompt for the key and set up SSH. The
env-file path is easier to audit per worktree, while `runpodctl config` stores
the key in your user-level RunPod CLI config.

### Pod bootstrap

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
RUN_TARGETED=0 \
scripts/training/run_cpline_runpod_curriculum.sh
```

The curriculum passes CPLine hard-negative line-loss settings through these env
vars:

```bash
LINE_HARD_NEGATIVE_WEIGHT=0.25
LINE_HARD_NEGATIVE_RATIO=0.05
LINE_HARD_NEGATIVE_MULTIPLIER=4.0
```

Raise `LINE_HARD_NEGATIVE_WEIGHT` for a focused dark-mode continuation if graph
eval shows the model is treating dark backgrounds or non-crease visual texture
as crease lines.

For the very first paid shakedown, optionally set `GRAPH_EVAL_COUNT=32` so each
stage vectorizes a bounded validation subset. Leave it unset when you want full
graph-eval summaries.

If 1024px graph vectorization dominates wall-clock time, set
`SKIP_GRAPH_EVAL=1` for the curriculum and run smaller posthoc graph evals from
the saved stage checkpoints. Pixel-loss validation still runs when graph eval is
skipped.

The script runs:

1. `stage-base` from scratch.
2. `stage-balanced` initialized from `stage-base`.
3. Optional targeted continuation initialized from `stage-balanced` when
   `RUN_TARGETED=1`.

Set `RUN_TARGETED=1` only after reviewing deterministic posthoc eval from
`stage-balanced`.

Each stage writes `summary.json`, `latest.pt`, `run_config.json`, and prediction
cache artifacts under its output directory. It also streams step metrics to
`train_history.jsonl` and mirrors stdout/stderr to `train.log`.

## Monitoring

From your laptop, check Pod status:

```bash
set -a; source configs/runpod.env; set +a
runpodctl pod list
runpodctl pod get "$RUNPOD_POD_ID" --include-machine
runpodctl ssh info "$RUNPOD_POD_ID"
```

On the Pod, run the curriculum inside `tmux` so disconnects do not stop
training:

```bash
tmux new -s cpline
OUTPUT_ROOT=checkpoints/runpod_phase3_curriculum \
MANIFEST=data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
TRAIN_COUNT=512 \
VAL_COUNT=64 \
BATCH_SIZE=1 \
NUM_WORKERS=4 \
IMAGE_SIZE=1024 \
BACKBONE=hrnet_w18 \
RUN_TARGETED=0 \
LOG_EVERY=50 \
SKIP_GRAPH_EVAL=1 \
scripts/training/run_cpline_runpod_curriculum.sh
```

Detach with `Ctrl-b d`, then reattach with:

```bash
tmux attach -t cpline
```

Watch live logs and step metrics from another shell on the Pod:

```bash
tail -f checkpoints/runpod_phase3_curriculum/stage-base/train.log
tail -f checkpoints/runpod_phase3_curriculum/stage-base/train_history.jsonl
watch -n 5 nvidia-smi
```

After a stage finishes, inspect its summary:

```bash
.venv/bin/python -m json.tool checkpoints/runpod_phase3_curriculum/stage-base/summary.json | less
```

For the staged run, change `stage-base` to `stage-balanced` as the second stage
begins.

## Review Gates

After each stage, inspect:

- `val_graph_sweep.best_threshold`
- clean and augmented edge F1
- clean and augmented structural validity
- predicted edge count versus ground-truth edge count
- clean, dark-mode, photo-light, and photo-dark examples once `stage-balanced`
  starts

Use targeted continuation only if a specific slice underperforms; keep the other
modes in the mix to avoid forgetting.

## Suggested Follow-Up

If the first RunPod curriculum is stable, increase `TRAIN_COUNT` and `VAL_COUNT`
before increasing augmentation aggressiveness.
