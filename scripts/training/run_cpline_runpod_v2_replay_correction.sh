#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v2_replay_correction}"
export PROFILE="${PROFILE:-v2-replay-corrective}"
export EVAL_PROFILE="${EVAL_PROFILE:-v2-replay-corrective}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/runpod_v2_continuation_final_w4/full/latest.pt}"
export FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-$INIT_CHECKPOINT}"
export CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-init}"
export RUN_WARMUP="${RUN_WARMUP:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export STEPS_FULL="${STEPS_FULL:-1200}"
export TRAIN_COUNT_FULL="${TRAIN_COUNT_FULL:-2048}"
export VAL_COUNT_FULL="${VAL_COUNT_FULL:-256}"
export LR="${LR:-0.00005}"
export SEED="${SEED:-31}"
export NO_PIN_MEMORY="${NO_PIN_MEMORY:-1}"
export LOG_MEMORY="${LOG_MEMORY:-1}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
export SKIP_GRAPH_EVAL="${SKIP_GRAPH_EVAL:-1}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"

exec scripts/training/run_cpline_runpod_v2_continuation.sh
