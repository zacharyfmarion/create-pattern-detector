#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"

export MAX_EDGES="${MAX_EDGES:-1200}"

PROMOTED_CHECKPOINT="${PROMOTED_CHECKPOINT:-checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max1200_probe_20260618/full/latest.pt}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max${MAX_EDGES}_probe_20260618}"
export PROFILE="${PROFILE:-v3-no-guide-grid-replay}"
export EVAL_PROFILE="${EVAL_PROFILE:-v3-no-guide-grid-replay}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROMOTED_CHECKPOINT}"
export FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-$INIT_CHECKPOINT}"
export CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-init}"
export REINIT_HEADS="${REINIT_HEADS:-}"

export JUNCTION_SIGMA_PX="${JUNCTION_SIGMA_PX:-1.5}"
export JUNCTION_OFFSET_RADIUS_PX="${JUNCTION_OFFSET_RADIUS_PX:-3.0}"
export JUNCTION_OFFSET_WEIGHT="${JUNCTION_OFFSET_WEIGHT:-0.5}"
export JUNCTION_FOCAL_ALPHA="${JUNCTION_FOCAL_ALPHA:-2.0}"
export JUNCTION_FOCAL_BETA="${JUNCTION_FOCAL_BETA:-4.0}"

export RUN_WARMUP="${RUN_WARMUP:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export STEPS_FULL="${STEPS_FULL:-1500}"
export TRAIN_COUNT_FULL="${TRAIN_COUNT_FULL:-2048}"
export VAL_COUNT_FULL="${VAL_COUNT_FULL:-256}"
export LR="${LR:-0.00005}"
export SEED="${SEED:-31}"
export NO_PIN_MEMORY="${NO_PIN_MEMORY:-1}"
export LOG_MEMORY="${LOG_MEMORY:-1}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
export SKIP_GRAPH_EVAL="${SKIP_GRAPH_EVAL:-1}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

if [[ -e "$OUTPUT_ROOT" && "${ALLOW_EXISTING_OUTPUT_ROOT:-0}" != "1" ]]; then
  echo "Are you sure? OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2
  echo "Set OUTPUT_ROOT to a new run directory, or set ALLOW_EXISTING_OUTPUT_ROOT=1 if you really intend to reuse it." >&2
  exit 2
fi

verify_run_config() {
  local run_config="$1"
  "$PYTHON" scripts/training/verify_cpline_run_config.py \
    --run-config "$run_config" \
    --expect-str "augment_profile=$PROFILE" \
    --expect-str "eval_augment_profile=$EVAL_PROFILE" \
    --expect-str "reinit_heads=$REINIT_HEADS" \
    --expect-str "init_checkpoint=$FULL_INIT_CHECKPOINT" \
    --expect-str "max_edges=$MAX_EDGES" \
    --expect-float "junction_sigma_px=$JUNCTION_SIGMA_PX" \
    --expect-float "junction_offset_radius_px=$JUNCTION_OFFSET_RADIUS_PX" \
    --expect-float "junction_offset_weight=$JUNCTION_OFFSET_WEIGHT" \
    --expect-float "junction_focal_alpha=$JUNCTION_FOCAL_ALPHA" \
    --expect-float "junction_focal_beta=$JUNCTION_FOCAL_BETA" \
    --expect-suffix "loaded_checkpoint=$PROMOTED_CHECKPOINT"
}

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-${OUTPUT_ROOT}_preflight}"
  (
    export OUTPUT_ROOT="$PREFLIGHT_OUTPUT_ROOT"
    export RUN_WARMUP=0
    export RUN_FULL=1
    export STEPS_FULL="${PREFLIGHT_STEPS:-1}"
    export TRAIN_COUNT_FULL="${PREFLIGHT_TRAIN_COUNT:-2}"
    export VAL_COUNT_FULL="${PREFLIGHT_VAL_COUNT:-1}"
    export NUM_WORKERS="${PREFLIGHT_NUM_WORKERS:-0}"
    export CHECKPOINT_EVERY="${PREFLIGHT_CHECKPOINT_EVERY:-1}"
    export SKIP_GRAPH_EVAL=1
    export SKIP_FINAL_EVAL=1
    export LOG_EVERY=1
    scripts/training/run_cpline_runpod_v2_continuation.sh
  )
  verify_run_config "$PREFLIGHT_OUTPUT_ROOT/full/run_config.json"
  if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
  fi
fi

scripts/training/run_cpline_runpod_v2_continuation.sh
verify_run_config "$OUTPUT_ROOT/full/run_config.json"
