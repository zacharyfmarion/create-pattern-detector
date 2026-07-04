#!/usr/bin/env bash
# V4 junction-recall run: search225 data + solid-geometry augmentations.
#
# What this changes vs the promoted tess15-weighted continuation:
#   - MANIFEST: cp_training_mix_v4_search225 (v3 sources + 6,451 exact 22.5deg
#     SEARCH-22.5 tilings; native easy/medium are 79%/84% 22.5-system).
#   - TRAIN_FAMILY_SAMPLING=v4-search225-20pct (treemaker/rabbit 32.5% each,
#     tessellations 15%, search225 20%).
#   - PROFILE=v4-solid-geometry-replay: no dashed/text/watermark obfuscators
#     (the v3 mix trained junctions against ink-free points on 19% of samples).
#   - A real training budget: STEPS_FULL=12000 over TRAIN_COUNT_FULL=8192
#     (vs the 1500x2048 continuations).
# Keeps: the close-pair junction recipe (sigma 1.5, offset radius 3.0, focal
# 2/4), max_edges=1200 envelope, warm start from the promoted checkpoint with
# no head reinit, batch-stats BatchNorm.
#
# Checkpoint selection: score with scripts/evals/junction_scorecard.py on the
# v4 mix val split (product-parity decode); look at native-cp-v1 only at
# promotion time.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
PROMOTED_CHECKPOINT="${PROMOTED_CHECKPOINT:-$("$PYTHON" scripts/checkpoint/current_checkpoint.py --field checkpoint)}"

export MANIFEST="${MANIFEST:-data/generated/synthetic/cp_training_mix_v4_search225/raw-manifest.jsonl}"
export TRAIN_FAMILY_SAMPLING="${TRAIN_FAMILY_SAMPLING:-v4-search225-20pct}"
export PROFILE="${PROFILE:-v4-solid-geometry-replay}"
export EVAL_PROFILE="${EVAL_PROFILE:-$PROFILE}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v4_search225_solid_geometry_${RUN_DATE}}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROMOTED_CHECKPOINT}"
export FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-$INIT_CHECKPOINT}"
export CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-init}"
export REINIT_HEADS="${REINIT_HEADS:-}"

export MAX_EDGES="${MAX_EDGES:-1200}"
export JUNCTION_SIGMA_PX="${JUNCTION_SIGMA_PX:-1.5}"
export JUNCTION_OFFSET_RADIUS_PX="${JUNCTION_OFFSET_RADIUS_PX:-3.0}"
export JUNCTION_OFFSET_WEIGHT="${JUNCTION_OFFSET_WEIGHT:-0.5}"
export JUNCTION_FOCAL_ALPHA="${JUNCTION_FOCAL_ALPHA:-2.0}"
export JUNCTION_FOCAL_BETA="${JUNCTION_FOCAL_BETA:-4.0}"

export RUN_WARMUP="${RUN_WARMUP:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export STEPS_FULL="${STEPS_FULL:-12000}"
export TRAIN_COUNT_FULL="${TRAIN_COUNT_FULL:-8192}"
export VAL_COUNT_FULL="${VAL_COUNT_FULL:-512}"
export LR="${LR:-0.00005}"
export SEED="${SEED:-41}"
export NO_PIN_MEMORY="${NO_PIN_MEMORY:-1}"
export LOG_MEMORY="${LOG_MEMORY:-1}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
export SKIP_GRAPH_EVAL="${SKIP_GRAPH_EVAL:-1}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

if [[ -e "$OUTPUT_ROOT" && "${ALLOW_EXISTING_OUTPUT_ROOT:-0}" != "1" ]]; then
  echo "Are you sure? OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2
  echo "Set OUTPUT_ROOT to a new run directory, or ALLOW_EXISTING_OUTPUT_ROOT=1 to reuse it." >&2
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
    --expect-str "train_family_sampling=$TRAIN_FAMILY_SAMPLING" \
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
    export TRAIN_COUNT_FULL="${PREFLIGHT_TRAIN_COUNT:-8}"
    export VAL_COUNT_FULL="${PREFLIGHT_VAL_COUNT:-2}"
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
