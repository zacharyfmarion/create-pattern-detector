#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_phase3_curriculum}"
DEVICE="${DEVICE:-cuda}"
BACKBONE="${BACKBONE:-hrnet_w18}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
BATCHNORM_MODE="${BATCHNORM_MODE:-batch-stats}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
TRAIN_COUNT="${TRAIN_COUNT:-512}"
VAL_COUNT="${VAL_COUNT:-64}"
MAX_EDGES="${MAX_EDGES:-300}"
TRAIN_FAMILY_SAMPLING="${TRAIN_FAMILY_SAMPLING:-balanced}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0003}"
SEED="${SEED:-7}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-0.5,0.65,0.8}"
GRAPH_EVAL_COUNT="${GRAPH_EVAL_COUNT:-}"
LOG_EVERY="${LOG_EVERY:-50}"
SKIP_GRAPH_EVAL="${SKIP_GRAPH_EVAL:-0}"
LINE_HARD_NEGATIVE_WEIGHT="${LINE_HARD_NEGATIVE_WEIGHT:-0.25}"
LINE_HARD_NEGATIVE_RATIO="${LINE_HARD_NEGATIVE_RATIO:-0.05}"
LINE_HARD_NEGATIVE_MULTIPLIER="${LINE_HARD_NEGATIVE_MULTIPLIER:-4.0}"

STEPS_BASE="${STEPS_BASE:-1200}"
STEPS_BALANCED="${STEPS_BALANCED:-3600}"
STEPS_TARGETED="${STEPS_TARGETED:-1200}"
RUN_TARGETED="${RUN_TARGETED:-0}"
TARGETED_PROFILE="${TARGETED_PROFILE:-stage-balanced}"

mkdir -p "$OUTPUT_ROOT"

run_stage() {
  local profile="$1"
  local steps="$2"
  local output_dir="$3"
  local init_checkpoint="${4:-}"
  local args=(
    scripts/training/train_cpline_smoke.py
    --device "$DEVICE"
    --backbone "$BACKBONE"
    --hidden-channels "$HIDDEN_CHANNELS"
    --batchnorm-mode "$BATCHNORM_MODE"
    --manifest "$MANIFEST"
    --output-dir "$output_dir"
    --image-size "$IMAGE_SIZE"
    --train-count "$TRAIN_COUNT"
    --val-count "$VAL_COUNT"
    --max-edges "$MAX_EDGES"
    --train-family-sampling "$TRAIN_FAMILY_SAMPLING"
    --max-steps "$steps"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --seed "$SEED"
    --log-every "$LOG_EVERY"
    --augment-profile "$profile"
    --eval-augment-profile "$profile"
    --eval-thresholds "$EVAL_THRESHOLDS"
    --line-hard-negative-weight "$LINE_HARD_NEGATIVE_WEIGHT"
    --line-hard-negative-ratio "$LINE_HARD_NEGATIVE_RATIO"
    --line-hard-negative-multiplier "$LINE_HARD_NEGATIVE_MULTIPLIER"
  )
  if [[ -n "$GRAPH_EVAL_COUNT" ]]; then
    args+=(--graph-eval-count "$GRAPH_EVAL_COUNT")
  fi
  if [[ "$SKIP_GRAPH_EVAL" == "1" ]]; then
    args+=(--skip-graph-eval)
  fi
  if [[ -n "$init_checkpoint" ]]; then
    args+=(--init-checkpoint "$init_checkpoint")
  fi
  mkdir -p "$output_dir"
  local log_path="$output_dir/train.log"
  echo "=== Running $profile -> $output_dir ===" | tee "$log_path"
  TQDM_DISABLE="${TQDM_DISABLE:-1}" "$PYTHON" "${args[@]}" 2>&1 | tee -a "$log_path"
}

run_stage "stage-base" "$STEPS_BASE" "$OUTPUT_ROOT/stage-base"
run_stage "stage-balanced" "$STEPS_BALANCED" "$OUTPUT_ROOT/stage-balanced" "$OUTPUT_ROOT/stage-base/latest.pt"

if [[ "$RUN_TARGETED" == "1" ]]; then
  run_stage "$TARGETED_PROFILE" "$STEPS_TARGETED" "$OUTPUT_ROOT/targeted" "$OUTPUT_ROOT/stage-balanced/latest.pt"
fi
