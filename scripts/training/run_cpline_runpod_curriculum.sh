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
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
TRAIN_COUNT="${TRAIN_COUNT:-512}"
VAL_COUNT="${VAL_COUNT:-64}"
MAX_EDGES="${MAX_EDGES:-300}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0003}"
SEED="${SEED:-7}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-0.5,0.65,0.8}"
GRAPH_EVAL_COUNT="${GRAPH_EVAL_COUNT:-}"
LOG_EVERY="${LOG_EVERY:-50}"

STEPS_LIGHT="${STEPS_LIGHT:-3000}"
STEPS_PRINT="${STEPS_PRINT:-1800}"
STEPS_DARK="${STEPS_DARK:-1800}"
STEPS_DARK_GRID="${STEPS_DARK_GRID:-1800}"
STEPS_MIXED="${STEPS_MIXED:-1200}"
RUN_MIXED="${RUN_MIXED:-0}"

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
    --manifest "$MANIFEST"
    --output-dir "$output_dir"
    --image-size "$IMAGE_SIZE"
    --train-count "$TRAIN_COUNT"
    --val-count "$VAL_COUNT"
    --max-edges "$MAX_EDGES"
    --max-steps "$steps"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --seed "$SEED"
    --log-every "$LOG_EVERY"
    --augment-profile "$profile"
    --eval-augment-profile "$profile"
    --eval-thresholds "$EVAL_THRESHOLDS"
  )
  if [[ -n "$GRAPH_EVAL_COUNT" ]]; then
    args+=(--graph-eval-count "$GRAPH_EVAL_COUNT")
  fi
  if [[ -n "$init_checkpoint" ]]; then
    args+=(--init-checkpoint "$init_checkpoint")
  fi
  mkdir -p "$output_dir"
  local log_path="$output_dir/train.log"
  echo "=== Running $profile -> $output_dir ===" | tee "$log_path"
  TQDM_DISABLE="${TQDM_DISABLE:-1}" "$PYTHON" "${args[@]}" 2>&1 | tee -a "$log_path"
}

run_stage "stage-light" "$STEPS_LIGHT" "$OUTPUT_ROOT/stage-light"
run_stage "stage-print" "$STEPS_PRINT" "$OUTPUT_ROOT/stage-print" "$OUTPUT_ROOT/stage-light/latest.pt"
run_stage "stage-dark" "$STEPS_DARK" "$OUTPUT_ROOT/stage-dark" "$OUTPUT_ROOT/stage-print/latest.pt"
run_stage "stage-dark-grid" "$STEPS_DARK_GRID" "$OUTPUT_ROOT/stage-dark-grid" "$OUTPUT_ROOT/stage-dark/latest.pt"

if [[ "$RUN_MIXED" == "1" ]]; then
  run_stage "mixed" "$STEPS_MIXED" "$OUTPUT_ROOT/mixed" "$OUTPUT_ROOT/stage-dark-grid/latest.pt"
fi
