#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v2_continuation}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt}"
DEVICE="${DEVICE:-cuda}"
BACKBONE="${BACKBONE:-hrnet_w18}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
BATCHNORM_MODE="${BATCHNORM_MODE:-batch-stats}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_EDGES="${MAX_EDGES:-300}"
TRAIN_FAMILY_SAMPLING="${TRAIN_FAMILY_SAMPLING:-balanced}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NO_PIN_MEMORY="${NO_PIN_MEMORY:-0}"
LR="${LR:-0.0002}"
SEED="${SEED:-17}"
LOG_EVERY="${LOG_EVERY:-50}"
LOG_MEMORY="${LOG_MEMORY:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-400}"
SKIP_GRAPH_EVAL="${SKIP_GRAPH_EVAL:-1}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-0.35,0.45,0.55,0.65,0.75,0.85}"
JUNCTION_SIGMA_PX="${JUNCTION_SIGMA_PX:-}"
JUNCTION_OFFSET_RADIUS_PX="${JUNCTION_OFFSET_RADIUS_PX:-0.0}"
JUNCTION_OFFSET_WEIGHT="${JUNCTION_OFFSET_WEIGHT:-0.25}"
JUNCTION_FOCAL_ALPHA="${JUNCTION_FOCAL_ALPHA:-0.0}"
JUNCTION_FOCAL_BETA="${JUNCTION_FOCAL_BETA:-4.0}"
REQUIRE_CLOSE_PAIR_OFFSETS="${REQUIRE_CLOSE_PAIR_OFFSETS:-0}"
REQUIRE_NO_GUIDE_GRID_PROFILE="${REQUIRE_NO_GUIDE_GRID_PROFILE:-0}"
REQUIRE_R1_INIT="${REQUIRE_R1_INIT:-0}"

PROFILE="${PROFILE:-v2-all-issue-mix}"
EVAL_PROFILE="${EVAL_PROFILE:-$PROFILE}"
TRAIN_COUNT_WARMUP="${TRAIN_COUNT_WARMUP:-512}"
VAL_COUNT_WARMUP="${VAL_COUNT_WARMUP:-64}"
STEPS_WARMUP="${STEPS_WARMUP:-1200}"
TRAIN_COUNT_FULL="${TRAIN_COUNT_FULL:-2048}"
VAL_COUNT_FULL="${VAL_COUNT_FULL:-256}"
STEPS_FULL="${STEPS_FULL:-4800}"
RUN_WARMUP="${RUN_WARMUP:-1}"
RUN_FULL="${RUN_FULL:-1}"
FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-$OUTPUT_ROOT/warmup/latest.pt}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-init}"
REINIT_HEADS="${REINIT_HEADS:-}"

LINE_HARD_NEGATIVE_WEIGHT="${LINE_HARD_NEGATIVE_WEIGHT:-0.25}"
LINE_HARD_NEGATIVE_RATIO="${LINE_HARD_NEGATIVE_RATIO:-0.05}"
LINE_HARD_NEGATIVE_MULTIPLIER="${LINE_HARD_NEGATIVE_MULTIPLIER:-4.0}"
NON_CREASE_WEIGHT="${NON_CREASE_WEIGHT:-0.1}"
LINE_STYLE_WEIGHT="${LINE_STYLE_WEIGHT:-0.1}"
BOUNDARY_CONTACT_WEIGHT="${BOUNDARY_CONTACT_WEIGHT:-0.75}"
BOUNDARY_CONTACT_POS_WEIGHT="${BOUNDARY_CONTACT_POS_WEIGHT:-12}"
BOUNDARY_CONTACT_CORNER_NEGATIVE_WEIGHT="${BOUNDARY_CONTACT_CORNER_NEGATIVE_WEIGHT:-4}"
BOUNDARY_CONTACT_HARD_NEGATIVE_WEIGHT="${BOUNDARY_CONTACT_HARD_NEGATIVE_WEIGHT:-0.2}"
BOUNDARY_CONTACT_HARD_NEGATIVE_RATIO="${BOUNDARY_CONTACT_HARD_NEGATIVE_RATIO:-0.02}"
BOUNDARY_CONTACT_HARD_NEGATIVE_MULTIPLIER="${BOUNDARY_CONTACT_HARD_NEGATIVE_MULTIPLIER:-8}"
BOUNDARY_CONTACT_HARD_NEGATIVE_MIN_PIXELS="${BOUNDARY_CONTACT_HARD_NEGATIVE_MIN_PIXELS:-256}"
VERTEX_TYPE_WEIGHT="${VERTEX_TYPE_WEIGHT:-0.5}"
VERTEX_TYPE_CLASS_WEIGHTS="${VERTEX_TYPE_CLASS_WEIGHTS:-0.02,10,10,2}"
VERTEX_TYPE_FOCAL_GAMMA="${VERTEX_TYPE_FOCAL_GAMMA:-2.0}"
BOUNDARY_SIDE_WEIGHT="${BOUNDARY_SIDE_WEIGHT:-0.0}"
BOUNDARY_OFFSET_WEIGHT="${BOUNDARY_OFFSET_WEIGHT:-0.0}"
BOUNDARY_COORD_WEIGHT="${BOUNDARY_COORD_WEIGHT:-0.2}"

if [[ "$REQUIRE_CLOSE_PAIR_OFFSETS" == "1" || "$REQUIRE_NO_GUIDE_GRID_PROFILE" == "1" || "$REQUIRE_R1_INIT" == "1" ]]; then
  "$PYTHON" - <<'PY'
import os
import sys


def fail(message: str) -> None:
    print(f"fatal: {message}", file=sys.stderr)
    sys.exit(2)


def require_float(name: str, expected: float, tolerance: float = 1e-9) -> None:
    raw = os.environ.get(name, "")
    try:
        value = float(raw)
    except ValueError:
        fail(f"{name} must be numeric, got {raw!r}")
    if abs(value - expected) > tolerance:
        fail(f"{name} must be {expected:g}, got {raw!r}")


if os.environ.get("REQUIRE_CLOSE_PAIR_OFFSETS") == "1":
    require_float("JUNCTION_SIGMA_PX", 1.5)
    require_float("JUNCTION_OFFSET_RADIUS_PX", 3.0)
    require_float("JUNCTION_OFFSET_WEIGHT", 0.5)
    require_float("JUNCTION_FOCAL_ALPHA", 2.0)
    require_float("JUNCTION_FOCAL_BETA", 4.0)
    reinit_heads = {
        item.strip()
        for item in os.environ.get("REINIT_HEADS", "").replace(",", " ").split()
        if item.strip()
    }
    if "non_crease_head" not in reinit_heads:
        fail("REINIT_HEADS must include non_crease_head for no-guide-grid warm-starts")
    forbidden = {"offset_head", "junction_offset_head", "junction_offset"}
    found = sorted(reinit_heads & forbidden)
    if found:
        fail(
            "REINIT_HEADS must not include offset heads when preserving R1 "
            f"close-pair offsets; found {', '.join(found)}"
        )

if os.environ.get("REQUIRE_NO_GUIDE_GRID_PROFILE") == "1":
    for name in ("PROFILE", "EVAL_PROFILE"):
        value = os.environ.get(name, "")
        if value != "v3-no-guide-grid-replay":
            fail(f"{name} must be v3-no-guide-grid-replay, got {value!r}")

if os.environ.get("REQUIRE_R1_INIT") == "1":
    init_checkpoint = os.environ.get("FULL_INIT_CHECKPOINT") or os.environ.get("INIT_CHECKPOINT", "")
    if not init_checkpoint.endswith("checkpoints/r1_close_pair_warmstart/latest.pt"):
        fail(
            "FULL_INIT_CHECKPOINT/INIT_CHECKPOINT must point to "
            "checkpoints/r1_close_pair_warmstart/latest.pt for the current "
            f"no-guide-grid close-pair warm-start, got {init_checkpoint!r}"
        )
PY
fi

mkdir -p "$OUTPUT_ROOT"

run_stage() {
  local stage_name="$1"
  local steps="$2"
  local train_count="$3"
  local val_count="$4"
  local output_dir="$5"
  local init_checkpoint="$6"
  local args=(
    scripts/training/train_cpline_smoke.py
    --device "$DEVICE"
    --backbone "$BACKBONE"
    --hidden-channels "$HIDDEN_CHANNELS"
    --batchnorm-mode "$BATCHNORM_MODE"
    --manifest "$MANIFEST"
    --output-dir "$output_dir"
    --image-size "$IMAGE_SIZE"
    --train-count "$train_count"
    --val-count "$val_count"
    --max-edges "$MAX_EDGES"
    --train-family-sampling "$TRAIN_FAMILY_SAMPLING"
    --max-steps "$steps"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --seed "$SEED"
    --log-every "$LOG_EVERY"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --augment-profile "$PROFILE"
    --eval-augment-profile "$EVAL_PROFILE"
    --eval-thresholds "$EVAL_THRESHOLDS"
    --reinit-heads "$REINIT_HEADS"
    --junction-offset-radius-px "$JUNCTION_OFFSET_RADIUS_PX"
    --junction-offset-weight "$JUNCTION_OFFSET_WEIGHT"
    --junction-focal-alpha "$JUNCTION_FOCAL_ALPHA"
    --junction-focal-beta "$JUNCTION_FOCAL_BETA"
    --line-hard-negative-weight "$LINE_HARD_NEGATIVE_WEIGHT"
    --line-hard-negative-ratio "$LINE_HARD_NEGATIVE_RATIO"
    --line-hard-negative-multiplier "$LINE_HARD_NEGATIVE_MULTIPLIER"
    --v2-heads
    --non-crease-weight "$NON_CREASE_WEIGHT"
    --line-style-weight "$LINE_STYLE_WEIGHT"
    --boundary-contact-weight "$BOUNDARY_CONTACT_WEIGHT"
    --boundary-contact-pos-weight "$BOUNDARY_CONTACT_POS_WEIGHT"
    --boundary-contact-corner-negative-weight "$BOUNDARY_CONTACT_CORNER_NEGATIVE_WEIGHT"
    --boundary-contact-hard-negative-weight "$BOUNDARY_CONTACT_HARD_NEGATIVE_WEIGHT"
    --boundary-contact-hard-negative-ratio "$BOUNDARY_CONTACT_HARD_NEGATIVE_RATIO"
    --boundary-contact-hard-negative-multiplier "$BOUNDARY_CONTACT_HARD_NEGATIVE_MULTIPLIER"
    --boundary-contact-hard-negative-min-pixels "$BOUNDARY_CONTACT_HARD_NEGATIVE_MIN_PIXELS"
    --vertex-type-weight "$VERTEX_TYPE_WEIGHT"
    --vertex-type-class-weights "$VERTEX_TYPE_CLASS_WEIGHTS"
    --vertex-type-focal-gamma "$VERTEX_TYPE_FOCAL_GAMMA"
    --boundary-side-weight "$BOUNDARY_SIDE_WEIGHT"
    --boundary-offset-weight "$BOUNDARY_OFFSET_WEIGHT"
    --boundary-coord-weight "$BOUNDARY_COORD_WEIGHT"
    --use-v2-observed-assignment
  )
  if [[ -n "$JUNCTION_SIGMA_PX" ]]; then
    args+=(--junction-sigma-px "$JUNCTION_SIGMA_PX")
  fi
  if [[ "$NO_PIN_MEMORY" == "1" ]]; then
    args+=(--no-pin-memory)
  fi
  if [[ "$LOG_MEMORY" == "1" ]]; then
    args+=(--log-memory)
  fi
  if [[ "$CHECKPOINT_LOAD_MODE" == "resume" ]]; then
    args+=(--resume-checkpoint "$init_checkpoint")
  else
    args+=(--init-checkpoint "$init_checkpoint")
  fi
  if [[ "$SKIP_GRAPH_EVAL" == "1" ]]; then
    args+=(--skip-graph-eval)
  fi
  if [[ "$SKIP_FINAL_EVAL" == "1" ]]; then
    args+=(--skip-final-eval)
  fi
  mkdir -p "$output_dir"
  local log_path="$output_dir/train.log"
  echo "=== Running $stage_name ($PROFILE) -> $output_dir ===" | tee "$log_path"
  TQDM_DISABLE="${TQDM_DISABLE:-1}" "$PYTHON" "${args[@]}" 2>&1 | tee -a "$log_path"
}

if [[ "$RUN_WARMUP" == "1" ]]; then
  run_stage "warmup" "$STEPS_WARMUP" "$TRAIN_COUNT_WARMUP" "$VAL_COUNT_WARMUP" "$OUTPUT_ROOT/warmup" "$INIT_CHECKPOINT"
fi

if [[ "$RUN_FULL" == "1" ]]; then
  run_stage "full" "$STEPS_FULL" "$TRAIN_COUNT_FULL" "$VAL_COUNT_FULL" "$OUTPUT_ROOT/full" "$FULL_INIT_CHECKPOINT"
fi
