#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_vertex_refiner_source_only_probe_${RUN_DATE}}"

AUXILIARY_MODE="${AUXILIARY_MODE:-zero}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_EDGES="${MAX_EDGES:-1200}"
TRAIN_COUNT="${TRAIN_COUNT:-512}"
VAL_COUNT="${VAL_COUNT:-64}"
PROPOSALS_PER_SAMPLE="${PROPOSALS_PER_SAMPLE:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHUFFLE_TRAIN_CROPS="${SHUFFLE_TRAIN_CROPS:-1}"
RENDERED_SAMPLE_CACHE_SIZE="${RENDERED_SAMPLE_CACHE_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-0.0003}"
BASE_CHANNELS="${BASE_CHANNELS:-48}"
MODEL_VERSION="${MODEL_VERSION:-v1}"
TRAIN_CROP_REFS="${TRAIN_CROP_REFS:-}"
VAL_CROP_REFS="${VAL_CROP_REFS:-}"
CROP_REF_PROGRESS_EVERY="${CROP_REF_PROGRESS_EVERY:-16}"
BOUNDARY_GT_ANCHOR_REPEATS="${BOUNDARY_GT_ANCHOR_REPEATS:-}"
if [[ -z "$BOUNDARY_GT_ANCHOR_REPEATS" ]]; then
  if [[ "$MODEL_VERSION" == "v2" ]]; then
    BOUNDARY_GT_ANCHOR_REPEATS="3"
  else
    BOUNDARY_GT_ANCHOR_REPEATS="0"
  fi
fi
BOUNDARY_GT_ANCHOR_JITTER_PX="${BOUNDARY_GT_ANCHOR_JITTER_PX:-6.0}"
SEED="${SEED:-41}"
VAL_SEED="${VAL_SEED:-$((SEED + 1000))}"
LOG_EVERY="${LOG_EVERY:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
HEATMAP_THRESHOLD="${HEATMAP_THRESHOLD:-0.25}"
MATCH_TOLERANCE_PX="${MATCH_TOLERANCE_PX:-2.0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
ABORT_LOSS_THRESHOLD="${ABORT_LOSS_THRESHOLD:-100000}"
EARLY_EVAL_EVERY="${EARLY_EVAL_EVERY:-250}"
EARLY_STOP_AFTER_STEP="${EARLY_STOP_AFTER_STEP:-500}"
EARLY_STOP_MIN_VAL_F1="${EARLY_STOP_MIN_VAL_F1:-}"
RUN_STANDALONE_EVAL="${RUN_STANDALONE_EVAL:-1}"
EVAL_FULL_PATTERN_DIAGNOSTICS="${EVAL_FULL_PATTERN_DIAGNOSTICS:-1}"

PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"
PREFLIGHT_TRAIN_COUNT="${PREFLIGHT_TRAIN_COUNT:-2}"
PREFLIGHT_VAL_COUNT="${PREFLIGHT_VAL_COUNT:-1}"
PREFLIGHT_PROPOSALS_PER_SAMPLE="${PREFLIGHT_PROPOSALS_PER_SAMPLE:-4}"
PREFLIGHT_BATCH_SIZE="${PREFLIGHT_BATCH_SIZE:-4}"
PREFLIGHT_NUM_WORKERS="${PREFLIGHT_NUM_WORKERS:-0}"

if [[ "$AUXILIARY_MODE" != "zero" && "${ALLOW_NONZERO_VERTEX_REFINER_AUX:-0}" != "1" ]]; then
  echo "fatal: Phase 4 source-only probe requires AUXILIARY_MODE=zero, got $AUXILIARY_MODE" >&2
  echo "Set ALLOW_NONZERO_VERTEX_REFINER_AUX=1 only for an intentional ablation." >&2
  exit 2
fi

if [[ -e "$OUTPUT_ROOT" && "${ALLOW_EXISTING_OUTPUT_ROOT:-0}" != "1" ]]; then
  echo "fatal: OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2
  echo "Use a fresh OUTPUT_ROOT or set ALLOW_EXISTING_OUTPUT_ROOT=1 intentionally." >&2
  exit 2
fi

INIT_CHECKPOINT_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
  INIT_CHECKPOINT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi
EARLY_STOP_ARGS=()
if [[ -n "$EARLY_STOP_MIN_VAL_F1" ]]; then
  EARLY_STOP_ARGS=(--early-stop-min-val-f1 "$EARLY_STOP_MIN_VAL_F1")
fi
CROP_REF_ARGS=()
if [[ -n "$TRAIN_CROP_REFS" ]]; then
  CROP_REF_ARGS+=(--train-crop-refs "$TRAIN_CROP_REFS")
fi
if [[ -n "$VAL_CROP_REFS" ]]; then
  CROP_REF_ARGS+=(--val-crop-refs "$VAL_CROP_REFS")
fi
EVAL_CROP_REF_ARGS=()
if [[ -n "$VAL_CROP_REFS" ]]; then
  EVAL_CROP_REF_ARGS+=(--crop-refs "$VAL_CROP_REFS")
fi
SHUFFLE_TRAIN_ARGS=()
if [[ "$SHUFFLE_TRAIN_CROPS" == "0" || "$SHUFFLE_TRAIN_CROPS" == "false" ]]; then
  SHUFFLE_TRAIN_ARGS=(--no-shuffle-train-crops)
else
  SHUFFLE_TRAIN_ARGS=(--shuffle-train-crops)
fi
RENDERED_SAMPLE_CACHE_ARGS=()
if [[ -n "$RENDERED_SAMPLE_CACHE_SIZE" ]]; then
  RENDERED_SAMPLE_CACHE_ARGS=(--rendered-sample-cache-size "$RENDERED_SAMPLE_CACHE_SIZE")
fi

if [[ "$DEVICE" == "cuda" ]]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  if [[ -z "$GPU_NAME" ]]; then
    echo "fatal: DEVICE=cuda but nvidia-smi did not report a GPU" >&2
    exit 2
  fi
  case "$GPU_NAME" in
    *H100*|*H200*|*B200*|*A100*|*GH200*|*RTX\ PRO\ 6000*|*RTX\ Pro\ 6000*)
      if [[ "${ALLOW_EXPENSIVE_VERTEX_REFINER_GPU:-0}" != "1" ]]; then
        echo "fatal: refusing expensive GPU for <$10 probe: $GPU_NAME" >&2
        echo "Use an RTX A5000, RTX 4090, RTX 3090, L4, A40, or similar 24GB+ GPU." >&2
        echo "Set ALLOW_EXPENSIVE_VERTEX_REFINER_GPU=1 only if the budget decision changed." >&2
        exit 2
      fi
      ;;
  esac
  echo "Using GPU: $GPU_NAME"
fi

mkdir -p "$OUTPUT_ROOT"

verify_run_config() {
  local run_config="$1"
  "$PYTHON" scripts/training/verify_vertex_refiner_run_config.py \
    --run-config "$run_config" \
    --expect-str "device=$DEVICE" \
    --expect-str "auxiliary_mode=$AUXILIARY_MODE" \
    --expect-int "image_size=$IMAGE_SIZE" \
    --expect-int "max_edges=$MAX_EDGES" \
    --expect-int "base_channels=$BASE_CHANNELS" \
    --expect-str "model_version=$MODEL_VERSION" \
    --expect-int "boundary_gt_anchor_repeats=$BOUNDARY_GT_ANCHOR_REPEATS" \
    --expect-bool "include_gt_training_anchors=true" \
    --expect-bool "include_val_gt_anchors=false"
}

if [[ "${RUN_PREFLIGHT:-0}" == "1" ]]; then
  PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-${OUTPUT_ROOT}_preflight}"
  mkdir -p "$PREFLIGHT_OUTPUT_ROOT"
  echo "Running VertexRefiner source-only CUDA preflight..."
  "$PYTHON" scripts/training/train_vertex_refiner.py \
    --manifest "$MANIFEST" \
    --output-dir "$PREFLIGHT_OUTPUT_ROOT" \
    --device "$DEVICE" \
    --image-size "$IMAGE_SIZE" \
    --train-count "$PREFLIGHT_TRAIN_COUNT" \
    --val-count "$PREFLIGHT_VAL_COUNT" \
    --max-edges "$MAX_EDGES" \
    --proposals-per-sample "$PREFLIGHT_PROPOSALS_PER_SAMPLE" \
    --batch-size "$PREFLIGHT_BATCH_SIZE" \
    --num-workers "$PREFLIGHT_NUM_WORKERS" \
    "${SHUFFLE_TRAIN_ARGS[@]}" \
    "${RENDERED_SAMPLE_CACHE_ARGS[@]}" \
    --max-steps "$PREFLIGHT_STEPS" \
    --lr "$LR" \
    --base-channels "$BASE_CHANNELS" \
    --model-version "$MODEL_VERSION" \
    --crop-ref-progress-every "$CROP_REF_PROGRESS_EVERY" \
    --boundary-gt-anchor-repeats "$BOUNDARY_GT_ANCHOR_REPEATS" \
    --boundary-gt-anchor-jitter-px "$BOUNDARY_GT_ANCHOR_JITTER_PX" \
    --auxiliary-mode "$AUXILIARY_MODE" \
    --include-gt-training-anchors \
    --no-include-val-gt-anchors \
    --checkpoint-every 1 \
    --log-every 1 \
    --heatmap-threshold "$HEATMAP_THRESHOLD" \
    --match-tolerance-px "$MATCH_TOLERANCE_PX" \
    --eval-max-batches 4 \
    --abort-loss-threshold "$ABORT_LOSS_THRESHOLD" \
    --early-eval-every 0 \
    "${INIT_CHECKPOINT_ARGS[@]}" \
    --seed "$SEED"
  verify_run_config "$PREFLIGHT_OUTPUT_ROOT/run_config.json"
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "Running VertexRefiner source-only probe..."
mkdir -p "$OUTPUT_ROOT/full"
"$PYTHON" scripts/training/train_vertex_refiner.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/full" \
  --device "$DEVICE" \
  --image-size "$IMAGE_SIZE" \
  --train-count "$TRAIN_COUNT" \
  --val-count "$VAL_COUNT" \
  --max-edges "$MAX_EDGES" \
  --proposals-per-sample "$PROPOSALS_PER_SAMPLE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "${SHUFFLE_TRAIN_ARGS[@]}" \
  "${RENDERED_SAMPLE_CACHE_ARGS[@]}" \
  --max-steps "$MAX_STEPS" \
  --lr "$LR" \
  --base-channels "$BASE_CHANNELS" \
  --model-version "$MODEL_VERSION" \
  --crop-ref-progress-every "$CROP_REF_PROGRESS_EVERY" \
  --boundary-gt-anchor-repeats "$BOUNDARY_GT_ANCHOR_REPEATS" \
  --boundary-gt-anchor-jitter-px "$BOUNDARY_GT_ANCHOR_JITTER_PX" \
  --auxiliary-mode "$AUXILIARY_MODE" \
  --include-gt-training-anchors \
  --no-include-val-gt-anchors \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --log-every "$LOG_EVERY" \
  --heatmap-threshold "$HEATMAP_THRESHOLD" \
  --match-tolerance-px "$MATCH_TOLERANCE_PX" \
  --eval-max-batches "$EVAL_MAX_BATCHES" \
  --abort-loss-threshold "$ABORT_LOSS_THRESHOLD" \
  --early-eval-every "$EARLY_EVAL_EVERY" \
  --early-stop-after-step "$EARLY_STOP_AFTER_STEP" \
  "${EARLY_STOP_ARGS[@]}" \
  "${CROP_REF_ARGS[@]}" \
  "${INIT_CHECKPOINT_ARGS[@]}" \
  --seed "$SEED" \
  2>&1 | tee "$OUTPUT_ROOT/full/train.log"
verify_run_config "$OUTPUT_ROOT/full/run_config.json"

echo "Running standalone VertexRefiner source-only eval..."
if [[ "$RUN_STANDALONE_EVAL" == "1" || "$RUN_STANDALONE_EVAL" == "true" ]]; then
  EVAL_DIAGNOSTIC_ARGS=()
  if [[ "$EVAL_FULL_PATTERN_DIAGNOSTICS" == "1" || "$EVAL_FULL_PATTERN_DIAGNOSTICS" == "true" ]]; then
    EVAL_DIAGNOSTIC_ARGS=(--full-pattern-diagnostics)
  fi
  "$PYTHON" scripts/evals/eval_vertex_refiner.py \
  --checkpoint "$OUTPUT_ROOT/full/latest.pt" \
  --manifest "$MANIFEST" \
  --split val \
  --limit "$VAL_COUNT" \
  --max-edges "$MAX_EDGES" \
  --image-size "$IMAGE_SIZE" \
  --proposals-per-sample "$PROPOSALS_PER_SAMPLE" \
  --batch-size "$BATCH_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --model-version "$MODEL_VERSION" \
  "${RENDERED_SAMPLE_CACHE_ARGS[@]}" \
  --device "$DEVICE" \
  --auxiliary-mode "$AUXILIARY_MODE" \
  --crop-ref-progress-every "$CROP_REF_PROGRESS_EVERY" \
  --no-include-gt-training-anchors \
  --heatmap-threshold "$HEATMAP_THRESHOLD" \
  --match-tolerance-px "$MATCH_TOLERANCE_PX" \
  "${EVAL_DIAGNOSTIC_ARGS[@]}" \
  "${EVAL_CROP_REF_ARGS[@]}" \
  --seed "$VAL_SEED" \
  --out "$OUTPUT_ROOT/full/eval.json" 2>&1 | tee "$OUTPUT_ROOT/full/eval.log"
else
  echo "Skipping standalone VertexRefiner source-only eval because RUN_STANDALONE_EVAL=$RUN_STANDALONE_EVAL"
fi

echo "VertexRefiner source-only probe complete: $OUTPUT_ROOT/full"
