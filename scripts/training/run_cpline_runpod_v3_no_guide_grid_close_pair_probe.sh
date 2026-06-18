#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v3_no_guide_grid_close_pair_probe}"
export STEPS_FULL="${STEPS_FULL:-800}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"

exec scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
