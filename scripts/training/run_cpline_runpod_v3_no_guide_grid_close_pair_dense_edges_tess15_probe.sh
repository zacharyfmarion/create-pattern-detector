#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MANIFEST="${MANIFEST:-data/generated/synthetic/cp_training_mix_v3_tessellation_15pct/raw-manifest.jsonl}"
export TRAIN_FAMILY_SAMPLING="${TRAIN_FAMILY_SAMPLING:-v3-tessellation-15pct}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_weighted_probe_${RUN_DATE}}"

scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh
