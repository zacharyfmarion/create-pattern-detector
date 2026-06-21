#!/usr/bin/env bash
set -euo pipefail

if [[ "${CP_ARE_YOU_SURE_LEGACY_NO_GUIDE_GRID_SCRIPT:-}" != "are-you-sure" ]]; then
  cat >&2 <<'EOF'
fatal: run_cpline_runpod_v3_no_guide_grid_full.sh is retired.

This old name previously launched without the radius-3 close-pair offset
settings, producing a checkpoint with junction_offset_radius_px=0.0.

Are you sure? Use the canonical close-pair-compatible launcher instead:

  scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh

To deliberately route this old name to the canonical launcher, set:

  CP_ARE_YOU_SURE_LEGACY_NO_GUIDE_GRID_SCRIPT=are-you-sure
EOF
  exit 2
fi

exec scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
