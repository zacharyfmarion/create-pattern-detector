# V5 junction training run — runbook

The combined BP + SEARCH-22.5 junction run. Everything is staged; this doc is
what an operator (human or agent) needs to execute it on a RunPod GPU, select
a checkpoint, and evaluate for promotion. Read alongside
`docs/runpod-quickstart.md` (pod mechanics) and `docs/model-training-history.md`
(promotion rules).

## Why this run (measured context, 2026-07-02 → 07-07)

All numbers are on the 563-CP `native-cp-v1` benchmark in ori-studio
(`tree-maker-rust`), strict 2px, 25s exact-solve, latest main there.

- Production today: **121/563** CPs fully recovered end-to-end.
- Perfect-junction ceiling (post selection-completion PR#75): **288/563** —
  junction detection is the binding constraint, worth +167 gross.
- Failure geography: native easy/medium are **79%/84% 22.5-system** CPs
  (residual isolated sub-threshold misses live there); native hard is **63%
  box-pleated** with junction recall **65%** and 76% of misses heatmap-absent
  (the model never trained on that density — the old `max_edges=1200` filter
  excluded the median hard-BP CP at 1,376 edges).
- Augmentation audit: 19% of the previous mix had every crease dashed while
  junction labels stayed full-strength (junctions labeled on inkless points);
  scoped out in `v4-solid-geometry-replay`.

## What this run changes vs the promoted tess15 checkpoint

| axis | promoted (tess15) | this run |
|---|---|---|
| data | 14k synthetic + tess | **33,827 rows**: legacy 14k + tess 2,471 + **search225 6,451** + **box-pleated 10,905** (native-matched density incl. 4× dense tail) |
| sampler | v3-tessellation-15pct | **v5-bp-search225** (treemaker/rabbit 22.5% ea, tess 15%, search225 20%, BP 20%) |
| augmentations | v3 replay (19% dashed, 27% obfuscated) | **v4-solid-geometry-replay** (no dash/text/watermark) + width cap ≤3px on >2500-edge CPs |
| edge envelope | max_edges 1200 | **5000** |
| budget | 1500 steps × 2048 | **12000 steps × 8192** |
| unchanged | — | close-pair junction recipe (σ1.5 / offset r3 / focal 2,4), warm start from promoted, no head reinit, batch-stats BN |

## Data assets (LOCAL, must be synced to the pod)

On Zach's Mac under `~/Documents/datasets/create-pattern-detector/synthetic/`:

- `cp_training_mix_v5_bp_search225/` — the training root. **It is built from
  symlinks**; sync with `rsync -L` (follow links). Resolved size ≈ **2.3 GB**.
- (Sources, only needed if rebuilding: `cp_training_mix_v1`,
  `tessellation_*_v2_15pct`, `search225_v1`, `box_pleated_v1`,
  `box_pleated_dense_v1`.)

Sync (after `runpodctl ssh info` gives host/port):

```bash
rsync -avL -e "ssh -p $POD_PORT" \
  ~/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v5_bp_search225/ \
  root@$POD_HOST:/workspace/create-pattern-detector/data/generated/synthetic/cp_training_mix_v5_bp_search225/
```

Also sync the warm-start checkpoint (139 MB) to the same relative path the
pointer resolves to:

```bash
.venv/bin/python scripts/checkpoint/current_checkpoint.py --field checkpoint
# checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_weighted_probe_20260619/full/latest.pt
```

## Credentials

RunPod API key lives ONLY in the ignored `configs/runpod.env` in this repo
checkout on Zach's Mac (verified present). `~/.runpod/config.toml` may exist
but has historically held an empty key — do not rely on it. Never print the key.

```bash
set -a; source configs/runpod.env; set +a
runpodctl user   # sanity check
```

## Launch

Pod: 24GB+ GPU (4090 / L40S / A5000), template `runpod-torch-v220`, follow
`docs/runpod-quickstart.md` for bootstrap + the torch/cu121 pin gotcha.

```bash
tmux new -s v5run
NUM_WORKERS=12 scripts/training/run_cpline_runpod_v5_bp_search225_solid_geometry.sh
```

Notes:

- The launcher preflights 1 step and verifies the run config
  (profile/sampler/junction recipe/max_edges) before the real run; it refuses
  an existing `OUTPUT_ROOT`.
- `NUM_WORKERS`: dense CPs are CPU-render-bound (fixed-size targets, per-edge
  render). 8–12 workers on a RunPod CPU allotment keeps the GPU fed.
- Wall-clock estimate: 12k steps at batch 1 ≈ a few hours on a 4090-class GPU;
  checkpoints land every 1000 steps under `$OUTPUT_ROOT/full/`.
- `BATCH_SIZE` is env-tunable if VRAM allows (>1 changes BN statistics
  behavior under batch-stats mode — fine, but note it in the run log).

## Checkpoint selection (do NOT select on native-cp-v1)

Score every checkpoint with the product-parity scorecard, on the v5 val split:

```bash
.venv/bin/python scripts/evals/junction_scorecard.py \
  --checkpoint checkpoints/<run>/full/step_<N>.pt \
  --manifest data/generated/synthetic/cp_training_mix_v5_bp_search225/raw-manifest.jsonl \
  --val-count 512 --max-edges 5000 \
  --out reports/junction-scorecard-v5-step<N>.json
```

Run it twice per candidate: `--profile clean` (geometry recall) and
`--profile v4-solid-geometry-replay` (augmentation robustness). Decide on:

1. **per-CP clean_rate** (tracks end-to-end recovery; baseline for the
   promoted model on the v4 val was 87.5% overall) and **extras_per_sample**;
2. **per-family recall**, especially `box-pleated` (the new capability) and
   `search225-tiling`;
3. miss taxonomy: the run's thesis is shrinking `sub_threshold.isolated` and
   `absent.*` — if those don't move, say so plainly.

Baseline scorecard for the promoted checkpoint:
`reports/junction-scorecard-baseline-tess15w-48.json` (48-sample v4 val slice;
re-run it at 512 on the v5 manifest for a like-for-like baseline).

## Promotion evaluation (one look, in ori-studio)

Native-cp-v1 is the held-out test set. Evaluate the selected checkpoint once:

1. Export ONNX per `docs/checkpoint-management.md` conventions (radius-3
   offset head → decoder contract peak-gated offset-cluster; manifest must
   carry `inference.junction_offset_radius_px: 3.0`).
2. In `tree-maker-rust`: regenerate the native dense cache with the new model
   (the current cache was built with the MPS PyTorch path — see
   `native-cp-v1-pytorch-mps-v3-tess15-weighted`), then run
   `compare_exact_solve_benchmark` at pure defaults (they match the product
   decode). **Build and run from the same worktree** (provenance guard).
3. Compare per-bucket `solve_recovered_original` against **121/563**
   (easy 70 / medium 49 / hard 2) and junction P/R against the peak-gate
   numbers (easy 99.2% / medium 98.9% / hard 65% BP).
4. If promoting: follow the Update Rules in `docs/model-training-history.md`
   (checkpoint manifest, history tables, ONNX export + SHA, decoder settings).

Expected outcomes, calibrated: easy/medium gains ride on calibration/extras
(junction headroom easy +90 / medium +63 end-to-end); hard-BP junction recall
should move substantially if the density thesis is right, but hard *recovery*
stays capped (~+14 oracle) until pipeline work beyond junctions. If native
sub-threshold misses don't shrink while val is clean, the residual gap is
style/rendering — that's the signal to revisit real-data training or renderer
diversity, not more synthetic volume.

## Known caveats

- box-pleated M/V: 3.7% of creases are Maekawa-masked to `U` at ingest
  (assignment head sees honest unknowns; junction/line heads unaffected).
- 22 mix rows exceed max_edges=5000 (filtered; negligible).
- Run-to-run medium variance on the native benchmark is ±2–3 recovered.
- `line_style` head's "dashed" class gets no positives under the v4 profile
  (deliberate; see the augmentation audit in the PR description).
