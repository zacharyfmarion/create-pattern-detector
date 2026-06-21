# Remove Guide-Grid Non-Crease Training

## Goal

Train the next CPLineNet model so legitimate horizontal and vertical crease
lines, especially box-pleated/grid-heavy designs, are not suppressed as
non-crease artifacts.

The immediate failure mode is visible before vectorization: raw `line_prob`
misses some real horizontal/vertical creases, while `non_crease_prob` fires on
those same strokes. That means inference-time non-crease suppression is not the
root cause by itself. The model has likely learned from synthetic guide-grid
training examples that grid-like orthogonal line evidence is background/artifact
evidence.

## Decision

Do not train future models with synthetic guide-grid overlays as positive
`v2_non_crease_mask` targets.

Also do not leave guide-grid overlays in the input image as unlabeled background.
If guide-grid pixels remain visible while `line_prob` target is zero, the line
head and hard-negative mining can still learn that grid-like strokes should be
ignored. For this fix, guide-grid rendering must be removed from the future
training mix, not merely removed from the auxiliary non-crease mask.

Keep the existing V2 guide-grid profiles available for historical checkpoint
reproduction and stress evaluation. Add a new guide-grid-free training profile
or mix for new models.

## Current Status

Status as of 2026-06-15:

- The promoted production/browser model is the V3 close-pair R1 warm-start
  checkpoint, not the later R3 from-scratch ablation.
- A deterministic native box-pleat eval set now exists at
  `eval_specs/box_pleat_native_v1.json`. It selects `179` candidates from the
  scraped native converted-FOLD corpus using path-independent canonical FOLD
  fingerprints.
- The product-side browser/Rust eval lives in `tree-maker-rust`, because the
  shipped ONNX model and post-processing pipeline are evaluated there:
  - `scripts/cp-detect/build-box-pleat-native-pack.py`
  - `scripts/cp-detect/evaluate-box-pleat-dense-cache.py`
  - `scripts/cp-detect/README.md#box-pleat-native-eval`
- A top-3 smoke run on the shipped V3 ONNX model supports the hypothesis:
  - orthogonal BP crease raw line recall: `0.3323`
  - orthogonal BP crease effective recall after non-crease suppression: `0.2962`
  - orthogonal BP crease non-crease conflict fraction: `0.6822`
  - diagonal/other crease raw line recall control: `0.8591`
  - strict topology F1 on the same smoke pack: `0.1103`
- The full 179-sample shipped-model BP dense baseline is complete. Generated
  product artifacts are ignored under `tree-maker-rust/artifacts/`:
  - pack:
    `cp-detect-correctness/packs/box-pleat-native-v1-baseline-v3-20260615/`
  - dense cache:
    `cp-detect-correctness/dense-cache/box-pleat-native-v1-baseline-v3-20260615-browser-onnx-v3/`
  - dense report:
    `cp-detect-correctness/reports/box-pleat-native-v1-baseline-v3-20260615-dense-heads/`
- Full baseline metrics:

  | Slice | Raw line recall | Effective recall | Recall drop | Non-crease conflict | Mean line prob | Mean non-crease prob |
  |---|---:|---:|---:|---:|---:|---:|
  | Orthogonal BP creases | `0.5130` | `0.4744` | `0.0386` | `0.5422` | `0.5323` | `0.5887` |
  | Diagonal/other creases | `0.8752` | `0.8587` | `0.0166` | `0.0902` | `0.8802` | `0.1334` |
  | All creases | `0.5964` | `0.5631` | `0.0333` | `0.4345` | `0.6125` | `0.4798` |

- The Rust strict-topology full pass was intentionally not used as the
  pre-training gate because degenerate topology can make post-processing slow.
  A partial run scored 24 samples and confirmed the pass is useful later, but
  the dense-head baseline above is the required pre-training comparison.
- A controlled 800-step RunPod probe using `v3-no-guide-grid-replay`,
  warm-starting from R1 and reinitializing `non_crease_head`, completed
  successfully. It was exported to ONNX and evaluated through the product
  browser dense-cache path on the full 179-sample BP pack. Probe artifacts:
  - checkpoint:
    `checkpoints/runpod_v3_no_guide_grid_probe/full/latest.pt`
  - product ONNX:
    `tree-maker-rust/apps/web/public/models/cp-detector-v3-no-grid-probe/`
  - dense cache:
    `tree-maker-rust/artifacts/cp-detect-correctness/dense-cache/box-pleat-native-v1-no-grid-probe-20260615-browser-onnx/`
  - dense report:
    `tree-maker-rust/artifacts/cp-detect-correctness/reports/box-pleat-native-v1-no-grid-probe-20260615-dense-heads/`
- Full BP dense probe metrics:

  | Slice | Raw line recall | Effective recall | Recall drop | Non-crease conflict | Mean line prob | Mean non-crease prob |
  |---|---:|---:|---:|---:|---:|---:|
  | Orthogonal BP creases | `0.5528` | `0.5525` | `0.0003` | `0.0049` | `0.5687` | `0.2699` |
  | Diagonal/other creases | `0.8756` | `0.8744` | `0.0012` | `0.0103` | `0.8811` | `0.2921` |
  | All creases | `0.6272` | `0.6267` | `0.0005` | `0.0061` | `0.6408` | `0.2737` |

- Compared with the shipped-model baseline, orthogonal BP effective recall
  improved by `+0.0781`, orthogonal non-crease conflict dropped by `-0.5373`,
  and orthogonal suppression recall drop fell from `0.0386` to `0.0003`. This
  strongly supports the guide-grid suppression hypothesis.
- The remaining BP tail is now mostly raw line-head weakness on very dense
  designs, not inference-time non-crease suppression. The next approved step is
  a full RunPod training candidate using the same recipe.

## Non-Goals

- Do not remove the `non_crease` head entirely. Text and watermark suppression
  can remain useful.
- Do not change the historical R1/R3 checkpoint manifests or their training
  provenance.
- Do not claim robust guide-grid-background rejection in the next model unless a
  separate evaluation proves it.
- Do not add orientation-specific inference heuristics like "never suppress
  horizontal or vertical lines." The fix should be learned from safer training
  targets and positive box-pleat coverage.

## Affected Areas

- `src/data/v2_augmentations.py`
  - `v2-guide-grid` and `v2-combined` currently call `_add_guide_grid`.
  - `v2-combined` also includes text/watermark/dashed/faint/ambiguous effects,
    so replacing only standalone `v2-guide-grid` is not enough.
- `src/data/cpline_augmentations.py`
  - `V2_AUGMENT_MIX`, `V2_DARK_AUGMENT_MIX`, `V2_ALL_AUGMENT_MIX`, and
    `V2_REPLAY_CORRECTIVE_MIX` include guide-grid profiles directly or through
    combined profiles.
- `tests/test_cpline_phase3.py`
  - Existing tests assert old V2 replay mix contents. Keep those tests for
    reproducibility and add new tests for the guide-grid-free mix.
- `scripts/training/run_cpline_runpod_v2_replay_correction.sh`
  - This script is historical for the promoted R1 model. Prefer adding a new
    script or explicitly documented profile for the no-grid run rather than
    silently changing the old recipe.
- `scripts/v2/generate_issue_benchmark.py` and `scripts/v2/run_issue_ablation.py`
  - Guide-grid false-positive cases should remain available as stress/eval
    slices, but should no longer be a primary training objective.

## Implementation Steps

1. Preserve the old V2 profiles.
   - Status: implemented in this branch.
   - Leave `v2-guide-grid`, `v2-dark-guide-grid`, `v2-combined`, and
     `v2-dark-combined` behavior unchanged.
   - This keeps R1/R3 and older V2 metrics reproducible.

2. Add guide-grid-free combined profiles.
   - Status: implemented in this branch as `v2-combined-no-grid` and
     `v2-dark-combined-no-grid`.
   - Add light/dark profiles such as:

     ```text
     v2-combined-no-grid
     v2-dark-combined-no-grid
     ```

   - Refactor `apply_v2_augmentation` so "combined" behavior is controlled by
     booleans instead of hard-coded `issue_profile == "v2-combined"` checks:
     - `combined_style = issue_profile in {"v2-combined", "v2-combined-no-grid"}`
     - `include_guide_grid = issue_profile == "v2-combined"`
     - `include_text = combined_style or issue_profile == "v2-text"`
     - `include_watermark = combined_style or issue_profile == "v2-watermark"`
     - similar booleans for dashed/faint/ambiguous.
   - For `v2-combined-no-grid`, keep dashed, faint, ambiguous-M/V, watermark, and
     text effects; skip `_add_guide_grid` entirely.

3. Add a guide-grid-free training mix.
   - Status: implemented in this branch as `v3-no-guide-grid-replay`.
   - Add a new mix name, for example:

     ```text
     v3-no-guide-grid-replay
     ```

   - Mirror the useful shape of `v2-replay-corrective` without any profile that
     renders guide grids:
     - replay old readable profiles from `stage-balanced`;
     - keep extra `line-style`;
     - include text/watermark/dashed/faint/ambiguous profiles;
     - use `v2-combined-no-grid` and `v2-dark-combined-no-grid` for combined
       stress.
   - Do not include `v2-guide-grid`, `v2-dark-guide-grid`, `v2-combined`, or
     `v2-dark-combined` in this mix.

4. Add target-safety tests.
   - Status: implemented and locally validated in this branch.
   - Assert `AUGMENT_MIXES["v3-no-guide-grid-replay"]` contains no profile with
     `"guide-grid"` and no original `v2-combined`/`v2-dark-combined`.
   - Assert `render_cpline_sample(..., augment_profile="v2-combined-no-grid")`
     produces metadata modes without `"guide_grid"`.
   - Assert combined-no-grid still emits non-crease positives for text/watermark
     and still marks dashed/faint/ambiguous line styles as expected.
   - Add a simple orthogonal-grid positive fixture or synthetic CP helper and
     assert all real horizontal/vertical creases remain in `v2_target_line_mask`
     and do not overlap positive `v2_non_crease_mask`.

5. Add a box-pleat/grid-bias diagnostic.
   - Done for the product pipeline in `tree-maker-rust`:
     `scripts/cp-detect/evaluate-box-pleat-dense-cache.py` reports dense-head
     evidence on the deterministic native BP pack.
   - The current diagnostic reports raw model activations on the curated
     box-pleat slice:
     - mean `line_prob` on GT edge pixels;
     - mean `non_crease_prob` on GT edge pixels;
     - fraction of GT edge samples where `non_crease_prob >= 0.65`;
     - fraction where both `non_crease_prob >= 0.65` and `line_prob < 0.85`;
     - line recall with and without inference-time non-crease suppression.
   - Optional improvement: add p10/p50 line-prob and p50/p90 non-crease-prob
     summaries if the full-run report needs more distribution detail.
   - Done: the full 179-sample BP baseline on the shipped V3 ONNX model is
     generated and summarized in Current Status.
   - Still add or identify a control slice of non-box-pleat readable CPs.
   - Keep a guide-grid artifact slice as a known tradeoff measurement, not a hard
     blocker.

6. Run a short training ablation.
   - Status: complete. The 800-step probe validated the recipe direction.
   - Run on RunPod; local training is only for smoke tests.
   - Budget note: keep this probe within the user's approximately `$15` RunPod
     budget. Locate credentials from existing worktree setup rather than adding
     secrets to git.
   - Used `scripts/training/run_cpline_runpod_v3_no_guide_grid_probe.sh`.
   - Start from the promoted R1 checkpoint:

     ```text
     checkpoints/r1_close_pair_warmstart/latest.pt
     ```

   - Use the new `v3-no-guide-grid-replay` mix.
   - Strong candidate run:
     - warm-start all heads from R1;
     - reinitialize `non_crease_head` to remove the legacy guide-grid detector;
     - train with the existing text/watermark non-crease loss;
     - keep the line head warm-started so it preserves existing crease recall.
   - Raw `line_prob` improved on aggregate but stays weak on the densest
     box-pleat samples. Do not change the recipe yet; first run a full-length
     candidate to separate undertraining from representation limits.

7. Run a full training candidate.
   - Status: approved for launch on 2026-06-15.
   - Use `scripts/training/run_cpline_runpod_v3_no_guide_grid_full.sh`.
   - Start clean from the promoted R1 checkpoint, not from the 800-step probe:

     ```text
     checkpoints/r1_close_pair_warmstart/latest.pt
     ```

   - Default run configuration:
     - `PROFILE=v3-no-guide-grid-replay`
     - `EVAL_PROFILE=v3-no-guide-grid-replay`
     - `REINIT_HEADS=non_crease_head`
     - `CHECKPOINT_LOAD_MODE=init`
     - `STEPS_FULL=5000`
     - `TRAIN_COUNT_FULL=2048`
     - `VAL_COUNT_FULL=256`
     - `LR=0.00005`
     - `SEED=31`
     - `CHECKPOINT_EVERY=500`
     - `SKIP_GRAPH_EVAL=1`
     - `SKIP_FINAL_EVAL=1`
   - Use an RTX 4090-class pod when available. The 800-step probe took about
     `1003s`, so 5000 steps should be roughly `1.75h` of training on the same
     GPU class before setup/copy overhead.

8. Evaluate promotion.
   - Compare against the full shipped-model BP baseline, not only the top-3
     smoke run.
   - Required improvement:
     - box-pleat/grid-heavy GT line recall improves;
     - `non_crease_prob` on real horizontal/vertical GT creases drops
       substantially;
     - missing horizontal/vertical edges visibly recover in overlays.
   - Required non-regression:
     - clean/line-style/dashed/faint readable CP metrics stay close to R1;
     - text/watermark artifact suppression remains useful;
     - assignment accuracy and border metrics do not collapse.
   - Expected tradeoff:
     - synthetic guide-grid false-positive rejection may regress. That is
       acceptable if box-pleat positives improve and no real target requires
       guide-grid suppression as a primary production feature.

9. Export and document only after promotion.
   - If the no-grid model wins, add a checkpoint manifest under
     `artifacts/checkpoints/`.
   - Update `docs/model-training-history.md`.
   - Export ONNX for `tree-maker-rust` and record:
     - ONNX manifest path;
     - ONNX SHA-256;
     - required decoder settings;
     - whether non-crease suppression should remain enabled, softened, or
       disabled for the new model.

## Acceptance Gates

Before promoting the new model:

- A box-pleat/grid-heavy eval slice exists and is repeatable.
- Raw `line_prob` on real horizontal/vertical GT creases improves versus R1.
- `non_crease_prob` no longer systematically fires on those GT creases.
- Effective vectorized edge recall improves on the failing box-pleat examples.
- The model does not lose the close-pair R1 gains without an explicit decision.
- `docs/model-training-history.md` identifies the promoted model and clearly
  states whether guide-grid artifact rejection is in or out of the supported
  envelope.

## Rollback

If the no-grid training run improves box-pleat recall but causes unacceptable
false positives elsewhere, do not restore guide-grid non-crease training as-is.
Instead choose one of these follow-up tracks:

- add box-pleat/grid-positive training examples and retry;
- keep text/watermark `non_crease` but disable or soften inference suppression;
- train a separate, context-aware guide-grid/background-artifact head;
- handle fused or over-generated grid candidates downstream with topology and
  quality-report logic rather than teaching the dense line head that orthogonal
  strokes are artifacts.
