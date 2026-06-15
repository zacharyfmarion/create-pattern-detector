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
- This is sufficient evidence to plan a controlled training run, but promotion
  evidence still requires a full 179-sample shipped-model BP baseline before the
  training result is interpreted.

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
   - Leave `v2-guide-grid`, `v2-dark-guide-grid`, `v2-combined`, and
     `v2-dark-combined` behavior unchanged.
   - This keeps R1/R3 and older V2 metrics reproducible.

2. Add guide-grid-free combined profiles.
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
   - Before training, run the full 179-sample BP baseline on the shipped V3 ONNX
     model and save the ignored product-side report.
   - Still add or identify a control slice of non-box-pleat readable CPs.
   - Keep a guide-grid artifact slice as a known tradeoff measurement, not a hard
     blocker.

6. Run a short training ablation.
   - Only start after the full shipped-model BP baseline is generated.
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
   - If raw `line_prob` stays weak on box-pleat creases, run a second ablation
     with explicit box-pleat/grid-positive synthetic examples or oversampling.

7. Evaluate promotion.
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

8. Export and document only after promotion.
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
