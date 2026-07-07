# High-Resolution Vertex Refiner V1

## Goal

Build a source-image-aware neural vertex refiner that makes crease-pattern
vertex detection as close to perfect as practical, especially for close pairs,
boundary contacts, dense box-pleat grids, and collinear split vertices.

The refiner is a second-stage model. CPLineNet remains the global dense evidence
model, but its stride-4 junction heatmap is no longer the final source of vertex
truth. The refiner looks at full-resolution local image crops and predicts
discrete vertices plus local topology signals.

Target product path:

```text
source image + CPLineNet junction evidence
  -> generous proposal crops
  -> high-res VertexRefiner
  -> refined vertices + incident rays
  -> tree-maker-rust junction-first candidate graph
  -> selection / exact solve
```

## Current Evidence

Clean-15 end-to-end ceiling tests in `tree-maker-rust` show that better
junctions are a major lever but not the only remaining lever:

| Run | Exact topo | Strict vertex F1 | Strict edge F1 | Missing | Extra | Merged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source lines + model junctions | `3/15` | `0.9852` | `0.9634` | `124` | `69` | `35` |
| source lines + GT junctions | `5/15` | `0.9945` | `0.9815` | `60` | `38` | `23` |
| source lines + GT junctions, merge radius 1 | `5/15` | `0.9974` | `0.9844` | `42` | `41` | `12` |

The exact-solve pass on the GT-junction, merge-radius-1 run accepted `13/15`
samples and solved `11/15`, while strict topology remained `5/15`. This means
near-perfect junctions should significantly improve the product pipeline, but
candidate span construction, assignments, and solver/selection details still
need separate follow-up after V1.

The prior image-only junction ablation is the warning sign: pure skeleton or
line-arrangement junctions are not good enough. The refiner should be neural,
but its primary input should be the original image and source-derived line
evidence rather than model line probability.

## Ownership

This repository owns:

- vertex-refiner architecture
- crop proposal and crop dataset construction
- labels and training targets
- training scripts and evaluation scripts
- checkpoint manifests
- ONNX export

`tree-maker-rust` owns:

- browser/runtime inference integration
- Rust/WASM decoding of refiner outputs
- candidate graph usage of refined vertices and incident rays
- product-side benchmarks and exact-solve reports

Do not train or checkpoint this model inside `tree-maker-rust`.

## Architecture Decision

Implement `VertexRefinerV1` as a small full-resolution U-Net over `96x96` crops.
V1 must not use model line probability as an input channel.

Phase 4 and the first product integration are source-only. Channels `4-6` below
are reserved by the contract but are hard-zeroed for training, validation, and
runtime. This avoids a serial dependency on the full-image CPLineNet/HRNet pass
and tests the clean architecture first:

```text
source image -> source-image proposals -> VertexRefiner crops -> graph
```

Only after that result is measured should we run an explicit source-plus-dense
ablation. Dense CPLineNet junction inputs are not part of the default V1
architecture.

Input tensor:

```text
shape: 8 x 96 x 96

channels:
1. original grayscale crop
2. processed source-image line/ink probability crop
3. source-image distance-to-ink crop
4. CPLineNet junction probability crop
5. CPLineNet junction offset dx crop
6. CPLineNet junction offset dy crop
7. normalized x coordinate within crop
8. normalized y coordinate within crop
```

Channels 4-6 are zero in source-only V1. Rendered GT junction labels are
permitted only for tests or explicit leakage diagnostics. Real CPLineNet dense
caches are reserved for a later ablation if source-only recall is not enough.

Backbone:

```text
encoder:
  conv 3x3, 48 channels
  residual block, 48
  downsample to 48x48, 96 channels
  residual block, 96
  downsample to 24x24, 192 channels
  residual block, 192
  residual block, 192

decoder:
  upsample to 48x48 + skip, 96 channels
  residual block, 96
  upsample to 96x96 + skip, 64 channels
  residual block, 64
  residual block, 64
```

Output heads at `96x96`:

```text
vertex_heatmap: 1 channel
vertex_offset: 2 channels
vertex_kind: 5 classes
degree: 9 classes
incident_rays: 36 multi-label direction bins
```

Vertex kinds:

```text
background
interior_junction
boundary_contact
corner
endpoint_or_dangling
```

Degree classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8+
```

Incident rays are 36 direction bins over 360 degrees, 10 degrees per bin. A
degree-4 vertex with arms leaving east, north, west, and south activates four
bins. The decoder can later prefer a candidate edge from A to B when A has a
ray toward B and B has a ray back toward A.

## Data And Targets

Add source files:

```text
src/models/vertex_refiner.py
src/data/vertex_refiner_dataset.py
src/data/vertex_refiner_targets.py
src/data/vertex_refiner_proposals.py
scripts/training/train_vertex_refiner.py
scripts/evals/eval_vertex_refiner.py
scripts/export/export_vertex_refiner_onnx.py
artifacts/checkpoints/current-vertex-refiner-model.json
```

Training examples are crops from existing synthetic FOLD manifests and, later,
real/native eval packs where GT graph data is available.

For each GT graph:

1. Render or load the source image at the target size.
2. Build processed source-image line evidence and distance-to-ink maps.
3. Zero CPLineNet auxiliary junction channels for source-only V1.
4. Generate high-recall crop proposals from source-image evidence.
5. Include GT-centered crop anchors in training only, for positive coverage.
6. Keep validation and eval proposals source-only by default.
7. For every crop, label all GT vertices inside the crop.

Target generation per GT vertex:

```text
vertex_heatmap: Gaussian sigma 1.0 px
vertex_offset: subpixel dx/dy near the peak
vertex_kind: derived from boundary position and incident edge count
degree: count of incident GT graph edges, capped at 8+
incident_rays: one 10-degree bin per incident edge direction
```

Boundary contacts should use the rendered square frame coordinates. Interior
vertices near the frame but not on the rendered square should not be mislabeled
as boundary contacts.

## Proposal Generation

Generate crops from the union of these high-recall sources:

1. Processed source-image line endpoints and branchpoints.
2. Source-image line-arrangement intersections with local ink support.
3. Boundary-contact candidates from source-image line hits near the square
   border.
4. GT-centered anchors for training positive coverage only.

Merge proposal centers within `12px`, preserving provenance. Crop `96x96`
around each center at original image resolution. Include positive, hard
negative, and empty/background crops.

Hard-crop oversampling:

- close vertex pairs under `8px`
- dense box-pleat orthogonal grids
- collinear degree-2 split vertices
- boundary contacts and near-boundary vertices
- low-contrast or very thin strokes
- prior model misses from clean-15 and box-pleat native reports

## Losses

Use a multi-task loss:

```text
heatmap focal loss
offset SmoothL1 loss, masked near GT vertices
vertex-kind cross-entropy
degree cross-entropy
incident-ray binary focal loss
close-pair repulsion loss
```

The close-pair repulsion loss should penalize fused peaks when two GT vertices
inside the crop are closer than `8px`.

Weights should start conservative:

```text
heatmap: 1.0
offset: 0.5
kind: 0.2
degree: 0.2
incident_rays: 0.5
close_pair_repulsion: 0.25
```

Tune by end-to-end strict topology, not by pixel loss alone.

## Inference Decode

Per crop:

1. Run local-max NMS on `vertex_heatmap`, radius `2px`.
2. Keep peaks above `0.25`.
3. Apply `vertex_offset`.
4. Read `vertex_kind`, `degree`, and incident-ray probabilities at each peak.
5. Map crop coordinates back to image coordinates.

Across overlapping crops:

1. Merge duplicate vertices with a score-gated global cluster pass. The first
   measured source-only setting was `radius=3px`, `min_score=0.4`,
   `min_support=1` at 512px. The current best seed-17 warm-continuation
   setting is `heatmap=0.075`, `radius=2px`, `min_score=0.25`, and
   `min_support_fraction=0.45`.
2. Do not merge two vertices if their incident-ray signatures strongly disagree.
3. Keep provenance, support count, cluster spread, and max/mean confidence.
4. Preserve low-confidence vertices as optional candidates only if they have
   strong source-image line support or ray agreement.

The first product integration can use refined points only. The second
integration should use incident rays to prefer or penalize candidate spans.

## Evaluation Strategy

Local ML eval in this repo:

- vertex precision/recall/F1 by crop
- coordinate error
- close-pair recall by separation bucket
- boundary-contact recall
- degree accuracy
- incident-ray precision/recall
- false positives per crop

Product eval in `tree-maker-rust`:

- clean-15 strict topology
- clean-15 candidate endpoint availability
- clean-15 candidate oracle recall
- box-pleat native dense and strict topology smoke/full runs
- close-pair recovery report
- exact-solve acceptance/solve counts

Promotion gates for V1:

```text
clean-15 strict vertex F1 >= 0.995
clean-15 strict edge F1 >= 0.980
clean-15 unmatched GT vertices <= 10
clean-15 exact topology count improves or stays stable
box-pleat native does not regress materially
candidate endpoint availability improves
```

The GT-junction ceiling remains the reference for whether remaining failures
are model-side or pipeline-side.

## Phases

### Phase 0: Baseline And Contracts

- [x] Freeze the clean-15 and box-pleat baseline reports to compare against.
- [x] Keep the GT-junction ceiling benchmark in `tree-maker-rust` as a reusable
      eval mode.
- [x] Document the refiner tensor contract and output coordinate convention.
- [x] Decide the versioned ONNX output names and manifest schema.

Phase 0 artifacts:

- `artifacts/evaluations/vertex-refiner-v1-phase0-baselines.json`
- `docs/vertex-refiner-v1-contract.md`
- `src/models/vertex_refiner_contract.py`
- `tests/test_vertex_refiner_contract.py`

Exit criteria:

- Baseline metrics are reproducible.
- Refiner output schema is stable enough for training and product integration.

### Phase 1: Crop Dataset And Target Renderer

- [x] Implement source-image line/ink probability crop generation.
- [x] Implement distance-to-ink crop generation.
- [x] Implement CPLineNet junction probability/offset crop loading.
- [x] Implement high-recall proposal generation.
- [x] Implement GT crop labels for heatmap, offset, kind, degree, and rays.
- [x] Add visual crop contact sheets for positives, close pairs, boundary
      contacts, and hard negatives.
- [x] Add unit tests for ray bins, degree labels, boundary kind labels, and
      close-pair target rendering.

Phase 1 artifacts:

- `src/data/vertex_refiner_targets.py`
- `src/data/vertex_refiner_proposals.py`
- `src/data/vertex_refiner_dataset.py`
- `scripts/visualize/vertex_refiner_crops.py`
- `tests/test_vertex_refiner_targets.py`
- `tests/test_vertex_refiner_proposals.py`
- `tests/test_vertex_refiner_dataset.py`

Exit criteria:

- Deterministic crop generation from a fixed manifest and seed.
- Visual inspection confirms target overlays align with source pixels.

### Phase 2: Model And Loss Implementation

- [x] Implement `VertexRefinerV1`.
- [x] Implement output decoding helpers.
- [x] Implement the multi-task loss.
- [x] Add channel dropout for CPLineNet junction channels.
- [x] Add smoke tests for forward pass, loss, decode, and ONNX-exportable ops.
- [x] Add a tiny overfit test on a handful of crops.

Phase 2 artifacts:

- `src/models/vertex_refiner.py`
- `src/models/losses/vertex_refiner_loss.py`
- `tests/test_vertex_refiner_model.py`

Exit criteria:

- Model can overfit a tiny crop set.
- Export path has no unsupported operators.

### Phase 3: Local Training Smoke

- [x] Train a small local smoke model on a limited synthetic slice.
- [x] Verify crop-level metrics move in the right direction.
- [x] Evaluate close-pair and boundary-contact slices.
- [x] Generate qualitative before/after crop overlays.

Phase 3 artifacts:

- `src/evaluation/vertex_refiner_eval.py`
- `scripts/training/train_vertex_refiner.py`
- `scripts/evals/eval_vertex_refiner.py`
- `tests/test_vertex_refiner_eval.py`

Phase 3 smoke result from 2026-06-22:

- Command: `scripts/training/train_vertex_refiner.py --device cpu --train-count 1 --val-count 1 --image-size 96 --proposals-per-sample 4 --batch-size 2 --max-steps 8 --base-channels 4 --lr 0.001 --output-dir /tmp/vertex_refiner_train_smoke_phase3`
- Train loss moved from `27.5704` to `25.3975`.
- Validation loss moved from `62.8290` to `57.7988`.
- Validation false positives moved from `301` to `266`; validation mean matched error moved from `1.6409px` to `1.1685px`.
- Validation F1 moved from `0.00424` to `0.00458`.
- The evaluator emitted crop slices for `close_pair`, `boundary_contact`, `corner`, `endpoint_or_dangling`, `gt_training_anchor`, and `positive`.
- Qualitative overlay: `/tmp/vertex_refiner_train_smoke_phase3/qualitative_before_after.png`.
- Standalone eval smoke: `scripts/evals/eval_vertex_refiner.py --device cpu --checkpoint /tmp/vertex_refiner_train_smoke_phase3/latest.pt --limit 1 --image-size 96 --proposals-per-sample 4 --batch-size 2 --base-channels 4 --out /tmp/vertex_refiner_train_smoke_phase3/eval.json`.

This smoke validates the model, loss, training, slice-eval, checkpoint, and
qualitative artifact plumbing. It is not a promotion-quality model and should
not be compared against the full product gates.

Exit criteria:

- The model beats simple local-max / source-CV junction baselines on crop-level
  recall and localization.

### Phase 4: Full GPU Training

- [x] Add RunPod launcher and config verification.
- [x] Make source-only auxiliary mode the default for training and eval.
- [x] Keep validation/eval GT proposal anchors disabled by default.
- [x] Train a bounded source-only RunPod budget probe on a selected
      `cp_training_mix_v1` slice.
- [x] Register checkpoint manifests under `artifacts/checkpoints/`.
- [x] Run local ML evals and failure mining for the budget probe.
- [x] Replace spatial-prefix proposal capping with quality-ranked,
      coverage-aware proposal selection.
- [x] Add full-pattern eval diagnostics that split no-crop misses from
      covered-but-not-matched misses.
- [x] Add a simple global merge/NMS pass for overlapping crop predictions.
- [x] Sweep score, support, and radius thresholds on the corrected source-only
      checkpoint.
- [x] Warm-start the corrected source-only checkpoint on a larger 4090 run.
- [x] Add explicit square-frame corner proposals.
- [x] Add visibility-normalized merge filtering and faster spatial-grid merge.
- [x] Re-run product-style full-pattern evals on seed-17 and seed-61 slices.
- [ ] Miss-mine covered-but-not-matched boundary, corner, and close-pair
      vertices from the warm-continuation checkpoint.
- [ ] Train on the current synthetic mix with hard-slice oversampling at the
      final 1024px/base-48 target.
- [ ] Replace threshold-only global merge with cluster scoring that uses crop
      visibility, source-line support, and topology consistency.
- [ ] Iterate on hard slices, loss weights, and proposal recall.

Phase 4 source-only setup artifacts:

- `scripts/training/run_vertex_refiner_runpod_source_only_probe.sh`
- `scripts/training/verify_vertex_refiner_run_config.py`
- `docs/vertex-refiner-phase-4-runpod.md`

Default Phase 4 probe invariants:

- `auxiliary_mode=zero`
- `include_gt_training_anchors=true`
- `include_val_gt_anchors=false`
- validation and standalone eval use source-image proposals only
- default `proposals_per_sample=128`
- launcher refuses H100/H200/B200/A100-class GPUs unless explicitly overridden

Phase 4 source-only budget probe from 2026-06-22:

- Registry: `artifacts/checkpoints/runpod-vertex-refiner-source-only-phase4-small-3090.json`
- Checkpoint: `checkpoints/runpod_vertex_refiner_source_only_probe_20260622_phase4_small/full/latest.pt`
- GPU: RunPod RTX 3090, pod `6a9ku2p8b6xsoc`, stopped after artifact copy-back.
- Cost: about `$0.35` account-balance movement across setup attempts and the
  completed probe, under the requested `$10` cap.
- Config: `image_size=512`, `base_channels=24`, `train_records=32`,
  `val_records=8`, `proposals_per_sample=8`, `max_steps=200`,
  `auxiliary_mode=zero`, `include_val_gt_anchors=false`.
- Training loss moved from `325.7701` at step 1 to `3.8574` at step 200.
- Sampled validation before/after moved from precision `0.0106`, recall
  `0.3725`, F1 `0.0207`, FP `8565`, loss `393.1651` to precision `0.8600`,
  recall `0.3482`, F1 `0.4957`, FP `14`, loss `3.9839`.
- Standalone source-only eval: precision `0.7158`, recall `0.2906`, F1
  `0.4134`, mean error `1.0020px`.
- Boundary-contact slice: precision `0.8125`, recall `0.2955`, F1 `0.4333`.
- Close-pair slice: precision `0.7879`, recall `0.1955`, F1 `0.3133`.
- Corner slice: precision `0.3333`, recall `0.1190`, F1 `0.1754`.

Phase 4 finding:

Source-only V1 is viable: it learns quickly, suppresses false positives, and
keeps localization near `1px` on matched vertices. It is not promotion-ready:
recall is the bottleneck, especially close pairs and corners. The next Phase 4
iteration should improve source-proposal recall and hard-slice sampling before
spending on the final 1024px/base-48 run.

Engineering finding:

Rendering/proposal generation was the paid-run bottleneck. The dataset now
caches rendered samples per record, which made repeated crop reads effectively
free after proposal construction. A future full run should add persistent
proposal/crop caches or precompute selected crop refs before launching a longer
GPU job.

Proposal-selection correction from 2026-06-22:

- The first budget probe used `proposals_per_sample=8` and the dataset capped
  by taking a spatially sorted prefix. That made crop coverage look like model
  recall and severely under-covered full-pattern GT vertices.
- `select_vertex_refiner_proposals` now keeps a deterministic quality-ranked,
  spatially diverse subset instead of slicing `proposals[:N]`.
- Standalone eval can now emit `proposal_coverage` and `full_pattern_metrics`.
- Local eval of the existing small checkpoint with the corrected selector and
  `PROPOSALS_PER_SAMPLE=128` covered `1200 / 1210` GT vertices (`99.17%`) on
  the 8-pattern val slice.
- With coverage fixed, full-pattern recall was still only `11.65%`; `1059 /
  1069` false negatives were covered-but-not-matched. The next bottleneck is
  model/training/decode quality on visible vertices, especially interior
  junctions, not broad crop proposal recall.
- Artifact:
  `artifacts/evaluations/vertex-refiner-proposal-selection-fix-20260622.json`.
- A corrected RunPod retry was attempted on L4 pod `f6w3yb8juvopp7` and RTX PRO
  4000 pod `du7ny360ld1gwf`; both stayed `RUNNING` but never reached SSH
  readiness. Both pods were stopped and `runpodctl pod list` returned empty.

Corrected source-only probe from 2026-06-22:

- Registry:
  `artifacts/checkpoints/runpod-vertex-refiner-source-only-phase4-fixed-selector-3090.json`
- Checkpoint:
  `checkpoints/runpod_vertex_refiner_source_only_probe_20260622_fixed_selector/full/latest.pt`
- GPU: RunPod RTX 3090 via official template `runpod-torch-v220`, pod
  `3ip8xq59ltlrqj`, stopped and deleted after artifact copy-back.
- Config: `image_size=512`, `base_channels=24`, `train_records=32`,
  `val_records=8`, `train_crops=3322`, `val_crops=740`,
  `proposals_per_sample=128`, `max_steps=500`, `auxiliary_mode=zero`,
  `include_val_gt_anchors=false`.
- Training loss moved from `161.1156` at step 1 to `1.1343` at step 500.
- Sampled validation before/after moved from precision `0.0249`, recall
  `0.3473`, F1 `0.0465`, FP `12359`, loss `283.5637` to precision `0.8717`,
  recall `0.8813`, F1 `0.8765`, FP `118`, loss `1.3751`.
- Standalone crop-level eval: precision `0.9022`, recall `0.9109`, F1
  `0.9065`, mean error `0.7098px`.
- Standalone proposal coverage: `1076 / 1085` GT vertices covered (`99.17%`).
- Standalone full-pattern eval before global NMS/merge: precision `0.1951`,
  recall `0.9558`, F1 `0.3240`, mean error `0.4911px`.
- Full-pattern recall by kind: boundary contact `0.8921`, corner `0.6563`,
  interior junction `0.9759`.

Updated Phase 4 finding:

The corrected source-only refiner largely solves the local vertex recall
question on the small 512px probe. The next bottleneck moved downstream:
overlapping proposal crops produce duplicate global predictions, so
full-pattern precision is low before a real global merge/NMS stage. Corner crop
coverage also remains weaker than interior and boundary-contact coverage.

Global merge sweep from 2026-06-22:

- Artifact:
  `artifacts/evaluations/vertex-refiner-global-merge-sweep-20260622.json`.
- Local sweep setup: corrected source-only checkpoint, `image_size=512`,
  `proposals_per_sample=128`, `heatmap_threshold=0.25`,
  `match_tolerance=2px`, no validation GT anchors.
- Local crop-level metrics on the same slice: precision `0.9111`, recall
  `0.9041`, F1 `0.9075`, mean matched error `0.6787px`.
- Local proposal coverage: `1200 / 1210` GT vertices covered (`99.17%`).
- Best simple full-pattern merge in the sweep: `radius=3px`,
  `min_score=0.4`, `min_support=1`, precision `0.9476`, recall `0.9116`,
  F1 `0.9292`, mean matched error `0.6159px`.
- High-precision thresholding is not enough: `min_score=0.7` reached precision
  `0.9934` but recall fell to `0.6174`.
- Absolute multi-crop support is also too blunt: `min_support=2` around
  `radius=3px` reached precision `0.9157` but recall fell to `0.8620`, with
  especially weak corner and boundary retention.

Updated merge finding:

The duplicate-overlap problem is real and simple global merge fixes most of
the precision collapse, but threshold-only merge is not exact-graph safe. The
next decoder should score clusters using local crop visibility, support
normalized by how many crops could have seen that location, max/mean score,
cluster spread, kind/degree/ray agreement, source-line support, and graph
consistency before rejecting candidates. Boundary and corner vertices need
separate handling because they naturally have less crop support than interior
junctions.

Warm continuation from 2026-06-23:

- Registry:
  `artifacts/checkpoints/runpod-vertex-refiner-warm-continue-20260623-4090.json`
- Checkpoint:
  `checkpoints/runpod_vertex_refiner_warm_continue_20260623/full/latest.pt`
- Init checkpoint:
  `checkpoints/runpod_vertex_refiner_source_only_probe_20260622_fixed_selector/full/latest.pt`
- GPU: RunPod RTX 4090, pod `1ctjmwchg23hmh`, stopped and deleted after
  artifact copy-back. `runpodctl` reported no pods, no network volumes, and
  `currentSpendPerHr=0` after cleanup.
- Config: `image_size=512`, `base_channels=24`, `train_records=64`,
  `val_records=8`, `train_crops=6811`, `val_crops=872`,
  `proposals_per_sample=128`, `max_steps=2000`, `lr=0.00005`,
  `auxiliary_mode=zero`, `include_val_gt_anchors=false`.
- Training loss moved from `0.9973` at step 1 to `0.1576` at step 2000.
- Standalone crop-level eval on the launcher seed: precision `0.9656`, recall
  `0.9569`, F1 `0.9612`, mean error `0.4535px`.
- Raw full-pattern eval before merge remained duplicate-heavy: precision
  `0.2088`, recall `0.9820`, F1 `0.3444`.

Square-frame corner proposal correction from 2026-06-23:

- Added `square_frame_corner_proposals` so the four known rectified-square
  corners always receive candidate crops when a square frame is known.
- Seed-17 proposal coverage improved to `1210 / 1210` GT vertices (`100%`),
  including `35 / 35` corners.
- Best seed-17 product-style full-pattern eval:
  `heatmap=0.075`, `radius=2px`, `min_score=0.25`,
  `min_support_fraction=0.45`, precision `0.9924`, recall `0.9678`, F1
  `0.9799`, TP `1171`, FP `9`, FN `39`.
- Seed-17 recall by kind at that setting: boundary contact `0.9551`, corner
  `0.9143`, interior junction `0.9715`.
- Seed-61 at the same setting: proposal coverage `100%`, precision `0.9887`,
  recall `0.9583`, F1 `0.9733`, TP `1493`, FP `17`, FN `65`.
- Seed-61 focused sweep best F1 setting:
  `heatmap=0.125`, `radius=2px`, `min_score=0.3`,
  `min_support_fraction=0.55`, precision `0.9980`, recall `0.9570`, F1
  `0.9771`, TP `1491`, FP `3`, FN `67`.

Updated Phase 4 blocker:

The corner proposal correction closes the measured no-crop gap on the seed-17
and seed-61 slices. The remaining misses are now covered-but-not-matched, so
further proposal overgeneration is not the main lever. The weak product slices
are boundary contacts and corners, with close pairs still requiring hard-slice
attention at crop level. The next meaningful improvement should come from
miss-mining, boundary/corner oversampling and loss weighting, larger/final
resolution training, or a better decoder head; threshold tuning has plateaued
around full-pattern F1 `0.98`.

Exit criteria:

- Refiner hits the clean-15 local vertex target before product integration.
- Checkpoint manifest records dataset, training args, checkpoint SHA, and eval
  reports.

### Phase 5: Product Export And Integration

- [ ] Export ONNX with fixed input/output names.
- [ ] Add model manifest and checker script.
- [ ] Integrate browser/runtime refiner inference in `tree-maker-rust`.
- [ ] Add Rust/WASM decode support for refined vertices.
- [ ] Feed refined vertices into `junction-first-v1`.
- [ ] Keep the first integration using points only; add incident-ray span
      scoring after the point-only baseline is measured.

Exit criteria:

- Browser/runtime decode can run CPLineNet plus refiner end to end.
- Product benchmarks can toggle model junctions vs refined junctions vs GT
  junctions.

### Phase 6: End-To-End Evaluation And Promotion

- [ ] Run clean-15 strict topology.
- [ ] Run clean-15 candidate coverage.
- [ ] Run close-pair recovery report.
- [ ] Run box-pleat native dense and strict topology evals.
- [ ] Compare against the current promoted CPLineNet-only pipeline.
- [ ] Promote only if strict vertex recall, candidate endpoint availability, and
      strict edge F1 improve without unacceptable box-pleat regression.

Exit criteria:

- A promoted vertex-refiner checkpoint is registered.
- Product model manifests point to the approved ONNX assets.
- `docs/model-training-history.md` records the refiner promotion and required
  decoder settings.

### Phase 7: Pipeline Follow-Up

After V1 reaches the GT-junction ceiling as closely as practical, remaining
errors should be handled outside the refiner:

- [ ] Use incident-ray agreement in candidate span scoring.
- [ ] Improve assignment evidence where strict assignment accuracy limits exact
      topology.
- [ ] Tune selection penalties for overlong spans, unsupported crossings, and
      boundary contacts.
- [ ] Re-run the GT-junction ceiling after each pipeline change to distinguish
      model limits from decoder limits.

Exit criteria:

- Remaining clean-15 failures are categorized as refiner, candidate generation,
  selection, assignment, or exact-solve issues.
- The next highest-leverage phase is explicit before more training is launched.

## Open Questions

- Should V1 include CPLineNet junction offsets as input from the start, or only
  the junction heatmap? The current plan includes both with 50% channel dropout.
- Should `endpoint_or_dangling` be kept as a class if production CPs should not
  have dangling interior endpoints? Keep it for robustness and hard negatives,
  then measure whether it helps.
- Should degree and rays be decoded from the peak pixel only, or averaged over a
  small window around the peak? Start with peak pixel; add local averaging if
  ray outputs are noisy.
- Should real/native box-pleat crops with GT graph data be included in training
  immediately? Start synthetic-first for reproducibility, then add a controlled
  real/native fine-tune if synthetic-trained V1 misses the same native slices.

## Artifact Policy

Generated crop caches, dense caches, training checkpoints, reports, and
visualizations are ignored artifacts. Commit code, configs, small deterministic
fixtures, checkpoint manifests, and this plan. Do not commit `.pt`, ONNX files,
large crop caches, or raw generated datasets.
