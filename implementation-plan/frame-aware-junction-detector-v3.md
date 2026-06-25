# Frame-Aware Junction Detector V3

## Goal

Build the next junction-refiner model for product-style rendered crease
patterns, using the V2 detector as the structural baseline but removing the
skeleton input channel.

V3 should detect local vertex topology inside `96 x 96` crops from already
rectified/cropped light-mode crease-pattern images. The product does not need
V3 to solve arbitrary page cleanup, dark-mode diagrams, hand-drawn CPs,
background grids, photos, watermarks, or text occlusions in this phase.

The detector should produce the same topology outputs as V2:

- subpixel vertex position
- vertex kind: `interior_junction`, `boundary_contact`, `corner`,
  `endpoint_or_dangling`
- degree class
- incident ray bins
- confidence
- boundary side for boundary contacts

## Current Evidence

V2 established that the frame-aware crop model is the right shape for junction
refinement:

- Boundary-contact recall improved materially on corrected native CPOOGLE
  clean-15.
- Increasing the proposal cap from `128` to `256` removed no-crop misses on
  the native clean-15 slice.
- The current best V2 checkpoint reached native clean-15 full-pattern F1
  `0.9908` at proposal cap `256`.

The same checkpoint is not robust to light rendered style variation:

| Profile | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `clean` | `0.9688` | `0.9553` | `0.9620` |
| `square-symmetry` | `0.9740` | `0.9521` | `0.9629` |
| `line-style` | `0.7325` | `0.8174` | `0.7726` |
| `print-light` | `0.7221` | `0.9165` | `0.8078` |
| `print-medium` | `0.3957` | `0.7669` | `0.5220` |
| `photo-light` | `0.5147` | `0.8657` | `0.6456` |

This tells us:

- Rotation/flip geometry generalizes.
- Render style does not.
- The main failure is precision collapse under thicker, noisier, lower-contrast,
  or compressed light-mode lines.
- We should not ask V3 to learn dark mode, grids, watermarks, arbitrary text, or
  folded-model clutter yet.

The accepted-image corpus confirms the V1 product target is narrower than the
old HRNet stress profiles:

- mostly light-background rendered CPs
- varied line color, line thickness, antialiasing, faintness, and compression
- occasional off-white backgrounds
- some page/crop clutter, but product-side crop/rectification owns that before
  `96 x 96` junction refinement

## Inference Contract

Runtime inputs are the same product-level inputs assumed by V2:

1. Rectified/cropped crease-pattern image.
2. Known square paper frame in the rectified image.
3. Soft source-derived image channels computed from pixels.

FOLD graphs are only needed for training and evaluation labels.

Runtime flow:

```text
rectified/cropped light-mode CP image
  -> compute soft source/frame channels
  -> generate source/frame proposals
  -> run VertexRefinerV3 on 96x96 crops
  -> frame-aware decode, snap, and merge
  -> junction topology candidates for graph construction
```

Important constraints:

- Do not hard-mask the crop with Hough, skeleton, or binary cleanup before the
  model.
- Do not require CPLineNet dense line probability or dense junction heatmaps.
- Do not feed a skeleton channel into the model.
- Keep derived channels soft enough that bad preprocessing cannot erase the
  original grayscale evidence.

## Architecture

Implement `VertexRefinerV3` as the V2 frame-aware crop architecture with the
`source_skeleton` channel removed.

Crop size:

```text
96 x 96
```

Initial V3 input channels:

```text
1. grayscale source image
2. source ink probability
3. distance to ink
4. local line orientation cos(2 theta)
5. local line orientation sin(2 theta)
6. signed distance to square frame, clipped and normalized
7. frame-edge mask
8. inside-paper mask
9. boundary-contact prior: source ink near frame
10. normalized local x coordinate
11. normalized local y coordinate
```

The only mandatory architecture change from V2 is removing channel 4,
`source_skeleton`. V2 had 12 channels; V3 starts with 11.

Rationale:

- Skeletonization is a brittle discrete transform. It can break faint or
  compressed lines, shift centerlines, add branches, and delete valid evidence.
- The model should always retain access to the original grayscale crop.
- `source_ink_probability`, `distance_to_ink`, and orientation are still soft
  hints, but they must not become single points of failure.

Channel robustness requirements:

- Add training-time dropout/noise for `source_ink_probability`,
  `distance_to_ink`, and orientation channels.
- Include a channel-ablation eval that can zero each derived channel family at
  validation time.
- If V3 depends too heavily on `distance_to_ink` or orientation, reduce the
  input further before the paid training run.

Backbone:

- Reuse the V2 high-resolution U-Net capacity initially.
- Add a new model/input contract rather than overloading V2 checkpoints.
- Keep output heads and loss surfaces compatible with V2.

Output heads:

```text
vertex_heatmap: global vertex heatmap
boundary_contact_heatmap: boundary-specific vertex heatmap
vertex_offset: subpixel dx/dy
vertex_kind: 5 classes
degree: 9 classes, 0..8+
incident_rays: 36 multi-label direction bins
boundary_side: top/right/bottom/left auxiliary class
```

Decode:

1. Decode peaks from `vertex_heatmap`.
2. Decode peaks from `boundary_contact_heatmap`.
3. Union and locally suppress duplicates.
4. For boundary-like predictions, snap the perpendicular coordinate to the
   square frame.
5. Merge boundary contacts by `(side, side_coordinate)`.
6. Merge interior vertices by Euclidean radius.

Start from the V2 decoder and cap-`256` proposal recommendation.

## Training Augmentation Target

V3 training should focus on light-mode rendered CP robustness, not broad
real-world image cleanup.

In scope:

- clean black/gray line art on white or off-white backgrounds
- red/blue/gray MV renders
- monochrome black-line renders
- varied line width and antialiasing
- faint low-contrast lines
- muted/pastel line colors
- mild Gaussian blur
- mild pixel noise
- JPEG/PNG compression artifacts
- small brightness/contrast shifts
- square rotations/flips
- dense box-pleat and tessellation-like line density

Out of scope for V3:

- dark mode
- background grids
- watermarks and logos
- text or model photos overlapping the CP
- hand-drawn CPs
- arbitrary page cleanup
- large perspective/photo rectification problems

Create a dedicated augmentation profile, tentatively:

```text
vertex-light-rendered
```

Proposed mix:

| Component | Weight | Notes |
| --- | ---: | --- |
| `clean` | `0.15` | Preserve clean benchmark strength. |
| `square-symmetry` | `0.10` | Rotations/flips only. |
| `line-style-light` | `0.25` | Width, opacity, MV/monochrome/muted colors. |
| `print-light` | `0.25` | Off-white background, mild blur/noise/JPEG. |
| `print-medium-lite` | `0.15` | Stronger light-mode style without grids/dark/text. |
| `faint-light` | `0.10` | Low contrast but still rendered line art. |

Do not reuse `v3-no-guide-grid-replay` for V3 junction training. It includes
issue profiles that are outside this V1 product target and produced severe
precision collapse in the current V2 checkpoint.

## Training Strategy

Train source-only. Do not feed CPLineNet dense outputs.

Crop mix:

- product-style source/frame proposals, cap `256`
- GT-centered and GT-jittered anchors for positive coverage
- boundary-contact oversampling, `3x` to `5x`
- dense-grid/box-pleat samples, but balanced so they do not dominate
- close-pair examples
- hard negatives from rendered light-mode line clutter near true lines and
  near the frame

Channel regularization:

- randomly zero `source_ink_probability` for a small fraction of crops
- randomly zero `distance_to_ink`
- randomly zero orientation channels as a pair
- add mild noise/blur to derived channels independently of grayscale
- never zero grayscale
- never zero frame channels in the primary run, but add a separate eval that
  verifies frame sensitivity

Losses:

- focal/BCE for global vertex heatmap
- weighted focal/BCE for boundary-contact heatmap
- SmoothL1 for offset
- cross entropy for vertex kind
- cross entropy for degree
- BCE for incident ray bins
- cross entropy for boundary side, only on boundary-contact targets

## Evaluation Bar

Primary gates before product integration:

| Eval | Target |
| --- | ---: |
| Native CPOOGLE clean-15 full-pattern F1 | `>= 0.990` |
| Native CPOOGLE clean-15 recall | `>= 0.990` |
| Native CPOOGLE clean-15 precision | `>= 0.985` |
| Light-style crop F1, aggregate | `>= 0.95` |
| `line-style` crop F1 | `>= 0.94` |
| `print-light` crop F1 | `>= 0.94` |
| `print-medium-lite` crop F1 | `>= 0.92` |
| Boundary-contact recall on native clean-15 | `>= 0.98` |
| Proposal coverage at cap `256` | `>= 0.999` |

Every full-pattern eval report must include:

- per-kind recall
- false negatives split into no-crop vs covered-but-missed
- false positives by predicted kind
- false positives by nearest GT kind
- boundary-contact misses by side
- tolerance sweep at `1.0`, `1.5`, `2.0`, `2.5`, `3.0`, `4.0`, and `5.0` px

Every augmentation eval report must include:

- per-profile precision/recall/F1
- boundary-contact slice metrics
- close-pair slice metrics
- false positives per crop
- visual examples of high-FP crops
- derived-channel ablation results

Accepted-image corpus usage:

- Use accepted images for visual/style validation and sampled crop review.
- Do not treat accepted images as metric ground truth unless labels are added.
- Build contact sheets of normalized `96 x 96` crops from accepted images before
  training, to verify the augmentation mix looks like the product target.

## Implementation Phases

### Phase 0: Lock The V3 Scope

- [x] Document the product target as light-mode rendered CPs only.
- [x] Keep dark mode, background grids, watermarks, text occlusion, hand-drawn
      CPs, and general page cleanup out of the V3 training/eval bar.
- [x] Keep V2 artifacts and reports intact for comparison.

Exit criteria:

- V3 scope is explicit enough that augmentation choices can be rejected for
  being too broad.

### Phase 1: Define The V3 Input Contract

- [x] Add a V3 input version with 11 channels and no `source_skeleton`.
- [x] Keep V2 input version loadable for existing checkpoints.
- [x] Update model contract docs with the V2-to-V3 channel mapping.
- [x] Add shape tests for V3 crops.
- [x] Add tests proving the V3 input tensor omits the skeleton channel.

Exit criteria:

- A V3 crop tensor has shape `11 x 96 x 96`.
- No V3 code path reads or stores `source_skeleton` as a model input.
- V2 checkpoints and evals still run.

### Phase 2: Add Light-Rendered Augmentation Profiles

- [x] Add `line-style-light`.
- [x] Add `print-medium-lite`.
- [x] Add `faint-light`.
- [x] Add the weighted `vertex-light-rendered` mix.
- [x] Ensure all geometry-affecting transforms update labels before rendering.
- [x] Ensure photometric transforms touch only the input image.
- [x] Exclude dark, grid, text, watermark, and photo-object issue profiles.

Exit criteria:

- The augmentation profile is deterministic under a seed.
- Sidecar metadata records selected profile, line width, colors, background,
  blur/noise/JPEG settings, and geometry settings.

### Phase 3: Visual QA Before Training

- [x] Render contact sheets for `clean`, `line-style-light`, `print-light`,
      `print-medium-lite`, `faint-light`, and `vertex-light-rendered`.
- [x] Render matching input-channel sheets: grayscale, ink, distance,
      orientation, frame channels, and boundary prior.
- [x] Render accepted-image `96 x 96` product-crop sheets for style comparison.
- [ ] Manually reject augmentation settings that look unlike the accepted
      light-mode target.

Generated QA artifacts:

- `visualizations/vertex_refiner_v3_qa/source_augmentation_contact_sheet.png`
- `visualizations/vertex_refiner_v3_qa/input_channels.png`
- `visualizations/vertex_refiner_v3_qa/accepted_image_crops.png`

QA note: the accepted-image directory contains some out-of-scope dark,
photo-like, and text-heavy examples. Use the accepted-image crop sheet for
style inspection, not as proof that the whole directory is V3-training-ready
without filtering.

Exit criteria:

- The user signs off that the augmentation sheets match the intended V1 target.
- Contact sheets show no dark mode, grids, watermarks, text, or folded-model
  clutter.

### Phase 4: Eval Harness Before Training

- [ ] Add `--augment-profile` to the VertexRefiner trainer, not only eval.
- [ ] Add V3 channel-ablation eval flags:
      `--zero-ink`, `--zero-distance`, `--zero-orientation`.
- [ ] Add per-profile aggregate eval for the light-rendered suite.
- [ ] Add high-FP crop visualization for augmented eval.
- [ ] Add run-config validation for `input_version=v3` and augmentation profile.

Exit criteria:

- The current V2 checkpoint can be evaluated against the V3 light-profile suite
  as a baseline.
- Reports clearly separate clean, line-style, print-light, print-medium-lite,
  and faint-light performance.

### Phase 5: Local Command-Path Validation

- [ ] Build local V3 crop-ref caches for the intended training and validation
      slices.
- [ ] Validate cached refs fail fast on mismatched `augment_profile`,
      `input_version`, seed, image size, proposal cap, or boundary-anchor config.
- [ ] Run `--max-steps 0` locally.
- [ ] Run a one-step local CPU/MPS pass only if it avoids paid GPU setup risk.
- [ ] Verify no RunPod pod or volume is required for these checks.

Exit criteria:

- Training reaches run-config creation and dataloader construction locally.
- Any setup problem is found before renting GPU time.

### Phase 6: First V3 Training Run

- [ ] Train an initial V3 model from scratch or warm-start compatible layers from
      V2, excluding the removed skeleton input weights.
- [ ] Use affordable GPU only: 3090, 4090, L4, L40S, or similar.
- [ ] Avoid H100/H200/B200/A100-class GPUs.
- [ ] Keep no persistent network volume unless explicitly needed.
- [ ] Copy back best checkpoints and logs before deleting the pod.
- [ ] Delete all pod resources after the run.
- [ ] Continue training while validation improves; stop if loss diverges,
      validation degrades persistently, or light-profile precision does not
      recover.

Exit criteria:

- Checkpoint, run_config, summary, logs, and eval reports exist locally.
- RunPod resources are cleaned up.

### Phase 7: Compare V3 Against V2

- [ ] Clean/native full-pattern eval at cap `256`.
- [ ] Light-rendered crop-level eval.
- [ ] Light-rendered full-pattern eval on a small slice if local runtime allows.
- [ ] Channel-ablation eval.
- [ ] Failure visual sheets for high-FP style crops and remaining native misses.

Exit criteria:

- V3 matches or beats V2 on native clean full-pattern metrics.
- V3 materially improves line-style/print/faint light-mode robustness.
- No evidence shows V3 depends on fragile derived channels as perfect truth.

### Phase 8: Product Export And Integration

- [ ] Export V3 ONNX.
- [ ] Add model pointer metadata.
- [ ] Document V3 runtime channels for `tree-maker-rust`.
- [ ] Integrate V3 preprocessing/channel construction in browser/runtime code.
- [ ] Run product-side graph/topology benchmarks.

Exit criteria:

- Product can run V3 from a rectified light-mode CP image without a FOLD file,
  skeleton input, or CPLineNet dense outputs.
- Product-side benchmarks show better junction and graph topology robustness on
  rendered CP styles than V2.

## Risks And Open Questions

- Removing skeleton should improve robustness, but may reduce recall on some
  very faint or broken line styles. The ablation suite must catch this.
- `distance_to_ink` and orientation remain derived from `source_ink_probability`;
  channel dropout and ablation are required to prove the model does not depend
  on them being perfect.
- Proposal generation still uses source-derived heuristics. V3 removes skeleton
  from the model input, not necessarily every external proposal heuristic. If
  augmented eval shows proposal coverage failures, proposal generation needs its
  own light-style sweep.
- Accepted images do not provide GT labels. They are valuable for visual style
  matching, but synthetic FOLD renders remain the metric source until labeled
  real crops exist.
- If the light-rendered augmentation suite is too broad, the model may trade
  clean precision for unnecessary robustness. Keep the scope narrow and inspect
  contact sheets before training.
