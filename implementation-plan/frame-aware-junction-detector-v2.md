# Frame-Aware Junction Detector V2

## Goal

Build a production-ready junction detector that uses the rectified source image
and known square paper frame, not CPLineNet dense line or junction outputs, as
the default inference path.

The detector should produce all vertex candidates needed by downstream graph
construction:

- subpixel vertex position
- vertex kind: `interior_junction`, `boundary_contact`, `corner`,
  `endpoint_or_dangling`
- degree class
- incident ray bins
- confidence
- boundary side for boundary contacts

The immediate product target is to close the current boundary-contact recall
gap without regressing the already-strong interior/corner detection.

## Current Evidence

The native CPOOGLE eval harness previously mislabeled native boundary vertices
as interior vertices because canonicalized graph assignments let `U` edges win
over `B` edges. After fixing canonical assignment priority to `M/V > B > U`,
the same native 15-pattern slice reports:

- `1910` interior junctions
- `473` boundary contacts
- `60` corners

Corrected full-pattern metrics on native CPOOGLE clean 15:

| Slice | Recall |
| --- | ---: |
| Overall | `0.9574` |
| Interior junction | `0.9890` |
| Corner | `0.9833` |
| Boundary contact | `0.8266` |

Proposal coverage is `100%`, so the dominant failure is not crop selection. The
remaining false negatives are mostly covered-but-missed boundary contacts:

- `82` boundary-contact FNs
- `21` interior-junction FNs
- `1` corner FN

Only `14 / 82` boundary-contact misses have another GT vertex within `5px`, so
the native product bottleneck is not primarily the synthetic near-coincident
boundary-pair issue.

## Inference Contract

At product inference time there is no FOLD file. That is okay.

Required runtime inputs come from the product image pipeline:

1. User-uploaded crease-pattern image.
2. Rectified/cropped square paper frame.
3. Source-derived line/ink evidence computed from pixels.

FOLD graphs are only needed for training and evaluation labels.

Runtime flow:

```text
uploaded image
  -> rectify/crop to square paper frame
  -> compute source-derived image channels
  -> generate source/frame proposals
  -> run VertexRefinerV2 on 96x96 crops
  -> frame-aware decode, snap, and merge
  -> junction topology candidates for graph construction
```

The critical assumption is that rectification has already identified the paper
frame accurately. If the frame is wrong, boundary-contact channels will be wrong.

## Architecture

Implement `VertexRefinerV2` as a source-only, frame-aware crop model. It should
not depend on CPLineNet dense outputs by default, so it can run in parallel with
or instead of full-image dense inference.

Crop size:

```text
96 x 96
```

Input channels:

```text
1. grayscale source image
2. source ink probability
3. distance to ink
4. skeleton mask
5. local line orientation cos(2 theta)
6. local line orientation sin(2 theta)
7. signed distance to square frame, clipped and normalized
8. frame-edge mask
9. inside-paper mask
10. boundary-contact prior: source ink near frame
11. normalized local x coordinate
12. normalized local y coordinate
```

The frame channels are the key V2 change. Current boundary crops are centered
correctly, but the model has to infer frame semantics from blank pixels outside
the paper. V2 should explicitly tell the model where the paper frame is.

Backbone:

- keep the V1 high-resolution U-Net shape initially
- increase input channel count to 12
- only change capacity after source/frame channels and losses are validated

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
5. Merge boundary contacts by `(side, side_coordinate)` rather than plain 2D
   distance.
6. Merge interior vertices by Euclidean radius.

## Training Strategy

Train source-only. Do not feed CPLineNet line probability or CPLineNet junction
probability in the default V2 experiment.

Crop mix:

- 50% product-style source/frame proposals
- 50% GT-centered or GT-jittered anchors
- oversample boundary contacts by `3x` to `5x`
- include hard negatives near the paper frame
- keep close-pair and dense-grid examples, but do not let them dominate the
  native boundary-contact objective

Losses:

- focal/BCE for global vertex heatmap
- weighted focal/BCE for boundary-contact heatmap
- SmoothL1 for offset
- cross entropy for vertex kind
- cross entropy for degree
- BCE for incident ray bins
- cross entropy for boundary side, only on boundary-contact targets

Boundary contacts should receive higher heatmap/kind weighting until native
boundary recall is no longer the dominant error class.

## Evaluation Bar

Primary gate: corrected native CPOOGLE clean eval.

Minimum product-integration bar:

| Metric | Target |
| --- | ---: |
| Overall precision | `>= 0.98` |
| Overall recall | `>= 0.98` |
| Interior recall | `>= 0.99` |
| Corner recall | `>= 0.99` |
| Boundary-contact recall | `>= 0.95` |
| Proposal coverage | `>= 0.999` |

Every eval report must include:

- per-kind recall
- false negatives split into no-crop vs covered-but-missed
- false positives by predicted kind
- false positives by nearest GT kind
- boundary-contact misses by side
- tolerance sweep at `1.0`, `1.5`, `2.0`, `2.5`, `3.0`, `4.0`, `5.0` px

Secondary gates:

- synthetic clean 15, for continuity with historical product benchmarks
- synthetic seed 61, as a stress test for near-coincident boundary pairs
- box-pleat/native larger slices, for dense-grid behavior
- visual failure sheets for the top boundary-contact FN records

## Implementation Phases

### Phase 0: Preserve Correct Native Boundary Accounting

- [x] Keep canonical graph assignment priority at `M/V > B > U`.
- [x] Keep regression coverage for native-like `B` frame edges overlapped by
      `U` line segments.
- [x] Add a native-eval sanity check that fails if CPOOGLE boundary labels
      collapse to all interior vertices.

Exit criteria:

- Native CPOOGLE eval reports nonzero `boundary_contact` and `corner` counts.
- Boundary/corner counts match direct FOLD graph inspection within expected
  canonicalization differences.

### Phase 1: Add Source/Frame Input Channels

- [x] Add skeleton mask generation from source ink.
- [x] Add local orientation channels from source pixels or skeleton/Hough
      evidence.
- [x] Add signed distance to square frame.
- [x] Add frame-edge mask.
- [x] Add inside-paper mask.
- [x] Add boundary-contact prior from ink near the known frame.
- [x] Add unit tests for channel shapes, value ranges, crop padding, and frame
      alignment.

Exit criteria:

- A boundary contact centered on the top edge appears at crop center with valid
  frame channels and padded outside-paper pixels.
- No FOLD-specific data is required to build inference inputs.

### Phase 2: Implement `VertexRefinerV2`

- [x] Add a V2 model contract with 12 input channels.
- [x] Reuse V1 U-Net capacity for the first experiment.
- [x] Add `boundary_contact_heatmap` head.
- [x] Add `boundary_side` auxiliary head.
- [x] Extend loss code for new heads and boundary weighting.
- [x] Add model shape and loss tests.

Exit criteria:

- Forward pass, loss, and decode tests pass on synthetic crops.
- V1 and V2 contracts can coexist without breaking existing checkpoints.

### Phase 3: Frame-Aware Decoder And Merge

- [x] Union global vertex peaks and boundary-contact peaks.
- [x] Snap boundary predictions to the known square frame.
- [x] Merge boundary contacts by `(side, side_coordinate)`.
- [x] Preserve interior Euclidean merge behavior.
- [x] Report predicted kind, degree, rays, side, support count, and support
      fraction.
- [x] Add decode tests for top/right/bottom/left contacts and near-corner cases.

Exit criteria:

- Boundary contacts no longer drift off-frame after decode.
- Close boundary contacts on different sides or near corners do not incorrectly
  merge by plain Euclidean distance.

### Phase 4: Training Run

- [x] Build a V2 training config using source-only inputs.
- [x] Oversample boundary contacts with training-only jittered GT anchors.
- [ ] Add explicit frame-near hard negatives if precision becomes the dominant
      post-training failure mode.
- [x] Keep RunPod GPU choice cost-conscious: 3090, 4090, L4, or similar; avoid
      H100/H200/B200/A100-class GPUs for this probe.
- [x] Save checkpoint metadata and exact command/config.
- [x] Skip the paid smoke run by default; validate the V2 command path locally
      with `--max-steps 0` before launching paid training.
- [x] Add full-run early-termination guardrails for non-finite/exploding loss
      and optional intermediate validation F1 checks.
- [x] Attempt one affordable RunPod training run and terminate early on setup
      issues without leaving resources running.
- [x] Add progress logging around dataset/crop-ref construction so future GPU
      logs show which phase is active before `run_config.json` exists.
- [x] Add a local precompute artifact for selected crop refs:
      record id, selected-record index, proposal center, score, provenance,
      crop origin, and training-only boundary jitter settings.
- [x] Add a dataset loader path that can consume precomputed crop refs and skip
      `_build_crop_refs` on the GPU pod.
- [x] Add run-config verification so a training run fails fast if cached refs
      were built for a mismatched split, seed, image size, proposal count,
      auxiliary mode, input version, or boundary-oversampling config.
- [x] Build local cached refs for the intended V2 1024px source-only run and
      upload the small cache artifact with the dataset before retrying RunPod.
- [x] Add bounded rendered-sample cache and record-ordered training option so a
      cached 1024px run does not grow toward all 512 rendered samples in RAM.
- [x] Run one full affordable training run after the CPU-side setup bottleneck
      is fixed, terminating early if the guardrails indicate it is going badly.

Exit criteria:

- Training reaches `run_config.json` and the first logged loss promptly on the
  GPU pod, instead of spending paid time in opaque CPU-only setup.
- Training completes within the intended budget, or stops early with a clear
  `early_stop_reason` in `summary.json` and checkpoint metadata.
- Checkpoint can be loaded locally for deterministic eval.

2026-06-24 local cache implementation:

- Added `scripts/data/precompute_vertex_refiner_crop_refs.py`.
- Added strict JSON crop-ref cache loading to `VertexRefinerCropDataset`.
- Added `--train-crop-refs`, `--val-crop-refs`, `--crop-ref-progress-every`,
  `--rendered-sample-cache-size`, and `--no-shuffle-train-crops` to the trainer.
- Built intended 1024px source-only V2 cache artifacts:
  - `checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json`
    has 512 records, 95,728 refs, and is about 33 MB.
  - `checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json`
    has 64 records, 7,096 refs, and is about 2.3 MB.
- The cache contract is `input_version=v2` and `auxiliary_mode=zero`.
  `auxiliary_mode=zero` means no CPLineNet dense channels; V2 source/frame
  channels are selected by `input_version=v2` and `model_version=v2`.
- Standalone eval with cached validation refs must use the validation seed,
  `VAL_SEED=SEED+1000`. For the current `SEED=41` run, the validation cache was
  built with seed `1041`; running eval with seed `41` correctly fails cache
  verification and is a harness/config error, not a model failure.
- Local full-size cached command-path validation passed with `--max-steps 0`.
- Local full-size cached one-step validation passed with
  `--no-shuffle-train-crops --rendered-sample-cache-size 4`; first-step loss was
  finite (`3691.9351`) and below the updated explosion threshold.
- Raised `ABORT_LOSS_THRESHOLD` default to `100000`, because a cold-start V2
  loss around 1,000-6,000 is normal and should not be treated as exploding.

2026-06-24 RunPod attempt:

- Pod `v18e53zyit0hrv` used an RTX PRO 4000 Blackwell in secure cloud at about
  `$0.584/hr`, with no network volume. It was stopped and deleted after abort
  artifact copy-back; `runpodctl pod list` and `runpodctl network-volume list`
  were empty afterward, and `currentSpendPerHr=0`.
- The old CUDA 12.1 PyTorch pin from the 4090 recipe imports but cannot execute
  kernels on Blackwell `sm_120` (`no kernel image is available for execution on
  the device`). `torch==2.8.0+cu128` and `torchvision==0.23.0+cu128` passed an
  actual CUDA matrix multiply after repinning `numpy<2` and OpenCV.
- Three launches were stopped before training because they stayed CPU-only and
  did not write `run_config.json`: `512/64` for more than five minutes,
  `128/16` for more than five minutes, and `32/8` for about four minutes. This
  happens before the first GPU step, during eager render/proposal/crop setup.
- Local abort notes are under:
  `checkpoints/runpod_vertex_refiner_v2_source_only_full_20260624_blackwell/`,
  `checkpoints/runpod_vertex_refiner_v2_source_only_bounded_20260624_blackwell/`,
  and `checkpoints/runpod_vertex_refiner_v2_source_only_signal_20260624_blackwell/`.

2026-06-24 cached RTX 4090 training result:

- Pod `2mfc9y7nxeoca1` used a GeForce RTX 4090 at about `$0.69/hr`, with
  `volumeInGb=0` and no network volume. It was stopped and deleted after
  artifact copy-back; `runpodctl pod list` and
  `runpodctl network-volume list` returned empty, and
  `currentSpendPerHr=0`.
- The first cached full run reached the training loop promptly and completed
  `2000` steps from scratch at `LR=0.0003`. Validation F1 improved from
  `0.9297` at the end of the first run to `0.9768` after warm continuation.
- The warm continuation from the first run used `LR=0.0001` for another
  `2000` steps. The final sampled validation point was the best point:
  precision `0.9683`, recall `0.9854`, F1 `0.9768`, boundary-contact F1
  `0.9565`, close-pair F1 `0.9323`.
- Local artifacts copied back before pod deletion:
  `checkpoints/runpod_vertex_refiner_v2_cached_20260624_4090/` and
  `checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/`.
  The current best checkpoint is
  `checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/full/latest.pt`;
  `checkpoint_step2000.pt`, `summary.json`, `run_config.json`, logs, and
  intermediate named checkpoints were copied beside it.
- The standalone full-pattern diagnostics eval with the corrected
  `VAL_SEED=1041` was CPU-bound for about 12 minutes after training, wrote no
  `eval.json`, and was stopped to avoid wasting paid GPU pod time. The launcher
  now supports `RUN_STANDALONE_EVAL=0` and
  `EVAL_FULL_PATTERN_DIAGNOSTICS=0` so continuation runs can skip that tail.
- A lower-LR `LR=0.00003` continuation was started because the last sampled
  point was still improving, but it was stopped almost immediately when the user
  asked to clean up the pod. It only wrote `run_config.json` and an empty
  `train.log`.
- Local MPS eval from the copied best checkpoint over the full cached validation
  crop set (`7,096` crops) completed after cleanup:
  precision `0.9692`, recall `0.9651`, F1 `0.9671`,
  boundary-contact F1 `0.9495`, close-pair F1 `0.9029`, corner F1 `0.9498`.
  This is crop-level cached validation, not the native full-pattern product
  gate.

### Phase 5: Corrected Native Eval And Failure Review

- [ ] Run corrected native CPOOGLE clean 15.
- [ ] Run a larger CPOOGLE slice if local runtime is acceptable, or run it on
      the same affordable GPU pod.
- [ ] Generate per-kind metrics and tolerance sweep.
- [ ] Generate visual failure sheets for boundary-contact FNs and FPs.
- [ ] Compare against V1 corrected baseline:
      - overall recall `0.9574`
      - boundary-contact recall `0.8266`
      - interior recall `0.9890`

Exit criteria:

- Boundary-contact recall improves materially without degrading interior/corner
  recall.
- Remaining largest error class is identified with visuals.

### Phase 6: Iterate To Product Bar

- [ ] If boundary recall remains weak, tune boundary loss, boundary prior, side
      head, and frame-channel normalization.
- [ ] If precision becomes weak, tune boundary peak threshold, support fraction,
      and hard-negative mining.
- [ ] If localization dominates, tune offset loss and boundary snap/merge.
- [ ] If proposal coverage regresses, add frame-grid or sliding-window fallback
      only for affected cases.

Exit criteria:

- Native CPOOGLE clean eval reaches the product bar or the remaining blocker is
  documented with visual evidence.

### Phase 7: Product Export And Integration

- [ ] Export V2 ONNX.
- [ ] Add pointer metadata for the current V2 checkpoint.
- [ ] Document required runtime channels for `tree-maker-rust`.
- [ ] Integrate browser/runtime preprocessing for source/frame channels.
- [ ] Integrate decoded vertices into downstream graph construction.
- [ ] Run product-side graph/topology benchmarks.

Exit criteria:

- Product can run junction detection from a rectified source image without a
  FOLD file or CPLineNet dense outputs.
- Product-side benchmark reports improved junction and graph topology metrics.

## Risks And Open Questions

- Frame rectification accuracy becomes more important because V2 explicitly uses
  frame channels.
- Boundary contacts near corners need careful side assignment and merge logic.
- Source skeleton/orientation channels may be brittle on noisy scans; keep them
  as helpful evidence, not sole truth.
- A larger native CPOOGLE eval may need to run on GPU because local full-pattern
  decoding is slow.
- If source-only V2 cannot reach the bar, run a controlled source-plus-dense
  ablation, but keep source-only as the default product architecture until
  evidence says otherwise.
