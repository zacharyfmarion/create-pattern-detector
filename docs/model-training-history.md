# Model Training History

This is the source-of-truth orientation doc for CPLineNet checkpoints and ONNX
exports. Read this before using, replacing, exporting, or retraining a model.

## Current Model

The current downstream/browser model is the V3 close-pair R1 checkpoint:

```text
artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json
checkpoints/r1_close_pair_warmstart/latest.pt
```

It is the banked close-pair model used for the `tree-maker-rust` / Ori Studio
browser path. In that downstream repo, the product default is:

```text
apps/web/public/models/cp-detector-v3/manifest.json
apps/web/public/models/cp-detector-v3/model.onnx
```

That ONNX export is byte-for-byte the same as the local
`cp-detector-v3-close-pair` asset in `tree-maker-rust`. It is not the R3
from-scratch export.

Important settings for R1:

- PyTorch checkpoint SHA-256:
  `f4223ac93f90034663cd2c738480f2c1e33fea8d0f274e708deb0ce41e0c2ed6`
- ONNX SHA-256 in `tree-maker-rust`:
  `39705fb5b17b1e58255f0721d7185a11348187f8ac2a56adaf8226949eafb4ee`
- Image size: `1024`
- Backbone: `hrnet_w18`
- V2 auxiliary heads: enabled
- BatchNorm: use batch-stats behavior; the browser export uses explicit
  per-image BatchNorm ops.
- Junction offset radius: `3.0` px. Decoders must use offset-vote clustering
  rather than legacy sub-pixel-only offset decoding.

## Do Not Confuse R1 And R3

R3 is the later chronological training run, but it is not the promoted model.
It was a clean from-scratch ablation that tested whether warm-starting was the
close-pair ceiling. The result refuted that hypothesis: R3 landed statistically
identical to R1 despite 3.3x the training time.

R3 artifacts:

```text
artifacts/checkpoints/runpod-v3-close-pair-scratch-r3-4090.json
checkpoints/r3_close_pair_scratch/latest.pt
```

R3 ONNX exists in `tree-maker-rust` for comparison:

```text
apps/web/public/models/cp-detector-v3-r3-scratch/manifest.json
apps/web/public/models/cp-detector-v3-r3-scratch/model.onnx
```

R3 should not be used as the default unless a future eval explicitly promotes
it.

## Close-Pair Verdict

| Config | Exact | Strict eF1 | Miss | Extra | Merged | Pair res. proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped V2 model, PR #55 state | 4/15 | 0.942 | 176 | 130 | 74 | 5.4% |
| R1 warm-start + clusters | 4/15 | 0.953 | 145 | 101 | 65-67 | 26.5% |
| R3 from-scratch + clusters | 4/15 | 0.952 | 144 | 109 | 71 | 21.5% |

Bank R1 and stop tuning this exact recipe. The remaining close-pair gap appears
to be a representation/input-resolution limit at 1024px with about 2px strokes:
many 5px-apart junction pairs are nearly one ink blob. Future attempts should be
new projects, such as higher-resolution dense heads, tiled crops, a dedicated
pair head, or pipeline-side fused-vertex splitting from incident-line geometry.

## Box-Pleat No-Guide-Grid Diagnostic

The 2026-06-16 no-guide-grid warm-start experiment tested whether BP
horizontal/vertical crease families were being learned as non-crease guide-grid
artifacts. Dense-head eval on the full 179-sample
`box-pleat-native-v1` set strongly supports that hypothesis for the
non-crease head:

| Model | Orthogonal raw recall | Orthogonal effective recall | Orthogonal non-crease conflict | Diagonal raw recall |
| --- | ---: | ---: | ---: | ---: |
| Shipped V3 R1 | 0.5130 | 0.4744 | 0.5422 | 0.8752 |
| No-grid probe, 800 steps | 0.5528 | 0.5525 | 0.0049 | 0.8756 |
| No-grid full, 5000 steps | 0.6012 | 0.6001 | 0.0154 | 0.8810 |

Interpretation:

- The non-crease suppression failure is essentially fixed: effective recall now
  matches raw recall, and non-crease conflict fell from 54.2% to 1.5%.
- Orthogonal BP creases are still not equal to diagonal/other creases
  (`0.6012` raw recall vs `0.8810`). Since non-crease conflict is low and the
  warm-start continued improving from the 800-step probe, the remaining gap is
  more likely a data-distribution/BP-coverage problem than an inability to
  unlearn the old non-crease behavior. The current synthetic mix is mostly
  TreeMaker/Rabbit-Ear/22.5-style data and does not explicitly generate
  box-pleat crease patterns.
- The 5000-step no-grid checkpoint is not promotion-ready because the launcher
  forgot to forward the close-pair offset recipe. It trained and exported with
  `junction_offset_radius_px=0.0` instead of the R1-required `3.0`, so its dense
  BP numbers are valid but its junction-offset semantics are incompatible with
  the current radius-3 product decoder.

Future no-guide-grid runs that should remain close-pair-compatible must use the
canonical launcher:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
```

That script sets and verifies the R1 close-pair parameters
(`junction_sigma_px=1.5`, `junction_offset_radius_px=3.0`,
`junction_offset_weight=0.5`, `junction_focal_alpha=2.0`,
`junction_focal_beta=4.0`) before launching the full run. The retired
`run_cpline_runpod_v3_no_guide_grid_{probe,full}.sh` names intentionally fail
with an "Are you sure?" message unless explicitly acknowledged.

## Timeline

| Date | Run | Checkpoint | Init | Outcome |
| --- | --- | --- | --- | --- |
| 2026-05-19 | Phase 3 V1 stage-balanced | `artifacts/checkpoints/phase3-v1-cpline.json` | Stage-base curriculum | Blessed V1 Python CLI baseline for readable 1024px inputs. No V2 artifact heads. |
| 2026-05-22 | V2 replay correction full | `artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json` | V2 continuation/replay | Best balanced V2 candidate before close-pair work; added V2 heads and artifact robustness, but not the current browser model. |
| 2026-06-10 | V3 close-pair R1 warm-start | `artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json` | V2 replay checkpoint, reinitialized `offset_head` | Current banked/product-exported model. Improved strict topology and close-pair proxy with radius-3 offset clustering. |
| 2026-06-10 | V3 close-pair R3 scratch | `artifacts/checkpoints/runpod-v3-close-pair-scratch-r3-4090.json` | None | From-scratch ablation. Similar metrics to R1, so not promoted. |
| 2026-06-16 | V3 no-guide-grid full R1 diagnostic | Not registered; local ignored checkpoint `checkpoints/runpod_v3_no_guide_grid_full_r1_20260615/full/latest.pt` | R1 close-pair checkpoint, reinitialized `non_crease_head` | Dense BP evidence improved substantially, but the launcher omitted radius-3 close-pair offset args and produced `junction_offset_radius_px=0.0`. Treat as a dense-head diagnostic only, not a promotable model. |

## Update Rules

When a future model becomes the default:

1. Add or update a checkpoint manifest under `artifacts/checkpoints/`.
2. Update this file's `Current Model`, comparison table, and timeline.
3. Update `docs/checkpoint-management.md` if the default role changes.
4. If exported to `tree-maker-rust`, record the ONNX manifest path, SHA-256, and
   required decoder settings here.
5. Link any new benchmark reports or eval artifacts from the relevant phase doc.

Do not infer the latest model from file modification time, chronological training
order, or the presence of an ONNX directory alone. The promoted model is the one
documented in this file and backed by a checkpoint manifest.
