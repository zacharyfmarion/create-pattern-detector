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

## Timeline

| Date | Run | Checkpoint | Init | Outcome |
| --- | --- | --- | --- | --- |
| 2026-05-19 | Phase 3 V1 stage-balanced | `artifacts/checkpoints/phase3-v1-cpline.json` | Stage-base curriculum | Blessed V1 Python CLI baseline for readable 1024px inputs. No V2 artifact heads. |
| 2026-05-22 | V2 replay correction full | `artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json` | V2 continuation/replay | Best balanced V2 candidate before close-pair work; added V2 heads and artifact robustness, but not the current browser model. |
| 2026-06-10 | V3 close-pair R1 warm-start | `artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json` | V2 replay checkpoint, reinitialized `offset_head` | Current banked/product-exported model. Improved strict topology and close-pair proxy with radius-3 offset clustering. |
| 2026-06-10 | V3 close-pair R3 scratch | `artifacts/checkpoints/runpod-v3-close-pair-scratch-r3-4090.json` | None | From-scratch ablation. Similar metrics to R1, so not promoted. |

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
