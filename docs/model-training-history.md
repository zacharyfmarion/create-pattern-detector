# Model Training History

This is the source-of-truth orientation doc for CPLineNet checkpoints and ONNX
exports. Read this before using, replacing, exporting, or retraining a model.

## Current Model

The current downstream/browser model is the V3 no-guide-grid close-pair dense
edges `MAX_EDGES=1200` checkpoint:

```text
artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max1200-l40s.json
checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max1200_probe_20260618/full/latest.pt
```

It warm-started from the previous dense-edge max700 checkpoint, kept all heads,
increased the training edge envelope from `max_edges=700` to `max_edges=1200`,
and preserved the close-pair radius-3 junction-offset recipe.
In `tree-maker-rust` / Ori Studio, the stable product default path is:

```text
apps/web/public/models/cp-detector-v3/manifest.json
apps/web/public/models/cp-detector-v3/model.onnx
```

The versioned export used to promote that default is:

```text
apps/web/public/models/cp-detector-v3-dense-edges-max1200-20260618/manifest.json
apps/web/public/models/cp-detector-v3-dense-edges-max1200-20260618/model.onnx
```

Important settings:

- PyTorch checkpoint SHA-256:
  `befa99edc4531919ffb8f36c933b2e394f8868cb6e0fb0be2d7b96eee74c9bac`
- ONNX SHA-256 in `tree-maker-rust`:
  `96ba3d56277f0ead32a6be813a31402434f29620f4b6edd113d3592e2c3ab145`
- Image size: `1024`
- Backbone: `hrnet_w18`
- V2 auxiliary heads: enabled
- BatchNorm: use batch-stats behavior; the browser export uses explicit
  per-image BatchNorm ops.
- Junction offset radius: `3.0` px. Decoders must use offset-vote clustering
  rather than legacy sub-pixel-only offset decoding.
- Promoted continuation envelope: `max_edges=1200`, `1500` steps, no head
  reinitialization from the prior max700 dense-edge checkpoint.

## Do Not Confuse Older Runs

The older close-pair R1 checkpoint was the previous promoted browser model. It
is still registered for comparison, but it is no longer the default:

```text
artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json
checkpoints/r1_close_pair_warmstart/latest.pt
```

R3 is a later chronological close-pair training run, but it is not the promoted
model. It was a clean from-scratch ablation that tested whether warm-starting was
the close-pair ceiling. The result refuted that hypothesis: R3 landed
statistically identical to the older R1 despite 3.3x the training time.

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

The no-guide-grid close-pair R1 was promoted after it fixed the original
guide-grid/non-crease suppression failure, but it has now been superseded by
dense-edge continuations:

```text
artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-full-r1-4090.json
checkpoints/runpod_v3_no_guide_grid_close_pair_full_r1_20260617/full/latest.pt
```

The `MAX_EDGES=700` dense-edge continuation was the previous promoted
browser/product model and is retained for comparison:

```text
artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max700-4090.json
checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max700_probe_20260618/full/latest.pt
```

The first 5000-step no-guide-grid full run was also not promoted. It fixed the
BP dense-head conflict, but the retired launcher omitted the close-pair offset
recipe and exported with `junction_offset_radius_px=0.0`. Treat it as a dense
diagnostic only:

```text
checkpoints/runpod_v3_no_guide_grid_full_r1_20260615/full/latest.pt
```

## Close-Pair Verdict

| Config | Exact | Strict eF1 | Miss | Extra | Merged | Pair res. proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped V2 model, PR #55 state | 4/15 | 0.942 | 176 | 130 | 74 | 5.4% |
| R1 warm-start + clusters | 4/15 | 0.953 | 145 | 101 | 65-67 | 26.5% |
| R3 from-scratch + clusters | 4/15 | 0.952 | 144 | 109 | 71 | 21.5% |

The older close-pair R1 was banked as the close-pair recipe endpoint. The
remaining close-pair gap appears to be a representation/input-resolution limit
at 1024px with about 2px strokes: many 5px-apart junction pairs are nearly one
ink blob. Future attempts should be new projects, such as higher-resolution
dense heads, tiled crops, a dedicated pair head, or pipeline-side fused-vertex
splitting from incident-line geometry.

The no-guide-grid close-pair model keeps the same radius-3 close-pair decoder
contract. Its promotion is for BP orthogonal crease recovery and clean-15
strict-topology improvement, not because it changes the close-pair ceiling.

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
| No-grid full diagnostic, 5000 steps, radius 0 | 0.6012 | 0.6001 | 0.0154 | 0.8810 |
| No-grid close-pair full R1, superseded | 0.6113 | 0.6104 | 0.0128 | 0.8816 |
| Dense-edge max700 continuation, superseded | 0.6482 | 0.6462 | 0.0197 | 0.8983 |
| Dense-edge max1200 continuation, promoted | 0.6778 | 0.6746 | 0.0285 | 0.9046 |

Interpretation:

- The non-crease suppression failure is essentially fixed: effective recall now
  tracks raw recall, and non-crease conflict fell from 54.2% in the shipped V3
  R1 to 2.9% in the current promoted max1200 model.
- Orthogonal BP creases are still not equal to diagonal/other creases
  (`0.6482` raw recall vs `0.8983`). Since non-crease conflict is low and the
  max700 continuation improved both BP and clean-15 metrics without adding new
  BP families, the remaining gap is still most likely a data-distribution and
  dense-vertical-coverage problem. The current synthetic mix is mostly
  TreeMaker/Rabbit-Ear/22.5-style data and does not explicitly generate
  box-pleat crease patterns.
- Clean-15 product strict topology improved on the current product pack:
  previous no-guide-grid close-pair R1 `0.9594` edge F1 / `126` missing / `89`
  extra / `72` merged; promoted max1200 dense-edge continuation `0.9655` edge
  F1 / `107` missing / `76` extra / `60` merged.
- The older 5000-step no-grid diagnostic remains non-promotable because it
  trained and exported with `junction_offset_radius_px=0.0`; the promoted run
  above used the verified close-pair launcher and has radius `3.0`.

The dense-edge continuations also improved the hardest BP angle buckets:

| Slice | Horizontal effective recall | Vertical effective recall | 45/135 effective recall |
| --- | ---: | ---: | ---: |
| All 179, previous no-guide-grid R1 | 0.6546 | 0.5736 | 0.8634 |
| All 179, max700 | 0.6874 | 0.6124 | 0.8812 |
| All 179, current max1200 | 0.7300 | 0.6272 | 0.8866 |
| Dense top quartile, previous no-guide-grid R1 | 0.4354 | 0.3757 | 0.8082 |
| Dense top quartile, max700 | 0.4763 | 0.4074 | 0.8403 |
| Dense top quartile, current max1200 | 0.5153 | 0.4247 | 0.8501 |
| Very dense `edge_count >= 2000`, previous no-guide-grid R1 | 0.4062 | 0.3473 | 0.8059 |
| Very dense `edge_count >= 2000`, max700 | 0.4446 | 0.3797 | 0.8365 |
| Very dense `edge_count >= 2000`, current max1200 | 0.4808 | 0.3968 | 0.8438 |

Future no-guide-grid runs that should remain close-pair-compatible must use a
launcher that verifies the close-pair recipe. The canonical full no-guide-grid
launcher is:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
```

That script sets and verifies the R1 close-pair parameters
(`junction_sigma_px=1.5`, `junction_offset_radius_px=3.0`,
`junction_offset_weight=0.5`, `junction_focal_alpha=2.0`,
`junction_focal_beta=4.0`) before launching the full run. The retired
`run_cpline_runpod_v3_no_guide_grid_{probe,full}.sh` names intentionally fail
with an "Are you sure?" message unless explicitly acknowledged.

For controlled dense-edge follow-up probes from the current promoted model, use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh
```

## Timeline

| Date | Run | Checkpoint | Init | Outcome |
| --- | --- | --- | --- | --- |
| 2026-05-19 | Phase 3 V1 stage-balanced | `artifacts/checkpoints/phase3-v1-cpline.json` | Stage-base curriculum | Blessed V1 Python CLI baseline for readable 1024px inputs. No V2 artifact heads. |
| 2026-05-22 | V2 replay correction full | `artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json` | V2 continuation/replay | Best balanced V2 candidate before close-pair work; added V2 heads and artifact robustness, but not the current browser model. |
| 2026-06-10 | V3 close-pair R1 warm-start | `artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json` | V2 replay checkpoint, reinitialized `offset_head` | Previous banked/product-exported model. Improved strict topology and close-pair proxy with radius-3 offset clustering. Superseded by the no-guide-grid and later dense-edge models. |
| 2026-06-10 | V3 close-pair R3 scratch | `artifacts/checkpoints/runpod-v3-close-pair-scratch-r3-4090.json` | None | From-scratch ablation. Similar metrics to R1, so not promoted. |
| 2026-06-16 | V3 no-guide-grid full R1 diagnostic | Not registered; local ignored checkpoint `checkpoints/runpod_v3_no_guide_grid_full_r1_20260615/full/latest.pt` | R1 close-pair checkpoint, reinitialized `non_crease_head` | Dense BP evidence improved substantially, but the launcher omitted radius-3 close-pair offset args and produced `junction_offset_radius_px=0.0`. Treat as a dense-head diagnostic only, not a promotable model. |
| 2026-06-17 | V3 no-guide-grid close-pair full R1 | `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-full-r1-4090.json` | R1 close-pair checkpoint, reinitialized `non_crease_head` | Superseded browser/product model. Keeps radius-3 close-pair decoder compatibility, fixes BP non-crease suppression, and improves current-pack clean-15 strict topology. |
| 2026-06-18 | V3 no-guide-grid close-pair dense-edge max700 | `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max700-4090.json` | No-guide-grid close-pair R1, no head reinit | Superseded browser/product model. Raises the training edge envelope to `max_edges=700`, improves dense BP horizontal/vertical recall, and improves clean-15 strict topology to `0.9623` strict edge F1. |
| 2026-06-18 | V3 no-guide-grid close-pair dense-edge max1200 | `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max1200-l40s.json` | Max700 checkpoint, no head reinit | Current promoted browser/product model. Raises the training edge envelope to `max_edges=1200`; improves BP orthogonal effective recall to `0.6746` and clean-15 strict edge F1 to `0.9655`, with modestly higher non-crease conflict. |

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
