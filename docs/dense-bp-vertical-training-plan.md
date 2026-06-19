# Dense BP Vertical Training Plan

Status: In progress, 2026-06-19. The `MAX_EDGES=700` and `MAX_EDGES=1200`
probes were both promoted and then superseded. The corrected 15% tessellation
weighted continuation is now the promoted default.
Primary goal: improve dense vertical crease-line detection on real box-pleated
crease patterns without regressing the current clean topology benchmark.

This plan tracks dense BP work after the original V3 no-guide-grid close-pair
model. It deliberately separates two hypotheses:

1. The model is undertrained on dense examples already present in the synthetic
   mix when training filters to `max_edges <= 300`.
2. The current synthetic mix still lacks enough BP/tessellation-like vertical
   crease evidence, so new procedural tessellation data is needed.

Do not change both variables in the same first experiment.

## Current Evidence

The baseline before Experiment A was:

- Checkpoint manifest:
  `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-full-r1-4090.json`
- Training mix: `cp_training_mix_v1`
- Training edge filter: `maxEdges: 300`
- Init: prior close-pair R1 checkpoint, with `non_crease_head` reinitialized
- Required decoder setting: `junction_offset_radius_px=3.0`

The current promoted model after the tessellation probe is:

- Checkpoint manifest:
  `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-tess15-weighted-4090.json`
- Checkpoint:
  `checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_weighted_probe_20260619/full/latest.pt`
- Training edge filter: `maxEdges: 1200`
- Init: superseded max1200 dense-edge model, no head reinitialization
- Training sampler: `v3-tessellation-15pct`
- Required decoder setting: `junction_offset_radius_px=3.0`

The no-guide-grid training fixed the original non-crease suppression failure:
orthogonal BP non-crease conflict dropped from about `0.542` to `0.013`.
Remaining BP misses are mostly low `line_prob`, not non-crease suppression.

The angle-bucket diagnostic on the full BP dense eval showed the initial
no-guide-grid model still detected vertical lines worse than horizontal lines,
and both were much worse than diagonal lines:

| Slice | Horizontal effective recall | Vertical effective recall | 45/135 effective recall |
| --- | ---: | ---: | ---: |
| All 179 BP samples | `0.6546` | `0.5736` | `0.8634` |
| Dense top quartile | `0.4354` | `0.3757` | `0.8082` |
| Very dense `edge_count >= 2000` | `0.4062` | `0.3473` | `0.8059` |

The current synthetic mix already contains many examples above the original
`maxEdges: 300` training envelope:

| Split | Total | `<=300` edges | `>300` edges | `>1000` edges | Max edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | `11899` | `5202` | `6697` | `87` | `1857` |
| Val | `1401` | `638` | `763` | `8` | `1318` |
| All | `14000` | `6164` | `7836` | `101` | `1857` |

This makes the first controlled experiment obvious: include denser examples
that already exist before generating new data.

## Experiment A: Raise Training Edge Envelope

Question: does simply training on denser existing synthetic examples improve
dense BP vertical recall?

Keep fixed:

- Dataset: `cp_training_mix_v1`
- Augment profile: current no-guide-grid profile
- Close-pair recipe:
  - `junction_sigma_px=1.5`
  - `junction_offset_radius_px=3.0`
  - `junction_offset_weight=0.5`
  - `junction_focal_alpha=2.0`
  - `junction_focal_beta=4.0`
- Init checkpoint for the first probe: superseded no-guide-grid close-pair R1
- Init checkpoint for follow-up probes: current promoted tess15 weighted model
- Do not reinitialize heads
- Do not add tessellation data

Vary only:

- `MAX_EDGES`

Recommended probes:

1. `MAX_EDGES=700`, short continuation.
2. If runtime and metrics look sane, `MAX_EDGES=1200`.
3. Only try `MAX_EDGES=2000` after confirming label rendering and GPU memory
   are still acceptable.

The canonical launcher for dense-edge follow-up probes is:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh
```

It defaults to `MAX_EDGES=1200`, initializes from the current promoted tess15
weighted checkpoint, leaves `REINIT_HEADS` empty, and verifies the radius-3
close-pair configuration after preflight and after training. Set an explicit
fresh `OUTPUT_ROOT` for any follow-up probe.

Suggested cheap probe shape:

- Steps: `1000` to `1500`
- Batch size: `1`
- Learning rate: low continuation LR, starting from `0.00005`
- Skip expensive graph eval during training
- Evaluate after export, not by training loss alone

Success gates:

- BP dense vertical effective recall improves meaningfully.
- Horizontal recall does not regress materially.
- 45/135 recall does not regress materially.
- Clean-15 strict topology remains at least comparable to the current promoted
  model.
- Non-crease conflict remains low.

Stop conditions:

- Label rendering/data loading is too slow for budget.
- GPU memory becomes unstable.
- Clean-15 strict topology regresses enough that the candidate cannot supersede
  the current model.
- Vertical recall does not move, suggesting edge envelope alone is not the main
  issue.

## Experiment B: Procedural Tessellation Data

Question: if higher `MAX_EDGES` alone is not enough, does adding synthetic
BP/tessellation-like data improve dense vertical recall?

Status: implementation in progress after Experiment A showed useful but
incomplete gains. Keep this as the next isolated variable: compare a
tessellation-mix continuation against the promoted `MAX_EDGES=1200` checkpoint.

Recommended generator family:

```text
tessellation-fold-program
```

Initial subfamilies:

- `orthogonal-bp-grid`
  - Directly targets dense real horizontal and vertical crease lines.
  - Oversamples vertical crease length in vertical-heavy examples.
- `miura-ori`
  - Known periodic tessellation/corrugation family.
  - Useful for repeated crease paths, scale variation, and non-BP controls.
  - Not implemented in `tessellation_fold_program_v1`.

Potential later subfamilies:

- `waterbomb-grid`
- `diamond-corrugation`
- `square-twist-grid`

Generator requirements:

- Emit normalized `.fold` files through the existing synthetic-generator path.
- Add metadata for:
  - `tessellation_family`
  - repeat counts
  - edge count
  - active crease count
  - horizontal/vertical/diagonal angle histogram
  - vertical crease-length fraction
  - minimum rendered crease spacing at 1024px
- Keep generated files deterministic from seed + recipe.
- Keep raw generated releases outside git.

Validation requirements:

- Schema and normalization pass.
- No zero-length edges.
- No crossing edges away from shared vertices.
- Reasonable local flat-foldability checks where applicable.
- Rendered contact sheets for visual review.
- Label smoke at 1024px.
- Angle histogram report confirming vertical-heavy coverage.

Training design:

- Add tessellation data to a new synthetic mix release, not by mutating
  `cp_training_mix_v1`.
- Start with a conservative mix weight, around 5-15%.
- Reuse the best `MAX_EDGES` envelope found in Experiment A.
- Compare against the Experiment A model, not only against the current promoted
  baseline.
- For the current 15% tessellation mix,
  `cp_training_mix_v3_tessellation_15pct`, use
  `TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct`, not `natural` or plain
  `balanced`. This preserves the old dense-edge base exposure while adding the
  requested tessellation slice: 42.5% TreeMaker, 42.5% Rabbit Ear, 12%
  orthogonal BP grid, and 3% Miura.

## Evaluation Matrix

Every candidate intended for promotion needs these evals:

- Full BP dense-head eval in `tree-maker-rust`.
- BP angle-bucket report with horizontal, vertical, 45, and 135 degree slices.
- Clean-15 strict topology eval in `tree-maker-rust`.
- At least a small visual before/after activation check on dense BP examples.

Promotion requires both:

- Improved BP dense vertical recall.
- No unacceptable regression on clean-15 strict topology.

Dense BP strict topology is useful for tracking direction, but it should not be
the only promotion gate because exactization/post-processing can be dominated by
very dense or degenerate topology.

## Launch Safety

Use the verified no-guide-grid close-pair launcher path or a new launcher that
copies its explicit checks. Any dense-edge experiment must verify these values
before training starts:

```text
junction_sigma_px=1.5
junction_offset_radius_px=3.0
junction_offset_weight=0.5
junction_focal_alpha=2.0
junction_focal_beta=4.0
```

The historical `run_cpline_runpod_v3_no_guide_grid_{probe,full}.sh` names are
retired because they launched a non-promotable dense-head diagnostic with
`junction_offset_radius_px=0.0`.

Before spending RunPod budget:

1. Confirm the selected manifest exists on the pod.
2. Confirm the selected checkpoint is the current promoted dense-edge/tess15
   model.
3. Print the resolved `MAX_EDGES`.
4. Print the resolved close-pair junction parameters.
5. Print `train_family_sampling` and the selected train source counts from
   `run_config.json`.
6. Run a tiny data-loader smoke with the selected `MAX_EDGES`.
7. Confirm the expected number of train/val examples after filtering.

For the 15% tessellation experiment, use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_probe.sh
```

The generic dense-edge launcher auto-selects
`TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct` when the manifest path contains
`cp_training_mix_v3_tessellation_15pct`. It fails with an "Are you sure?"
message if a non-default tessellation sampler is used without
`ALLOW_NONDEFAULT_TESS15_SAMPLING=1`.

## Open Questions

- Is `MAX_EDGES=700` enough to move dense vertical recall, or do we need
  `1200+`?
- Does including denser existing TreeMaker examples help BP, or are they still
  too unlike real box pleats?
- Does a higher edge envelope change the loss balance enough that dense samples
  need reweighting instead of only inclusion?
- Should BP/tessellation data be oversampled by family, by vertical crease
  length, or by minimum rendered crease spacing?

## Experiment Log

### 2026-06-18: `MAX_EDGES=700` Probe

Status: completed and previously promoted; superseded by max1200 and then by
tess15 weighted.

Run:

- Launcher:
  `scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh`
- RunPod pod: `vuqivzyw6omfb5`, RTX 4090, stopped after artifacts were copied
  back.
- Cost estimate: about 1.25 pod-hours at `$0.69/hr`, under `$1`.
- Checkpoint:
  `checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max700_probe_20260618/full/latest.pt`
- Registry:
  `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max700-4090.json`
- Checkpoint SHA-256:
  `332b39ded02797e1f70746b733647c8dfa28e246e607eb418448e97c959e9ddf`
- Init checkpoint:
  `checkpoints/runpod_v3_no_guide_grid_close_pair_full_r1_20260617/full/latest.pt`
- ONNX export:
  `tree-maker-rust/apps/web/public/models/cp-detector-v3-dense-edges-max700-20260618/model.onnx`
- ONNX SHA-256:
  `323d640b504072262f172e10e45d2b85d377fee6d9c18e658f4d93bea1f9a400`

Training config:

- `MAX_EDGES=700`
- `max_steps=1500`
- `train_samples=2048`
- `val_samples=256`
- `lr=0.00005`
- `REINIT_HEADS=""`
- `junction_sigma_px=1.5`
- `junction_offset_radius_px=3.0`
- `junction_offset_weight=0.5`
- `junction_focal_alpha=2.0`
- `junction_focal_beta=4.0`

Training completed in `3296.8s`. First logged train loss was `0.4715`; last
logged train loss was `0.4000`.

Eval reports:

- BP dense heads:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/box-pleat-native-v1-dense-edges-max700-probe-20260618-dense-heads/summary.json`
- BP angle buckets:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/box-pleat-native-v1-angle-buckets-dense-edges-max700-probe-20260618/summary.json`
- Clean-15 strict topology:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/clean-1024-s15-strict-v3-dense-edges-max700-probe-20260618/summary.json`

Dense BP aggregate:

| Metric | Promoted no-grid R1 | `MAX_EDGES=700` probe |
| --- | ---: | ---: |
| Orthogonal effective recall | `0.6104` | `0.6462` |
| Orthogonal non-crease conflict | `0.0128` | `0.0197` |
| Diagonal effective recall | `0.8796` | `0.8954` |
| All crease effective recall | `0.6718` | `0.7030` |

BP angle buckets:

| Slice | Direction | Promoted no-grid R1 | `MAX_EDGES=700` probe |
| --- | --- | ---: | ---: |
| All 179 | Horizontal | `0.6546` | `0.6874` |
| All 179 | Vertical | `0.5736` | `0.6124` |
| All 179 | 45/135 | `0.8634` | `0.8812` |
| Dense top quartile | Horizontal | `0.4354` | `0.4763` |
| Dense top quartile | Vertical | `0.3757` | `0.4074` |
| Dense top quartile | 45/135 | `0.8082` | `0.8403` |
| Very dense `edge_count >= 2000` | Horizontal | `0.4062` | `0.4446` |
| Very dense `edge_count >= 2000` | Vertical | `0.3473` | `0.3797` |
| Very dense `edge_count >= 2000` | 45/135 | `0.8059` | `0.8365` |

Clean-15 strict topology:

| Metric | Promoted no-grid R1 | `MAX_EDGES=700` probe |
| --- | ---: | ---: |
| Strict edge F1 | `0.9594` | `0.9623` |
| Missing edges | `126` | `112` |
| Extra edges | `89` | `88` |
| Merged edges | `72` | `66` |
| Exact topology samples | `3/15` | `4/15` |
| Exact topology + assignment samples | `3/15` | `3/15` |

Conclusion:

The first controlled experiment supports the dense-edge-envelope hypothesis.
Without adding tessellation data, simply including denser existing synthetic
examples improved BP vertical recall and did not regress clean-15 topology. It
was promoted because it also improves clean-15 strict topology while preserving
the radius-3 close-pair decoder contract.

The next decision is whether to try a still larger dense-edge probe or move on
to BP/tessellation data.

### 2026-06-18: `MAX_EDGES=1200` Probe

Status: completed and previously promoted; superseded by tess15 weighted.

Run:

- Launcher:
  `scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh`
- RunPod pod: `54aa1fvtypz4l1`, NVIDIA L40S, stopped after artifacts were
  copied back.
- Cost estimate: about 1.17 pod-hours at `$0.99/hr`, around `$1.16`.
- Checkpoint:
  `checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max1200_probe_20260618/full/latest.pt`
- Registry:
  `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-max1200-l40s.json`
- Checkpoint SHA-256:
  `befa99edc4531919ffb8f36c933b2e394f8868cb6e0fb0be2d7b96eee74c9bac`
- Init checkpoint:
  `checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_max700_probe_20260618/full/latest.pt`
- ONNX export:
  `tree-maker-rust/apps/web/public/models/cp-detector-v3-dense-edges-max1200-20260618/model.onnx`
- ONNX SHA-256:
  `96ba3d56277f0ead32a6be813a31402434f29620f4b6edd113d3592e2c3ab145`

Training config:

- `MAX_EDGES=1200`
- `max_steps=1500`
- `train_samples=2048`
- `val_samples=256`
- `lr=0.00005`
- `REINIT_HEADS=""`
- `junction_sigma_px=1.5`
- `junction_offset_radius_px=3.0`
- `junction_offset_weight=0.5`
- `junction_focal_alpha=2.0`
- `junction_focal_beta=4.0`

Training completed in `3381.6s`. First logged train loss was `0.4282`; last
logged train loss was `0.5521`.

Eval reports:

- BP dense heads:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/box-pleat-native-v1-dense-edges-max1200-probe-20260618-dense-heads/summary.json`
- BP angle buckets:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/box-pleat-native-v1-angle-buckets-dense-edges-max1200-probe-20260618/summary.json`
- Clean-15 strict topology:
  `tree-maker-rust/artifacts/cp-detect-correctness/reports/clean-1024-s15-strict-v3-dense-edges-max1200-probe-20260618/summary.json`

Dense BP aggregate:

| Metric | Superseded max700 | Promoted `MAX_EDGES=1200` |
| --- | ---: | ---: |
| Orthogonal effective recall | `0.6462` | `0.6746` |
| Orthogonal non-crease conflict | `0.0197` | `0.0285` |
| Diagonal effective recall | `0.8954` | `0.9007` |
| All crease effective recall | `0.7030` | `0.7261` |

BP angle buckets:

| Slice | Direction | Superseded max700 | Promoted `MAX_EDGES=1200` |
| --- | --- | ---: | ---: |
| All 179 | Horizontal | `0.6874` | `0.7300` |
| All 179 | Vertical | `0.6124` | `0.6272` |
| All 179 | 45/135 | `0.8812` | `0.8866` |
| Dense top quartile | Horizontal | `0.4763` | `0.5153` |
| Dense top quartile | Vertical | `0.4074` | `0.4247` |
| Dense top quartile | 45/135 | `0.8403` | `0.8501` |
| Very dense `edge_count >= 2000` | Horizontal | `0.4446` | `0.4808` |
| Very dense `edge_count >= 2000` | Vertical | `0.3797` | `0.3968` |
| Very dense `edge_count >= 2000` | 45/135 | `0.8365` | `0.8438` |

Clean-15 strict topology:

| Metric | Superseded max700 | Promoted `MAX_EDGES=1200` |
| --- | ---: | ---: |
| Strict edge F1 | `0.9623` | `0.9655` |
| Missing edges | `112` | `107` |
| Extra edges | `88` | `76` |
| Merged edges | `66` | `60` |
| Exact topology samples | `4/15` | `4/15` |
| Exact topology + assignment samples | `3/15` | `3/15` |

Conclusion:

The second controlled experiment continues to support the dense-edge-envelope
hypothesis. Raising to `MAX_EDGES=1200` improves BP recall and clean-15 strict
topology over max700. The tradeoff is a modest increase in non-crease conflict
and slightly lower clean assignment accuracy (`0.9875 -> 0.9855`). It is now
promoted because the BP and strict-topology gains are larger than that tradeoff.

### 2026-06-18: Tessellation Source Data Implementation

Status: in progress.

Implemented the first `tessellation-fold-program` subfamily,
`orthogonal-bp-grid`, to target dense horizontal and vertical crease-line
evidence that is missing from the current synthetic mix. The first recipe is:

- `recipes/synthetic/tessellation_fold_program_v1.yaml`
- family: `tessellation-fold-program`
- subfamily: `orthogonal-bp-grid`
- active crease buckets: small through superdense, up to `2200`
- validation: dense, tessellation-structure, local flat-foldability, and Rabbit
  Ear global solver checks
- label provenance: `tessellation-fold-program`
- assignment convention: whole M/V crease lines alternate along one axis, with
  connector segments preserving a 3-to-1 split at every interior grid junction

This data should remain a standalone source release at:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/tessellation_fold_program_v1
```

The first mixed root with tessellations is:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v2_tessellation
```

It contains 12,000 TreeMaker samples, 2,000 Rabbit Ear samples, and 1,000
tessellation samples. The next training experiment should use this as an
explicit new dataset variable and compare against the promoted `MAX_EDGES=1200`
model with the same close-pair decoder contract.

Correction note: the first implementation used a segment-level parity pattern
that passed initial Rabbit Ear spot checks but made vertical-heavy examples look
like one-color ladders. The release was regenerated with line-level alternation
and the recipe now requires every accepted tessellation row to pass both
`local-flat-foldability` and `rabbit-ear-solver`.

### 2026-06-19: Tessellation Sampler Safety Fix

The first 15% tessellation probe accidentally used
`TRAIN_FAMILY_SAMPLING=natural` on
`cp_training_mix_v3_tessellation_15pct`. That did add about 15% tessellations,
but it also changed the old base exposure from balanced TreeMaker/Rabbit Ear to
the manifest-natural distribution: roughly 73% TreeMaker, 12% Rabbit Ear, and
15% tessellation. Its BP vertical recall gains therefore cannot be cleanly
separated from the Rabbit Ear exposure reduction, and it should not be promoted.

The corrected sampler is now explicit:

```text
TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct
```

For a 2048-sample training slice at `MAX_EDGES=1200`, seed `31`, this selects:

| Source | Records |
| --- | ---: |
| `treemaker-tree` from `cp_training_mix_v1` | `870` |
| `rabbit-ear-fold-program` from `cp_training_mix_v1` | `870` |
| `tessellation_orthogonal_bp_grid_v2_15pct` | `246` |
| `tessellation_miura_ori_v2_15pct` | `62` |

The launcher now records `train_selection_summary` in `run_config.json` and
fails with an "Are you sure?" message if the tessellation mix is launched with
a non-default sampler without an explicit override.

### 2026-06-19: Corrected 15% Tessellation Probe

The corrected run used the dedicated launcher:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_probe.sh
```

Run details:

- Checkpoint: `checkpoints/runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_weighted_probe_20260619/full/latest.pt`
- Registry: `artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-tess15-weighted-4090.json`
- Init: superseded max1200 dense-edge checkpoint
- GPU: RunPod RTX 4090, pod `rtljlswig1uksp`
- Steps: `1500`
- Training sampler: `v3-tessellation-15pct`
- Train selection: `870` TreeMaker, `870` Rabbit Ear, `246` orthogonal BP grid, `62` Miura
- Close-pair contract: `junction_offset_radius_px=3.0`

Dense BP results:

| Run | Orthogonal eRecall | Orthogonal non-crease conflict | All crease eRecall |
| --- | ---: | ---: | ---: |
| max1200 promoted baseline | `0.6746` | `0.0285` | `0.7261` |
| first tess15 natural-sampler ablation | `0.7277` | `0.0303` | `0.7651` |
| corrected tess15 weighted probe | `0.7547` | `0.0266` | `0.7854` |

Angle-bucket dense BP results:

| Cohort | Run | Horizontal eRecall | Vertical eRecall | 45/135 eRecall |
| --- | --- | ---: | ---: | ---: |
| all 179 | max1200 | `0.7300` | `0.6272` | `0.8866` |
| all 179 | corrected tess15 weighted | `0.7529` | `0.7634` | `0.8801` |
| dense top quartile | max1200 | `0.5153` | `0.4247` | `0.8501` |
| dense top quartile | corrected tess15 weighted | `0.5578` | `0.5311` | `0.8465` |
| very dense, edges >= 2000 | max1200 | `0.4808` | `0.3968` | `0.8438` |
| very dense, edges >= 2000 | corrected tess15 weighted | `0.5238` | `0.4875` | `0.8389` |

Clean-15 strict gate:

| Run | Exact topology | Exact topology + assignment | Strict eF1 | Missing | Extra | Merged | Split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max1200 promoted baseline | `4/15` | `3/15` | `0.9655` | `107` | `76` | `60` | `9` |
| first tess15 natural-sampler ablation | `3/15` | `3/15` | `0.9611` | `117` | `89` | `67` | `11` |
| corrected tess15 weighted probe | `4/15` | `3/15` | `0.9651` | `108` | `77` | `55` | `16` |

Interpretation: the corrected sampler removes the confound and improves the
actual BP target more than the accidental natural-sampler run, especially on
vertical dense BP lines. Clean-15 is effectively tied with max1200: one more
missing edge, one more extra edge, five fewer merges, and seven more splits.
The net result was promoted on 2026-06-19. The stable downstream
`cp-detector-v3` path now points at this ONNX export, and the ignored checkpoint
has been mirrored into the canonical local detector checkout.
