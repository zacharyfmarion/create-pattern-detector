# Box-Pleat Native Eval V1

This eval is a deterministic first-pass pull of box-pleat-like crease patterns
from the scraped native converted-FOLD corpus. It exists to test the grid-line
suppression hypothesis: real horizontal and vertical crease families should not
be learned or decoded as non-crease guide grids.

The source of truth is the recipe, not a path list:

- Spec: `eval_specs/box_pleat_native_v1.json`
- Finder: `scripts/data/find_box_pleat_candidates.py`
- Generated artifacts: `artifacts/box_pleat_eval/` ignored by git

## Regenerate

Set up the shared Python environment first:

```bash
scripts/setup_python_env.sh
```

If the scraped dataset is linked into the worktree, the spec default root works:

```bash
scripts/data/link_shared_scraped_data.sh
.venv/bin/python scripts/data/find_box_pleat_candidates.py \
  --eval-spec eval_specs/box_pleat_native_v1.json \
  --verify-spec
```

To use an explicit local dataset root:

```bash
.venv/bin/python scripts/data/find_box_pleat_candidates.py \
  --eval-spec eval_specs/box_pleat_native_v1.json \
  --fold-root /path/to/scraped/native/converted_fold \
  --verify-spec
```

Successful verification means the regenerated corpus and selected candidates
match the pinned path-independent fingerprints in the spec.

## Outputs

The finder writes:

- `ranked_candidates.jsonl`: every parsed FOLD, sorted by BP score
- `ranked_candidates.csv`: spreadsheet-friendly copy of the same ranking
- `review_candidates.json`: all selected non-weak candidates
- `fingerprints.json`: path-independent counts and aggregate hashes
- `verification.json`: comparison against the spec when `--eval-spec` is used
- `top_candidates_contact_sheet.png`: visual review sheet for the top candidates

These files are generated artifacts and should stay out of git.

## Product-Side Evaluation

This repo defines and verifies the deterministic BP candidate set. The
browser/Rust product repo owns the downstream eval over the shipped ONNX model
and post-processing pipeline. In `tree-maker-rust`, read
`scripts/cp-detect/README.md#box-pleat-native-eval` and build a correctness
pack from this spec with `scripts/cp-detect/build-box-pleat-native-pack.py`.

That split keeps the open-source source of truth path-independent: this repo
commits the recipe and fingerprints, while generated product packs, dense
caches, and reports stay ignored under the product repo's `artifacts/` tree.

## Shipped V3 Dense Baseline

The shipped V3 browser model was evaluated on the full `179` candidate BP set on
2026-06-15. The generated product artifacts are ignored in `tree-maker-rust`:

- Pack: `artifacts/cp-detect-correctness/packs/box-pleat-native-v1-baseline-v3-20260615/`
- Dense cache: `artifacts/cp-detect-correctness/dense-cache/box-pleat-native-v1-baseline-v3-20260615-browser-onnx-v3/`
- Dense report: `artifacts/cp-detect-correctness/reports/box-pleat-native-v1-baseline-v3-20260615-dense-heads/`

Dense-head summary:

| Slice | Raw line recall | Effective recall | Recall drop | Non-crease conflict | Mean line prob | Mean non-crease prob |
|---|---:|---:|---:|---:|---:|---:|
| Orthogonal BP creases | `0.5130` | `0.4744` | `0.0386` | `0.5422` | `0.5323` | `0.5887` |
| Diagonal/other creases | `0.8752` | `0.8587` | `0.0166` | `0.0902` | `0.8802` | `0.1334` |
| All creases | `0.5964` | `0.5631` | `0.0333` | `0.4345` | `0.6125` | `0.4798` |

Use these numbers as the pre-training baseline for no-guide-grid training
probes. The key gate is improving orthogonal BP raw/effective recall and
reducing non-crease conflict without collapsing the diagonal/control slice.

## No-Guide-Grid And Dense-Edge Promoted Runs

R1 warm-start no-guide-grid experiments were evaluated on the same full
179-sample BP pack in `tree-maker-rust`:

- Probe: 800 steps, no guide-grid training profiles, reinitialized
  `non_crease_head`.
- Full diagnostic: 5000 steps with the same no-guide-grid profile, but launched
  by the retired script that omitted the radius-3 close-pair offset recipe.
- No-guide-grid close-pair full R1: 5000 steps launched by the verified
  no-guide-grid close-pair script, reinitialized `non_crease_head`, and kept
  `junction_offset_radius_px=3.0`.
- Dense-edge max700 continuation: 1500 steps warm-started from the
  no-guide-grid close-pair R1, kept all heads, raised `MAX_EDGES` from `300` to
  `700`, and kept `junction_offset_radius_px=3.0`.
- Dense-edge max1200 probe: 1500 steps warm-started from the then-promoted max700
  checkpoint, kept all heads, raised `MAX_EDGES` to `1200`, and kept
  `junction_offset_radius_px=3.0`. It was later superseded by tess15 weighted.
- Dense-edge tess15 weighted: 1500 steps warm-started from max1200, kept all
  heads, kept `MAX_EDGES=1200`, added the corrected 15% tessellation mix with
  `TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct`, and kept
  `junction_offset_radius_px=3.0`. It is now the promoted default.

Dense-head comparison:

| Model | Slice | Raw line recall | Effective recall | Recall drop | Non-crease conflict | Mean line prob | Mean non-crease prob |
|---|---|---:|---:|---:|---:|---:|---:|
| Shipped V3 R1 | Orthogonal BP creases | `0.5130` | `0.4744` | `0.0386` | `0.5422` | `0.5323` | `0.5887` |
| No-grid probe, 800 steps | Orthogonal BP creases | `0.5528` | `0.5525` | `0.0003` | `0.0049` | `0.5687` | `0.2699` |
| No-grid full diagnostic, 5000 steps, radius 0 | Orthogonal BP creases | `0.6012` | `0.6001` | `0.0011` | `0.0154` | `0.6159` | `0.1062` |
| No-grid close-pair full R1, superseded | Orthogonal BP creases | `0.6113` | `0.6104` | `0.0009` | `0.0128` | `0.6306` | `0.1216` |
| Dense-edge max700, superseded | Orthogonal BP creases | `0.6482` | `0.6462` | `0.0019` | `0.0197` | `0.6646` | `0.1121` |
| Dense-edge max1200, superseded | Orthogonal BP creases | `0.6778` | `0.6746` | `0.0031` | `0.0285` | `0.6924` | `0.1209` |
| Dense-edge tess15 weighted, promoted | Orthogonal BP creases | `0.7582` | `0.7547` | `0.0035` | `0.0266` | `0.7719` | `0.0939` |
| Shipped V3 R1 | Diagonal/other creases | `0.8752` | `0.8587` | `0.0166` | `0.0902` | `0.8802` | `0.1334` |
| No-grid probe, 800 steps | Diagonal/other creases | `0.8756` | `0.8744` | `0.0012` | `0.0103` | `0.8811` | `0.2921` |
| No-grid full diagnostic, 5000 steps, radius 0 | Diagonal/other creases | `0.8810` | `0.8788` | `0.0023` | `0.0160` | `0.8833` | `0.0826` |
| No-grid close-pair full R1, superseded | Diagonal/other creases | `0.8816` | `0.8796` | `0.0021` | `0.0130` | `0.8868` | `0.0827` |
| Dense-edge max700, superseded | Diagonal/other creases | `0.8983` | `0.8954` | `0.0029` | `0.0157` | `0.9013` | `0.0610` |
| Dense-edge max1200, superseded | Diagonal/other creases | `0.9046` | `0.9007` | `0.0038` | `0.0197` | `0.9075` | `0.0586` |
| Dense-edge tess15 weighted, promoted | Diagonal/other creases | `0.8989` | `0.8950` | `0.0039` | `0.0194` | `0.9002` | `0.0494` |

This confirms the original grid-suppression hypothesis for the non-crease head:
orthogonal BP crease pixels stopped being classified as non-crease guide-grid
evidence, and effective recall now tracks raw recall. The later max700
continuation shows that the remaining dense-orthogonal gap also responds to
including denser existing synthetic examples.

The remaining orthogonal-vs-diagonal recall gap is not fixed:
`0.7582` raw recall for orthogonal BP creases versus `0.8989` for
diagonal/other creases in the current promoted run. Since non-crease conflict
is now low, the residual gap is more likely due to data distribution and dense
vertical coverage than to non-crease suppression still being baked into the
warm-start.

The promoted checkpoint is registered at
`artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-tess15-weighted-4090.json`.
It also improved clean-15 strict topology in the product repo's current-pack
eval relative to the previous no-guide-grid close-pair R1 (`0.9594 -> 0.9651`
edge F1, missing `126 -> 108`, extra `89 -> 77`, merged `72 -> 55`).

The dense-edge continuations also improved direction-specific BP recall:

| Slice | Direction | No-grid close-pair R1 | Dense-edge max700 |
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

The max1200 model improved these same direction-specific recall slices again,
and tess15 weighted then specifically improved the vertical BP target:

| Slice | Direction | Dense-edge max700 | Dense-edge max1200 | Tess15 weighted promoted |
| --- | --- | ---: | ---: | ---: |
| All 179 | Horizontal | `0.6874` | `0.7300` | `0.7529` |
| All 179 | Vertical | `0.6124` | `0.6272` | `0.7634` |
| All 179 | 45/135 | `0.8812` | `0.8866` | `0.8801` |
| Dense top quartile | Horizontal | `0.4763` | `0.5153` | `0.5578` |
| Dense top quartile | Vertical | `0.4074` | `0.4247` | `0.5311` |
| Dense top quartile | 45/135 | `0.8403` | `0.8501` | `0.8465` |
| Very dense `edge_count >= 2000` | Horizontal | `0.4446` | `0.4808` | `0.5238` |
| Very dense `edge_count >= 2000` | Vertical | `0.3797` | `0.3968` | `0.4875` |
| Very dense `edge_count >= 2000` | 45/135 | `0.8365` | `0.8438` | `0.8389` |

The earlier 5000-step full diagnostic accidentally trained with
`junction_offset_radius_px=0.0` because the old no-grid launcher did not forward
the R1 close-pair offset parameters. Its dense-head BP metrics are valid, but it
must not be promoted or exported as the product default. Future compatible
training runs must use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
```

For dense-edge follow-up probes from the promoted tess15 weighted checkpoint,
use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh
```

## Selection Recipe

The scorer ignores boundary edges, then fits the best rotated orthogonal frame
to non-boundary crease length. It scores each pattern with:

- fraction of crease length near the two orthogonal axes
- balance between both fitted axes
- repeated row/column coordinate evidence after rotating into the fitted frame
- crease density
- a weak filename prior for explicit box-pleat/BP naming

The V1 candidate set includes tiers:

- `strong`
- `review`
- `review-name-prior`

For the pinned dataset snapshot this yields:

- Parsed FOLD files: `640`
- Strong candidates: `104`
- Review candidates: `74`
- Name-prior review candidates: `1`
- Total selected candidates: `179`

## Fingerprints

The spec intentionally does not save absolute paths or relative path manifests.
Instead, the script hashes canonicalized FOLD geometry and assignments, then
builds aggregate digests:

- `all_inputs_canonical_sha256`: all parsed input FOLD content hashes
- `selected_canonical_sha256`: selected candidate content hashes
- `selected_ranked_features_sha256`: selected pathless feature rows in ranked order

If a future branch regenerates from the same dataset snapshot with the same
recipe, these fingerprints should match exactly.

## Manual Curation

This eval is still a candidate pull, not a final hand-labeled BP benchmark. If a
future branch promotes a smaller manually reviewed set, store labels by
canonical FOLD hash, not by local file path. Keep rendered contact sheets and
other large review artifacts ignored.
