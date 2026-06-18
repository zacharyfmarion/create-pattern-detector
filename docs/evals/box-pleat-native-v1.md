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

## No-Guide-Grid Dense Diagnostic

Two R1 warm-start no-guide-grid experiments were evaluated on the same full
179-sample BP pack in `tree-maker-rust` on 2026-06-16:

- Probe: 800 steps, no guide-grid training profiles, reinitialized
  `non_crease_head`.
- Full diagnostic: 5000 steps with the same no-guide-grid profile.

Dense-head comparison:

| Model | Slice | Raw line recall | Effective recall | Recall drop | Non-crease conflict | Mean line prob | Mean non-crease prob |
|---|---|---:|---:|---:|---:|---:|---:|
| Shipped V3 R1 | Orthogonal BP creases | `0.5130` | `0.4744` | `0.0386` | `0.5422` | `0.5323` | `0.5887` |
| No-grid probe, 800 steps | Orthogonal BP creases | `0.5528` | `0.5525` | `0.0003` | `0.0049` | `0.5687` | `0.2699` |
| No-grid full, 5000 steps | Orthogonal BP creases | `0.6012` | `0.6001` | `0.0011` | `0.0154` | `0.6159` | `0.1062` |
| Shipped V3 R1 | Diagonal/other creases | `0.8752` | `0.8587` | `0.0166` | `0.0902` | `0.8802` | `0.1334` |
| No-grid probe, 800 steps | Diagonal/other creases | `0.8756` | `0.8744` | `0.0012` | `0.0103` | `0.8811` | `0.2921` |
| No-grid full, 5000 steps | Diagonal/other creases | `0.8810` | `0.8788` | `0.0023` | `0.0160` | `0.8833` | `0.0826` |

This confirms the original grid-suppression hypothesis for the non-crease head:
orthogonal BP crease pixels stopped being classified as non-crease guide-grid
evidence, and effective recall now tracks raw recall.

The remaining orthogonal-vs-diagonal recall gap is not fixed:
`0.6012` raw recall for orthogonal BP creases versus `0.8810` for
diagonal/other creases in the full diagnostic. Since non-crease conflict is now
low, the residual gap is more likely due to data distribution and missing
synthetic BP-style crease families than to non-crease suppression still being
baked into the warm-start. A warm-start-only line-head bias is still possible,
but the 800-step to 5000-step improvement argues against a hard inability to
unlearn the old behavior.

Important caveat: the 5000-step full diagnostic accidentally trained with
`junction_offset_radius_px=0.0` because the old no-grid launcher did not forward
the R1 close-pair offset parameters. Its dense-head BP metrics are valid, but it
must not be promoted or exported as the product default. Future compatible
training runs must use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
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
