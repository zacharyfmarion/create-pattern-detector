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
