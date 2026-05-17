# Rabbit Ear Fold-Program Supplement Dataset

## Decision

TreeMaker remains the primary production synthetic family. Rabbit Ear is added
only as a capped supplement for strict geometric diversity:

- target mix: 12,000 TreeMaker + 2,000 Rabbit Ear = 14,000 fold-only samples;
- target share: about 14% Rabbit Ear;
- family: `rabbit-ear-fold-program`;
- recipe: `recipes/synthetic/rabbit_ear_fold_program_v1.yaml`;
- no image augmentation in this phase.

This path uses the maintained `rabbit-ear@0.9.32` API under
`tools/synthetic-generator`; it does not revive `data/ts-generation`, old
classic/base spam, fake grid outputs, or single-vertex-only outputs.

## Generator Contract

Each sample starts from `ear.graph.square()` and applies deterministic seeded
Rabbit Ear fold operations:

1. sample an axiom line from Rabbit Ear axioms 1, 2, 3, 4, or 7;
2. apply `ear.graph.flatFold(graph, line.vector, line.origin, assignment)`;
3. repeat until the target active M/V crease bucket is reached;
4. normalize the graph and store Rabbit Ear provenance metadata.

The supplement has three active-crease buckets:

- `small`: 40-90 active M/V creases;
- `medium`: 90-220 active M/V creases;
- `dense`: 220-700 active M/V creases.

Canonical metadata:

- `rabbit_ear_metadata.generator = "rabbit-ear-fold-program"`;
- `rabbit_ear_metadata.rabbitEarApi = "ear.graph.flatFold"`;
- applied/attempted fold counts;
- axiom usage counts;
- target and actual active crease counts;
- requested bucket.

## Validation

Accepted Rabbit Ear supplement samples must pass:

- complete border;
- no duplicate, degenerate, or unsplit-crossing geometry;
- local Kawasaki/Maekawa;
- Rabbit Ear global layer solver;
- finite flat-folded coordinates;
- Rabbit Ear fold-program metadata/provenance checks;
- bucket minimums, rejecting sparse/toy outputs.

The generator is allowed to produce less designer-like CPs than TreeMaker, but
it must be strict, nontrivial, and useful for supplemental geometry diversity.

## Shared Dataset Layout

External dataset roots:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v1
```

Future worktrees should link the mixed training root with:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
```

The mix builder symlinks fold and metadata files instead of copying them:

```bash
python3.10 scripts/data/build_synthetic_training_mix.py \
  --out /Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v1 \
  --recompute-splits \
  /Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1 \
  /Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1
```

It preserves row provenance with `sourceDataset`, `sourceFoldPath`, and
`sourceMetadataPath`, rejects duplicate sample ids, and writes merged QA counts.

## Release Checks

Minimum release checks:

```bash
bun run --cwd tools/synthetic-generator typecheck
bun test --cwd tools/synthetic-generator
PYTHONPATH=. python3.10 -m pytest tests/test_synthetic_mix_builder.py
python3.10 scripts/data/synthetic_fold_report.py --root <rabbit-root>
python3.10 scripts/data/synthetic_fold_report.py --root <mix-root>
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py --root <mix-root>
```

Folded previews should be generated for Rabbit Ear samples for visual QA, but
the release is fold-only and should not include rendered image augmentation.
