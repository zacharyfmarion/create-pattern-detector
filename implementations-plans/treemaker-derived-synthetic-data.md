# TreeMaker-Derived Synthetic CP Dataset

## Current Decision

Use the real legacy TreeMaker engine as an external dependency, not a mock and
not a reimplementation. The production path is:

1. Sample a symbolic tree spec in `tools/synthetic-generator`.
2. Call an external TreeMaker CLI via `TREEMAKER_CLI`.
3. Let TreeMaker optimize/triangulate/build the CP.
4. Accept only `HAS_FULL_CP` outputs.
5. Convert TreeMaker crease output to canonical `.fold`.
6. Render images, manifests, QA, and folded previews where Rabbit Ear can
   interpret the exported graph.

TreeMaker source and build artifacts live outside this repo, like scraped or
generated data. The repo contains only the thin wrapper source, setup script,
adapter, recipe, tests, and documentation needed to reproduce the external
toolchain.

## Why The Legacy Branch

The current working external source is:

- repo: `https://github.com/AndrewKvalheim/treemaker.git`
- branch: `legacy-environment`

The modern `vishvish/treemaker` fork builds on macOS, but the headless model
calls tested in this spike produced empty/invalid CPs. The legacy branch,
with the same wrapper logic plus small modern-Clang patches, produced real
TreeMaker CPs:

- `@optimized`: 104 creases, `HAS_FULL_CP`;
- `@gusset`: 24 creases, `HAS_FULL_CP`;
- `tmModelTester_5.tmd5`: 434 creases, `HAS_FULL_CP`;
- sampled tree spec with `--triangulate`: 299 creases, `HAS_FULL_CP`.

Until the fork behavior difference is understood, use the legacy branch.

## Setup

```bash
python3.10 tools/treemaker-adapter/scripts/setup_external_treemaker_cli.py
export TREEMAKER_CLI=~/.cache/cp-detector/treemaker-legacy/build/treemaker-json-cli
export TREEMAKER_CLI_ARGS=--triangulate
```

`--triangulate` is required for sampled specs. Without it, TreeMaker may emit
partial polygon content with `POLYS_NOT_VALID`.

## Dataset Recipe

The V1 recipe is `recipes/synthetic/treemaker_tree_v1.yaml`.

Symmetry target:

- diagonal: 42.5%;
- middle-axis: 42.5%;
- asymmetric: 15%.

Middle-axis is split between vertical/horizontal. Diagonal is split between
main/anti diagonal.

Topology target:

- `radial-star`: shallow baseline trees;
- `hubbed-limbs`: paired or asymmetric hubs with terminal child flaps;
- `spine-chain`: multi-hub chains with paired terminals along a symmetry axis;
- `branched-hybrid`: spine plus side hubs.

The goal is diverse `.fold` graph structure before any image augmentation.
Image noise/style augmentation belongs to Phase 3 training, not this graph
generation phase.

## Acceptance Rules

Required before a generated sample becomes training data:

- external TreeMaker `ok=true`;
- `optimization.success=true`;
- `foldedForm.success=true`;
- complete border;
- no degenerate/duplicate/unsplit geometry after conversion;
- TreeMaker crease kinds preserved in `edges_treemakerKind`;
- rendered manifest row points to a real image and `.fold`;
- folded-preview QA is attempted and may be skipped when Rabbit Ear cannot solve
  a TreeMaker export that TreeMaker itself marked as a full CP.

Rabbit Ear global layer solving is optional for this TreeMaker-derived family because
TreeMaker is the CP-generation authority. Rabbit Ear preview remains useful QA
when compatible.

## Current Smoke Result

Command:

```bash
TREEMAKER_CLI=/tmp/cp-detector-treemaker-setup-smoke/build/treemaker-json-cli \
TREEMAKER_CLI_ARGS=--triangulate \
bun run --cwd tools/synthetic-generator generate -- \
  --recipe ../../recipes/synthetic/treemaker_tree_v1.yaml \
  --count 4 \
  --out /tmp/treemaker_tree_v1_real \
  --max-attempts 80
```

Result after removing unused BP code from the PR branch:

- accepted: 4 / 43 attempts;
- edge counts: 58 to 240 arranged edges;
- fold distribution report generated;
- Rabbit Ear folded-preview manifest generated with skipped rows because the
  layer solver rejected these TreeMaker exports as incompatible preview input.

## Scale Workflow

Use shard-level generation with different seeds and external TreeMaker timeout:

```bash
TREEMAKER_CLI=~/.cache/cp-detector/treemaker-legacy/build/treemaker-json-cli \
TREEMAKER_CLI_ARGS=--triangulate \
TREEMAKER_TIMEOUT_MS=5000 \
bun run --cwd tools/synthetic-generator generate -- \
  --recipe ../../recipes/synthetic/treemaker_tree_v1.yaml \
  --seed 9170001 \
  --count 250 \
  --out /path/to/synthetic/treemaker_tree_v1/shards/shard-000 \
  --max-attempts 10000
```

Merge accepted fold shards without rendering:

```bash
python3.10 scripts/data/merge_synthetic_fold_shards.py \
  --out /path/to/synthetic/treemaker_tree_v1/merged \
  --recompute-splits \
  /path/to/synthetic/treemaker_tree_v1/shards/shard-*
```

Create graph distribution QA:

```bash
python3.10 scripts/data/synthetic_fold_report.py \
  --root /path/to/synthetic/treemaker_tree_v1/merged
```

The report tracks vertices, edges, assignments, degree histogram, active-degree
histogram, angle histogram, border intersections, archetype, symmetry, topology,
branch depth, and terminal count.

## Current 12k Release

The durable fold-only TreeMaker V1 release lives outside the repo at:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1
```

It contains 12,000 accepted FOLD samples generated from 12 shards. The split is
10,200 train / 1,200 val / 600 test. All accepted samples report TreeMaker
optimization success and folded-form success.

Future worktrees should link it with:

```bash
scripts/data/link_shared_synthetic_data.sh treemaker_tree_v1
```

Lightweight smoke, before installing torch:

```bash
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py
```

## Rabbit Ear Supplement

TreeMaker remains the primary production family. The only approved supplement is
the strict Rabbit Ear fold-program family documented in
`implementations-plans/rabbit-ear-fold-program-supplement.md`.

Target fold-only training mix:

- 12,000 TreeMaker samples from `treemaker_tree_v1`;
- 2,000 Rabbit Ear fold-program samples from `rabbit_ear_fold_program_v1`;
- mixed symlinked root: `cp_training_mix_v1`.

Use the mix root for training setup once it exists:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py --root data/generated/synthetic/cp_training_mix_v1
```

## Next Work

1. After installing the full ML dependencies, verify a `CreasePatternDataset`
   train/val/test smoke can render labels from
   `data/generated/synthetic/treemaker_tree_v1/folds` without pre-rendered
   images.
2. Use clean rendering for CPLineNet training first; image augmentation belongs
   to Phase 3 and is intentionally not part of the fold-only release.
3. Decide whether `F`/unfolded hinge lines should be included in every render
   or reserved for full-CP variants only.
4. Keep the no-torch dataset smoke available for lightweight worktrees:
   `PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py`.
