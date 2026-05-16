# TreeMaker-Derived Synthetic CP Dataset

## Current Decision

Use the real legacy TreeMaker engine as an external dependency, not a mock and
not a reimplementation. The production path is:

1. Sample a symbolic tree spec in `tools/synthetic-generator`.
2. Call an external TreeMaker CLI via `TREEMAKER_CLI`.
3. Let TreeMaker optimize/triangulate/build the CP.
4. Accept only `HAS_FULL_CP` outputs.
5. Convert TreeMaker crease output to canonical `.fold`.
6. Render images, manifests, QA, and folded previews.

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
- folded-preview QA can be generated.

Rabbit Ear global layer solving is optional for this non-BP family because
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

Result:

- accepted: 4 / 16 attempts;
- edge counts: 102 to 159 arranged edges;
- symmetry: 2 diagonal, 2 middle-axis;
- rendered rows: 12;
- folded previews: 4 / 4.

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

## Next Work

1. Run a 100-accepted-graph pilot and inspect the fold distribution report.
2. Tune the sampler to reduce rejection rate while preserving the 85% symmetry
   distribution.
3. Generate a 1k accepted graph calibration set and verify topology/archetype
   diversity before training.
4. Add a manifest-loader smoke for rendered rows after clean rendering is
   re-enabled for CPLineNet training.
5. Decide whether `F`/unfolded hinge lines should be included in every render
   or reserved for full-CP variants only.
