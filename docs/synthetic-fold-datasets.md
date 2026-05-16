# Synthetic Fold Datasets

This document covers fold-only synthetic dataset generation for the current
training corpus.

The production mix is TreeMaker-primary, with a small strict Rabbit Ear
fold-program supplement:

```text
12,000 TreeMaker samples + 2,000 Rabbit Ear samples = 14,000 fold-only samples
```

No image augmentation is part of this step. The goal here is diverse, validated
`.fold` graph data. Rendering style/noise augmentation belongs to the later
image-training phase.

## What Was Added

- `tools/synthetic-generator/src/rabbit-ear-fold-program.ts`
  - Starts from `ear.graph.square()`.
  - Samples deterministic Rabbit Ear axiom lines.
  - Applies `ear.graph.flatFold(...)` operations with seeded M/V assignments.
  - Writes `rabbit_ear_metadata` and label provenance into each `.fold`.

- `recipes/synthetic/rabbit_ear_fold_program_v1.yaml`
  - Enables only `rabbit-ear-fold-program`.
  - Defines `small`, `medium`, and `dense` active-crease buckets.
  - Requires local flat-foldability, Rabbit Ear global solver, and finite folded
    coordinates.

- `scripts/data/build_synthetic_training_mix.py`
  - Builds a fold-only mixed dataset from external source roots.
  - Symlinks source `.fold` and metadata files instead of copying them.
  - Merges raw manifests and preserves source provenance.

- `implementations-plans/rabbit-ear-fold-program-supplement.md`
  - Records the supplement decision, validation contract, and release checks.

## Dataset Roots

Shared datasets live outside the git worktree:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v1
```

Future worktrees should access them through ignored symlinks:

```bash
scripts/data/link_shared_synthetic_data.sh treemaker_tree_v1
scripts/data/link_shared_synthetic_data.sh rabbit_ear_fold_program_v1
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
```

The mixed root is the recommended default for training setup once both source
datasets exist:

```text
data/generated/synthetic/cp_training_mix_v1
```

## Setup

Install the Bun generator dependencies:

```bash
bun install --cwd tools/synthetic-generator
```

Use Python 3.10 or the shared Python environment for fold reports and smokes:

```bash
scripts/setup_python_env.sh
```

Then use either:

```bash
.venv/bin/python
```

or:

```bash
python3.10
```

depending on the machine.

## Rabbit Ear Smoke

Run a small strict smoke before generating a release:

```bash
rm -rf /tmp/rabbit_ear_fold_program_v1_smoke

bun run --cwd tools/synthetic-generator generate -- \
  --recipe ../../recipes/synthetic/rabbit_ear_fold_program_v1.yaml \
  --count 64 \
  --out /tmp/rabbit_ear_fold_program_v1_smoke \
  --max-attempts 2500
```

Create a fold distribution report:

```bash
python3.10 scripts/data/synthetic_fold_report.py \
  --root /tmp/rabbit_ear_fold_program_v1_smoke
```

Create folded-preview QA:

```bash
bun run --cwd tools/synthetic-generator folded-preview -- \
  --root /tmp/rabbit_ear_fold_program_v1_smoke \
  --limit 24 \
  --skip-failures

PYTHONPATH=. python3.10 scripts/data/render_folded_preview.py \
  --root /tmp/rabbit_ear_fold_program_v1_smoke \
  --limit 24 \
  --output /tmp/rabbit_ear_fold_program_v1_smoke/qa/folded/contact_sheet_24.png
```

No-torch label smoke:

```bash
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py \
  --root /tmp/rabbit_ear_fold_program_v1_smoke \
  --samples-per-split 2 \
  --image-size 128
```

Expected smoke properties:

- `qa.json` has `familyCounts.rabbit-ear-fold-program = 64`.
- `rabbitEarStrictPassRate` is `1`.
- accepted rows include `rabbitEarMetadata`.
- folded preview emits finite folded coordinates for the checked rows.

## Generate The Rabbit Ear Release

Generate the 2,000-sample supplement outside the repo. A simple shard layout is
recommended so interrupted runs can be resumed or regenerated independently:

```bash
export CP_DATA_ROOT=/Users/zacharymarion/Documents/datasets/create-pattern-detector
export RABBIT_RELEASE=$CP_DATA_ROOT/synthetic/rabbit_ear_fold_program_v1
export RABBIT_SHARDS=$CP_DATA_ROOT/synthetic/.shards/rabbit_ear_fold_program_v1

rm -rf "$RABBIT_SHARDS"
mkdir -p "$RABBIT_SHARDS"
```

Generate 8 shards of 250 accepted samples:

```bash
for shard in 0 1 2 3 4 5 6 7; do
  seed=$((9917000 + shard))
  printf -v shard_name "shard-%03d" "$shard"

  bun run --cwd tools/synthetic-generator generate -- \
    --recipe ../../recipes/synthetic/rabbit_ear_fold_program_v1.yaml \
    --seed "$seed" \
    --count 250 \
    --out "$RABBIT_SHARDS/$shard_name" \
    --max-attempts 10000
done
```

Merge the shards into the durable release root:

```bash
rm -rf "$RABBIT_RELEASE.build"

python3.10 scripts/data/merge_synthetic_fold_shards.py \
  --out "$RABBIT_RELEASE.build" \
  --recompute-splits \
  "$RABBIT_SHARDS"/shard-*

python3.10 scripts/data/synthetic_fold_report.py \
  --root "$RABBIT_RELEASE.build"

rm -rf "$RABBIT_RELEASE"
mv "$RABBIT_RELEASE.build" "$RABBIT_RELEASE"
```

Link and smoke the release:

```bash
scripts/data/link_shared_synthetic_data.sh rabbit_ear_fold_program_v1

PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py \
  --root data/generated/synthetic/rabbit_ear_fold_program_v1 \
  --samples-per-split 2 \
  --image-size 128
```

## Build The Mixed Training Root

Build the 14k fold-only training mix from the two external roots:

```bash
export CP_DATA_ROOT=/Users/zacharymarion/Documents/datasets/create-pattern-detector
export MIX_RELEASE=$CP_DATA_ROOT/synthetic/cp_training_mix_v1

rm -rf "$MIX_RELEASE.build"

python3.10 scripts/data/build_synthetic_training_mix.py \
  --out "$MIX_RELEASE.build" \
  --recompute-splits \
  "$CP_DATA_ROOT/synthetic/treemaker_tree_v1" \
  "$CP_DATA_ROOT/synthetic/rabbit_ear_fold_program_v1"

python3.10 scripts/data/synthetic_fold_report.py \
  --root "$MIX_RELEASE.build"

rm -rf "$MIX_RELEASE"
mv "$MIX_RELEASE.build" "$MIX_RELEASE"
```

Link and smoke the mix:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1

PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py \
  --root data/generated/synthetic/cp_training_mix_v1 \
  --samples-per-split 2 \
  --image-size 128
```

## Outputs

Each generator release root contains:

```text
folds/*.fold
metadata/*.json
raw-manifest.jsonl
recipe.json
qa.json
qa/fold-distribution-report.json
```

The mixed root contains symlinks instead of copied source files:

```text
folds/<sourceDataset>--<sample>.fold -> source fold
metadata/<sourceDataset>--<sample>.json -> source metadata
raw-manifest.jsonl
qa.json
qa/mix-summary.json
qa/fold-distribution-report.json
```

Mixed manifest rows preserve provenance:

- `family`: source generator family, such as `treemaker-tree` or
  `rabbit-ear-fold-program`;
- `sourceDataset`: external source root name;
- `sourceFoldPath`: absolute original fold path;
- `sourceMetadataPath`: absolute original metadata path;
- `foldPath`: symlink path relative to the mixed root;
- `metadataPath`: symlink path relative to the mixed root.

## Interpreting QA

Important `qa.json` fields for Rabbit Ear:

- `accepted`: accepted sample count.
- `attempts`: generation attempts used.
- `acceptanceRate`: accepted / attempts.
- `rabbitEarStrictPassRate`: share of rows that passed the Rabbit Ear global
  solver check.
- `rabbitEarAxiomCounts`: aggregate axiom-line usage.
- `rabbitEarRequestedBucketCounts`: accepted counts by requested crease bucket.

Important fold distribution fields:

- `counts.family`: TreeMaker vs Rabbit Ear share.
- `counts.bucket`: complexity bucket distribution.
- `vertices` and `edges`: graph size distribution.
- `assignment_totals`: M/V/B/U/F totals.
- `angle_histogram`: broad orientation diversity.
- `degree_histogram`: graph junction complexity.

For the intended production mix:

- total rows should be 14,000;
- `treemaker-tree` should be 12,000 rows;
- `rabbit-ear-fold-program` should be 2,000 rows;
- split should be recomputed globally as roughly 85/10/5.

## Programmatic Use

The mixed dataset is still just a raw-manifest fold dataset. A lightweight parse
and label smoke looks like:

```python
from pathlib import Path

from src.data.annotations import GroundTruthGenerator
from src.data.fold_parser import FOLDParser

root = Path("data/generated/synthetic/cp_training_mix_v1")
row = next(line for line in (root / "raw-manifest.jsonl").read_text().splitlines() if line)

import json

record = json.loads(row)
cp = FOLDParser().parse(root / record["foldPath"])
gt = GroundTruthGenerator(image_size=128, padding=8, line_width=1).generate(cp)

print(cp.num_vertices, cp.num_edges)
print(gt["segmentation"].shape)
```

For full model training, use the linked mixed root as the fold source and keep
render/image augmentation in the later image-data phase.
