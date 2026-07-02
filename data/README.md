# Data Directory

This directory contains all crease pattern generation code, validation logic, and output data for the CP Extractor project.

## Directory Structure

```
data/
├── ts-generation/         # TypeScript generator + validator
│   ├── src/
│   │   ├── generators/           # CP generation algorithms
│   │   ├── validators/           # Validation logic
│   │   ├── types/                # TypeScript types
│   │   ├── utils/                # Utilities
│   │   ├── generate-dataset.ts   # Synthetic CP generator
│   │   └── validate-scraped.ts   # Scraped CP validator
│   ├── package.json
│   ├── tsconfig.json
│   └── README.md
│
├── output/
│   ├── synthetic/        # Generated synthetic patterns
│   │   ├── raw/
│   │   │   ├── tier-s/          # Gold: Passes all validation
│   │   │   ├── tier-a/          # Silver: Passes local validation
│   │   │   └── rejected/        # Failed validation
│   │   └── rendered/            # PNG renders for HAWP training
│   │       ├── tier-s/
│   │       └── tier-a/
│   │
│   └── scraped/          # Real patterns scraped from web
│       ├── raw/                 # Unvalidated files (YOU PUT FILES HERE)
│       ├── validated/           # After validation
│       └── rejected/            # Invalid files
│
└── examples/             # Reference examples
    ├── eagle.fold
    ├── penguin.fold
    └── test_pattern.fold
```

## Workflows

### 1. Generate Synthetic Crease Patterns

```bash
cd ts-generation

# Generate 100 patterns (default)
bun run src/generate-dataset.ts

# Generate custom dataset
bun run src/generate-dataset.ts \
  --count 1000 \
  --method mixed \
  --min-creases 10 \
  --max-creases 50

# Output: ../output/synthetic/raw/tier-{s,a}/
```

### 2. Scrape and Validate Real FOLD Files

**Step 1: Add scraped files**
```bash
# Put your scraped .fold files here:
output/scraped/raw/*.fold
```

**Step 2: Validate**
```bash
cd ts-generation

# Validate all files in raw/
bun run src/validate-scraped.ts

# Valid files → ../output/scraped/validated/
# Invalid files → ../output/scraped/rejected/
```

## Validation Tiers

### Tier S (Gold) ⭐
- Passes all local validation (Maekawa, Kawasaki, 2-colorability)
- Passes global validation (FOLD CLI flat-foldability check)
- Highest quality for training

### Tier A (Silver)
- Passes all local validation
- May fail global validation (complex flat-foldability)
- Still useful for training

### Rejected ❌
- Fails local validation
- Mathematical constraints violated
- Not suitable for training

## Synthetic dataset roots

Each synthetic source is a self-contained "dataset root" (lives under
`~/Documents/datasets/create-pattern-detector/synthetic/<name>/`, symlinked into the
repo via `data/output`) with this layout:

```
<root>/
  folds/<id>.fold            # canonical training files (geometry + M/V/B/U/F)
  metadata/<id>.json         # per-sample config + validation + embedded fold
  raw-manifest.jsonl         # one row/sample (id, foldPath, family, bucket, split, …)
  recipe.json                # how the root was generated
  qa.json                    # summary stats
```

Roots are combined into a training mix with `scripts/data/build_synthetic_training_mix.py`,
which symlinks folds/metadata, tags each row with `sourceDataset`, dedupes by `id`, and
(optionally) recomputes train/val/test splits.

Current sources:

| Source root | Family | Origin |
|-------------|--------|--------|
| `rabbit_ear_fold_program_v1` | `rabbit-ear-fold-program` | In-repo TS generator (`data/ts-generation`) |
| `treemaker_tree_v1` | `treemaker-tree` | External TreeMaker CLI, normalized to a root |
| `search225_v1` | `search225-tiling` | Local SEARCH-22.5 db files (see below) |

### SEARCH-22.5 / ExplOri 22.5 patterns

Exact, flat-foldable 22.5° grid tilings from SEARCH-22.5 / ExplOri 22.5
(<https://github.com/theplantpsychologist/SEARCH-22.5>). These satisfy
Maekawa/Kawasaki by construction and ship with valid M/V/B assignments, giving a
distinct distribution (tessellation/modular-style) from the tree-based sources.
Hinge creases whose M/V is undetermined by the reconstruction are preserved as `F`
(mapped to `U` by the fold parser, and rendered).

**Strictly offline — never fetch from the public site.** The site runs on a single
machine in the author's home; the per-(N, symmetry) `tilings_{N}_{sym}.db` SQLite
files were provided directly by the author (local copies:
`~/Documents/open source/origami-designer/explori_db/`, plus `tilings_3_diag.db` in
`~/Downloads`). Patterns are reconstructed locally with `db_to_fold.py` in the
SEARCH-22.5 repo (its venv has the compiled `math225_core` extension), then ingested
into a standard dataset root:

```bash
# 1. db -> FOLD staging (in the SEARCH-22.5 repo)
cd ~/Documents/code/SEARCH-22.5
for db in ~/Documents/open\ source/origami-designer/explori_db/*.db ~/Downloads/tilings_3_diag.db; do
  .venv/bin/python db_to_fold.py "$db" --out /tmp/search225_staging
done

# 2. Staging -> dataset root (in this repo)
python scripts/data/build_search225_dataset.py \
  --folds /tmp/search225_staging \
  --out ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1

# 3. Fold it into a new training mix alongside the existing sources:
python scripts/data/build_synthetic_training_mix.py --recompute-splits \
  --out ~/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v4 \
  ~/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1 \
  ~/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1 \
  ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1
```

Adapter code: `src/data/search225/` (local-folds ingestion only; the old API client
was removed on purpose).

### Legacy synthetic patterns (Rabbit Ear)
- Generated using Rabbit Ear library
- Axiom-based folding simulation
- Guaranteed local flat-foldability
- Variable complexity and styles

### Scraped Patterns
Common sources:
- **origami-database.com** - Large collection of CPs
- **GitHub repositories** - fold-examples, origami-simulations
- **FOLD format repo** - Example patterns
- **Academic papers** - Published crease patterns

## Quality Guidelines

### When Adding Scraped Files

1. **File Format**: Must be valid FOLD JSON
2. **Required Fields**: `vertices_coords`, `edges_vertices`
3. **Assignments**: Should have `edges_assignment` (M/V/B)
4. **Size**: Reasonable number of vertices (< 10,000)
5. **Provenance**: Document source and licensing

### When Generating Synthetic Data

1. **Diversity**: Mix symmetry types and complexity levels
2. **Quality**: Prefer tier-S, but tier-A is acceptable
3. **Volume**: Generate enough for training (5,000+)
4. **Validation**: Always validate before using for training

## Licensing

- Synthetic patterns: Generated code (MIT/Apache-2.0)
- Scraped patterns: **Varies by source** - document provenance
- Always respect original creator licenses