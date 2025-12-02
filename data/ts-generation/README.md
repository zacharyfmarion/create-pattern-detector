# Synthetic Crease Pattern Generator

TypeScript/Bun-based generator for synthetic origami crease patterns with multi-tier validation.

## Features

- **Multiple Generation Methods**:
  - Tree method: Incremental crease addition with validity checking
  - Box pleating: Grid-based patterns
  - Classic bases: Waterbomb, bird, frog, preliminary, fish

- **Symmetry Support**:
  - 2-fold (mirror symmetry)
  - 4-fold (rotational symmetry)
  - Asymmetric patterns

- **Multi-Tier Validation**:
  - **Tier S (Gold)**: Passes all local + global validation (FOLD CLI)
  - **Tier A (Silver)**: Passes all local validation only
  - **REJECT**: Fails local validation

- **Local Validation** (using Rabbit Ear):
  - Maekawa's theorem: |M - V| = 2 at interior vertices
  - Kawasaki's theorem: Alternating angle sum = 0
  - No self-intersections
  - Complete border
  - 2-colorability (flat-foldability)

- **Global Validation** (using FOLD CLI):
  - FOLD format compliance
  - Global flat-foldability check

## Installation

```bash
# Install dependencies
bun install

# Install FOLD CLI globally
npm install -g fold
```

## Usage

### Generate Test Dataset (100 CPs)

```bash
bun run generate-test
```

### Generate Custom Dataset

```bash
bun run src/generate-dataset.ts \
  --count 1000 \
  --method mixed \
  --symmetry mixed \
  --min-creases 10 \
  --max-creases 50

# Or specify custom output:
bun run src/generate-dataset.ts \
  --count 1000 \
  --output ../output/synthetic/raw \
  --method mixed
```

### Options

- `--count, -c`: Number of patterns to generate (default: 100)
- `--output, -o`: Output directory (default: ../output/synthetic/raw)
- `--method`: Generation method: `mixed`, `tree`, `box-pleating`, `classic-bases` (default: mixed)
- `--symmetry`: Symmetry type: `mixed`, `none`, `2-fold`, `4-fold` (default: mixed)
- `--width`: Pattern width in pixels (default: 1024)
- `--height`: Pattern height in pixels (default: 1024)
- `--min-creases`: Minimum number of creases (default: 10)
- `--max-creases`: Maximum number of creases (default: 50)
- `--skip-global`: Skip global validation (faster, only Tier A) (default: false)
- `--parallel`: Run validations in parallel (default: false)

## Output Structure

```
../output/synthetic/raw/
├── tier-s/           # Gold tier (passes all validation)
│   ├── cp-*.fold     # FOLD format files
│   └── cp-*.json     # Metadata (config + validation results)
├── tier-a/           # Silver tier (passes local validation)
│   ├── cp-*.fold
│   └── cp-*.json
└── rejected/         # Failed validation
    ├── cp-*.fold
    └── cp-*.json
```

## Next Steps

After generation, render to images and HAWP annotations:

```bash
cd ../..
python scripts/render_dataset.py \
  --input ./data/output/synthetic/raw \
  --output ./data/output/synthetic/rendered
```

## Project Structure

```
ts-generation/
├── src/
│   ├── types/
│   │   └── crease-pattern.ts     # TypeScript types
│   ├── utils/
│   │   └── fold-helpers.ts       # FOLD utilities
│   ├── validators/
│   │   ├── local.ts              # Rabbit Ear validation
│   │   ├── global.ts             # FOLD CLI integration
│   │   └── validator.ts          # Orchestrator
│   ├── generators/
│   │   ├── tree-method.ts        # Tree method generator
│   │   ├── box-pleating.ts       # Box pleating generator
│   │   ├── classic-bases.ts      # Classic bases
│   │   └── symmetric.ts          # Symmetry utilities
│   └── generate-dataset.ts       # Main script
├── package.json
├── tsconfig.json
└── README.md
```

## Development

```bash
# Run tests
bun test

# Type checking
bun run tsc --noEmit
```

## References

- [Rabbit Ear](https://github.com/robbykraft/Origami) - Origami mathematics library
- [FOLD Format](https://github.com/edemaine/fold) - Origami file format specification
- [HAWP](https://github.com/cherubicXN/hawp) - Wireframe detection model
