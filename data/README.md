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

## Data Sources

### Synthetic Patterns
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