# Phase 2 Deterministic Vectorizer

Phase 2 is a deterministic baseline for turning clean crease-pattern label
fields into a planar FOLD-style graph.

It does not train or run an image model. It starts from real `.fold` files,
renders very clean ground-truth labels, then asks:

```text
clean rendered labels -> PlanarGraphBuilder -> predicted graph -> metrics + overlays
```

This isolates vectorization quality before adding real-image complications such
as photos, scans, perspective distortion, augmentation, texture, lighting, or
stroke noise.

## What Was Added

- `src/vectorization/evidence.py`
  - Renders clean vectorizer evidence from a `CreasePattern`.
  - Produces `line_prob`, `angle`, `junction_heatmap`, and
    `assignment_labels`.
  - Canonicalizes metric-side edges so raw FOLD spans that pass through explicit
    vertices are compared as planar adjacent edges.

- `src/vectorization/planar_graph_builder.py`
  - Adds `VectorizerEvidence`, `PlanarGraphBuilderConfig`, and
    `PlanarGraphResult`.
  - Extracts line hypotheses, finds junctions, builds supported edges, votes
    assignments, and runs deterministic planar cleanup.

- `src/vectorization/metrics.py`
  - Adds vertex/edge matching metrics.
  - Adds assignment accuracy and FOLD structural validity checks.
  - Uses a spatial index for crossing checks on larger graphs.

- `scripts/vectorization/build_phase2_real_fixtures.py`
  - Builds deterministic `smoke`, `curated_gate`, and `full_stress` manifests
    from the final scraped real FOLD corpus.
  - Writes preflight corpus stats, a complexity histogram, and a curated contact
    sheet.

- `scripts/vectorization/run_phase2_vectorizer.py`
  - Runs the vectorizer over a manifest.
  - Writes metrics, overlays, contact sheets, worst-case sheets, and failure
    overlays.

## Setup

Use the shared Python environment setup:

```bash
scripts/setup_python_env.sh
```

For a worktree with an existing local `.venv` directory:

```bash
scripts/setup_python_env.sh --adopt-local
```

or replace it with the shared environment:

```bash
scripts/setup_python_env.sh --replace-local
```

Use the shared scraped dataset symlink. Do not copy the scraped dataset into the
worktree:

```bash
scripts/data/link_shared_scraped_data.sh
```

By default, the symlink is:

```text
data/output/scraped -> /Users/zacharymarion/Documents/datasets/create-pattern-detector/scraped
```

If the dataset lives somewhere else:

```bash
export CP_SHARED_DATA_ROOT=/path/to/create-pattern-detector-datasets
# or
export CP_SCRAPED_DATASET=/path/to/create-pattern-detector-datasets/scraped

scripts/data/link_shared_scraped_data.sh
```

## Regenerate Fixture Manifests

Fixture manifests are tracked. Raw FOLD files and generated visual reports are
not tracked.

```bash
.venv/bin/python scripts/vectorization/build_phase2_real_fixtures.py
```

This writes:

```text
fixtures/phase2_real_folds/smoke.json
fixtures/phase2_real_folds/curated_gate.json
fixtures/phase2_real_folds/full_stress.json
```

and ignored preflight artifacts:

```text
visualizations/phase2_vectorizer/preflight/corpus_stats.json
visualizations/phase2_vectorizer/preflight/complexity_histogram.png
visualizations/phase2_vectorizer/preflight/curated_gate_contact_sheet.png
```

## Run The Vectorizer

Smoke run:

```bash
.venv/bin/python scripts/vectorization/run_phase2_vectorizer.py \
  --manifest fixtures/phase2_real_folds/smoke.json \
  --output-dir visualizations/phase2_vectorizer/smoke \
  --image-size 1024 \
  --padding 32
```

Curated gate:

```bash
.venv/bin/python scripts/vectorization/run_phase2_vectorizer.py \
  --manifest fixtures/phase2_real_folds/curated_gate.json \
  --output-dir visualizations/phase2_vectorizer/curated_gate \
  --image-size 1024 \
  --padding 32
```

Full real stress telemetry:

```bash
.venv/bin/python scripts/vectorization/run_phase2_vectorizer.py \
  --manifest fixtures/phase2_real_folds/full_stress.json \
  --output-dir visualizations/phase2_vectorizer/full_stress \
  --image-size 1024 \
  --padding 32
```

The full stress run processes all final real FOLD files and can take a while.
Use it for telemetry and failure mining, not as a quick development loop.

## Outputs

Each run writes:

```text
summary.json
per_file_metrics.jsonl
contact_sheet.png
worst_vertex_recall_contact_sheet.png
worst_edge_recall_contact_sheet.png
slowest_contact_sheet.png
overlays/*.png
failures/*.png
```

Overlay panels are arranged as:

```text
rendered GT labels | detected line hypotheses and junctions | final predicted graph
```

The failure directory copies overlays for files that miss the current metric
thresholds:

- vertex recall < 99%
- vertex precision < 99%
- edge recall < 98%
- edge precision < 98%
- assignment accuracy < 99%
- structural validity failure

## Interpreting Metrics

The main metrics are:

- `vertex_precision`: predicted vertices that match GT vertices.
- `vertex_recall`: GT vertices recovered by the prediction.
- `edge_precision`: predicted edges that match GT edges after vertex matching.
- `edge_recall`: GT edges recovered by the prediction.
- `assignment_accuracy`: matched edges with the correct `M/V/B/U` assignment.
- `structural_validity_rate`: files that parse and avoid duplicate edges,
  zero-length edges, illegal crossings, and incomplete detected borders.

`bucket_summary` groups results by complexity:

- `tiny`: up to 99 edges.
- `small`: 100 to 249 edges.
- `medium`: 250 to 599 edges.
- `large`: 600 to 1499 edges.
- `stress`: 1500 or more edges.

The current practical Phase 2 gate is the non-stress curated slice. The stress
bucket is retained in the curated manifest and full corpus as telemetry for the
long complexity tail.

## Current Best Results

Best known run artifacts:

```text
visualizations/phase2_vectorizer/curated_gate_final_v2/
visualizations/phase2_vectorizer/full_stress_final/
```

Curated gate, non-stress records only:

- 52 files.
- Vertex precision/recall: 100.00% / 99.67%.
- Edge precision/recall: 98.82% / 98.61%.
- Assignment accuracy: 100.00%.
- Structural validity: 100.00%.

Full real stress, all 582 records:

- Vertex precision/recall: 99.997% / 97.91%.
- Edge precision/recall: 93.60% / 85.80%.
- Assignment accuracy: 99.999%.
- Structural validity: 96.74%.

Dense diagrams above roughly 1.7k edges remain the main known failure mode.

## Programmatic Use

```python
from pathlib import Path

from src.data.fold_parser import FOLDParser
from src.vectorization import PlanarGraphBuilder, render_vectorizer_evidence
from src.vectorization.metrics import evaluate_graph

cp = FOLDParser().parse(Path("path/to/file.fold"))
rendered = render_vectorizer_evidence(
    cp,
    image_size=1024,
    padding=32,
)

result = PlanarGraphBuilder().build(rendered.evidence)
metrics = evaluate_graph(
    result,
    gt_vertices=rendered.pixel_vertices,
    gt_edges=rendered.edges,
    gt_assignments=rendered.assignments,
)

print(result.vertices_coords)
print(result.edges_vertices)
print(metrics.to_dict())
```

`PlanarGraphResult` includes:

- `vertices_coords`: canonical `[0, 1]` coordinates.
- `edges_vertices`: undirected edge endpoint indices.
- `edges_assignment`: integer assignments using the repo FOLD parser mapping.
- `edge_support`: sampled line support per edge.
- `vertex_support`: vertex support diagnostics.
- `pixel_vertices`: image-space vertex coordinates.
- `debug`: masks, Hough segments, line hypotheses, junction peaks, and cleanup
  stats.

## Important Limitations

- This is not an image detector. It uses clean rendered labels from FOLD
  geometry.
- It does not perform augmentation, stretching, perspective transforms, stroke
  noise, or photo simulation.
- Downstream Rabbit Ear/base-computation validation is still recorded as skipped
  until the CLI setup is restored without colliding with Unix `fold`.
- The scraped real corpus is mostly `U` plus `B`, so M/V assignment quality
  still needs synthetic or manually labeled real-image benchmarks.
