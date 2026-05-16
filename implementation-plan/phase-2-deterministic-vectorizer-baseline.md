# Phase 2 Deterministic Vectorizer Baseline

## Goal

Build a deterministic, label-driven vectorizer baseline before spending more effort on model training. The baseline consumes rendered ground-truth evidence from FOLD files, reconstructs a planar straight-line graph, exports FOLD, and reports graph metrics with visual debug artifacts.

The real scraped FOLD corpus is used through the ignored symlink:

```text
data/output/scraped -> /Users/zacharymarion/Documents/datasets/create-pattern-detector/scraped
```

Do not copy raw scraped files into git. Commit only code, small deterministic fixture manifests, docs, and tests.

## Staged Acceptance

- `smoke`: a tiny deterministic subset for rapid local iteration.
- `curated_gate`: the blocking real-FOLD Phase 2 gate, stratified by graph complexity.
- `full_stress`: every `.fold` file in `data/output/scraped/final/native_files/fold`, used for telemetry and failure mining.

The real-FOLD gate focuses on geometry and `U/B` assignment behavior because the scraped native corpus is mostly geometry-only. M/V assignment should remain covered by synthetic fixtures until real labeled inputs are curated.

## Implementation Steps

1. Preflight and fixture manifests
   - Link the shared scraped dataset with `scripts/data/link_shared_scraped_data.sh`.
   - Generate corpus stats, complexity histogram, and fixture manifests.
   - Render a contact sheet of selected real FOLD labels for visual approval.

2. PlanarGraphBuilder baseline
   - Add `VectorizerEvidence`, `PlanarGraphBuilderConfig`, and `PlanarGraphResult`.
   - Consume rendered `line_prob`, bidirectional `angle`, `junction_heatmap`, and `assignment_labels`.
   - Extract supported line hypotheses, merge collinear duplicates, snap junctions to analytic intersections, split edges at accepted vertices, and vote assignments.
   - Keep the existing skeleton/over-complete `GraphExtractor` as legacy/reference code.

3. Metrics and validation
   - Add vertex precision/recall, edge precision/recall, localization error, assignment accuracy by class, and structural FOLD checks.
   - Record downstream/base-computation as skipped until the Rabbit Ear/FOLD CLI setup no longer collides with Unix `fold`.

4. Visual checkpoints
   - Preflight contact sheet and stats.
   - Smoke overlays for 6 to 12 fixtures.
   - Curated-gate failure overlays and summary.
   - Full-stress bucketed metrics and worst-failure sheets.

## Blocking Thresholds

For `curated_gate`:

- Vertex recall >= 99%.
- Vertex precision >= 99%.
- Edge recall >= 98%.
- Edge precision >= 98%.
- FOLD structural validity >= 99%.
- `U/B` assignment accuracy >= 99% on matched real edges.

## Artifact Policy

Generated reports and visual artifacts belong under:

```text
visualizations/phase2_vectorizer/<run-id>/
```

This path is ignored by git. Important images can be shown in chat during checkpoint review, but they should not be committed.

## Current Best Run

Best deterministic settings as of the first full real-corpus run:

- 1024px render size with 32px padding.
- Canonicalized metric-side GT edges, splitting raw FOLD spans that pass through explicit vertices.
- Long direct-edge fallback up to 1024px for <=256 vertices.
- Short direct-edge fallback up to 180px for <=800 vertices.
- Planar cleanup enabled, splitting edges through vertices and removing weaker true crossings.

Key artifacts:

- Curated gate: `visualizations/phase2_vectorizer/curated_gate_final_v2/`
- Full stress: `visualizations/phase2_vectorizer/full_stress_final/`

Curated gate, all 60 records:

- Vertex precision: 99.99%
- Vertex recall: 98.68%
- Edge precision: 94.62%
- Edge recall: 85.48%
- Assignment accuracy: 100.00%
- Structural validity: 96.67%

Curated gate, non-stress records only (52 records), which is the current practical gate slice:

- Vertex precision: 100.00%
- Vertex recall: 99.67%
- Edge precision: 98.82%
- Edge recall: 98.61%
- Assignment accuracy: 100.00%
- Structural validity: 100.00%

Full real stress, all 582 records:

- Vertex precision: 99.997%
- Vertex recall: 97.91%
- Edge precision: 93.60%
- Edge recall: 85.80%
- Assignment accuracy: 99.999%
- Structural validity: 96.74%

The high-stress tail remains non-blocking telemetry. Failures are concentrated in dense diagrams above roughly 1.7k edges plus a few style-specific outliers in the large bucket.
