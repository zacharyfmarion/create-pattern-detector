# Crease Pattern Detector Roadmap

## Architecture Decision

The production system should be vector-first, not mask-first.

Current prototype:

```text
FOLD -> rendered image -> pixel segmentation/orientation/junction heads
     -> skeletonize mask -> trace graph -> GNN cleanup -> FOLD export
```

Locked production architecture:

```text
input image
  -> SquareRectifier
  -> CPLineNet
  -> PlanarGraphBuilder
  -> EdgeAssignmentClassifier
  -> OrigamiConstraintRepair
  -> FOLDWriter
  -> validation report + debug overlay
```

The learned model should detect visual evidence for lines, junctions, and assignments. The final graph should be built by deterministic geometry and validated as a planar FOLD graph.

## Self-Critique

The earlier architecture was directionally right but too generous to the current implementation.

- Pixel segmentation is useful supervision, but it is the wrong primary representation. A crease pattern is a planar straight-line graph. Skeletonizing a mask turns a geometric problem into fragile pixel topology.
- A GNN cleanup stage is appealing, but it can only choose among candidates. If skeletonization misses a crease or invents bad junctions, the GNN has no reliable way to recover a correct FOLD graph.
- End-to-end training is premature. The project needs deterministic contracts and graph-quality metrics before large GPU runs.
- M/V assignment should be separated from geometry. Some real crease pattern images do not encode mountain/valley visually, so assignment recovery can be ambiguous even when geometry is correct.
- Generic wireframe detectors are relevant but not sufficient. CPs need square-domain normalization, all-intersection graph construction, origami validation, and FOLD export.

Rejected main paths:

- Pure semantic segmentation + skeleton tracing as the production vectorizer.
- A transformer that directly emits all FOLD vertices/edges as a sequence.
- A GNN that serves as the primary source of graph correctness.
- Training on huge synthetic data before the generator and evaluator are trusted.

## Production Contract

The production command should eventually look like:

```bash
cp-detect input.png --output output.fold --report output.report.json --debug-dir debug/
```

Inputs:

- A photo, scan, or rendered crease pattern image.
- Optional flag for already-rectified square images.

Outputs:

- `output.fold`: canonical FOLD JSON with vertices in `[0, 1] x [0, 1]`.
- `output.report.json`: confidence, validation results, known violations, and ambiguity warnings.
- Debug overlays: rectified image, line evidence, detected line hypotheses, final graph, assignment overlay.

Success means not just a visually plausible line drawing. It means a FOLD graph that downstream origami tooling can consume.

## Exact Architecture

### 1. SquareRectifier

Purpose: convert the input into a canonical square image and preserve the inverse transform.

Initial implementation:

- If `--rectified` is passed, trust the input and resize/pad to canonical resolution.
- Otherwise detect the outer square using border/contour/Hough evidence.
- Estimate a homography to a `1024 x 1024` canonical image.
- Fail gracefully if corner confidence is low.

Outputs:

- `rectified_rgb`: `(1024, 1024, 3)`.
- `homography_image_to_square`.
- `rectification_confidence`.

### 2. CPLineNet

Purpose: predict dense evidence fields that support vectorization.

Use the existing HRNet code only as a starting backbone. Prefer `hrnet_w18` or another lightweight high-resolution backbone for the first serious model. The model is not a final graph predictor.

Inputs:

- Rectified RGB image at `1024 x 1024`.

Heads:

- `line_prob`: one channel, probability of any crease or border line.
- `angle`: two channels, bidirectional angle encoded as `(cos 2 theta, sin 2 theta)`.
- `junction_heatmap`: one channel for graph vertices, crossings, border intersections, and square corners.
- `junction_offset`: two channels for sub-pixel vertex refinement.
- `assignment_logits`: four channels for `M/V/B/U`, supervised only on line pixels.

Important separation:

- Pixel classes are `background + line evidence`.
- Edge assignment classes are `M/V/B/U`.
- Do not couple assignment class count to segmentation class count.

### 3. PlanarGraphBuilder

Purpose: turn dense evidence into an exact planar straight-line graph.

Algorithm:

1. Extract weighted line hypotheses from `line_prob` and `angle`.
2. Fit straight line segments with orientation-aware Hough/RANSAC or a DeepLSD-style line proposal backend.
3. Merge duplicate collinear hypotheses.
4. Intersect all supported lines with each other and with the square boundary.
5. Snap junction heatmap peaks to nearby analytic intersections.
6. Split lines at every accepted vertex.
7. Keep an edge only if sampled line evidence along the segment exceeds threshold.
8. Remove duplicate vertices/edges with fixed tolerances in canonical coordinates.
9. Return a planar graph, not a mask trace.

Outputs:

- `vertices_coords`: canonical `[0, 1]` coordinates.
- `edges_vertices`: undirected edges.
- `edge_support`: support/confidence per edge.
- `vertex_support`: support/confidence per vertex.

This stage is the heart of the project.

### 4. EdgeAssignmentClassifier

Purpose: classify the already-built graph edges.

Initial implementation:

- Sample image pixels, CPLineNet features, `assignment_logits`, and line evidence along each edge.
- Pool samples into a small edge feature vector.
- Predict `M/V/B/U` and confidence.

Rules:

- If M/V evidence is absent or uncertain, output `U`.
- Do not hallucinate mountain/valley labels solely to satisfy constraints unless explicitly requested.

### 5. OrigamiConstraintRepair

Purpose: validate and minimally repair the graph before export.

Checks:

- Vertices inside the square.
- Complete square boundary.
- No crossings except at graph vertices.
- Duplicate and near-zero-length edges removed.
- Interior even-degree check.
- Kawasaki angle check where applicable.
- Maekawa check only when M/V assignments are present.

Repair operations:

- Snap near-border vertices to border.
- Merge near-duplicate vertices.
- Split edges at missed intersections.
- Remove unsupported spurs.
- Optionally solve missing M/V labels as a constrained optimization problem, but report ambiguity.

### 6. FOLDWriter

Purpose: write a valid, minimal FOLD file.

Required fields:

- `file_spec`
- `file_creator`
- `file_classes`
- `frame_classes`
- `vertices_coords`
- `edges_vertices`
- `edges_assignment`

Optional but useful:

- `faces_vertices` after planar face construction.
- Custom confidence arrays in namespaced metadata.

## What To Keep From The Current Repo

Keep and refactor:

- `src/data/fold_parser.py`
- FOLD rendering and annotation generation concepts from `src/data/annotations.py`
- Parts of `src/data/transforms.py`
- HRNet wrapper as a possible CPLineNet backbone
- Validation scripts as references for future contract tests
- TypeScript synthetic generator as a starting point, but not as trusted final data

Park or replace:

- Skeleton tracing as the primary graph builder
- Dynamic graph extraction inside graph training
- Current graph head as production-critical architecture
- Current Tier S validation until the CLI collision with Unix `fold` is resolved

## Roadmap

### Phase 0: Repo Stabilization

Goal: make the repo runnable and measurable.

Tasks:

- Fix `.python-version` or document a working Python environment.
- Restore `data/ts-generation/package.json` and `tsconfig.json`.
- Add a single smoke command that parses FOLD, renders labels, vectorizes, exports FOLD, and validates.
- Unpack or regenerate a small checked-in fixture set outside ignored training data.
- Add `pytest` tests for FOLD parse/export and canonical coordinate transforms.

Exit criteria:

- Fresh setup can run smoke tests locally.
- No GPU required.

### Phase 1: Synthetic Data Generator V2

Goal: generate data that teaches the right task.

Tasks:

- Rewrite generator around explicit families: simple bases, single-vertex patterns, grids, tessellations, box-pleating-like structures, and real-style noisy renders.
- Track manifests with generator family, complexity, seed, validation tier, render style, and license/provenance.
- Fix global validation. Do not call Unix `/usr/bin/fold` by accident.
- Add distribution reports: vertices, edges, degree histogram, angle histogram, M/V balance, border intersections, and image style.
- Produce paired outputs: canonical FOLD, clean render, noisy render, dense labels, and vector labels.

Exit criteria:

- 10k high-quality training examples.
- 1k held-out validation examples.
- Complexity buckets are balanced and inspectable.

### Phase 2: Deterministic Vectorizer Baseline

Goal: solve clean synthetic vectorization before training more ML.

Tasks:

- Implement `PlanarGraphBuilder` against ground-truth rendered labels first.
- Build metrics for vertex recall/precision, edge recall/precision, assignment accuracy, FOLD validity, and downstream base-computation success.
- Add debug overlays for every failed fixture.
- Tune canonical tolerances for snapping, merging, and support sampling.

Exit criteria on clean synthetic labels:

- Vertex recall >= 99%.
- Edge recall >= 98%.
- FOLD validity >= 99%.
- Failures are explainable with saved overlays.

### Phase 3: CPLineNet Training

Goal: replace ground-truth labels with model-predicted fields.

Tasks:

- Implement CPLineNet heads with separate geometry and assignment targets.
- Train first on clean renders, then progressively add render noise.
- Evaluate through the full vectorizer, not only pixel IoU.
- Cache predictions and graph-builder intermediate artifacts for reproducibility.

Exit criteria on held-out synthetic renders:

- Clean edge recall >= 95%.
- Noisy synthetic edge recall >= 90%.
- Final valid FOLD rate >= 90%.
- Median vertex localization error <= 2 px at 1024 resolution.

### Phase 4: Edge Assignment And Constraint Repair

Goal: produce usable FOLD assignments and honest ambiguity reports.

Tasks:

- Train/evaluate edge assignment separately from geometry.
- Implement graph validation and repair.
- Add optional constrained M/V completion for unassigned or low-confidence edges.
- Report whether M/V labels are observed, inferred, ambiguous, or impossible.

Exit criteria:

- Assignment accuracy >= 95% when visual M/V colors are present in synthetic renders.
- Unknown/unassigned behavior is correct when M/V colors are absent.
- Constraint repair improves validity without large geometry drift.

### Phase 5: Production Inference CLI

Goal: one command from image to FOLD.

Tasks:

- Implement `src/inference` pipeline using the exact component contract above.
- Add CLI command `cp-detect`.
- Save debug artifacts and JSON reports.
- Add batch inference mode.
- Add regression tests on fixed images.

Exit criteria:

- `cp-detect examples/*.png` produces FOLD files and reports.
- Failures are explicit and diagnosable.
- No silent invalid FOLD output.

### Phase 6: Real Image Robustness

Goal: handle scans/photos without destroying synthetic performance.

Tasks:

- Build a small real benchmark of 25 to 100 manually corrected CP images.
- Add rectification tests.
- Add scanner/photo augmentations only after clean vectorization is stable.
- Fine-tune CPLineNet on mixed synthetic and real examples.

Exit criteria:

- Real benchmark valid FOLD rate improves release over release.
- Synthetic regression suite does not degrade.

### Phase 7: Optional Graph Refinement Model

Goal: use a graph model only after deterministic candidates are strong.

Tasks:

- Cache candidate graphs from PlanarGraphBuilder.
- Train an edge scorer/refiner on deterministic candidates.
- Compare against deterministic thresholds.

Exit criteria:

- Graph model improves final FOLD validity or edge F1 on held-out data.
- If it does not beat deterministic repair, keep it out of production.

## RunPod Budget Policy

Do not use paid GPU until Phases 0 through 2 pass locally.

Recommended order:

1. CPU/local smoke tests.
2. Tiny local overfit tests.
3. Short RunPod 24GB GPU run for CPLineNet.
4. Longer run only after the first short run improves full-pipeline graph metrics.

Budget guardrails:

- First serious training milestone target: under $50.
- Stop runs early if full-pipeline validation does not improve.
- Store datasets/checkpoints outside RunPod long-term storage.

## Production Definition Of Done

The project is production-ready when:

- A user can run one command on a CP image and receive a FOLD file.
- The report states whether the graph is valid, repaired, ambiguous, or failed.
- Debug overlays make failures actionable.
- The system has fixed synthetic and real regression suites.
- Downstream origami-base computation succeeds on the intended benchmark set.

