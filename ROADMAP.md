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
- Current Stage 5 flag: `--rectified` for readable CP images or page/screenshot
  images where the CP panel has a visible border. Later phases should remove the
  need for that hint on arbitrary photos/scans.

Outputs:

- `output.fold`: canonical FOLD JSON with vertices in `[0, 1] x [0, 1]`.
- `output.report.json`: confidence, validation results, known violations, and ambiguity warnings.
- Debug overlays: rectified image, line evidence, detected line hypotheses, final graph, assignment overlay.

Success means not just a visually plausible line drawing. It means a FOLD graph that downstream origami tooling can consume.

## Eventual Browser Target

The eventual user-facing target should be a browser-only app:

```text
upload image
  -> browser-side rectification and preprocessing
  -> CPLineNet exported to ONNX and run with WebGPU/WASM
  -> PlanarGraphBuilder in TypeScript or Rust/WASM
  -> EdgeAssignmentClassifier in browser
  -> OrigamiConstraintRepair in browser
  -> FOLDWriter
  -> download .fold + view validation/debug overlays
```

This should not be the first implementation target. The Python/CLI pipeline comes first because it is faster to debug, easier to profile, and better for model/geometry iteration.

Browser constraints to plan for:

- No image data should leave the user's machine.
- Model weights can be downloaded as static app assets, then cached by a PWA for offline use.
- Inference should prefer ONNX Runtime Web with WebGPU and fall back to WASM.
- Geometry code must avoid Python dependencies; keep the production algorithms portable to TypeScript or Rust/WASM.
- Debug overlays should be a first-class browser feature, not just a CLI artifact.

Browser success criteria:

- User can upload a CP image and download a `.fold` file without API calls.
- The app works offline after first load if model weights are cached.
- The browser output matches CLI output within fixed graph tolerances on regression fixtures.
- If local compute is too slow or unsupported, the UI reports that clearly instead of silently degrading quality.

## Exact Architecture

### 1. SquareRectifier

Purpose: convert the input into a canonical square image and preserve the inverse transform.

Implemented Stage 5 behavior:

- `--rectified` means the input is a readable CP image or page/screenshot with a
  visible CP panel. The rectifier no longer blindly pads the whole page first.
- It applies EXIF orientation, normalizes RGB/grayscale/RGBA inputs, composites
  transparency with an explicit or inferred alpha matte, and infers padding color
  from the image instead of assuming a white background. This preserves
  dark-mode crease-pattern support.
- It detects visible square/quadrilateral CP-panel borders using
  contour/long-line evidence, square/aspect scoring, interior crease density,
  and line coverage, then estimates a homography to a `1024 x 1024` canonical
  image.
- It maps the detected CP border to the Phase 3 training-style inset margin,
  currently 32 px at 1024, and fills the outside ring with the inferred padding
  color so thin border pixels remain detectable instead of being clipped at the
  exact image edge.
- If panel confidence is low, it falls back to resize/pad and records that
  lower-confidence transform instead of pretending a crop occurred.
- Phase 6 remains responsible for arbitrary real-photo/document rectification,
  faint/occluded panels, multiple competing panels, and benchmark-driven
  robustness.

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

Status: complete.

Goal: solve clean synthetic vectorization before training more ML.

Tasks:

- Implemented `PlanarGraphBuilder` against ground-truth rendered labels first.
- Built metrics for vertex recall/precision, edge recall/precision, assignment accuracy, FOLD validity, and downstream base-computation status.
- Added debug overlays, contact sheets, worst-case sheets, bucket summaries, and failure overlays.
- Tuned canonical tolerances for snapping, merging, support sampling, metric-edge canonicalization, and planar cleanup.

Exit criteria on clean synthetic labels:

- Vertex recall >= 99%: passed on the practical curated real gate slice.
- Edge recall >= 98%: passed on the practical curated real gate slice.
- FOLD validity >= 99%: passed on the practical curated real gate slice.
- Failures are explainable with saved overlays: passed.

Phase 2 was completed using clean rendered labels from the scraped real FOLD corpus, not augmented/noisy images. The practical curated gate excludes the intentionally high-stress telemetry tail and achieved:

- 52 non-stress curated real fixtures.
- Vertex precision/recall: 100.00% / 99.67%.
- Edge precision/recall: 98.82% / 98.61%.
- Assignment accuracy: 100.00%.
- Structural validity: 100.00%.

The full 582-file real stress run completed as non-blocking telemetry:

- Vertex precision/recall: 99.997% / 97.91%.
- Edge precision/recall: 93.60% / 85.80%.
- Assignment accuracy: 99.999%.
- Structural validity: 96.74%.

Saved artifacts:

- Curated gate: `visualizations/phase2_vectorizer/curated_gate_final_v2/`
- Full stress: `visualizations/phase2_vectorizer/full_stress_final/`

### Phase 3: CPLineNet Training

Goal: replace ground-truth labels with model-predicted fields.

Status: V1 complete for the supported readable-input envelope. See
`docs/phase-3-v1-status.md` for the completion decision, eval numbers, and the
dense/tiny geometry V2 caveat. The current blessed checkpoint is registered in
`artifacts/checkpoints/phase3-v1-cpline.json`; use
`docs/checkpoint-management.md` for checkpoint organization, checksums, and
replacement rules.

Tasks:

- Implement CPLineNet heads with separate geometry and assignment targets.
- Train first on clean renders, then progressively add CPLine-specific augmentations.
- Add render-time augmentation profiles before larger local/RunPod training:
  `square-symmetry`, `line-style`, `dark-mode`, `print-light`,
  `print-medium`, `photo-light`, and `photo-dark`.
- Keep augmentation vector-first: apply geometric perturbations to graph vertices,
  then render matching input images and dense CPLineNet labels from that geometry.
- Include exact non-identity square-domain rotations/flips: 90/180/270-degree
  rotation, horizontal/vertical flip, and both diagonal reflections. Preserve
  M/V labels because the rendered colors/line assignments move with the
  transformed graph.
- Keep dark-mode augmentation focused on background and crease-line palette
  variation. Background guide grids are out of scope for V1.
- Add visual augmentation contact sheets and JSON sidecars before scaling image size.
- Run local visual/performance gates for augmentations before paid GPU training.
- Add configurable augmentation mixes so curriculum stages can sample only the
  approved profiles for that stage instead of jumping straight to full `mixed`:
  `stage-base` and `stage-balanced`.
- Use the generated fold-only `raw-manifest.jsonl` contract for CPLine training:
  rows provide `foldPath` relative to the manifest root plus explicit
  train/val/test `split` values. The old Phase 2 `records[].path` fixture format
  remains only for deterministic vectorizer telemetry.
- Evaluate through the full vectorizer, not only pixel IoU.
- Cache predictions and graph-builder intermediate artifacts for reproducibility.

Current local finding:

- The 14k `cp_training_mix_v1` root now exists locally and is the default
  CPLine manifest. It contains 12k TreeMaker rows and 2k Rabbit Ear rows.
- Earlier 384px tiny-backbone staged MPS gates proved the architecture path on
  the mixed manifest. The current curriculum supersedes the old sequential
  light/print/dark schedule with a short `stage-base` warmup followed by
  `stage-balanced` training.
- The mixed-data tiny model still overproduces edges heavily, so local edge F1
  is low after these short gates. This is acceptable for architecture proof, not
  a quality result.
- Dark augmented validation remains the hardest active slice. RunPod runs should
  monitor predicted edge count versus ground truth on dark examples before
  enabling long full-`mixed` training.
- A 1024px `hrnet_w18` preflight with batch size 1 ran locally for two MPS
  steps on the mixed manifest. Loss moved from 3.694 to 2.543, proving the
  full-size path and memory shape, not model quality.
- A first 1024px RunPod `hrnet_w18` curriculum checkpoint was followed by a
  focused hard-negative line-loss continuation after grid-like dark backgrounds
  exposed false positives. This produced the current best Phase 3 V1 checkpoint
  artifact for readable crease patterns.
- Deterministic single-worker dark eval before removing grid augmentation showed
  that background guide grids create a distracting line-detection problem that is
  out of scope for V1. Treat multi-worker augmented eval numbers before `f639532`
  with caution because worker-copied RNG state could sample duplicated
  augmentation streams.
- Oracle graph extraction from perfect dense CPLine targets reaches about 98%
  edge recall and 99% vertex recall on the same clean and dark samples, so the
  remaining gap is model evidence quality, especially junction/line evidence
  under style variation, not an impossible graph-builder ceiling.
- Broad `mixed` continuation and a dark-mode-only high hard-negative pass both
  destabilized validation in this run. The next RunPod pass should use smaller
  segmented checkpoints, deterministic augmented eval, and a junction/recall
  recovery plan rather than pushing full `mixed` longer by default.
- V1 completion decision: proceed to Phase 4 with an explicit supported-input
  envelope. V1 targets readable, rectified 1024px CP inputs with resolvable line
  spacing across light/dark/print/photo-like render styles. Do not claim robust
  recovery for extreme dense tiny-fold geometry, dark guide-grid backgrounds, or
  partial occlusion.
- A local family/geometry diagnostic on the `max_edges <= 300` validation slice
  showed that the current clean 1024px checkpoint passes TreeMaker but not
  Rabbit Ear: 24-sample stratified eval was 95.7% edge recall for TreeMaker and
  86.1% for Rabbit Ear. The Rabbit Ear gap is not just dataset imbalance.
  Failures correlate strongly with tiny/close geometry: for Rabbit Ear,
  `tiny_edge_frac_lt8` has Spearman rho -0.84 against edge recall and
  `close_vertex_frac_lt8` has rho -0.87. Reserve that tail for a
  higher-resolution or scale-aware V2 stage instead of blocking Phase 4.

Augmentation curriculum before larger sizes:

1. `stage-base`: short warmup on clean, line-style, and exact square symmetries
   to establish geometry and assignment labels.
2. `stage-balanced`: main training with clean, line-style, print-light,
   print-medium, photo-light, dark-mode, and photo-dark samples all present
   together so dark is not treated as a final afterthought.
3. Targeted continuation only after deterministic eval identifies a specific
   weakness. Keep the other modes in the mix to avoid forgetting.

Run `stage-balanced` by initializing from the passing `stage-base` checkpoint
with `--init-checkpoint`; restarting every stage from random weights is only
useful as an extra stress test.

Do not add occlusion augmentation for V1. It requires an explicit completion
contract and confidence reporting so the model does not learn to hallucinate
hidden geometry.

V1 exit criteria on held-out synthetic renders:

- Clean edge recall >= 95% on the supported readable-input slice.
- Noisy synthetic evaluation is no longer a single blocking number; light,
  dark, print, and photo-like profiles are implemented and visually QAed, while
  real-photo benchmark collection remains Phase 6.
- Augmentation contact sheets are visually approved at 256 and 384 resolution.
- Dark-mode backgrounds render without guide grids for V1.
- Each curriculum stage passed local graph-eval or pixel/shape gates before
  moving to paid GPU training.
- Final valid FOLD rate is >= 90%; the current stratified clean eval had 100%
  structurally valid predicted FOLD graphs.
- Median vertex localization error is <= 2 px at 1024 resolution on the
  stratified clean evals.
- Dense tiny Rabbit Ear-style examples remain tracked as a V2 regression suite,
  not a V1 Phase 3 blocker.

### Phase 4: Edge Assignment And Constraint Repair

Goal: produce usable FOLD assignments and honest ambiguity reports.

Implementation handoff and reproduction commands live in
`docs/phase-4-stage4-handoff.md`.

Phase 4 principle: build the honesty layer before clever origami completion.
The system should be willing to say "geometry is outside the V1 envelope" or
"M/V is visually ambiguous" instead of silently inventing a confident FOLD.

Recommended work order:

1. **Edge assignment sampler and confidence.**
   Aggregate evidence per vectorized edge from CPLineNet `assignment_logits`,
   `line_prob`, the input image, and edge-support samples. Emit:
   - `assignment`: `M`, `V`, `B`, or `U`.
   - `assignment_confidence`: calibrated per-edge confidence.
   - `assignment_source`: `observed`, `unknown`, or later `inferred`.
   - `edge_support`: line-evidence support along the edge.
   - `assignment_margin`: gap between the top two assignment probabilities.
2. **Graph quality report.**
   Add a structured report object that classifies each output as `valid`,
   `repaired`, `ambiguous`, `outside_v1_envelope`, or `failed`. Include warnings
   for incomplete borders, weak/short edges, crowded junctions, low-confidence
   assignments, illegal crossings, duplicate/zero-length edges, and dense
   Rabbit Ear-style tiny geometry.
3. **Assignment eval suite.**
   Evaluate assignment separately from geometry on fixed synthetic fixtures:
   colored M/V, monochrome/no-color, dark-mode, print/photo-like, and
   geometry-correct but color-ambiguous examples. Monochrome examples should
   become `U` or low-confidence, not hallucinated M/V.
4. **Conservative graph repair.**
   Implement only repairs that do not invent origami semantics by default:
   dedupe edges, remove zero-length edges, snap or complete obvious border
   fragments, drop unsupported edges, and downgrade low-confidence M/V labels to
   `U`. Keep geometry drift explicit and minimal.
5. **Optional constrained completion.**
   Add M/V completion behind an explicit flag such as `--infer-assignments`.
   Mark inferred labels separately from observed labels and report ambiguity when
   multiple valid completions remain possible.

Implementation tasks:

- Train/evaluate edge assignment separately from geometry.
- Implement an edge-level assignment sampler over predicted logits and image
  evidence.
- Add per-edge confidence/source fields to predicted FOLD metadata.
- Implement graph validation and repair.
- Add a `report.json` contract for validation status, warnings, repair actions,
  confidence summaries, and V1-envelope checks.
- Add optional constrained M/V completion for unassigned or low-confidence edges.
- Report whether M/V labels are observed, inferred, ambiguous, or impossible.
- Preserve dense/tiny Rabbit Ear examples as a non-blocking V2 regression suite.

Exit criteria:

- Assignment accuracy >= 95% when visual M/V colors are present in synthetic renders.
- Monochrome or visually ambiguous CPs do not hallucinate M/V; they produce `U`,
  low confidence, or an ambiguity warning.
- Every evaluated graph produces a quality report with one of `valid`,
  `repaired`, `ambiguous`, `outside_v1_envelope`, or `failed`.
- Conservative repair improves validity without large geometry drift or invented
  M/V semantics.
- Dense/tiny Rabbit Ear-style cases trigger an out-of-envelope or low-confidence
  warning rather than silently producing a confident bad graph.
- Optional M/V completion is disabled by default and marks inferred labels
  distinctly when enabled.

### Phase 5: Production Inference CLI

Goal: one command from image to FOLD.

Status: implemented and updated. The Stage 5 implementation adds the Python
inference package and `cp-detect` CLI around the blessed Phase 3 V1 checkpoint
and Stage 4 assignment/repair/report/FOLD export stack. The supported mode is
still `--rectified`, but it now includes visible CP-panel isolation and
perspective warping for page/screenshot inputs; arbitrary real-photo discovery
and benchmarked robustness remain Phase 6.

Tasks:

- Implement `src/inference` pipeline using the exact component contract above.
- Add CLI command `cp-detect`.
- Support `cp-detect --rectified` for already-square readable CP inputs and
  page/screenshot inputs with a visible CP-panel border; full arbitrary photo
  rectification and benchmark-driven robustness remain Phase 6.
- Save debug artifacts and JSON reports.
- Add batch inference mode.
- Add regression tests on fixed images.
- Recover the blessed local checkpoint from the main checkout when needed:
  `/Users/zacharymarion/Documents/code/create-pattern-detector/checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt`.
- Verify the checkpoint against `artifacts/checkpoints/phase3-v1-cpline.json`
  before using it for inference or checkpoint-gated tests.
- Keep alpha handling dark-mode friendly: opaque images preserve their pixels;
  transparent inputs use `--alpha-matte auto|white|black`, defaulting to auto
  matte inference with report warnings when it must fall back.
- Crop titles, model diagrams, and surrounding page content out of detected CP
  panels before CPLineNet inference; record source quads and homographies in the
  report metadata.
- Preserve CP border pixels during panel warp by targeting a small inset border
  margin and recording both source and target quads in metadata.
- Make `failed` outputs explicit by writing reports/debug artifacts without
  silently writing invalid `.fold` files.

Exit criteria:

- `cp-detect examples/*.png` produces FOLD files and reports.
- Failures are explicit and diagnosable.
- No silent invalid FOLD output.

### Phase 6: Real Image Robustness

Goal: handle scans/photos without destroying synthetic performance.

Tasks:

- Build a small real benchmark of 25 to 100 manually corrected CP images.
- Add rectification tests for difficult real photos, faint borders, curved pages,
  multiple CP panels, and no-border examples that exceed the Stage 5 heuristic
  envelope.
- Fine-tune CPLineNet on mixed synthetic and real examples after Phase 3
  augmentation gates are stable.

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

### Phase 8: Browser-Only App

Goal: package the stable inference pipeline into a local-first browser experience.

Tasks:

- Export CPLineNet to ONNX and validate numerical parity against PyTorch.
- Build a static web app with upload, preview, rectification controls, debug overlays, and `.fold` download.
- Port `PlanarGraphBuilder`, `OrigamiConstraintRepair`, and `FOLDWriter` to TypeScript or Rust/WASM.
- Run inference with ONNX Runtime Web using WebGPU, with WASM fallback.
- Add PWA caching for model weights and app assets.
- Add a browser regression suite that compares CLI and browser outputs on fixed fixtures.

Exit criteria:

- No API calls are required for inference.
- Uploaded images stay local to the browser.
- Browser output matches CLI output within agreed tolerances.
- Offline PWA mode works after first load.

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
- A user can also upload an image in the browser and receive a FOLD file without API calls.
- The report states whether the graph is valid, repaired, ambiguous, or failed.
- Debug overlays make failures actionable.
- The system has fixed synthetic and real regression suites.
- Downstream origami-base computation succeeds on the intended benchmark set.
