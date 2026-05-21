# Crease Pattern Detector V2 Roadmap

## Goal State

V2 should produce structurally correct square origami crease-pattern FOLD graphs
for most readable inputs in the supported envelope.

The target is not a nicer line drawing. The target is:

- A canonical unit-square outer boundary.
- Every crease/border intersection represented as a graph vertex.
- The square boundary split at every boundary-contact vertex.
- No duplicate edges, zero-length edges, or unsplit crossings.
- Border edges fixed to `B`.
- Interior graph topology that is planar and exact enough for downstream origami
  tooling.
- M/V assignments that are observed when visually supported and inferred only
  when constraint solving makes them forced.
- Honest ambiguity or failure reports when visual evidence is insufficient.
- Rectification and crop confidence reported explicitly when the source square
  is uncertain or partially missing.

V2 should optimize for square origami crease patterns, not general wireframe
line extraction.

## What V1 Taught Us

The recent border experiments are the clearest V2 signal.

Current label-only border repair:

```text
border precision: 0.8482
border recall:    0.7949
border F1:        0.8207
```

Production square-border reconstruction with safety gates:

```text
border precision: 0.8471
border recall:    0.8259
border F1:        0.8364
```

Ground-truth oracle border chain:

```text
border precision: 1.0000
border recall:    1.0000
border F1:        1.0000
```

Conclusion:

- The evaluator and FOLD representation can score perfect borders.
- The current postprocess can recover only a small slice of the missing border
  topology.
- The gap is mostly missing or unstable boundary-contact geometry, not just bad
  B labels.
- Higher resolution may help dense creases, but retraining the same heads and
  decoder is unlikely to reach 0.99.
- Real-world failures also happen before topology decoding: text is detected as
  crease evidence, guide grids become false folds, dashed lines lose support,
  rectifier crops can remove unrecoverable border evidence, and naturally dense
  CPs can fall below the effective image scale.

The system needs to model the square-domain graph directly, but it also needs
better evidence hygiene and rectification confidence.

## V2 Architecture Principle

Use the known square paper boundary as a hard structural prior after
rectification.

For rectified square CPs, the border is not a detected object. It is the unit
square. Before rectification, however, the source panel still has to be found,
cropped conservatively, padded when needed, and reported honestly when uncertain
or damaged.

After rectification, the hard problem is identifying which crease lines,
junctions, and boundary-contact points belong in the graph while rejecting
non-crease visual lines.

V2 architecture:

```text
input image
  -> SquareRectifier-V2 + crop/confidence report
  -> rectified square image
  -> CPLineNet-V2 dense evidence
  -> artifact/non-crease evidence suppression
  -> SquareTopologyDecoder
  -> BoundaryChainBuilder
  -> EdgeAssignmentSolver
  -> FlatFoldabilityValidator
  -> FOLDWriter + report
```

CPLineNet should supply evidence. The decoder/solver should enforce the square
crease-pattern structure.

## Product Reliability Principles

The square prior is powerful, but it does not solve all V1 failures by itself.

- Rectification is a first-class component. V2 must measure whether the source
  square was found, whether crop margins preserved the full border, and whether
  homography confidence is high enough to trust unit-square decoding.
- Text, labels, watermarks, guide grids, page borders, and compression artifacts
  must be treated as hard negatives or separate non-crease evidence, not merely
  left for topology cleanup.
- Dashed or gapped crease styles need carrier-level support. Sampling every
  pixel along a segment as if the line is continuous will under-score valid
  dashed creases.
- Dense structures need an explicit scale policy, but not a synthetic
  degradation augmentation. V2 should learn dense CPs from the real/synthetic
  FOLD distribution, train at larger sizes, and know when minimum readable
  spacing requires higher-resolution or tiled reprocessing.
- Cropped-away borders and missing source regions are not recoverable visual
  evidence. V2 should detect/report these as unsupported or low-confidence
  inputs, not train a model to hallucinate the border that would have been
  there.
- Symmetry completion is out of scope for V2. It may become a V3 recovery mode,
  but V2 should not silently add mirrored creases or optimize on synthetic
  one-sided-miss examples.
- Flat-foldability should validate geometry and optionally solve assignments
  after strict gates. It should not add missing creases by default.

## V2 Metrics

Add metrics before major training changes. Border F1 alone is too blunt.

Required metrics:

- `boundary_contact_precision/recall/f1`: predicted crease endpoints on the
  square boundary, matched by side and side coordinate.
- `corner_recall`: all four square corners are present and fixed.
- `border_chain_precision/recall/f1`: B edges after deterministic boundary
  splitting.
- `interior_vertex_precision/recall`: excludes border vertices.
- `interior_edge_precision/recall`: excludes border edges.
- `carrier_recall`: whether each underlying straight crease line has a matching
  predicted line carrier before graph splitting.
- `dashed_carrier_recall`: carrier recall on dashed/gapped crease styles.
- `artifact_false_positive_rate`: predicted crease/carrier evidence overlapping
  known text, guide-grid, watermark, or page-border negatives.
- `rectifier_frame_iou`: detected source square/panel agreement when labels are
  available.
- `crop_margin_recall`: whether the full source border survives rectification
  with enough margin for decoding.
- `homography_error_px`: corner or frame reprojection error on labeled panels.
- `minimum_readable_spacing_px`: smallest predicted/GT spacing bucket per
  example, used to separate model errors from resolution-envelope failures.
- `unsplit_crossing_count`: predicted crossings that are not graph vertices.
- `flat_foldable_vertex_rate`: interior vertices passing local Kawasaki and
  Maekawa where assignments are known or inferred.
- `valid_fold_rate`: parseable, planar, complete-border FOLD outputs.
- `ambiguous_rate`: outputs that are structurally plausible but assignment
  ambiguous.

Keep the old aggregate score, but do not use it as the only promotion gate.
Every V2 candidate should report the issue-slice metrics separately.

## Phase V2.0: Issue Benchmark And Bottleneck Map

Goal: turn the observed V1 failures into a frozen synthetic benchmark and prove
which part of the system is limiting the 0.99 target before paying for
high-resolution training.

There are no labeled real-world CP issue examples yet. The first V2.0 benchmark
must therefore be synthetic: render known square CP graphs, inject the observed
failure modes, save the exact clean target graph, and produce visual contact
sheets for manual approval. Real-world examples can be added later as visual QA
or lightly labeled fixtures, but they must not be treated as labeled metrics
until annotations exist.

Build a V2 issue benchmark from generated square CP examples that exercise:

- Text/watermark false positives.
- Guide-grid false positives.
- Dashed/gapped crease styles.
- Faint or low-contrast crease evidence.
- Monochrome or visually ambiguous M/V assignments.

Do not include a `dense_scale_collapse` or low-resolution downsample
augmentation. Dense/tiny CPs are already present in the underlying `.fold`
distribution and should be handled through sampling, minimum-spacing buckets,
higher-resolution training, and supported-envelope reporting. The benchmark may
still include naturally dense examples as clean geometry cases, but it should
not train or score the model on deliberately destroyed pixel evidence.

Do not include cropped/missing borders or one-sided symmetry misses as positive
recovery slices. They are hallucination traps: if source pixels are gone, V2
should report `outside_supported_envelope`, `low_rectifier_confidence`, or
`incomplete_source_square` rather than learn to invent structure. Symmetry
completion should stay out of V2 and be reconsidered only as an explicit V3
hypothesis mode.

Run these ablations on the same held-out `.fold` validation slice:

1. Oracle rectification frame/crop, predicted evidence and topology.
2. Oracle artifact mask that suppresses text/grid/watermark lines.
3. Oracle boundary contacts, predicted carriers and interior graph.
4. Oracle line carriers, predicted boundary contacts/support/assignments.
5. Oracle junctions/intersections, predicted carriers and assignments.
6. Oracle topology, predicted M/V/B/U assignments.
7. Predicted topology, oracle assignments.
8. Oracle dashed/gapped carrier support on dashed-line examples.

Questions to answer:

- If rectification is perfect, how much real-world error remains?
- If text/grid artifacts are masked out, how much false geometry disappears?
- If boundary contacts are perfect, does border F1 approach 0.99?
- If line carriers are perfect, does interior edge recall approach 0.99?
- If dashed carrier support is perfect, do dashed examples stop failing?
- If topology is perfect, are M/V assignments already good enough?
- Are current failures mostly boundary contacts, naturally dense interior
  creases, artifact evidence, dashed support, faint evidence, or assignment
  ambiguity?

Exit criteria:

- A written bottleneck table for TreeMaker, Rabbit Ear, and real-world examples.
- A frozen synthetic V2 issue benchmark with per-issue metrics, manifest rows,
  clean references, issue images, oracle masks/targets where applicable, and
  contact sheets for visual inspection.
- Real-world issue examples remain a separate unlabeled visual-inspection queue
  until manual annotations are added.
- A ranked list of which model heads or decoder changes buy the largest metric
  gain.
- No high-resolution GPU run until this table exists.

## Phase V2.1: Square CP Ground-Truth Schema

Goal: generate labels for the graph structure we actually need.

From each FOLD example, derive:

- Vertex type:
  - `corner`
  - `boundary_contact`
  - `interior_intersection`
  - `interior_endpoint` if present
- Boundary side for every boundary vertex: `top/right/bottom/left`.
- Boundary side coordinate in `[0, 1]`.
- Crease carrier ids: maximal straight-line crease supports before splitting.
- Carrier-boundary intersections.
- Carrier-carrier intersections.
- Expected graph edges after splitting all carriers and the square boundary.
- Local sector angles around every interior vertex.
- Assignment constraints:
  - observed `M/V/B/U`
  - unknown/ambiguous
  - locally forced labels where applicable
- Render/evidence metadata:
  - dashed or gapped line style
  - guide-grid/text/watermark negative regions when synthetically generated or
    manually labeled
  - minimum edge length and closest-vertex spacing buckets
  - source-frame/crop labels for report-only rectification fixtures

This schema should be emitted as small deterministic sidecars next to training
records and eval fixtures. It should be inspectable without rendering images.

Exit criteria:

- Label-generation tests for corners, boundary contacts, dense near-boundary
  creases, diagonal creases, and multi-line intersections.
- Contact sheets that overlay vertex types and carrier ids.
- Distribution reports for boundary contacts per side, smallest side spacing,
  carrier count, and smallest interior edge length.
- Negative-evidence overlays for synthetic text, guide-grid, watermark, and page
  border artifacts.
- Dashed/gapped carrier labels that preserve the full intended crease carrier
  even when rendered pixels are discontinuous.
- Minimum-spacing buckets from naturally dense CPs, without generating
  low-resolution collapse augmentations.

## Phase V2.2: CPLineNet-V2 Evidence Heads

Goal: train the model to expose boundary and topology evidence explicitly.

Keep the current dense heads as useful evidence:

- `line_prob`
- `angle`
- `junction_heatmap`
- `junction_offset`
- `assignment_logits`

Add V2 heads:

- `vertex_type_logits`: corner, boundary contact, interior intersection,
  background.
- `boundary_contact_heatmap`: 2D heatmap focused near the square frame.
- `boundary_side_logits`: top/right/bottom/left for boundary-contact pixels.
- `boundary_offset`: subpixel offset to the exact boundary-contact coordinate.
- `boundary_side_coordinate`: side-conditioned 1D coordinate distribution for
  contacts along the square frame.
- `carrier_support`: optional line-carrier evidence for long straight crease
  supports.
- `line_style_logits`: solid/dashed/gapped/unknown crease style evidence.
- `non_crease_line_logits`: visual line evidence that should not become a CP
  crease, such as text, guide grids, watermarks, page borders, or UI chrome.
- `endpoint_tangent`: optional local crease direction at boundary contacts.

De-risking runs before high resolution:

1. Tiny local smoke at 384 or 512 just to validate label shapes and loss
   plumbing.
2. Short 1024 RunPod run on clean + line-style only.
3. Short 1024 RunPod run on the full V1 render profile mix.
4. Boundary-contact eval through the decoder, not just heatmap loss.
5. Hard-negative run on text/grid/watermark synthetic overlays.
6. Dashed/gapped-line run with carrier-level metrics.
7. Naturally dense sample run at the chosen input size, reported by
   minimum-spacing bucket rather than by an artificial downsample augmentation.

Exit criteria before high-res training:

- Boundary-contact recall >= 0.97 on clean/line-style at 1024.
- Boundary-contact recall >= 0.94 on print/photo/dark profiles at 1024.
- False boundary-contact rate low enough that deterministic border-chain
  reconstruction improves F1 without profile/family regressions.
- Contact errors are mostly localization noise, not missing whole sides or
  hallucinated side vertices.
- Artifact false-positive rate drops on the V2 issue benchmark without reducing
  clean crease recall.
- Dashed carrier recall is measured separately and improves over V1 support
  sampling.

Progress as of May 20, 2026:

- V2.0 synthetic issue benchmark tooling is in place for text, watermark, guide
  grid, dashed/gapped support, faint/low-contrast evidence, and visually
  ambiguous M/V. Cropped-border, symmetry-recovery, and dense-scale-collapse
  hallucination targets are intentionally excluded.
- V2.1 label sidecars now emit square-frame/vertex/contact/carrier metadata,
  artifact masks, dashed/faint/monochrome style metadata, observed-vs-latent
  assignments, and natural spacing stats. Label tests cover boundary contacts,
  carriers, artifacts, and ambiguous assignments.
- V2 render-time augmentations are wired into the training path as
  `v2-issue-mix` and `v2-dark-issue-mix`, with visual QA contact sheets under
  `visualizations/v2_augmentations/`.
- CPLineNet has optional V2 auxiliary heads for `non_crease_logits`,
  `line_style_logits`, `boundary_contact_heatmap`, `vertex_type_logits`,
  `boundary_side_logits`, `boundary_offset`, and `boundary_side_coordinate`.
  The smoke trainer can enable them with `--v2-heads`, `--non-crease-weight`,
  `--line-style-weight`, the boundary-head loss weights, and
  `--use-v2-observed-assignment`.
- Local CPU plumbing smokes completed at 128px for both light and dark V2 mixes
  without graph eval: light total loss `5.2911 -> 3.8934`; dark total loss
  `5.2109 -> 3.9599`. Prediction/target QA sheets were generated under
  `visualizations/v2_training_validation_smoke/`.
- Boundary-head local CPU plumbing smokes completed at 128px for both light and
  dark V2 mixes without graph eval: light total loss `5.1879 -> 4.2881`,
  validation boundary-contact loss `0.9281`; dark total loss `5.2147 ->
  4.3133`, validation boundary-contact loss `0.9454`. Target/prediction QA
  sheets were generated under `visualizations/v2_boundary_validation_smoke/`.
  The prediction columns are still diffuse after this tiny smoke and should not
  be treated as learned localization quality yet.
- This is only a wiring validation. No 1024 de-risking run, carrier-support
  head, decoder boundary-contact eval, or high-resolution training run has been
  launched yet.

## Phase V2.3: SquareTopologyDecoder

Goal: upgrade `PlanarGraphBuilder` into a square-aware graph decoder.

Decoder contract:

1. Start with the exact unit square and four fixed corners.
2. Extract candidate crease carriers from `line_prob`, `angle`, and optional
   `carrier_support`.
3. Add boundary-contact candidates from the boundary heads.
4. Suppress carriers that overlap high-confidence non-crease evidence unless
   there is strong contradictory crease evidence.
5. Intersect carriers analytically with:
   - other carriers
   - the square boundary
6. Snap predicted junctions to nearby analytic intersections.
7. Split every accepted carrier at every accepted vertex.
8. Split the square boundary at every accepted boundary-contact vertex.
9. Score every candidate segment by sampled line evidence, gap-tolerant dashed
   support, assignment evidence,
   geometric consistency, and local topology.
10. Output a planar graph with no unsplit crossings.

Important constraints:

- Do not infer the square frame from predicted border fragments once the input
  is rectified. The frame is known.
- Do not treat a border edge as a free line proposal.
- Do not connect arbitrary nearby points. Candidate edges must lie on an
  accepted crease carrier or on the square boundary.
- The boundary chain should be deterministic after boundary contacts are chosen.
- Symmetry completion is not part of the V2 decoder.

Exit criteria:

- With oracle boundary contacts and predicted carriers, border F1 >= 0.99.
- With predicted boundary contacts and oracle carriers, border F1 >= 0.97.
- With predicted contacts and predicted carriers, border F1 beats V1 by at
  least +0.10 absolute before high-res training.
- Unsplit crossing count is near zero on clean synthetic validation.
- Text/grid/watermark issue slices do not regress relative to V1.
- Dashed-line issue slices improve carrier recall without over-connecting
  unrelated broken evidence.

## Phase V2.4: Assignment And Flat-Foldability Solver

Goal: make flat-foldability a constrained solve, not a warning-only afterthought.

V2 should separate geometry validity from assignment certainty.

Assignment pipeline:

1. Sample model assignment logits along each graph edge.
2. Fix border edges to `B`.
3. Preserve confident observed M/V labels.
4. Mark uncertain labels as `U`.
5. Run a constrained MAP solver for optional completion:
   - Maekawa at interior vertices.
   - Kawasaki sector-angle residual thresholds.
   - model logits as unary costs.
   - label-change penalties for observed labels.
   - explicit ambiguity when multiple completions are plausible.
6. Report whether each assignment is `observed`, `inferred`, `ambiguous`, or
   `unknown`.

Do not force a globally plausible M/V pattern when the image gives no visual
evidence and local constraints do not make it unique.

Do not use flat-foldability to invent missing geometry by default. Kawasaki and
Maekawa are valid only after the topology is plausible; when geometry is wrong,
they should downgrade confidence or report inconsistency rather than add
creases.

Exit criteria:

- Complete-border, planar graphs are always exported even when M/V is ambiguous.
- On colored synthetic CPs, assignment accuracy >= 98%.
- On monochrome CPs, hallucinated M/V rate remains low; `U` or ambiguity is
  preferred.
- Inferred assignment mode remains separately flagged and can be disabled for
  conservative production outputs.
- Reports include ambiguity count or equivalent solver evidence when multiple
  label completions are plausible.
- Flat-foldability status is actionable:
  - `flat_foldable_observed`
  - `flat_foldable_inferred`
  - `assignment_ambiguous`
  - `geometry_invalid`
  - `outside_supported_envelope`

## Phase V2.5: Higher-Resolution Training

Goal: recover dense/tiny crease geometry that 1024px V1 cannot resolve.

Do this after V2.0-V2.3 prove the targets and decoder.

Recommended sequence:

1. 1024 V2-head baseline.
2. 1536 V2-head training on the same curriculum.
3. Tiled or multiscale inference experiment for dense regions where a global
   resize destroys spacing.
4. 2048 experiment only if 1536 or multiscale inference improves dense-tail
   recall without unacceptable
   runtime/memory cost.

Training strategy:

- Keep a mixed-resolution curriculum so the model does not overfit to one
  scale.
- Oversample dense Rabbit Ear/tiny-edge examples, but keep TreeMaker and simple
  bases in the mix to avoid forgetting.
- Track tiny-edge buckets by minimum edge length and closest-vertex spacing.
- Add adaptive reprocessing when the first pass reports spacing below the
  supported envelope.
- Consider patch/crop auxiliary losses for dense regions, but final inference
  must still reconcile one global square graph.
- Keep square rotations/reflections and render-style augmentations from V1, but
  do not add symmetry-completion or dense-scale-collapse targets.
- Add real-world render issues only when labels and eval exist:
  faint scans, page margins around an already-rectified square, compression,
  and dark backgrounds.
- Do not train CPLineNet-V2 on skew/perspective as a normal augmentation. The
  rectifier owns perspective correction; residual homography/skew should be
  measured as rectifier confidence/error and reported before square-domain
  decoding.

High-res promotion gates:

- Dense Rabbit Ear edge recall improves by >= +0.08 absolute over 1024 V2.
- Boundary-contact recall does not regress.
- TreeMaker clean/line-style metrics do not regress beyond the comparison gate.
- The system reports `outside_supported_envelope` when minimum readable spacing
  is below the selected resolution's proven limit.
- Runtime remains acceptable for the intended CLI/browser path, or high-res is
  exposed as a slower quality mode.

## Phase V2.6: Real-World Validation Loop

Goal: stop trusting synthetic-only metrics for production readiness.

Use the scraped and manually inspected real-world CP examples as a separate
track:

- Keep visual verification first.
- Add lightweight labels only where they answer a bottleneck question:
  - square frame
  - boundary contacts
  - obvious missing/extra creases
  - M/V availability
- Build issue tags:
  - cropped border or incomplete source square, reported as unrecoverable unless
    a human supplies missing structure
  - uncertain rectification
  - missing boundary contact
  - extra watermark/text/grid line
  - dashed/gapped crease
  - faint crease
  - dense tiny region
  - ambiguous M/V
  - non-square or non-CP input

Real-world exit criteria:

- Stage Inspector can show boundary-contact predictions, carrier proposals,
  final topology, and flat-foldability diagnostics.
- Manual review shows failures map to known report statuses rather than silent
  confident bad FOLD outputs.
- At least one curated real-world validation set is frozen before claiming V2
  production readiness.
- Rectification failures are distinguishable from topology failures in reports
  and overlays.

## Phase V2.7: Productization And Browser Path

Goal: make the V2 pipeline portable and inspectable.

Tasks:

- Keep the Python implementation as the reference.
- Keep decoder algorithms deterministic and portable to TypeScript or Rust/WASM.
- Export CPLineNet-V2 to ONNX after the Python path stabilizes.
- Add browser-side parity tests against Python outputs on frozen fixtures.
- Make debug overlays first-class:
  - boundary contacts
  - carriers
  - analytic intersections
  - rejected candidate segments
  - final FOLD graph
  - assignment/flat-foldability status

## Definition Of Done For V2

V2 is complete when a frozen validation suite meets:

- Border F1 >= 0.99 on supported synthetic square CPs.
- Rectifier frame/crop metrics meet the supported-input gate or the output is
  marked outside-envelope.
- Boundary-contact recall >= 0.99 on clean/line-style and >= 0.97 on
  print/photo/dark synthetic profiles.
- Artifact false-positive rate is acceptably low on text/grid/watermark issue
  slices.
- Dashed carrier recall meets a separate dashed-line gate.
- Interior edge F1 >= 0.96 overall, with dense-tail metrics reported
  separately.
- Valid FOLD rate >= 0.995 for supported synthetic inputs.
- Unsplit crossing rate near zero.
- Flat-foldability report present for every output.
- Assignment solver does not hallucinate M/V on visually ambiguous inputs.
- Real-world curated set has no silent catastrophic failures; failures are
  reported as ambiguous, outside-envelope, or failed with inspectable reasons.

## Near-Term Next Steps

1. Review the generated V2 augmentation and prediction/target contact sheets as
   the final visual QA gate for the current augmentation set.
2. Add the missing V2 structural heads needed for the square prior:
   boundary-contact heatmap/type/side targets first, then carrier support if
   contact metrics show carrier evidence is the next bottleneck.
3. Add Stage Inspector overlays for boundary contacts, non-crease/artifact
   evidence, line style, and observed-vs-latent assignment targets.
4. Run the first short 1024 de-risking run with V2 heads enabled on the
   approved issue mix. Compare artifact false-positive rate, dashed carrier
   recall, faint-line recall, and ambiguous-assignment behavior against V1.
5. Use the 1024 results to decide whether to add carrier-support heads,
   adjust augment weights, or proceed to the square topology decoder changes.
6. Only after 1024 graph metrics improve should we launch 1536/2048 or tiled
   inference experiments.
