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

Progress as of May 22, 2026:

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
- A 1024 V2 continuation candidate has been trained and pulled locally as
  `artifacts/checkpoints/runpod-v2-continuation-final-w4.json`. Against the
  Phase 3 baseline, it improved the synthetic V2 issue mix substantially:
  edge F1 `0.5548 -> 0.8269`, vertex F1 `0.6434 -> 0.8834`, border F1
  `0.5197 -> 0.8463`, structural validity `0.9688 -> 0.9844`, and mean
  vertex error `1.3308 -> 1.1161` px. On clean examples it slightly regressed:
  edge F1 `0.9318 -> 0.9140`, vertex F1 `0.9716 -> 0.9579`, border F1
  `0.8893 -> 0.8931`, assignment accuracy `0.9861 -> 0.9825`.
- The result is useful but incomplete. It proves V2 issue training improves
  robustness under synthetic artifacts, but it does not prove the architecture
  can reach structurally exact CP output because the production vectorizer still
  mostly used V1 evidence paths.
- The immediate implementation priority is now decoder wiring, not more
  training: make inference/eval consume V2 non-crease and boundary-contact
  evidence, then build the full square-aware topology decoder.
- The first local V2 evidence bridge is implemented for inference, checkpoint
  eval, training graph eval, and Stage Inspector recompute. It carries the V2
  auxiliary outputs and lets the existing graph builder suppress high-confidence
  non-crease pixels and merge boundary-contact heatmaps into junction
  candidates.
- A small local 1024 validation slice (`n=4/profile`) comparing Phase 3 to the
  V2 continuation with the bridge enabled showed strong issue-profile gains:
  aggregate edge F1 `0.6128 -> 0.8419`, vertex F1 `0.7152 -> 0.9039`, border
  F1 `0.6345 -> 0.8875`, and mean vertex error `1.4709 -> 1.2646` px.
  `v2-guide-grid`, `v2-dashed`, and `v2-combined` improved the most. Clean was
  effectively flat, but the older `line-style` profile regressed badly:
  edge F1 `0.8923 -> 0.6952`, border F1 `0.9027 -> 0.7203`.
- A targeted bridge ablation on selected examples showed that regression is
  mostly in the V2 checkpoint/curriculum rather than the bridge itself:
  disabling V2 auxiliary evidence still left the V2 model over-generating on
  `line-style`. The bridge is useful plumbing, but it is not yet the
  square-topology solution.
- A corrective replay curriculum was defined as `v2-replay-corrective`:
  40% `stage-balanced` replay, 25% extra `line-style`, 25% `v2-all-issue-mix`,
  and 10% extra `v2-combined`/`v2-dark-combined` stress. It starts from
  `runpod-v2-continuation-final-w4` with a fresh optimizer and lower learning
  rate, so it can preserve V2 issue robustness while restoring the old readable
  profile distribution.
- The corrective replay curriculum has now been trained on the full available
  synthetic FOLD dataset pack at 1024px:
  `artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json`.
  The run used 5,202 train examples, 638 validation examples, `max_edges=300`,
  batch size 1, 800 steps, and the full `v2-replay-corrective` mix. The
  training loader memory leak found during the interrupted RunPod attempts was
  fixed before the full run.
- The current reusable comparison snapshot is committed at
  `artifacts/evaluations/v2-checkpoint-comparison-20260522.json`, with a human
  summary in `docs/v2-checkpoint-metrics.md`. This means future checkpoint
  screens can compare against the stored Phase 3, V2 issue-only, partial replay,
  and full replay numbers without rerunning those older checkpoints.
- On a 1024px Stage 4 checkpoint screen (`n=16/profile`, profiles `clean`,
  `line-style`, `v2-all-issue-mix`, and `v2-dark-issue-mix`), the full replay
  checkpoint is the best balanced candidate so far:
  aggregate edge P/R `0.877/0.877`, border P/R/F1 `0.860/0.892/0.876`,
  assignment accuracy `0.987`, and structural validity `1.000`. It improves
  over the V2 issue-only checkpoint by `+0.114` edge precision, `+0.042` edge
  recall, and `+0.054` border F1.
- The replay run restored most of the old-profile `line-style` regression:
  border F1 moved from `0.721` on V2 issue-only to `0.862` on full replay.
  Clean border F1 is now `0.906`; V2 all-issue border F1 is `0.857`; V2 dark
  issue border F1 is `0.878`.
- Boundary-contact heads are not the main bottleneck on the current synthetic
  contact eval. The full replay checkpoint is around `0.94` contact F1 across
  clean, line-style, V2 issue, and V2 dark issue slices. The remaining gap to
  0.99 structural output is downstream topology: carrier selection, analytic
  intersections, deterministic boundary-chain splitting, artifact rejection,
  dashed support, and compile/flat-foldability gates.
- The full replay checkpoint should be treated as the current V2 candidate for
  Stage Inspector visual review and decoder work, not as a production-ready V2
  model. Many outputs are parseable and complete-border, but quality reports
  still often land in `ambiguous` or `outside_v1_envelope` because of short
  edges, crowded junctions, Kawasaki/Maekawa warnings, and assignment
  uncertainty.
- `SquareTopologyDecoder` is now implemented and wired into Stage 4 checkpoint
  eval, Stage Inspector recompute, and the production inference builder. It
  uses a fixed rectified square frame, fixed corners, border-carrier
  suppression, V2 non-crease/boundary-contact evidence, junction snapping to
  analytic carrier intersections, deterministic side-sorted border chains, and
  planar cleanup. It deliberately does not add every analytic carrier-carrier
  intersection by default, because that over-generated unreadable graphs.
- A square-topology Stage Inspector bundle was generated at
  `visualizations/v2_square_topology_inspector/eval` with 66 synthetic examples
  across clean, `line-style`, isolated V2 issue profiles, combined, dark
  combined, and replay-corrective profiles. Aggregate structural validity is
  `1.000`, edge P/R/F1 is `0.865/0.811/0.837`, and border P/R/F1 is
  `0.924/0.892/0.908`. Clean border F1 is `0.951`; `line-style` border F1 is
  `0.923`; combined and dark-combined remain lower at roughly `0.861` and
  `0.864`.
- Stage Inspector V2 evidence overlays are implemented for the current square
  decoder path: line evidence, non-crease/artifact evidence, line style,
  boundary contacts, assignment labels, and final topology can be toggled during
  review.
- Exact square-CP compile gates are now part of the quality report. In addition
  to parse/duplicate/zero/crossing checks, reports can flag missing square
  border, missing square corners, non-square border frame/edges, invalid border
  cycle, and boundary contacts that reach the square without splitting the
  border chain.
- The first square-topology review of the 66-example synthetic inspector bundle
  found no parseability failures, but still found square-compile warnings:
  `boundary_contact_not_split` on 20 examples, `invalid_border_cycle` on 9,
  `missing_square_corners` on 8, and `non_square_border_edges` on 1. These are
  concentrated in combined, dark-combined, and replay-corrective profiles and
  point to topology splitting/border-chain recovery rather than FOLD export.
- `SquareTopologyDecoder` now has conservative dashed-support scoring: dashed
  style evidence can rescue a gapped segment only when there is still line
  evidence beneath it. Faint style does not rescue segments by itself.
- Real-world Stage 5 visual review is still a separate queue in this worktree:
  `visualizations/stage5_scraped_inspector/eval` and the shared scraped-data
  symlink are not present here, so the completed review above is synthetic V2
  issue coverage only.

## Phase V2.3: SquareTopologyDecoder

Goal: upgrade `PlanarGraphBuilder` into a square-aware graph decoder.

Current implementation plan:

1. Done: add a V2 evidence adapter that carries optional `non_crease`,
   `line_style`, `boundary_contact`, `vertex_type`, `boundary_side`,
   `boundary_offset`, and `boundary_coord` predictions through inference,
   training graph eval, checkpoint eval, and Stage Inspector recompute.
2. Done: add a conservative V2 bridge inside `PlanarGraphBuilder`:
   - suppress high-confidence non-crease pixels before Hough/carrier extraction;
   - merge boundary-contact heatmap peaks into junction candidates;
   - keep existing repair/reporting as the final safety guard.
3. Done: re-run Phase 3 vs V2 checkpoint metrics with the V2 evidence bridge
   enabled, then train and compare a corrective replay checkpoint. The stored
   snapshot is `artifacts/evaluations/v2-checkpoint-comparison-20260522.json`.
4. Done: implement the first real `SquareTopologyDecoder`:
   - fixed unit-square frame and four corners;
   - accepted crease carriers only, with border-carrier suppression;
   - deterministic side-sorted boundary-contact chain;
   - artifact-aware carrier scoring through non-crease evidence;
   - gap-tolerant dashed support scoring;
   - analytic carrier/boundary intersections and graph splitting;
   - junction snapping to analytic intersections without adding all
     carrier-carrier intersections as vertices by default.
5. Done: add Stage Inspector overlays for V2 evidence so failures can be
   assigned to the model heads or the decoder rather than guessed from the final
   graph.
6. Done: visually inspect the square-topology Stage Inspector bundle and record
   current synthetic failure categories. The weakest slices are combined,
   dark-combined, and replay-corrective; the most actionable square-topology
   warnings are unsplit boundary contacts, invalid border cycles, and missing
   square corners after repair.
7. Done: add exact square-CP compile gates to the quality report.
8. Next: broaden visual review to cached real-image Stage 5 outputs once they
   exist in this worktree, then tune carrier selection/rejection and
   boundary-chain splitting on the examples that still show obvious missing
   creases or square-cycle warnings.

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

## V2 RunPod Philosophy

V2 paid GPU work should optimize for the least expensive GPU that can complete
the run reliably without making turnaround painfully slow. The default target
for 1024px HRNet W18 continuation runs is a 24GB-class GPU: RTX 4090 first,
then comparable 24GB cards such as RTX A5000 or L4 if the 4090 is unavailable.
Use larger cards such as L40S/A6000 only when the 24GB class fails a measured
memory or throughput gate. Do not use premium H100/H200/A100-class GPUs for V2
continuation or replay runs unless the roadmap explicitly records the measured
need and the user has approved the cost.

Dataset logistics should not drive GPU selection. The synthetic training mix is
small enough to re-upload when needed; re-uploading data is preferable to
renting an expensive GPU solely because a network volume already exists in that
region. Network volumes are useful for repeated cheap runs in the same region,
but convenience does not override GPU cost.

Every V2 RunPod run should follow this order:

1. Run local CPU/MPS smoke tests and objective/label checks.
2. Create a short-lived 24GB-class pod with an auto-stop.
3. Upload only the required dataset slice/checkpoint artifacts to ephemeral
   storage unless a cheap matching-region volume is already available.
4. Run a tiny CUDA preflight that verifies dataset links, checkpoint loading,
   one training step, CUDA memory, and expected loss keys.
5. Launch the short corrective or validation run, monitor early steps for memory
   and throughput, and stop early if graph metrics are not likely to improve.
6. Copy back checkpoints/logs/metrics, then stop or delete the pod promptly.

Budget intent: spend money where it buys information about graph quality, not
where it buys idle convenience or unused VRAM. For this V2 replay correction,
the appropriate run target is a 1024px, batch-size-1, 24GB-class GPU run from
the V2 checkpoint, followed by metric comparison against Phase 3 and the
issue-only V2 checkpoint.

## Near-Term Next Steps

1. Generate or restore the cached Stage 5 scraped inspector bundle for
   real-world examples, then review it with the same V2 evidence overlays. The
   synthetic inspector work is complete enough to classify failure types, but
   real-world visual review is blocked here until
   `visualizations/stage5_scraped_inspector/eval` exists.
2. Tune `SquareTopologyDecoder` on the remaining synthetic weak slices:
   combined, dark-combined, and replay-corrective. Prioritize carrier
   selection/rejection, boundary-contact snapping, deterministic border-chain
   splitting, rejected-segment diagnostics, and artifact overlap scoring.
3. Add first-class inspector overlays for rejected candidate segments and
   carrier proposals, so missing obvious lines can be separated from rejected
   evidence versus missing model evidence.
4. Run the stored checkpoint screen only for new candidate checkpoints, using
   `artifacts/evaluations/v2-checkpoint-comparison-20260522.json` as the cached
   baseline for older checkpoints.
5. Use the decoder results to decide whether carrier-support heads, dashed
   carrier heads, or 1536/2048 training should be next. Do not treat more 1024
   training as the main path to 0.99 border metrics unless the decoder has
   already consumed the V2 heads and compile gates.
