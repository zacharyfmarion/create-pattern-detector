# Realistic Box-Pleat Production Data Plan

## Summary

The current synthetic BP generator is not production-grade. It creates strict, solver-valid crease graphs, but the visual distribution is still dominated by square grids plus isolated X/triangle cells. That is not how designer box-pleated CPs are constructed. Real BP CPs are organized around tree structure, flap/river layout, contour boundaries, stretch gadgets, ridge paths, staircase/fan motifs, and manual aesthetic cleanup.

This plan replaces the fake BP generator with a BP Studio-centered production data pipeline. BP Studio is MIT-licensed, actively maintained, and already implements the design abstractions we need: tree structures, flap/gadget layout, GOPS/Kamiya-style stretch math, layout optimization, and CP/FOLD export. We should use that code as the primary geometry teacher and keep Rabbit Ear strict validation as the final training-label gate.

The work is intentionally decomposed into parallel streams so multiple agents can work independently with narrow ownership.

## Grounding And Source Audit

BP Studio references:

- App/docs: https://bp-studio.github.io/
- Manual: https://bp-studio.github.io/manual.html
- Notes/bibliography: https://bp-studio.github.io/notes.html
- Source: https://github.com/bp-studio/box-pleating-studio

Important findings from source audit:

- License is MIT, so code reuse is viable with attribution.
- Current source version audited: `box-pleating-studio` `0.7.14`.
- Core geometry/layout code is mostly separable from Vue/Pixi UI:
  - `src/core/design/layout/*`
  - `src/core/design/layout/pattern/*`
  - `src/core/math/gops.ts`
  - `src/core/math/kamiya.ts`
  - `src/core/controller/layoutController.ts`
  - `src/core/service/state.ts`
- CP export is already implemented:
  - `src/core/controller/layoutController.ts` collects border, flap/rivers contours, ridges, and axis-parallel creases.
  - `src/client/plugins/cp/index.ts` exports ORIPA CP or FOLD.
  - `src/client/plugins/cp/fold.ts` maps BP Studio crease types to `B/M/V/F/U`.
- Tests are valuable executable documentation:
  - `test/specs/pattern/twoFlap.spec.ts`
  - `test/specs/pattern/threeFlap.spec.ts`
  - `test/specs/plugins/cp.spec.ts`
  - `test/specs/plugins/optimizer.spec.ts`
- BP Studio itself warns that CP export is a starting point rather than guaranteed flat-foldable output. We must keep strict Rabbit Ear validation and never silently accept non-strict outputs.

## Option Evaluation

### Option A: Continue hand-writing synthetic BP molecules

Pros:

- Fast to edit inside the existing generator.
- Full control over strictness and metadata.
- No third-party integration complexity.

Cons:

- We already saw the failure mode: local validity is easy, in-distribution semantics are hard.
- Recreating BP design theory piecemeal will take longer than integrating BP Studio.
- Visual QA will keep catching fake-looking structures late.

Verdict: use only for negative controls, ablations, and validation stress tests. Do not use as production training distribution.

### Option B: Drive BP Studio as a headless app

Pros:

- Fastest path to real BP Studio outputs.
- Minimal source surgery.
- Keeps behavior close to the designer-facing tool.
- Good for screenshots/contact sheets and export parity checks.

Cons:

- Browser automation can be slow and brittle.
- Harder to run large-scale generation.
- UI state and async worker flows can make deterministic generation harder.
- Harder to extract rich internal metadata.

Verdict: useful as an early oracle and regression comparison, but not ideal as the long-term generator.

### Option C: Vendor or submodule BP Studio and write a Node/Bun adapter around its core/test utilities

Pros:

- Reuses BP Studio’s actual tree/layout/stretch/pattern code.
- Avoids UI automation for scale.
- Tests show headless construction is plausible: BP Studio tests already call tree creation, stretch completion, and CP export without the UI.
- Gives us access to internal pattern metadata: repositories, configurations, devices, gadgets, contours.

Cons:

- BP Studio uses TS path aliases, global-ish state, and worker/client split. Integration will require build-system work.
- Some modules live under `client` even when logically export-related.
- Need to pin a commit and maintain compatibility.

Verdict: recommended main path.

### Option D: Port selected BP Studio core modules into our generator

Pros:

- Clean long-term API tailored to our dataset generator.
- Can remove UI/build dependencies and simplify state.
- Makes deterministic CI easier.

Cons:

- Risk of accidental semantic drift.
- Upfront port is larger than wrapping.
- We may port the wrong subset before learning what data works.

Verdict: do after the adapter proves distribution quality. Do not start here.

### Option E: Build a real/reference corpus and train mostly on real labels

Pros:

- Best distribution fidelity.
- Avoids synthetic-theory gaps.

Cons:

- Licensing and label quality are serious blockers.
- Most real CP images do not come with clean FOLD graph labels.
- Manual annotation is slow and expensive.

Verdict: use real references for calibration and visual QA now. Add licensed real labels later if available.

## Recommended Architecture

Use a staged hybrid:

1. Pin BP Studio as a third-party source dependency.
2. Build `tools/bp-studio-adapter/`, a headless TypeScript package that can:
   - create BP Studio projects from generated tree/layout specs;
   - run BP Studio's optimizer and stretch/GOPS metadata paths;
   - export scaffold CP/FOLD lines for debug/reference overlays only;
   - emit internal metadata for optimized flaps, tree edges, rivers, stretches, devices, gadgets, contours, and pattern choices.
3. Regularize the BP Studio tree/layout proposal into the restricted compiler contract:
   - one horizontal or vertical uniaxial spine in V1;
   - integer or half-grid flap/body/river regions;
   - explicit corridors, ports, parity, and rejection reasons.
4. Compile final training labels with our own certified molecule/fold-program compiler:
   - flap contours;
   - river and hinge corridors;
   - corner fans;
   - diagonal staircases;
   - diamond/chevron connectors;
   - GOPS/Elias-like stretch gadgets;
   - body panels.
5. Normalize the compiler output into this repo's canonical FOLD graph.
6. Run Rabbit Ear strict validation:
   - local Kawasaki/Maekawa;
   - Rabbit Ear layer solver;
   - finite folded coordinates;
   - no unsplit crossings;
   - no bad border/degenerate/duplicate edges.
7. Render and QA only accepted strict outputs.
8. Use real/BPStudio-like references only for calibration, distribution metrics, and human visual review unless explicit licensing permits training.

This gives us BP Studio's tree-aware layout realism first, with strict labels generated only by our compiler.

## Target Deliverables

- `tools/bp-studio-adapter/`
  - Pinned BP Studio dependency or submodule integration.
  - Headless CLI for tree/layout -> CP/FOLD export.
  - Adapter tests based on BP Studio two-flap, three-flap, CP export, and optimizer fixtures.
- `tools/synthetic-generator/`
  - Diagnostic family: `bp-studio-realistic`.
  - Production candidate family: `bp-studio-completed`.
  - New recipe: `recipes/synthetic/bp_studio_realistic_v1.yaml`.
  - New recipe: `recipes/synthetic/bp_completed_uniaxial_v1.yaml`.
  - Removal or demotion of current fake `realistic-box-pleat` from production recipes.
- `data/references/bp_clean_v1/`
  - Calibration-only manifest schema.
  - Scripts to render/compare reference images and generated images.
- QA:
  - Contact sheets grouped by archetype, edge count, gadget type, rejection reason, and reviewer status.
  - Distribution reports versus references.
  - Failed-attempt logs with structured reasons.
- Dataset:
  - Strict accepted FOLD files.
  - Rendered variants.
- Manifest rows containing BP Studio project metadata and strict validation metadata.
  - Manifest rows containing BP Studio project metadata, compiler metadata, molecule counts, and strict validation metadata.

## Implementation Status

Current checkpoint:

- BP Studio is pinned as `third_party/bp-studio` and wrapped by `tools/bp-studio-adapter/`.
- `bp-studio-realistic` exists as a diagnostic/calibration generator family with `recipes/synthetic/bp_studio_realistic_v1.yaml`.
- `bp-studio-completed` exists as the first restricted compiler-backed generator family with `recipes/synthetic/bp_completed_uniaxial_v1.yaml`.
- The old hand-written generator families, old recipes, and strict-completion fallback have been removed from the production synthetic path.
- The diagnostic generator samples BP Studio-style tree/layout specs, runs the BP Studio adapter, normalizes raw exports, and fails unless the raw BP Studio-derived graph passes local precheck and the normal strict validator.
- Raw-only generation remains blocked and should stay calibration-only: current raw smokes produce `0` accepted samples because the BP Studio export geometry is not locally strict after arrangement.
- The production candidate path now samples a tree/layout, runs the BP Studio adapter/optimizer, regularizes the optimized layout, emits a restricted certified fold program, and validates the compiler-generated FOLD graph.
- A current 8-sample strict smoke for `bp_completed_uniaxial_v1` produced `8/8` accepted samples after `12` attempts, `121-353` folded creases, `53-193` faces, and `100%` Rabbit Ear strict pass rate.
- Diagnostic tooling now exists: `cd tools/synthetic-generator && bun run bp-local-diagnostics -- --fold <export.fold>` reports bad vertices by BP Studio source kind, role, assignment, degree, folded degree, and auxiliary policy.

Important blocker discovered during implementation:

- Raw BP Studio CP exports are useful scaffold/calibration artifacts, but they can fail local Kawasaki/Maekawa after arrangement. They must not be treated as production labels.
- Source-aware diagnostics show the problem is not merely F/U auxiliary mapping. Representative preserved exports have thousands of failures dominated by active `device-draw-ridge`, `device-axis-parallel`, and `node-ridge` lines; some interior vertices have impossible odd active folded degree. Assignment solving over existing edges is therefore necessary but not sufficient.
- Greedy deletion or unassignment repair is explicitly not a production path. The accepted path is restricted compilation from regularized tree/layout plus certified molecules.
- See `implementations-plans/bp-studio-raw-export-rca.md` for the current RCA.

## Parallel Workstreams

These are designed for sub-agents with disjoint write scopes.

| Stream | Ownership | Primary Outputs | Can Run In Parallel With |
| --- | --- | --- | --- |
| A. BP Studio Adapter | `tools/bp-studio-adapter/` | Headless wrapper, project schema, CP/FOLD export | B, C, D, E |
| B. Reference Corpus And Metrics | `data/references/`, `scripts/data/`, `src/data/synthetic/` | Reference manifest, metrics, contact sheets | A, C, D |
| C. Tree And Layout Sampler | `tools/synthetic-generator/src/bp-studio-tree-*` | Archetype grammars, layout specs, optimizer requests | A, B, E |
| D. Strict Validation And QA | `tools/synthetic-generator/src/validate.ts`, folded preview and QA modules | Validation gates, visual warnings, failure taxonomy | A, B, C |
| H. Completion Contracts | `tools/synthetic-generator/src/bp-completion-contracts.ts` | `CompletionLayout`, `MoleculeTemplate`, `Port`, `PortJoin`, `CompletionResult` | A, C, D |
| I. Certified Molecule Compiler | `tools/synthetic-generator/src/bp-completion.ts`, tests | Molecule/fold-program templates, fixture families, composition rules | H after contracts |
| E. Rendering And Dataset Integration | `src/data/synthetic/`, `scripts/data/` | Manifest schema, renderer styles, loader smoke | A, C, D |
| F. Human QA Tooling | `scripts/data/`, optional small web/static viewer | Review workflow, accept/reject labels | B, E |
| G. Training Curriculum | `recipes/synthetic/`, training configs | Small/medium/dense/superdense schedules | B, C, D, E |

Sub-agents should not share write ownership. If a stream needs another stream's data contract, define an interface file first and then proceed independently.

## Phase 0: Integration Spike And Source Lock

Goal: prove BP Studio can be used headlessly before any large rewrite.

Parallel tasks:

- Agent A1: create a temporary adapter spike outside production paths that imports BP Studio test utilities and reproduces one two-flap and one three-flap fixture.
- Agent A2: audit BP Studio path aliases/build config and propose one dependency strategy:
  - git submodule under `third_party/bp-studio`;
  - pinned tarball downloaded by setup script;
  - vendored selected modules with MIT notice.
- Agent A3: audit legal/attribution requirements and add third-party notice plan.
- Agent D1: define the strict validation contract expected of BP Studio exports.

Recommended decision:

- Use a pinned git submodule or pinned source checkout first.
- Keep copied code out of the repo until adapter shape is stable.
- If submodule friction is high, vendor selected MIT modules later with a clear notice.

Exit criteria:

- One command can produce a BP Studio CP/FOLD from a known two-flap fixture.
- Output can be parsed by our FOLD parser.
- We know whether BP Studio F labels appear and how `useAuxiliary=false` changes them.
- Attribution requirements are documented.

## Phase 1: Calibration Corpus And Metrics

Goal: make "in distribution" measurable before generating at scale.

Parallel tasks:

- Agent B1: define `data/references/bp_clean_v1/manifest.jsonl` schema.
- Agent B2: gather a small local calibration set of permitted clean digital references, or placeholder manifests with source/license fields if images are not yet licensed.
- Agent B3: implement graph/raster metrics:
  - empty-space ratio;
  - local density variance;
  - connected diagonal run lengths;
  - staircase/fan motif counts;
  - long diagonal ridge segment counts;
  - contour-to-grid ratio;
  - degree histogram;
  - role ratios;
  - repeated-cell penalty;
  - full-sheet ruled-grid penalty.
- Agent F1: build contact sheets that group generated and reference images side by side.

Exit criteria:

- Current fake V3 scores visibly worse than references on motif and repetition metrics.
- Metrics produce stable JSON and contact sheets.
- Human reviewers can tag samples as `in_distribution`, `too_gridlike`, `bad_stretch`, `invalid_visual`, `unclear`.

## Phase 2: BP Studio Adapter V1

Goal: build a production-facing headless adapter around BP Studio core behavior.

Agent ownership:

- Agent A1 owns adapter package scaffolding:
  - `tools/bp-studio-adapter/package.json`
  - `tools/bp-studio-adapter/tsconfig.json`
  - path alias config
  - scripts
- Agent A2 owns project construction:
  - input schema for tree edges, flap dimensions, flap positions, sheet/grid type;
  - deterministic seed handling;
  - BP Studio state reset and tree creation.
- Agent A3 owns CP/FOLD export:
  - call `LayoutController.getCP(border, useAuxiliary=false)`;
  - convert CP lines to our canonical FOLD;
  - preserve BP Studio line roles where possible.
- Agent A4 owns fixture tests:
  - two-flap universal GPS;
  - double relay;
  - three-flap relay;
  - join/split patterns;
  - CP export expectations from BP Studio tests.

Adapter CLI:

```bash
bun run bpstudio-generate -- \
  --spec /tmp/spec.json \
  --out /tmp/out.fold \
  --metadata /tmp/out.metadata.json
```

Adapter output contract:

- `fold`: canonical-ish BP Studio export, not yet strict-guaranteed.
- `bpStudioMetadata`:
  - source commit;
  - tree edges;
  - flap positions/dimensions;
  - stretches;
  - repository/configuration/pattern counts;
  - device/gadget counts;
  - export options;
  - missing-pattern flags.

Exit criteria:

- Adapter passes BP Studio fixture parity tests.
- Adapter can export at least 50 deterministic fixture variations.
- No browser UI is required.

## Phase 3: Tree And Layout Sampler

Goal: generate realistic BP Studio input projects rather than drawing CP lines directly.

Parallel tasks:

- Agent C1: archetype tree grammars:
  - insects: body, 6 legs, antennae, abdomen/tail, optional wings;
  - quadrupeds: body, 4 legs, head/neck, tail, ears/horns;
  - birds: wings, tail fan, head/neck, legs;
  - dragons/complex fantasy: wings, legs, tail, horns, claws;
  - objects/abstract bases: handles, prongs, symmetric appendages.
- Agent C2: branch length and flap size sampler:
  - integer branch lengths;
  - width/height flaps for elevated/body regions;
  - complexity buckets with target leaf counts and sheet utilization.
- Agent C3: layout proposal generator:
  - symmetric initial layouts;
  - asymmetric variants;
  - boundary and corner-biased flap placements;
  - body-centered layouts;
  - reference-inspired terminal distributions.
- Agent C4: BP Studio optimizer integration:
  - build optimizer requests;
  - run random and view modes;
  - retain top-k layouts by utilization and aesthetic score.

Exit criteria:

- Produce hundreds of BP Studio project specs across archetypes.
- Layouts have non-overlap, reasonable margins, nontrivial flap clusters, and varied sheet utilization.
- Specs are deterministic by seed.

## Phase 4: Restricted Completion Compiler

Goal: turn sampled tree/layout specs and BP Studio optimizer proposals into compiler-generated strict CP labels.

Parallel tasks:

- Agent H1: maintain the shared compiler contracts:
  - `CompletionLayout`;
  - `MoleculeTemplate`;
  - `Port`;
  - `PortJoin`;
  - `CompletionResult`;
  - label provenance policy.
- Agent I1: implement certified molecule templates:
  - flap contours;
  - river corridors;
  - hinge corridors;
  - corner fans;
  - diagonal staircases;
  - diamond/chevron connectors;
  - GOPS/Elias-like stretch gadgets;
  - body panels.
- Agent I2: implement typed port composition:
  - reject incompatible orientation/width/parity joins;
  - fail completion when required ports cannot be joined;
  - preserve port checks and rejected-candidate counts in metadata.
- Agent C5: regularize BP Studio optimizer output:
  - map optimized adapter IDs back to sampler node IDs;
  - snap flaps/body/rivers to the compiler grid;
  - reject layouts that cannot fit the restricted uniaxial model.
- Agent E1: emit canonical compiler labels:
  - final M/V/B assignments come from templates/fold programs only;
  - raw BP Studio CP colors remain debug/reference metadata only;
  - compiler edge source is separate from BP Studio scaffold source.

Exit criteria:

- Fixture families compile to strict FOLD or fail with structured reasons.
- Port joins cannot be silently ignored.
- Raw BP Studio exports are never marked training eligible.
- Contact sheets show completed CP structures, not BP Studio scaffolds.

## Phase 5: Strict Validation, QA, And Acceptance

Goal: validate and promote only compiler-generated strict labels.

Parallel tasks:

- Agent D1: strict validation harness:
  - no missing borders;
  - no duplicate/degenerate edges;
  - all intersections split;
  - local Kawasaki/Maekawa;
  - Rabbit Ear solver pass;
  - finite folded coordinates;
  - assignment completeness.
- Agent D2: arrangement and normalization:
  - robustly split compiler output intersections;
  - dedupe collinear fragments without losing roles;
  - normalize coordinates to `[0, 1]`;
  - preserve BP Studio scaffold metadata separately.
- Agent D3: production rejection policy:
  - reject incompatible ports;
  - reject scaffold-only outputs;
  - reject odd active-degree interior vertices;
  - reject uniform diagonal wallpaper/full ruled grids;
  - never use greedy unassignment/deletion repair as production.
- Agent D4: solver performance:
  - timeout handling;
  - per-sample solver timing;
  - complexity caps by bucket;
  - deterministic retry policy.

Exit criteria:

- Accepted samples all pass strict validation.
- Rejected samples retain structured failure metadata.
- No validation downgrade exists in the production recipe.
- Every accepted row has `label_policy.trainingEligible=true` and `labelSource=compiler`.
- Folded preview contact sheets are generated for every smoke sample.

## Phase 6: Visual QA And Distribution Tuning

Goal: make production data quality a reviewable dataset property, not a vibe.

Parallel tasks:

- Agent B4: reference comparison report.
- Agent F2: review UI/contact sheets:
  - grouped by archetype;
  - grouped by motif/gadget type;
  - grouped by edge count;
  - sorted by realism score and rejection reason.
- Agent F3: reviewer label ingestion:
  - store reviewer status in QA JSON or sidecar JSONL;
  - prevent reviewed rejects from entering training.
- Agent C6: tuning loop:
  - adjust tree/layout sampler based on visual rejection reasons;
  - rebalance motif/archetype distributions.

Exit criteria:

- Human-reviewed acceptance rate for generated strict samples is at least 70% on a 100-sample smoke.
- No accepted production sample is dominated by uniform X cells, plain square grid, or repeated triangle wallpaper.
- Contact sheets show structures like long stair diagonals, nested fans, flap tips, body panels, river corridors, and stretch gadgets.

## Phase 7: Production Dataset Recipes

Goal: expose repeatable commands for training-ready data.

Recipes:

- `recipes/synthetic/bp_studio_realistic_small_v1.yaml`
- `recipes/synthetic/bp_studio_realistic_medium_v1.yaml`
- `recipes/synthetic/bp_studio_realistic_dense_v1.yaml`
- `recipes/synthetic/bp_studio_realistic_mixed_v1.yaml`

Parallel tasks:

- Agent G1: curriculum bucket definitions:
  - small: 4-8 flaps, 80-300 edges after arrangement;
  - medium: 8-16 flaps, 300-900 edges;
  - dense: 16-32 flaps, 900-2500 edges;
  - superdense: 32+ flaps, 2500-6000 edges, optional after solver performance is stable.
- Agent G2: split policy and manifest schema.
- Agent E3: renderer variants:
  - clean M/V/B colors;
  - monochrome;
  - faint blueprint;
  - scanned/noisy later, after clean digital distribution works.
- Agent G3: dataset loader smoke and training preview.

Exit criteria:

- One command can generate, render, validate, preview, and report a 256-sample dataset.
- Every manifest row loads through `SyntheticManifestDataset`.
- QA includes strict pass rate, visual review rate, archetype/motif distribution, edge/vertex/face distributions, and solver timings.

## Phase 8: Scale-Up And Training Feedback

Goal: close the loop between data quality and model performance.

Parallel tasks:

- Agent G4: generate 10k strict accepted samples with balanced archetypes.
- Agent G5: train pixel model on clean digital BP data.
- Agent G6: evaluate on held-out BP Studio-like references and any licensed real samples.
- Agent G7: analyze model failures:
  - missed long stair diagonals;
  - confused axis-parallel valleys;
  - bad dense intersections;
  - graph pruning mistakes around stretch gadgets.
- Agent C7/D5: feed failure modes back into sampler and validation/repair.

Exit criteria:

- Model performance improves on real/BPStudio-like reference images, not just synthetic validation.
- Distribution reports remain stable across large generations.
- Synthetic data includes enough hard cases to expose graph reconstruction errors before real deployment.

## Suggested Sub-Agent Tickets

These tickets are intentionally small enough to run in parallel.

1. `bp-adapter-build`: scaffold `tools/bp-studio-adapter` and import BP Studio modules from a pinned checkout.
2. `bp-fixture-parity`: port BP Studio two-flap/three-flap fixture tests into adapter tests.
3. `bp-export-normalizer`: convert BP Studio CP/FOLD lines into canonical FOLD with roles and metadata.
4. `bp-tree-grammar-insect`: implement insect tree grammar and deterministic seeds.
5. `bp-tree-grammar-quadruped-bird`: implement quadruped and bird grammars.
6. `bp-layout-optimizer`: wrap BP Studio optimizer bridge and expose deterministic layout proposals.
7. `bp-reference-metrics`: implement reference manifest, motif metrics, and distribution reports.
8. `bp-contact-review`: build grouped contact sheets and reviewer sidecar format.
9. `bp-strict-arrangement`: harden line arrangement for BP Studio exports.
10. `bp-validation-repair`: implement strict validation failure taxonomy and safe repairs.
11. `bp-render-manifest`: extend manifest/rendering for BP Studio metadata.
12. `bp-production-recipes`: add smoke and production recipes with curriculum buckets.
13. `bp-training-smoke`: run a small training/eval smoke and report model-visible failure modes.

## Risks And Mitigations

Risk: BP Studio exports are often not Rabbit Ear strict.

- Mitigation: treat BP Studio as a generator of high-quality candidates, not labels. Only strict-passing outputs enter training.

Risk: Strict validation rejects too much of the BP Studio distribution.

- Mitigation: start with simpler two/three-flap and moderate multi-flap archetypes; add safe repair; keep rejection metadata; tune sampler toward strict-compatible layouts.

Risk: BP Studio integration is brittle because of path aliases/global state.

- Mitigation: first wrap exactly what tests use; pin source; isolate adapter package; avoid broad porting until adapter is proven.

Risk: Visual metrics can be gamed.

- Mitigation: human review remains a gate for production recipe promotion; metrics are triage, not final truth.

Risk: Real references are unlicensed.

- Mitigation: keep them calibration-only with source/license metadata; do not train on them unless explicitly cleared.

Risk: This becomes too big for one agent.

- Mitigation: keep streams disjoint, define schemas early, and use adapter CLI contracts between streams.

## Definition Of Done For Production V1

Production V1 is done when:

- BP Studio adapter can generate CP/FOLD candidates from deterministic tree/layout specs.
- At least 1,000 generated samples pass strict local/global validation.
- At least 70% of a 100-sample reviewed strict smoke is marked in-distribution.
- Contact sheets show authentic BP structures: flap contours, long stair diagonals, nested fans, river corridors, stretch gadgets, body panels, and appendage clusters.
- Dataset rows load through `SyntheticManifestDataset`.
- Folded previews are finite for all smoke samples.
- Training on the dataset improves evaluation on BP Studio-like held-out references compared with the current fake dense/V3 generators.

## Current Implementation Status

Status as of 2026-05-15:

- The strict compiler has a `v0.7` clipped-terminal-fan baseline. Terminal fans keep BP-like orthogonal pleat axes, nested flap contour diamonds, and the body corridor diagonal, but no longer emit all eight sheet-wide rays.
- The previous auxiliary pleat-strip stars and bounded corridor diamonds were removed from active generation because they were not independently certified. They only passed when old global rays happened to pass through them, so they were scaffold-dependent rather than true local molecules.
- A 32-sample BP Studio-backed smoke passes strict generation, rendering, folded preview, and `SyntheticManifestDataset` loading, but the contact sheet remains too simple and repetitive for production. Crease counts are now in the 136-160 range for the current smoke.
- Next implementation work should certify new bounded density molecules in isolation: contour-completing terminal wrappers, bounded staircase cells, and bounded corridor/diamond connectors with explicit port joins. These should not be added to sampled generation until fixture, pairwise, and smoke tests pass without relying on sheet-wide rays.

## Non-Goals

- Do not use the current fake `realistic-box-pleat` as production data.
- Do not lower strict validation to improve acceptance rate.
- Do not train on copyrighted real CPs without license review.
- Do not clone the BP Studio UI experience; we only need its geometry/layout/export engine.
- Do not attempt arbitrary all-designer-level supercomplexity in the first production release. Start with authentic, strict, moderate BP designs and scale up.
