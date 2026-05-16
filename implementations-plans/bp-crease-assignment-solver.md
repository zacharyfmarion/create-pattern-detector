# Box-Pleat Crease Assignment Solver

## Purpose

This document is the working source of truth for how the BP Studio-guided compiler should assign M/V creases. Update it whenever the algorithm, contracts, or rejection policy changes.

BP Studio provides optimized tree/flap layout and scaffold metadata. It does **not** provide final strict training labels. Final M/V/B assignments must come from our own compiler and must pass local Kawasaki/Maekawa, Rabbit Ear global layer solving, and folded-coordinate QA before entering training.

## Core Idea

Do not assign M/V labels one crease at a time. Assign **region and molecule states**.

A molecule state is a locally meaningful BP pattern template plus its transform and phase:

- geometry: local vertices/segments on the BP grid;
- assignments: M/V/B emitted by the molecule template;
- roles: ridge, hinge, axis, stretch, border;
- ports: typed connection boundaries with orientation, width, parity, phase, and exposed assignment sequence.

Example corridor port:

```text
orientation: horizontal
width: 5 lanes
parity: integer
phase: 0
sequence: V M V M V
```

The same corridor with phase 1 exposes:

```text
sequence: M V M V M
```

Two regions join only when their ports are compatible. If ports disagree, the solver must try a different molecule state, flip a phase, insert a connector, or backtrack. It must not "fix" conflicts by recoloring arbitrary individual creases.

## Pipeline

```text
BP Studio tree + optimized packing
  -> macro region extraction
  -> candidate molecule states per region
  -> port constraint graph
  -> backtracking search over states/phases/connectors
  -> arranged full crease graph
  -> local + global strict validation
  -> accepted FOLD or structured rejection
```

## Macro Region Graph

The regularized BP Studio layout should become a graph of macro regions:

- terminal fan regions around flap endpoints;
- river/corridor bands along tree edges;
- body/hub panels where several tree edges meet;
- staircase or turn regions where corridors offset;
- stretch/gadget regions where valid BP layout contacts/overlaps force extra material;
- empty-space regions that should remain mostly uncreased.

Edges in this graph are **ports**. A port records the boundary between two regions and the crease sequence that must be reconciled there.

## Search Variables

Each macro region becomes a variable. Its domain is the set of certified molecule states that can fill that region:

- template kind, such as straight corridor, terminal fan, body hub, staircase, chevron connector, or stretch gadget;
- rotation and mirror transform;
- grid scale and dimensions;
- phase of alternating pleats;
- optional connector choice, such as phase shifter or chevron bridge.

The solver assigns one state to each region. A final assignment is valid only if every adjacent pair of assigned regions has compatible ports.

## Constraint Checks

Port compatibility must check at least:

- same physical boundary location after transform;
- compatible orientation and side;
- compatible width and lane count;
- compatible integer/half-grid parity;
- compatible M/V sequence or an inserted phase-shift connector;
- no dangling active crease endpoints except at sheet border or declared closed boundaries;
- no overlap between placed molecule interiors unless an explicit stretch/gadget region owns the overlap;
- no overlap between flap allocation circles/territories. BP Studio-style packing treats overlapping flap allocations as invalid; if two required regions collide, the layout must be optimized/regularized differently or rejected.

These checks happen before full FOLD arrangement so bad layouts fail cheaply.

Current implementation detail:

- `bp-studio-packing-validity` extracts optimized flap centers from BP Studio metadata and uses the corresponding tree-edge length as the flap allocation circle radius.
- Pairwise circle overlap is a hard error because it means the BP Studio-style flap territories collide.
- Circle overflow beyond the sheet boundary is reported as a warning for now, not a production hard gate, because BP Studio can place zero-area terminal points on the sheet boundary while showing the relevant in-sheet arc. If we later confirm that full-circle containment is required for the specific production scaffold, promote this warning to a hard rejection and update this document.
- The compiler lattice for optimized layouts is derived from BP Studio's optimized sheet dimensions. A `7 x 7` optimized sheet uses a `28` compiler grid by default, which preserves BP Studio integer coordinates and gives four compiler lanes per BP Studio cell. Do not fall back to a standalone `32/128` lattice for optimized BP Studio layouts, because that makes circles, corridors, and debug grids visually and geometrically disagree.
- BP Studio final contour bounds are scaffold/region metadata, not optimized internal-node coordinates. Do not use the center of a contour bounding box as a body/hub point. The debug packing panel must show only optimized terminal points and tree-length allocation circles as BP Studio source truth; inferred internal points may be used only as candidate-construction internals and must be gated by circle/region overlap checks.
- Optimized terminal final contour bounds are now carried through `CompletionTerminal.sourceContour` and `FlapRegion.sourceContourRect` as source metadata. They are clamped to the sheet and snapped to the BP Studio-derived compiler grid. This metadata is a target for future bounded terminal/fan closure, not active crease geometry by itself; do not add contour boundaries to the active graph unless a certified molecule state owns that boundary.
- Candidate body panels and corridor strips must not overlap the interior of any flap allocation circle. Touching a circle boundary is allowed; invading the circle interior is a structured rejection because it steals paper allocated to that flap.
- A tree edge is not always representable as one straight corridor. When the BP Studio terminal lane does not pass through the inferred body panel, the region compiler now emits a routed corridor made of a terminal-facing strip plus a body-approach strip. Body panels are expanded enough to expose separate side ports for multiple incident corridors, and those ports are clamped away from flap allocation circles so corridor thickness does not invade flap territory. Small overlaps between routed strips that share an endpoint are treated as branch/hub manifold overlap in the candidate scaffold; arbitrary unrelated strip overlaps remain rejections.
- Fixture and BP-sourced body panels are reserved regions. A corridor may enter its endpoint body, but it must not run through a non-endpoint sourced body panel. Compiler-inferred body panels are still drawn and reported, but non-endpoint overlaps against them are warnings until the adapter exposes reliable source body/hub contours for the active root body.
- Pleat strips for unrelated tree edges may only overlap inside a body-owned manifold where both strips touch that body. Small overlaps outside the body panel are no longer treated as harmless just because the strips share a hub endpoint; they are structured `pleat-strip-overlap` rejections and must be solved by route selection or an explicit connector molecule.
- Body panels reserve one extra corridor-width of spacing for same-side ports so adjacent pleat strips are not packed with guaranteed overlap. Expanded body panels are shifted away from flap allocation circles when possible, and same-side ports that collapse onto one coordinate after obstacle clamping are separated again. If a corridor or body still invades a flap allocation circle, the candidate remains a structured rejection rather than being repaired visually.
- Corridor lanes are now rerouted against the expanded region view by treating BP Studio flap allocation circles as the highest-priority source truth. A lane may be shifted to the nearest outside/tangent grid line so its corridor strip touches but does not invade a flap circle. Inferred body panels no longer force a route to invade a flap allocation circle just to avoid an inferred body obstacle.
- If a rerouted corridor core no longer physically touches its body panel, the region compiler adds a same-axis `body-connector` strip from the body edge to the corridor core. This keeps the scaffold connected instead of showing floating corridor fragments. These connectors are still incomplete hub-side scaffold pieces until a certified body/hub molecule owns their active endpoints.
- Inferred body-panel overlaps are now warnings, not hard source-truth rejections, because BP Studio does not currently expose a reliable source contour for the active root body/hub in this adapter path. This keeps BP Studio flap allocation circles and final contours as the hard geometry source while making compiler-inferred body problems visible for tuning. BP-sourced or fixture body overlaps remain hard rejections.
- Region candidates now emit pleat lines along the BP Studio-derived route axis for each corridor/approach strip. The earlier rectangle-long-axis heuristic made short terminal pieces flip direction and overlap unrelated routes; route-axis pleats keep active lanes aligned with the intended tree edge even when the rectangle is square or short.
- Corridor strip width is derived from the BP Studio/tree edge length on the optimized sheet, not from a fixed visual lane width. For example, on a `7 x 7` optimized simple-animal layout, a length-1 leg/head corridor is `1/7` of the sheet and the length-3 tail corridor is `3/7`. This is a correctness rule, not a style setting: using a skinny fixed strip changes the represented flap length and produces scaffold-like CPs with too few long alternating pleats.
- A new source-line fold-program checkpoint can take BP Studio's final scaffold line families, sort them by source role, and apply them as deterministic Rabbit Ear `flatFold` operations. This produces a strict local/global FOLD graph from BP Studio source geometry and replaces the sparse waterbomb-like generator output as the current e2e checkpoint. It is still not the final bounded molecule compiler: it can introduce fold-program completion creases beyond the exact raw export, and it does not yet prove a TreeMaker-style terminal-distance base. Keep using it as a strict source-derived checkpoint while terminal fans, hub joins, and stretch molecules are completed.
- The previous deterministic stair-boundary overlay has been removed from active region candidates. It was not a certified connector molecule in this context and produced misleading Kawasaki failures. Future diagonal staircases, terminal fans, and hub/turn closures must enter as certified molecule states with port contracts, not as decorative caps.
- Region candidates now add two conservative bounded lane-continuation passes before turn closure:
  - `hub-closure` fills only collinear pleat-lane gaps whose entire interval is already inside the BP Studio-derived body/strip manifold. It does not create new lane endpoints and is intended to represent a pleat lane continuing through a body/hub panel.
  - `terminal-closure` extends a degree-1 pleat endpoint to the sheet border only when the endpoint belongs to that strip's terminal and the whole extension segment remains inside the relevant BP Studio terminal source zone: either the tree-length allocation circle or the snapped final-contour bounds. This represents a bounded terminal fan/border continuation, not a sheet-wide sweep, and large neighboring flap circles must not claim unrelated strip endpoints.
- Region candidates add a bounded turn-closure pass for active Kawasaki failures caused by simple elbows and T-junctions: perpendicular elbows receive a two-diagonal fan on the opposite side of the turn, and three-way T junctions receive the missing fourth cardinal spoke. The pass now searches for a nearby safe fan endpoint and skips candidates that land on or cross existing active creases; if no safe fan exists, the candidate remains locally incomplete until a certified connector molecule owns that junction.
- Region candidate JSON includes an active local probe with `kawasakiBad`, `maekawaBad`, and `failureReasons`. This is a scaffold/completion diagnostic only; `layout-valid` means the BP Studio-derived regions do not violate packing/overlap gates, not that the active crease content is finished or locally flat-foldable.

## Backtracking Strategy

Use constraint search rather than a fixed left-to-right placement order.

1. Build domains for all regions from geometry and role requirements.
2. Choose the next unassigned region using **minimum remaining values**: assign the region with the fewest currently valid states first.
3. Prefer high-degree regions first when tied: body hubs and stretch gadgets constrain more neighbors than straight corridors.
4. After assigning a region, propagate constraints to neighboring domains by filtering incompatible port states.
5. If a domain becomes empty, backtrack.
6. If all regions are assigned, arrange the full graph and run strict validation.
7. If strict validation fails, record the failure and backtrack to the newest decision that can plausibly affect it.

Suggested region priority when domains are similar:

```text
body hubs / stretch gadgets
  -> long corridors
  -> terminal fans
  -> turns / staircases
  -> phase shifters / boundary closures
```

## Reconciliation Options

When two ports disagree, try reconciliation in this order:

1. Flip the phase of an unconstrained corridor or corridor chain.
2. Use a mirrored/rotated variant of the adjacent terminal fan or hub.
3. Swap to an alternate body-hub variant with the required exposed sequence.
4. Insert a certified phase-shift, chevron, diamond, or staircase connector.
5. Swap stretch-gadget variant if the conflict is inside an overlap region.
6. Backtrack to an earlier region decision.
7. Reject the regularized BP Studio layout with a structured reason.

Never repair production outputs by deleting creases, downgrading M/V to U, or silently ignoring an incompatible port.

## Validation Boundary

Intermediate region assignments may be incomplete and may not be globally flat-foldable. That is acceptable.

The final arranged graph must pass:

- complete sheet border;
- no duplicate, degenerate, or unsplit crossing geometry;
- no invalid dangling active endpoints;
- local Kawasaki/Maekawa;
- Rabbit Ear global layer solver;
- finite folded coordinates;
- visual QA showing finished BP structure, not scaffold-only geometry.

## Implementation Notes

The current code already has early concepts such as `Port`, `BoundaryPort`, `phase`, and `startAssignment`, but the current compiler mostly emits deterministic alternating strips. The next implementation step is to introduce an explicit region-variable solver over molecule states and port phases.

Current certified primitive checkpoints:

- `sheet pleat primitive`: a sheet-spanning accordion field with alternating M/V parallel lines. It is strict-flat-foldable and useful as a reference for long pleat runs, but it is not a finished BP layout because the strips terminate only on the sheet border.
- `region sheet-sweep lab`: takes a BP Studio-derived region candidate, extends each long-axis corridor pleat lane to the sheet border, then runs deterministic Maekawa assignment solving and Rabbit Ear strict validation. This proves the corrected long-axis lane field can be made globally flat-foldable, but it is lab-only and `trainingEligible=false` because sheet-wide extension can invade flap allocation territories and does not solve bounded terminal/hub closures.
- `simple quadruped line-field checkpoint`: the BP Studio optimized six-flap animal fixture (`simpleQuadrupedBPStudioSpec(7)`, optimizer layout `view`) now has a regression test proving the line-field probe passes complete-border, edge geometry, local flat-foldability, Rabbit Ear global solver, and finite folded-preview generation. This is the current strict proof target for the same layout shown in route-debug visualizations. It remains `trainingEligible=false`; the bounded compiler still has unresolved terminal/hub closures before this can become production data.
- `diagonal staircase cap primitive`: a 45-degree ridge with alternating axis/hinge partner creases at every pleat endpoint. All four sheet-corner orientations pass local Kawasaki/Maekawa, Rabbit Ear global solving, and folded-coordinate preview.
- `staircase bridge primitive`: two diagonal caps are composed as geometry first, then a deterministic Maekawa assignment solve chooses final M/V labels for the completed molecule. This passes local Kawasaki/Maekawa, Rabbit Ear global solving, and folded-coordinate preview for the certified small bridge. It is lab-only and must not be treated as production BP data because the whole-sheet cap overlay visually/semantically resembles overlapping flap allocation territories, which BP Studio packing would reject.

Important composition finding:

- A staircase cap is not freely composable with another cap. Naively overlaying two cap orientations with fixed template colors creates locally invalid interior grid intersections. Cap-to-cap and cap-to-corridor joins therefore require an explicit connector/hub/stretch molecule, a molecule-local final assignment solve, or a solver rejection. Do not combine cap primitives by geometric union and assume final flat-foldability.
- Molecule templates may contain provisional color preferences when they are being composed. Final training labels must come from a certified molecule state or fixture-level assignment solve, not from arbitrary post-hoc repair. In practice this means the solver can assign M/V after arranging a bounded molecule composition, but the resulting assignment becomes part of the certified molecule/state before production generation can use it.
- Final flat-foldability is not enough for production. A solved fixture that violates BP Studio layout semantics, such as overlapping flap allocation circles/territories, remains a lab artifact or rejection case even if Rabbit Ear can fold it.

When implementing, keep the solver deterministic by seed:

- stable region ordering;
- stable state ordering;
- bounded attempt count;
- structured rejection codes for exhausted search, incompatible ports, failed arrangement, local failure, global solver failure, and visual-distribution failure.

## Context Reset Protocol

After context compaction or when a new agent resumes BP compiler work, read these files before editing:

1. `AGENTS.md`
2. `implementations-plans/realistic-box-pleat-production-data.md`
3. `implementations-plans/bp-studio-raw-export-rca.md`
4. `implementations-plans/bp-crease-assignment-solver.md`

If the implementation diverges from this algorithm, update this document in the same change.
