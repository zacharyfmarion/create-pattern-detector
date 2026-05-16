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
- Candidate body panels and corridor strips must not overlap the interior of any flap allocation circle. Touching a circle boundary is allowed; invading the circle interior is a structured rejection because it steals paper allocated to that flap.
- A tree edge is not always representable as one straight corridor. When the BP Studio terminal lane does not pass through the inferred body panel, the region compiler now emits a routed corridor made of a terminal-facing strip plus a body-approach strip. Body panels are expanded enough to expose separate side ports for multiple incident corridors, and those ports are clamped away from flap allocation circles so corridor thickness does not invade flap territory. Small overlaps between routed strips that share an endpoint are treated as branch/hub manifold overlap in the candidate scaffold; arbitrary unrelated strip overlaps remain rejections.
- Body panels are also reserved regions. A corridor may enter its endpoint body, but it must not run through a non-endpoint body panel. Route selection and body-port clamping therefore treat other body panels as obstacles, and the region compiler rejects non-endpoint body overlaps with `pleat-strip-body-overlap`.

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
