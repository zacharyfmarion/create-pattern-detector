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
- stretch/gadget regions where flap circles overlap or force extra material;
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
- no overlap between placed molecule interiors unless an explicit stretch/gadget region owns the overlap.

These checks happen before full FOLD arrangement so bad layouts fail cheaply.

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
