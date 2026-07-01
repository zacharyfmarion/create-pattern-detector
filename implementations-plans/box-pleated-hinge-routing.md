# Box-Pleated Hinge Routing & Coloring

## Problem

After axials and ridges are colored (axials by reflected-crease chains, ridges by
the Big-Little-Big lemma seeded from the axials), some interior vertices still fail
flat-foldability and need **hinge** creases to balance them. The current pipeline
places hinges purely for geometry (even degree / Kawasaki), colors them greedily
by an alternate-from-origin walk, then patches Maekawa with a greedy flip search.
That fails in two ways:

- **Problem A (even-distance hinge):** a hinge's M/V alternates one flip per unit
  step (it flips wherever it crosses a line). Over an **even** number of unit steps
  its two ends land on **opposite** colors, so the far endpoint receives the wrong
  color and can't satisfy Maekawa. A single flip can't fix it (the hinge's two ends
  demand opposite colors), so the greedy repair never commits.
- **Missing / mis-routed hinges:** hinges stop at ridges instead of reflecting, so
  many clean routes (reflect off a ridge, continue to the paper edge) don't exist.

## Model (confirmed with the author)

- A hinge is a **reflecting ray**: axis-aligned, **reflects off ridges exactly like
  an axial** (mirror reflection), until it terminates.
- Its M/V **alternates one flip per unit step** (each grid-line crossing). The color
  at any point is the origin color flipped `#unit-steps` times, so the terminus
  color is decided by **path-length parity**.
- The origin-segment color is forced by Maekawa at the origin vertex, given the
  already-fixed creases there.
- A vertex may carry **up to 3 hinge creases**.
- A single hinge ray changes its origin vertex's degree by **+1**, so the number of
  rays a vertex needs is coupled to its degree parity:
  - **odd-degree** vertex: one ray (to the edge, or connecting to another odd vertex)
    makes it even and can balance it.
  - **even-degree** vertex failing only Maekawa: needs an **even** number of rays
    (minimum two) to shift the color balance without breaking degree parity. The
    pair may be **any two axis directions** (not necessarily collinear) as long as
    local constraints are satisfied.
- **Saturated vertices are out of scope.** A degree-8 vertex has all four axis
  directions (and the 45° slots) occupied — no room for an axis-aligned hinge. The
  router never adds to those; they are reported as deferred/unresolved. Their
  residual conflicts are expected and are a separate future problem.
- **Local constraints only** for now (even degree, Kawasaki, Maekawa). Global
  (layer-ordering) flat-foldability is deferred to v2.

## Core restructure

Split assignment into two phases with a hard boundary:

- **Phase 1 — Axials + ridges, geometry and color, with NO hinges present.**
- **Phase 2 — Hinge creation + coloring together**, as a backtracking route search
  over the frontier of vertices Phase 1 leaves unsatisfied.

Today ridge coloring runs with hinges already in the planar graph; decoupling is
the enabling change.

## Phase 1 (mostly exists, needs isolation)

1. Axial/pleat geometry (`propagateAxials`, `propagateAllAxialOffsets`).
2. Axial colors (`axialChainColors`) — locked.
3. Ridge geometry from the packing.
4. Ridge colors via Big-Little-Big, seeded by axials, on the planar graph of
   **boundary + ridges + axials only** (no hinges).
5. Output: a fully M/V-colored axial+ridge CP, plus the **frontier** = non-saturated
   interior vertices that do not yet satisfy (even degree ∧ Kawasaki ∧ Maekawa).

## Phase 2 — the router

**Route classification** (preference order; take the first that discharges cleanly):

1. **Terminates on the paper edge** → no far-end obligation. Try first.
2. **Reflects off ridge(s) → reaches edge** → same, clean.
3. **Ends on another hinge's endpoint**, then check:
   - **odd** path length AND target has **< 3** hinges → compatible, discharges.
   - **even** parity, OR target already at capacity → discharges *this* vertex but
     pushes a **new obligation** at the terminus (needs a further hinge, within the
     3-per-vertex cap).

**Validity constraints on any route:**

- Every reflection/termination lands on the integer grid (reject routes that reflect
  off a non-45° stretch ridge to an off-grid point — reuse the existing guard).
- No vertex exceeds **3** incident hinges.
- Never route into a **saturated** (no free axis direction) vertex.

**Search (backtracking DFS):**

```
solve(frontier, hinges):
  if frontier empty: return SUCCESS
  V = pick a failing vertex           # order is a tunable knob; correctness from backtracking
  for route in candidateRoutes(V) ordered [edge, reflect->edge, odd-hinge, even-hinge]:
     place hinge(route) with alternating colors from c0(V)
     if V still needs another ray (even-degree case): re-queue V at the FRONT
     push any new obligation (even/saturated terminus)
     if solve(updatedFrontier, hinges+route) == SUCCESS: return SUCCESS
     undo route                       # backtrack
  return FAIL                         # V unroutable -> caller backtracks
```

- **Success:** every non-saturated frontier vertex is locally flat-foldable.
- **Residual:** saturated vertices remain conflicted by design (deferred).
- **Objective:** first feasible solution; the preference order is the tie-breaker
  toward simpler completions (most edge-terminations, fewest hinges). No numeric
  optimizer. A convergence heuristic for frontier ordering is a later experiment
  over many CPs.

## Engine change (Stage 1, in isolation)

Make hinges **reflect off ridges** instead of stopping, reusing the axial
reflection path (`marchRay`/`reflect`); keep the off-grid-reflection rejection.
Validate that hinges reach the edge via reflection before any coloring logic.

## Code changes

- **New:** `box-pleated-hinge-router.ts` — reflecting marcher + backtracking search
  + coloring.
- **Restructure:** `assignCreases` splits — Phase-1 coloring standalone; Phase 2
  calls the router.
- **Remove (replaced by the router):** `assignHinges`, `repairByFlipping`, the
  ridge-crossing repair loop in `assignMolecule`, `propagateHinges`.
- **Reuse:** `planarize`, Kawasaki / even-degree / `maekawaConflicts` checks,
  `marchRay` / `reflect`.

## Staging (each validated with an `.ori` of the 5 sample CPs)

1. Hinges reflect off ridges → reach edges (geometry only).
2. Phase-1 isolation: colored axial+ridge CP + frontier list; confirm the frontier
   matches today's non-saturated conflict set.
3. Router: edge-preference + reflection + parity + 3-cap + free pairs + skip
   saturated + backtracking; generate `.ori`, compare to hand solutions.
4. Delete dead code; migrate the assignment tests off the old zero-conflict
   baselines onto the router.

## Deferred to v2

- Global (layer-ordering) flat-foldability beyond local Kawasaki/Maekawa.
- Resolving saturated (degree-8) vertices.
- Optimization beyond "first feasible + simplicity tie-break", including a learned
  frontier-ordering heuristic.
