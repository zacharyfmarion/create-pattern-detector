// Stage D: mountain/valley assignment for a box-pleated molecule.
//
// Given a fixture (paper boundary + ridges + packing), this builds the full
// crease geometry (axials, pleats, hinges) and assigns each crease M/V using the
// deterministic box-pleat rules:
//
//   - Axial family (axials + pleats): one color per crease, by pleat-level
//     parity. The outermost ring (closest to the paper edge) is mountain and the
//     colors alternate inward.
//   - Ridges: a ridge is cut into pieces by its crossings with axial-family
//     creases. Propagating outward from the polygon CENTER (where axial-0
//     emanates), the first piece matches the color of the axial it meets and the
//     color flips at each subsequent crossing until the ridge terminates.
//   - Hinges: each hinge originates at the junction it was created to fix. Its
//     first segment is colored so Maekawa holds at that origin, then the color
//     alternates along the hinge until it terminates.
//
// What remains after this pass are the flap-center vertices, where Maekawa needs
// a 3-1 split of the four ridge arms that the per-ridge rule cannot produce on
// its own; those are reported as conflicts and resolved in a later repair stage.

import type { GridPoint, OriFixture, OriSegment } from "./ori-parser.ts";
import {
  findFlapCenters,
  hingeEndpoint,
  planarize,
  propagateAxials,
  propagateAllAxialOffsets,
  propagateHinges,
  failingJunctions,
  traceHingeRay,
  segmentKey,
} from "./box-pleated-molecule.ts";
import { axialChainColors } from "./box-pleated-axial-coloring.ts";

const AXIS_DIRS: GridPoint[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 },
];

export type Assignment = "M" | "V" | "B";
export type CreaseType = "axial" | "ridge" | "hinge" | "boundary";

export interface AssignedEdge {
  a: GridPoint;
  b: GridPoint;
  type: CreaseType;
  /** Mountain/valley/boundary, or null if unassigned. */
  mv: Assignment | null;
}

export interface BoxPleatedMolecule {
  sheet: { width: number; height: number };
  boundary: OriSegment[];
  ridges: OriSegment[];
  /** axials + edge-axials + pleats. */
  axialFamily: OriSegment[];
  hinges: OriSegment[];
  /** Flap centers used for axial seeding (river-filtered). */
  centers: GridPoint[];
  /** Flap centers including paper-corner ones (used to seed ridge assignment). */
  ridgeSeeds: GridPoint[];
  /** Color of an axial-family segment by pleat-level parity (outermost = M). */
  axialColorOf: (seg: OriSegment) => Assignment | null;
}

const EPS = 1e-6;

/** Build the full crease geometry plus the axial color function. */
export function buildMolecule(f: OriFixture): BoxPleatedMolecule {
  const centers = findFlapCenters(f.ridges, f.packing, { boundary: f.boundary, sheet: f.sheet });
  // Ridge propagation needs every polygon center, including paper-corner flaps
  // that the river filter drops (they would create axial crossings if seeded as
  // axials, but they are genuine flap centers for ridge purposes).
  const ridgeSeeds = findFlapCenters(f.ridges, f.packing);
  const { axials, edgeAxials } = propagateAxials(f.ridges, f.sheet, centers);
  const pleats = propagateAllAxialOffsets(f.ridges, f.sheet, axials, edgeAxials);
  const axialFamily = [...axials, ...edgeAxials, ...pleats];
  const base = [...f.boundary, ...f.ridges, ...axialFamily];
  const { hinges } = propagateHinges(f.ridges, axialFamily, centers, f.sheet, base);

  const axialColorOf = axialChainColors(axialFamily, f.ridges, f.sheet);

  return { sheet: f.sheet, boundary: f.boundary, ridges: f.ridges, axialFamily, hinges, centers, ridgeSeeds, axialColorOf };
}

/** Assign M/V to every crease and return the planarized unit edges. */
export function assignCreases(m: BoxPleatedMolecule): AssignedEdge[] {
  const ridge = assignRidges(m);

  const adj = planarize([...m.boundary, ...m.ridges, ...m.axialFamily, ...m.hinges]);
  const edges = planarUnitEdges(adj);

  for (const e of edges) {
    const mid = midpoint(e);
    const segId = segKey(e.a, e.b);
    if (onAny(mid, m.boundary)) {
      e.type = "boundary";
      e.mv = "B";
    } else if (onAny(mid, m.ridges)) {
      e.type = "ridge";
      // Clean (45/90) ridges colour by exact unit key (first-write-wins, the
      // center-out alternation). A non-45 Pythagorean stretch ridge's unit edges
      // do not land on lattice steps, so fall back to the geometric span that
      // contains the midpoint.
      e.mv = ridge.colors.get(segId) ?? ridge.spans.find((s) => pointOnSegment(mid, s))?.color ?? null;
    } else if (onAny(mid, m.hinges)) {
      e.type = "hinge";
      e.mv = null; // assigned below
    } else {
      e.type = "axial";
      e.mv = axialColorAt(mid, m);
    }
  }

  // Derive ridge phase from the locked axials via the Big-Little-Big lemma. This
  // overrides the center-out walk above wherever the lemma determines a ridge
  // (which is everywhere it is connected to an axial through strict-local-min
  // sectors); the walk's color only survives on edges the lemma leaves free.
  applyBigLittleBig(m, edges);

  assignHinges(m, edges);
  return edges;
}

/**
 * Big-Little-Big lemma (Hull): around a flat-foldable vertex, if a sector angle
 * is a strict local minimum, the two creases bounding it have opposite M/V. With
 * the axials locked, this pins every ridge's phase (Maekawa alone leaves it free).
 *
 * We build a signed union-find over the unit edges: each strict-local-min sector
 * unions its two bounding edges as "opposite"; every axial edge is tied to a
 * virtual Mountain ground by its locked color. Any ridge edge that ends up in the
 * ground's component then reads its color off the accumulated parity.
 */
function applyBigLittleBig(m: BoxPleatedMolecule, edges: AssignedEdge[]): void {
  const GROUND = edges.length;
  const parent = new Map<number, number>();
  const par = new Map<number, number>(); // parity from a node to its parent
  const find = (x: number): [number, number] => {
    let root = x;
    let p = 0;
    while ((parent.get(root) ?? root) !== root) {
      p ^= par.get(root) ?? 0;
      root = parent.get(root)!;
    }
    return [root, p];
  };
  const union = (a: number, b: number, rel: number): void => {
    const [ra, pa] = find(a);
    const [rb, pb] = find(b);
    if (ra === rb) return; // already related (consistency conflicts are ignored)
    parent.set(ra, rb);
    par.set(ra, pa ^ pb ^ rel);
  };

  // Seed: each axial edge is tied to the Mountain ground by its locked color.
  edges.forEach((e, i) => {
    if (e.type === "axial" && (e.mv === "M" || e.mv === "V")) union(i, GROUND, e.mv === "M" ? 0 : 1);
  });

  // Incidence: edge directions at each vertex.
  const key = (p: GridPoint): string => `${Math.round(p.x * 1e3) / 1e3},${Math.round(p.y * 1e3) / 1e3}`;
  const inc = new Map<string, Array<{ i: number; ang: number }>>();
  edges.forEach((e, i) => {
    for (const [p, o] of [[e.a, e.b], [e.b, e.a]] as const) {
      const k = key(p);
      (inc.get(k) ?? inc.set(k, []).get(k)!).push({ i, ang: Math.atan2(o.y - p.y, o.x - p.x) });
    }
  });

  // Rule Y - Y-junction: at a vertex of two ridges + one axial, both ridge arms
  // take the axial's colour. Neither the bisector nor Big-Little-Big applies there,
  // so this is the anchor. Apply first so it wins over any far-end colour. We allow
  // any number of hinges at the same vertex (a hinge often emanates from the same
  // flap-center junction): the ridge/axial relationship is unchanged, and requiring
  // strict degree 3 previously skipped these, leaving an edge-ward ridge uncoloured.
  // The vertex must be ridges + axial + hinges only (a boundary vertex is handled
  // elsewhere).
  for (const [, list] of inc) {
    const ridges = list.filter((x) => edges[x.i].type === "ridge");
    const axials = list.filter((x) => edges[x.i].type === "axial");
    const hinges = list.filter((x) => edges[x.i].type === "hinge");
    if (ridges.length !== 2 || axials.length !== 1) continue;
    if (ridges.length + axials.length + hinges.length !== list.length) continue;
    union(ridges[0].i, axials[0].i, 0);
    union(ridges[1].i, axials[0].i, 0);
  }

  // Rule 0 - Axial-bisector anchor: a ridge that bisects the angle between its two
  // neighbouring axials is OPPOSITE them when that angle is the small (< 180, the
  // acute/right) wedge, and SAME when it is the large (> 180, reflex) wedge. Since
  // axial colours are fixed, this is what ties ridges to the ground; the other two
  // rules only fix relationships between creases.
  for (const [, list] of inc) {
    if (list.length < 3) continue;
    const L = [...list].sort((a, b) => a.ang - b.ang);
    const n = L.length;
    const gaps = L.map((_, j) => {
      let g = L[(j + 1) % n].ang - L[j].ang;
      if (g <= 0) g += 2 * Math.PI;
      return g;
    });
    for (let i = 0; i < n; i++) {
      if (edges[L[i].i].type !== "ridge") continue;
      const prev = L[(i - 1 + n) % n];
      const next = L[(i + 1) % n];
      if (edges[prev.i].type !== "axial" || edges[next.i].type !== "axial") continue;
      const angle = gaps[(i - 1 + n) % n] + gaps[i];
      const rel = angle < Math.PI - 1e-9 ? 1 : 0; // small wedge -> opposite, large -> same
      union(L[i].i, prev.i, rel);
      union(L[i].i, next.i, rel);
    }
  }

  // Rule 1 - Ridge alternation: a ridge that passes straight through a vertex
  // flips wherever another crease crosses it, so its two collinear halves are
  // opposite. Applies at every crossing (axial or ridge), not just axial ones.
  for (const [, list] of inc) {
    const ridges = list.filter((x) => edges[x.i].type === "ridge");
    if (ridges.length < 2) continue;
    for (let a = 0; a < ridges.length; a++) {
      for (let b = a + 1; b < ridges.length; b++) {
        let d = Math.abs(ridges[a].ang - ridges[b].ang) % (2 * Math.PI);
        if (d > Math.PI) d = 2 * Math.PI - d;
        if (Math.abs(d - Math.PI) >= 1e-6) continue; // not collinear - not a straight-through ridge
        // A crossing exists if some incident crease is not parallel to this ridge.
        const crossed = list.some((x) => {
          let e = Math.abs(x.ang - ridges[a].ang) % (2 * Math.PI);
          if (e > Math.PI) e = 2 * Math.PI - e;
          return e > 1e-6 && Math.abs(e - Math.PI) > 1e-6;
        });
        if (crossed) union(ridges[a].i, ridges[b].i, 1);
      }
    }
  }

  for (const [k, list] of inc) {
    if (list.length < 3) continue;
    if (list.length % 2 === 1) continue; // odd-degree vertex is not flat-foldable: no lemma
    const L = [...list].sort((a, b) => a.ang - b.ang);
    const n = L.length;
    const gaps = L.map((_, j) => {
      let g = L[(j + 1) % n].ang - L[j].ang;
      if (g <= 0) g += 2 * Math.PI;
      return g;
    });
    // Kawasaki gate: Big-Little-Big is a theorem about FLAT-FOLDABLE vertices only.
    // Apply it just where the alternating sector sums are equal (= pi). A vertex
    // that fails Kawasaki (e.g. a 45-degree ridge crossing a straight axial, sectors
    // 45,135,45,135) is not flat-foldable - it needs a hinge - so the lemma must
    // not run there (it would contradict the ridge-alternation rule).
    let evenSum = 0;
    for (let j = 0; j < n; j += 2) evenSum += gaps[j];
    if (Math.abs(evenSum - Math.PI) > 1e-6) continue;
    // Big-Little-Big proper: only a STRICT local-minimum sector forces its two
    // bounding creases opposite. (Equal-angle runs are NOT "alternate across the
    // run" - that over-extension produced Maekawa-invalid colours; the straight-
    // through cases are handled by the ridge-alternation rule instead.) Never
    // constrain through a boundary half-edge.
    const EPS_A = 1e-9;
    for (let j = 0; j < n; j++) {
      const prev = gaps[(j - 1 + n) % n];
      const cur = gaps[j];
      const next = gaps[(j + 1) % n];
      if (cur >= prev - EPS_A || cur >= next - EPS_A) continue; // not a strict local min
      const a = L[j].i;
      const b = L[(j + 1) % n].i;
      if (edges[a].type !== "boundary" && edges[b].type !== "boundary") union(a, b, 1);
    }
  }

  // Read each ridge/hinge colour from its parity to the Mountain ground.
  const { width: W, height: H } = m.sheet;
  const paperEdges = (pt: GridPoint): string[] => {
    const es: string[] = [];
    if (Math.abs(pt.x) < EPS) es.push("L");
    if (Math.abs(pt.x - W) < EPS) es.push("R");
    if (Math.abs(pt.y) < EPS) es.push("B");
    if (Math.abs(pt.y - H) < EPS) es.push("T");
    return es;
  };
  const [rg, pg] = find(GROUND);
  edges.forEach((e, i) => {
    if (e.type !== "ridge" && e.type !== "hinge") return;
    const [r, p] = find(i);
    if (r === rg) {
      e.mv = (p ^ pg) === 0 ? "M" : "V";
      return;
    }
    // Not anchored to any axial. Default ONLY a ridge that spans a paper corner
    // (its two endpoints lie on two different paper edges) - that one is genuinely
    // free, valid either way, so choose Mountain. Every other undetermined crease
    // stays null, to be handled separately.
    if (e.type === "ridge") {
      const ea = paperEdges(e.a);
      const eb = paperEdges(e.b);
      const cornerSpan = ea.length > 0 && eb.length > 0 && !ea.some((x) => eb.includes(x));
      e.mv = cornerSpan ? "M" : null;
    } else {
      e.mv = null;
    }
  });

  // Maekawa completion: resolve any ridge still undetermined at an even-degree
  // interior vertex where it is the only unknown, by choosing the colour that
  // makes |M-V|=2. Iterate to a fixpoint.
  const vEdges = new Map<string, number[]>();
  for (const [k, list] of inc) vEdges.set(k, list.map((x) => x.i));
  for (let pass = 0; pass < 64; pass++) {
    let changed = false;
    for (const [k, idxs] of vEdges) {
      const [x, y] = k.split(",").map(Number);
      if (isBoundaryVertex({ x, y }, m.sheet)) continue;
      const fold = idxs.filter((i) => edges[i].type !== "boundary");
      if (fold.length % 2 !== 0) continue; // odd degree: Maekawa cannot resolve it
      let mk = 0;
      let vk = 0;
      const unknown: number[] = [];
      for (const i of fold) {
        if (edges[i].mv === "M") mk++;
        else if (edges[i].mv === "V") vk++;
        else unknown.push(i);
      }
      if (unknown.length !== 1 || edges[unknown[0]].type !== "ridge") continue;
      const asM = Math.abs(mk + 1 - vk) === 2;
      const asV = Math.abs(mk - (vk + 1)) === 2;
      if (asM && !asV) {
        edges[unknown[0]].mv = "M";
        changed = true;
      } else if (asV && !asM) {
        edges[unknown[0]].mv = "V";
        changed = true;
      }
    }
    if (!changed) break;
  }
}

export interface AssignmentResult {
  molecule: BoxPleatedMolecule;
  edges: AssignedEdge[];
  conflicts: GridPoint[];
}

/**
 * Full assignment with the ridge-crossing repair. The deterministic pass leaves
 * Maekawa failures at degree-4 vertices where four ridges cross with no axial or
 * hinge to balance them (the meeting point of four flaps' ridges, e.g. a
 * pinwheel). Such a vertex cannot satisfy |M-V|=2 with four ridges alone, so we
 * add a single straight hinge line through it (two collinear arms), turning it
 * into a degree-6 vertex that can. Arms are marched to the nearest ridge or the
 * paper edge, preferring the axis whose arms terminate on the edge. We re-assign
 * after each repair and loop until no such vertex remains.
 */
export function assignBoxPleated(f: OriFixture): AssignmentResult {
  return assignMolecule(buildMolecule(f), f.sheet);
}

/**
 * Full M/V assignment with ridge-crossing repair, operating on an already-built
 * molecule. Split from {@link assignBoxPleated} so callers that construct the
 * molecule themselves (the packing pipeline) can reuse the same assignment.
 */
// ---------------------------------------------------------------------------
// Hinge routing (Phase 2): place + color hinges by reflecting-ray search.
// ---------------------------------------------------------------------------

const HINGE_AXES: GridPoint[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 },
];

/** Does a crease leave vertex v going in direction d (so a hinge there would overlap)? */
function dirOccupied(v: GridPoint, d: GridPoint, segs: OriSegment[]): boolean {
  return segs.some((c) => {
    const ex = c.b.x - c.a.x;
    const ey = c.b.y - c.a.y;
    if (Math.abs(ex * d.y - ey * d.x) > EPS) return false; // not parallel
    if (Math.abs((c.a.x - v.x) * d.y - (c.a.y - v.y) * d.x) > EPS) return false; // not on v's ray line
    const ta = (c.a.x - v.x) * d.x + (c.a.y - v.y) * d.y;
    const tb = (c.b.x - v.x) * d.x + (c.b.y - v.y) * d.y;
    return Math.max(ta, tb) > EPS && Math.min(ta, tb) < 1 - EPS; // the segment leaves v forward
  });
}

/** Axis directions at v not already carrying a crease (the <= 3 usable hinge dirs). */
function freeAxes(v: GridPoint, m: BoxPleatedMolecule, hinges: OriSegment[]): GridPoint[] {
  const segs = [...m.ridges, ...m.axialFamily, ...hinges];
  return HINGE_AXES.filter((d) => !dirOccupied(v, d, segs));
}

/** Vertices still needing a hinge: geometry- or Maekawa-failing, interior, non-saturated. */
function hingeFrontier(
  m: BoxPleatedMolecule,
  hinges: OriSegment[],
  edges: AssignedEdge[],
  sheet: { width: number; height: number },
): GridPoint[] {
  const adj = planarize([...m.boundary, ...m.ridges, ...m.axialFamily, ...hinges]);
  const geom = failingJunctions(adj, sheet).map((f) => ({ x: f.x, y: f.y }));
  const mae = maekawaConflicts(edges, sheet);
  const seen = new Set<string>();
  const out: GridPoint[] = [];
  for (const v of [...geom, ...mae]) {
    const k = `${v.x},${v.y}`;
    if (seen.has(k)) continue;
    seen.add(k);
    if (isBoundaryVertex(v, sheet)) continue;
    if (freeAxes(v, m, hinges).length === 0) continue; // saturated: deferred
    out.push(v);
  }
  return out;
}

/** Degree of vertex v in the current planar graph (count of incident unit edges). */
function vertexDegree(v: GridPoint, edges: AssignedEdge[]): number {
  let n = 0;
  for (const e of edges) if (samePoint(e.a, v) || samePoint(e.b, v)) n++;
  return n;
}

/** Reflecting hinge rays from v that discharge cleanly (terminate at edge or hinge). */
function hingeRays(
  v: GridPoint,
  m: BoxPleatedMolecule,
  hinges: OriSegment[],
  sheet: { width: number; height: number },
): OriSegment[][] {
  return freeAxes(v, m, hinges)
    .map((d) => traceHingeRay(v, d, m.ridges, hinges, m.axialFamily, sheet))
    .filter((t): t is NonNullable<typeof t> => t !== null && (t.terminusType === "edge" || t.terminusType === "hinge"))
    .map((t) => t.path);
}

/** Candidate hinge placements at v: one ray if odd degree, a pair if even. */
function hingePlacements(
  v: GridPoint,
  m: BoxPleatedMolecule,
  hinges: OriSegment[],
  edges: AssignedEdge[],
  sheet: { width: number; height: number },
): OriSegment[][] {
  const single = hingeRays(v, m, hinges, sheet);
  return vertexDegree(v, edges) % 2 === 1
    ? single
    : single.flatMap((a, i) => single.slice(i + 1).map((b) => [...a, ...b]));
}

export interface HingeStep {
  hinges: OriSegment[];
  edges: AssignedEdge[];
  frontier: GridPoint[];
  /** The vertex chosen this step and the rays placed to discharge it (empty when stuck). */
  vertex: GridPoint | null;
  placed: OriSegment[];
}

/**
 * Greedy forward trace of the hinge router (no backtracking) for debugging: each
 * step records the current hinges, the colored edges, the frontier, and the
 * placement it commits (the resolving placement that leaves the fewest new failing
 * vertices). Stops when the frontier is empty or no vertex is resolvable.
 */
export function routeHingesSteps(m: BoxPleatedMolecule, sheet: { width: number; height: number }): HingeStep[] {
  const steps: HingeStep[] = [];
  let hinges: OriSegment[] = [];
  for (let i = 0; i < 200; i++) {
    const edges = assignCreases({ ...m, hinges });
    const fr = hingeFrontier(m, hinges, edges, sheet);
    let chosen: { v: GridPoint; add: OriSegment[] } | null = null;
    for (const v of fr) {
      const scored = hingePlacements(v, m, hinges, edges, sheet)
        .map((add) => ({ add, nfr: hingeFrontier(m, [...hinges, ...add], assignCreases({ ...m, hinges: [...hinges, ...add] }), sheet) }))
        .filter((c) => !c.nfr.some((p) => samePoint(p, v)))
        .sort((a, b) => a.nfr.length - b.nfr.length);
      if (scored.length) {
        chosen = { v, add: scored[0].add };
        break;
      }
    }
    steps.push({ hinges: [...hinges], edges, frontier: fr, vertex: chosen?.v ?? null, placed: chosen?.add ?? [] });
    if (!chosen || fr.length === 0) break;
    hinges = [...hinges, ...chosen.add];
  }
  return steps;
}

/**
 * Route hinges with a backtracking, best-first reflecting-ray search.
 *
 * At each step pick a failing (non-saturated, interior) vertex and enumerate hinge
 * placements from it - a single ray when its degree is odd (one ray makes it even),
 * a pair of rays when its degree is even (a pair preserves degree parity while
 * shifting the M/V balance). Each ray reflects off ridges and terminates at the
 * edge / another hinge / a junction. A placement is only taken if it resolves the
 * chosen vertex; the search then recurses on whatever new obligations it created,
 * and backtracks when a branch dead-ends. Candidates are tried in increasing
 * resulting-frontier order (edge-terminating routes tend to sort first). Falls back
 * to the best partial hinge set seen if no full solution is found within budget.
 */
export function routeHinges(m: BoxPleatedMolecule, sheet: { width: number; height: number }): OriSegment[] {
  let budget = 2500;
  let best: { hinges: OriSegment[]; score: number } = { hinges: [], score: Infinity };

  const rays = (v: GridPoint, hinges: OriSegment[]): OriSegment[][] => {
    const dirs = freeAxes(v, m, hinges);
    return dirs
      .map((d) => traceHingeRay(v, d, m.ridges, hinges, m.axialFamily, sheet))
      // A hinge must discharge at the paper edge or on another hinge. Terminating
      // at a ridge junction is a dead-end that leaves a new odd-degree vertex.
      .filter((t): t is NonNullable<typeof t> => t !== null && (t.terminusType === "edge" || t.terminusType === "hinge"))
      .map((t) => t.path);
  };

  const solve = (hinges: OriSegment[], edges: AssignedEdge[], fr: GridPoint[]): OriSegment[] | null => {
    if (fr.length < best.score) best = { hinges: [...hinges], score: fr.length };
    if (fr.length === 0) return hinges;
    if (budget-- <= 0) return null;

    // Branch on the first frontier vertex that has a placement resolving it; a
    // vertex with none is (for now) unroutable and deferred like the saturated
    // ones - skip it rather than failing the whole search.
    for (const v of fr) {
      const single = rays(v, hinges);
      // One ray for an odd-degree vertex, a pair for an even-degree one.
      const placements: OriSegment[][] =
        vertexDegree(v, edges) % 2 === 1
          ? single
          : single.flatMap((a, i) => single.slice(i + 1).map((b) => [...a, ...b]));

      const scored = placements
        .map((add) => {
          const next = [...hinges, ...add];
          const nedges = assignCreases({ ...m, hinges: next });
          return { next, nedges, nfr: hingeFrontier(m, next, nedges, sheet) };
        })
        .filter((c) => !c.nfr.some((p) => samePoint(p, v))) // must discharge v
        .sort((a, b) => a.nfr.length - b.nfr.length);

      if (scored.length === 0) continue; // v not resolvable now - try another vertex

      for (const c of scored) {
        const r = solve(c.next, c.nedges, c.nfr);
        if (r) return r;
        if (budget <= 0) return null;
      }
      return null; // committed to v, exhausted its options -> backtrack
    }
    return null; // no frontier vertex resolvable
  };

  const edges0 = assignCreases({ ...m, hinges: [] });
  const solved = solve([], edges0, hingeFrontier(m, [], edges0, sheet));
  return solved ?? best.hinges;
}

export function assignMolecule(m: BoxPleatedMolecule, sheet: { width: number; height: number }): AssignmentResult {
  // Phase 2: route hinges from scratch (axials + ridges are already colored) with
  // the reflecting-ray search, then do the final coloring with those hinges.
  const hinges = routeHinges(m, sheet);
  const edges = assignCreases({ ...m, hinges });
  return { molecule: { ...m, hinges }, edges, conflicts: maekawaConflicts(edges, sheet) };
}

/** Degree-4 interior vertices whose incident creases are all ridges and that fail Maekawa. */
function ridgeCrossingFailures(edges: AssignedEdge[], sheet: { width: number; height: number }): GridPoint[] {
  const incidence = new Map<string, AssignedEdge[]>();
  for (const e of edges) {
    for (const p of [e.a, e.b]) {
      const k = pointKey(p);
      if (!incidence.has(k)) incidence.set(k, []);
      incidence.get(k)!.push(e);
    }
  }
  const out: GridPoint[] = [];
  for (const [k, incident] of incidence) {
    const v = parsePoint(k);
    if (isBoundaryVertex(v, sheet)) continue;
    if (incident.length !== 4) continue;
    if (!incident.every((e) => e.type === "ridge")) continue;
    const M = incident.filter((e) => e.mv === "M").length;
    const V = incident.filter((e) => e.mv === "V").length;
    if (Math.abs(M - V) !== 2) out.push(v); // not a valid 3-1 split
  }
  return out;
}

/**
 * Min-conflicts repair. Flips non-ridge creases (axials/pleats/hinges) to clear
 * remaining Maekawa conflicts: repeatedly pick the incident flip that most
 * reduces the total conflict count (allowing a least-bad non-improving move to
 * escape a local minimum), then chase any newly broken vertex. A flip only
 * changes its two endpoint vertices, so each candidate is cheap to score. Ridge
 * and boundary creases are never flipped. Mutates `edges` in place.
 */
function repairByFlipping(edges: AssignedEdge[], sheet: { width: number; height: number }, maxIters = 4000): void {
  const incidence = new Map<string, AssignedEdge[]>();
  for (const e of edges) {
    for (const p of [e.a, e.b]) {
      const k = pointKey(p);
      if (!incidence.has(k)) incidence.set(k, []);
      incidence.get(k)!.push(e);
    }
  }
  const conflictAt = (vk: string): boolean => {
    const incident = incidence.get(vk);
    if (!incident) return false;
    const v = parsePoint(vk);
    if (isBoundaryVertex(v, sheet)) return false;
    if (isStraightDegree2(v, incident)) return false;
    let M = 0;
    let V = 0;
    for (const e of incident) {
      if (e.mv === "M") M++;
      else if (e.mv === "V") V++;
    }
    return Math.abs(M - V) !== 2;
  };

  // Axials carry the chain + alternation coloring, which is correct by
  // construction; flipping a single axial unit edge would break a crease's
  // one-color invariant. Only hinges (the balancing creases) may be flipped.
  const flippable = (e: AssignedEdge): boolean => e.type === "hinge";
  // Tabu the most recent flips so we do not immediately undo a non-improving move.
  const tabu: string[] = [];
  const tabuSize = 6;

  for (let iter = 0; iter < maxIters; iter++) {
    const conflicted = [...incidence.keys()].filter(conflictAt);
    if (conflicted.length === 0) break;

    const candidateEdges = new Set<AssignedEdge>();
    for (const vk of conflicted) for (const e of incidence.get(vk)!) if (flippable(e)) candidateEdges.add(e);

    let best: { edge: AssignedEdge; delta: number } | null = null;
    for (const e of candidateEdges) {
      const ek = segKey(e.a, e.b);
      const before = (conflictAt(pointKey(e.a)) ? 1 : 0) + (conflictAt(pointKey(e.b)) ? 1 : 0);
      e.mv = e.mv === "M" ? "V" : "M";
      const after = (conflictAt(pointKey(e.a)) ? 1 : 0) + (conflictAt(pointKey(e.b)) ? 1 : 0);
      e.mv = e.mv === "M" ? "V" : "M"; // restore
      const delta = after - before;
      const tabued = tabu.includes(ek) && delta >= 0;
      if (tabued) continue;
      if (!best || delta < best.delta) best = { edge: e, delta };
    }
    if (!best) break;

    best.edge.mv = best.edge.mv === "M" ? "V" : "M";
    tabu.push(segKey(best.edge.a, best.edge.b));
    if (tabu.length > tabuSize) tabu.shift();
  }
}

function isOnEdge(p: GridPoint, sheet: { width: number; height: number }): boolean {
  return Math.abs(p.x) < EPS || Math.abs(p.y) < EPS || Math.abs(p.x - sheet.width) < EPS || Math.abs(p.y - sheet.height) < EPS;
}

/** Interior vertices where Maekawa (|M - V| = 2) fails, given the assignment. */
export function maekawaConflicts(edges: AssignedEdge[], sheet: { width: number; height: number }): GridPoint[] {
  const incidence = new Map<string, AssignedEdge[]>();
  for (const e of edges) {
    for (const p of [e.a, e.b]) {
      const k = pointKey(p);
      if (!incidence.has(k)) incidence.set(k, []);
      incidence.get(k)!.push(e);
    }
  }
  const bad: GridPoint[] = [];
  for (const [k, incident] of incidence) {
    const v = parsePoint(k);
    if (isBoundaryVertex(v, sheet)) continue;
    if (isStraightDegree2(v, incident)) continue;
    let M = 0;
    let V = 0;
    let unknown = 0;
    for (const e of incident) {
      if (e.mv === "M") M++;
      else if (e.mv === "V") V++;
      else if (e.mv === null) unknown++;
    }
    if (unknown > 0) continue; // a still-unassigned crease here - not a Maekawa failure yet
    if (Math.abs(M - V) !== 2) bad.push(v);
  }
  return bad;
}

// ---------------------------------------------------------------------------
// Ridge assignment (center-out alternation)
// ---------------------------------------------------------------------------

/** A coloured stretch of a ridge between two consecutive axial-family crossings. */
interface RidgeSpan {
  a: GridPoint;
  b: GridPoint;
  color: Assignment;
}

interface RidgeColors {
  /** Exact unit-edge colours for clean 45/90 ridges (center-out, first-write-wins). */
  colors: Map<string, Assignment>;
  /** Per-piece coloured spans, used as a geometric fallback for non-45 ridges. */
  spans: RidgeSpan[];
}

function assignRidges(m: BoxPleatedMolecule): RidgeColors {
  const colors = new Map<string, Assignment>();
  const spans: RidgeSpan[] = [];
  for (const ridge of mergeCollinear(m.ridges)) {
    // A single straight ridge can pass through several flap centers (e.g. the
    // main diagonals of a pinwheel). Split it at every center and seed each
    // sub-span from its own center, alternating outward and terminating at the
    // next center. Seeding from the wrong center would arrive at the far end
    // with flipped parity; the per-span seeds agree on shared boundaries.
    const centers = m.ridgeSeeds.filter((c) => pointOnSegment(c, ridge));
    if (centers.length === 0) {
      // No flap center on this ridge (a stretch boundary, or a ridge whose seed
      // is itself an axial crossing). Anchor the phase to the axial color at the
      // ridge's start, so the first piece equals the axial color there.
      const dir = { x: ridge.b.x - ridge.a.x, y: ridge.b.y - ridge.a.y };
      emitRidgeArm(ridge.a, ridge.b, m, colors, spans, axialThrough(ridge.a, dir, m));
      continue;
    }
    const dir = { x: ridge.b.x - ridge.a.x, y: ridge.b.y - ridge.a.y };
    const along = (p: GridPoint): number => (p.x - ridge.a.x) * dir.x + (p.y - ridge.a.y) * dir.y;
    const stops = [ridge.a, ...centers, ridge.b]
      .map((p) => ({ p, t: along(p) }))
      .sort((a, b) => a.t - b.t);
    const uniq: Array<{ p: GridPoint; t: number }> = [];
    for (const s of stops) if (!uniq.some((x) => Math.abs(x.t - s.t) < EPS)) uniq.push(s);
    const isCenter = (p: GridPoint): boolean => centers.some((c) => samePoint(c, p));
    for (let i = 0; i + 1 < uniq.length; i++) {
      const s0 = uniq[i].p;
      const s1 = uniq[i + 1].p;
      if (isCenter(s0)) emitRidgeArm(s0, s1, m, colors, spans);
      else if (isCenter(s1)) emitRidgeArm(s1, s0, m, colors, spans);
    }
  }
  return { colors, spans };
}

/** A clean (axis-aligned or 45-degree) segment, vs a non-45 stretch span. */
function isCleanSeg(a: GridPoint, b: GridPoint): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.abs(dx) < EPS || Math.abs(dy) < EPS || Math.abs(Math.abs(dx) - Math.abs(dy)) < EPS;
}

/** Color of an axial that passes through `p` and is not collinear with the ridge. */
function axialThrough(p: GridPoint, ridgeDir: GridPoint, m: BoxPleatedMolecule): Assignment | null {
  for (const ax of m.axialFamily) {
    if (!pointOnSegment(p, ax)) continue;
    const ad = { x: ax.b.x - ax.a.x, y: ax.b.y - ax.a.y };
    if (Math.abs(ad.x * ridgeDir.y - ad.y * ridgeDir.x) < EPS) continue; // runs along the ridge
    const c = m.axialColorOf(ax);
    if (c) return c;
  }
  return null;
}

function emitRidgeArm(
  center: GridPoint,
  far: GridPoint,
  m: BoxPleatedMolecule,
  colors: Map<string, Assignment>,
  spans: RidgeSpan[],
  seedColor: Assignment | null = null,
): void {
  // Axial-family crossings along center -> far, with the axial's color, sorted
  // outward from the center. Keep the actual intersection point (not a
  // distance-reconstructed one) so the cut lands exactly on the crease.
  const cuts: Array<{ point: GridPoint; d: number; color: Assignment | null }> = [];
  for (const ax of m.axialFamily) {
    const x = intersect(center, far, ax.a, ax.b);
    if (!x) continue;
    const d = Math.hypot(x.x - center.x, x.y - center.y);
    if (d < EPS) continue;
    cuts.push({ point: x, d, color: m.axialColorOf(ax) });
  }
  cuts.sort((p, q) => p.d - q.d);
  const unique: Array<{ point: GridPoint; color: Assignment | null }> = [];
  for (const c of cuts) if (!unique.some((u) => Math.abs(Math.hypot(u.point.x - c.point.x, u.point.y - c.point.y)) < EPS)) unique.push(c);

  const bounds: GridPoint[] = [center, ...unique.map((c) => c.point), far];

  // First piece matches the first axial crossing's color; flip at each crossing.
  // Clean pieces colour their unit edges by exact key (preserving the original
  // first-write-wins behaviour); every piece is also recorded as a span so a
  // non-45 piece (whose unit edges miss the lattice) can be coloured by midpoint.
  const color0: Assignment = seedColor ?? (unique.length ? (unique[0].color ?? "M") : "M");
  let color: Assignment = color0;
  for (let i = 0; i < bounds.length - 1; i++) {
    spans.push({ a: bounds[i], b: bounds[i + 1], color });
    const p = snapPoint(bounds[i]);
    const q = snapPoint(bounds[i + 1]);
    if (isCleanSeg(bounds[i], bounds[i + 1])) {
      const sd = { x: Math.sign(q.x - p.x), y: Math.sign(q.y - p.y) };
      const n = Math.round(Math.max(Math.abs(q.x - p.x), Math.abs(q.y - p.y)));
      for (let j = 0; j < n; j++) {
        const a = snapPoint({ x: p.x + sd.x * j, y: p.y + sd.y * j });
        const b = snapPoint({ x: p.x + sd.x * (j + 1), y: p.y + sd.y * (j + 1) });
        const k = segKey(a, b);
        if (!colors.has(k)) colors.set(k, color);
      }
    } else if (!samePoint(p, q)) {
      // A non-45 (Pythagorean stretch) ridge's planar edge runs exactly between
      // two consecutive crossings, i.e. this whole span. Its unit edges miss the
      // lattice so there is nothing to step; register the span's own endpoints as
      // an exact color key so it resolves like a clean edge instead of via the
      // fragile first-geometric-match span fallback.
      const k = segKey(p, q);
      if (!colors.has(k)) colors.set(k, color);
    }
    color = color === "M" ? "V" : "M";
  }
}

// ---------------------------------------------------------------------------
// Hinge assignment
// ---------------------------------------------------------------------------

function assignHinges(m: BoxPleatedMolecule, edges: AssignedEdge[]): void {
  const byVertex = new Map<string, AssignedEdge[]>();
  for (const e of edges) {
    for (const p of [e.a, e.b]) {
      const k = pointKey(p);
      if (!byVertex.has(k)) byVertex.set(k, []);
      byVertex.get(k)!.push(e);
    }
  }
  for (const hinge of m.hinges) {
    // The hinge originates where it was created (hinge.a, the failing junction).
    const origin = hinge.a;
    // Walk the hinge's ACTUAL creases: the hinge-type edges lying on the hinge
    // segment, ordered outward from the origin. planarize splits a hinge at every
    // crossing, and those crossings are sub-grid when the hinge runs along a non-45
    // Pythagorean stretch direction (e.g. a hinge reflected off a stretch ridge). So
    // we must break the hinge at its real crossings and alternate colours over those
    // pieces, rather than step along a synthetic integer axis/45-degree grid (which
    // walks off a non-45 hinge and matches nothing, leaving it unassigned).
    const parts = edges
      .filter((e) => e.type === "hinge" && onSegment(e.a, hinge.a, hinge.b) && onSegment(e.b, hinge.a, hinge.b))
      .sort((e, f) => distTo(midpoint(e), origin) - distTo(midpoint(f), origin));

    // First color satisfies Maekawa at the origin given already-assigned creases.
    const incident = byVertex.get(pointKey(origin)) ?? [];
    let M = 0;
    let V = 0;
    for (const e of incident) {
      if (e.mv === "M") M++;
      else if (e.mv === "V") V++;
    }
    let color: Assignment;
    if (Math.abs(M + 1 - V) === 2) color = "M";
    else if (Math.abs(M - (V + 1)) === 2) color = "V";
    else color = M <= V ? "M" : "V";

    // Alternate M/V outward along the hinge; keep any color the constraint solve set.
    for (const part of parts) {
      if (part.mv == null) part.mv = color;
      color = color === "M" ? "V" : "M";
    }
  }
}

/** True when point p lies on the segment [a,b] (collinear and within its extent). */
function onSegment(p: GridPoint, a: GridPoint, b: GridPoint): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 < EPS) return false;
  const cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
  if (Math.abs(cross) > EPS * Math.sqrt(len2)) return false; // off the line
  const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
  return t >= -EPS && t <= 1 + EPS;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

function planarUnitEdges(adj: Map<string, GridPoint[]>): AssignedEdge[] {
  const out: AssignedEdge[] = [];
  const seen = new Set<string>();
  for (const [vk, neighbors] of adj) {
    const v = parsePoint(vk);
    for (const n of neighbors) {
      const k = segKey(v, n);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ a: v, b: n, type: "axial", mv: null });
    }
  }
  return out;
}

function axialColorAt(mid: GridPoint, m: BoxPleatedMolecule): Assignment | null {
  for (const ax of m.axialFamily) {
    if (pointOnSegment(mid, ax)) return m.axialColorOf(ax);
  }
  return null;
}

function mergeCollinear(ridges: OriSegment[]): OriSegment[] {
  const groups = new Map<string, OriSegment[]>();
  for (const s of ridges) {
    const dx = s.b.x - s.a.x;
    const dy = s.b.y - s.a.y;
    const slope = Math.sign(dx) * Math.sign(dy);
    const off = slope > 0 ? Math.round(s.a.x - s.a.y) : Math.round(s.a.x + s.a.y);
    const k = `${slope}:${off}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(s);
  }
  const out: OriSegment[] = [];
  for (const segs of groups.values()) {
    const dir = { x: segs[0].b.x - segs[0].a.x, y: segs[0].b.y - segs[0].a.y };
    const len = Math.hypot(dir.x, dir.y);
    const u = { x: dir.x / len, y: dir.y / len };
    const t = (p: GridPoint): number => p.x * u.x + p.y * u.y;
    const ivs = segs.map((s) => [Math.min(t(s.a), t(s.b)), Math.max(t(s.a), t(s.b))] as [number, number]).sort((a, b) => a[0] - b[0]);
    const merged: Array<[number, number]> = [];
    let cur: [number, number] = [...ivs[0]];
    for (let i = 1; i < ivs.length; i++) {
      if (ivs[i][0] <= cur[1] + EPS) cur[1] = Math.max(cur[1], ivs[i][1]);
      else {
        merged.push(cur);
        cur = [...ivs[i]];
      }
    }
    merged.push(cur);
    const base = segs[0].a;
    const t0base = t(base);
    for (const [lo, hi] of merged) {
      out.push({
        a: snapPoint({ x: base.x + u.x * (lo - t0base), y: base.y + u.y * (lo - t0base) }),
        b: snapPoint({ x: base.x + u.x * (hi - t0base), y: base.y + u.y * (hi - t0base) }),
      });
    }
  }
  return out;
}

function intersect(p1: GridPoint, p2: GridPoint, p3: GridPoint, p4: GridPoint): GridPoint | null {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;
  const den = d1x * d2y - d1y * d2x;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / den;
  const s = ((p3.x - p1.x) * d1y - (p3.y - p1.y) * d1x) / den;
  if (t < -1e-7 || t > 1 + 1e-7 || s < -1e-7 || s > 1 + 1e-7) return null;
  return { x: p1.x + d1x * t, y: p1.y + d1y * t };
}

function onAny(p: GridPoint, segs: OriSegment[]): boolean {
  return segs.some((s) => pointOnSegment(p, s));
}

function pointOnSegment(p: GridPoint, s: OriSegment): boolean {
  const cross = (s.b.x - s.a.x) * (p.y - s.a.y) - (s.b.y - s.a.y) * (p.x - s.a.x);
  if (Math.abs(cross) > EPS) return false;
  const dot = (p.x - s.a.x) * (s.b.x - s.a.x) + (p.y - s.a.y) * (s.b.y - s.a.y);
  const len2 = (s.b.x - s.a.x) ** 2 + (s.b.y - s.a.y) ** 2;
  return dot > -EPS && dot < len2 + EPS;
}

function isBoundaryVertex(v: GridPoint, sheet: { width: number; height: number }): boolean {
  return Math.abs(v.x) < EPS || Math.abs(v.y) < EPS || Math.abs(v.x - sheet.width) < EPS || Math.abs(v.y - sheet.height) < EPS;
}

function isStraightDegree2(v: GridPoint, incident: AssignedEdge[]): boolean {
  if (incident.length !== 2) return false;
  const angle = (e: AssignedEdge): number => {
    const o = samePoint(e.a, v) ? e.b : e.a;
    return Math.atan2(o.y - v.y, o.x - v.x);
  };
  return Math.abs(Math.abs(angle(incident[0]) - angle(incident[1])) - Math.PI) < EPS;
}

function midpoint(s: OriSegment): GridPoint {
  return { x: (s.a.x + s.b.x) / 2, y: (s.a.y + s.b.y) / 2 };
}

function distTo(p: GridPoint, q: GridPoint): number {
  return Math.hypot(p.x - q.x, p.y - q.y);
}

function samePoint(a: GridPoint, b: GridPoint): boolean {
  return Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS;
}

function snapPoint(p: GridPoint): GridPoint {
  const snap = (v: number): number => {
    const r = Math.round(v);
    return Math.abs(v - r) < 1e-4 ? r : Math.round(v * 1e4) / 1e4;
  };
  return { x: snap(p.x), y: snap(p.y) };
}

function pointKey(p: GridPoint): string {
  return `${p.x},${p.y}`;
}

function parsePoint(k: string): GridPoint {
  const [x, y] = k.split(",").map(Number);
  return { x, y };
}

function segKey(a: GridPoint, b: GridPoint): string {
  const ka = pointKey(a);
  const kb = pointKey(b);
  return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
}
