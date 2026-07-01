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

  for (const [k, list] of inc) {
    if (list.length < 3) continue;
    const L = [...list].sort((a, b) => a.ang - b.ang);
    const n = L.length;
    const gaps = L.map((_, j) => {
      let g = L[(j + 1) % n].ang - L[j].ang;
      if (g <= 0) g += 2 * Math.PI;
      return g;
    });
    // Group the sectors into maximal runs of equal angle (cyclically). A run whose
    // angle is strictly smaller than the sectors bounding it on both ends is a
    // minimum (a crimp when it has length > 1); the creases spanning it alternate,
    // so union every consecutive pair across the WHOLE run - not just its ends,
    // which would break the alternation chain for runs of length >= 3.
    const EPS_A = 1e-9;
    const eq = (a: number, b: number): boolean => Math.abs(a - b) < EPS_A;
    // Constrain only between two fold creases: at a paper-edge vertex the boundary
    // half-edges shape the sector geometry but carry no M/V.
    const link = (s: number): void => {
      const a = L[s].i;
      const b = L[(s + 1) % n].i;
      if (edges[a].type !== "boundary" && edges[b].type !== "boundary") union(a, b, 1);
    };
    if (!gaps.every((g) => eq(g, gaps[0]))) {
      // Start at a run boundary so runs do not wrap.
      let s0 = 0;
      while (eq(gaps[s0], gaps[(s0 - 1 + n) % n])) s0++;
      let idx = s0;
      do {
        const val = gaps[idx];
        const run = [idx];
        let nx = (idx + 1) % n;
        while (eq(gaps[nx], val)) {
          run.push(nx);
          nx = (nx + 1) % n;
        }
        const before = gaps[(run[0] - 1 + n) % n];
        const after = gaps[nx];
        if (val < before - EPS_A && val < after - EPS_A) for (const s of run) link(s);
        idx = nx;
      } while (idx !== s0);
    }
  }

  const [rg, pg] = find(GROUND);
  const walk = new Map<number, Assignment | null>();
  edges.forEach((e, i) => {
    if (e.type !== "ridge") return;
    walk.set(i, e.mv); // remember the fallback walk color
    const [r, p] = find(i);
    e.mv = r === rg ? ((p ^ pg) === 0 ? "M" : "V") : null; // lemma color, else undetermined
  });

  // Maekawa completion: a ridge the lemma did not reach (e.g. a stretch ridge
  // whose only anchor is the paper edge) is still forced once the other creases at
  // an interior, even-degree vertex are known. Any valid M/V assignment is fine -
  // we just need |M-V|=2 - so fill a single undetermined ridge wherever that forces
  // a unique value, and iterate to a fixpoint.
  const vEdges = new Map<string, number[]>();
  for (const [k, list] of inc) vEdges.set(k, list.map((x) => x.i));
  for (let pass = 0; pass < 64; pass++) {
    let changed = false;
    for (const [k, idxs] of vEdges) {
      const [x, y] = k.split(",").map(Number);
      if (isBoundaryVertex({ x, y }, m.sheet)) continue;
      const fold = idxs.filter((i) => edges[i].type !== "boundary");
      if (fold.length % 2 !== 0) continue;
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

  // Anything still undetermined falls back to the walk color.
  edges.forEach((e, i) => {
    if (e.type === "ridge" && e.mv == null) e.mv = walk.get(i) ?? null;
  });
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
export function assignMolecule(m: BoxPleatedMolecule, sheet: { width: number; height: number }): AssignmentResult {
  const hinges = [...m.hinges];
  let edges = assignCreases({ ...m, hinges });

  for (let iter = 0; iter < 64; iter++) {
    const targets = ridgeCrossingFailures(edges, sheet);
    if (targets.length === 0) break;
    let added = false;
    for (const c of targets) {
      // Resolve the crossing by adding two hinge arms (degree 4 -> 6, so |M-V|=2
      // becomes reachable). March each axis direction with the shared hinge
      // marcher and pick the two arms that reach the paper edge first - an
      // edge-terminating arm has no far-end junction to balance, so it is easier
      // to assign. An arm that stops at a ridge (inward) is still valid as a
      // fallback. At a corner-adjacent crossing the two best arms are the
      // perpendicular pair toward the nearest edges, e.g. (2,2) -> (0,2),(2,0).
      const arms = AXIS_DIRS.map((d) => hingeEndpoint(c, d, m.ridges, m.ridgeSeeds, hinges, m.axialFamily, sheet))
        .filter((end): end is GridPoint => end !== null && !samePoint(end, c))
        .map((end) => ({ end, onEdge: isOnEdge(end, sheet) }));
      arms.sort((a, b) => Number(b.onEdge) - Number(a.onEdge));
      const chosen = arms.slice(0, 2).map((arm) => ({ a: c, b: arm.end }));
      if (chosen.length === 0) continue;
      const trial = assignCreases({ ...m, hinges: [...hinges, ...chosen] });
      if (!maekawaConflicts(trial, sheet).some((p) => samePoint(p, c))) {
        hinges.push(...chosen);
        edges = trial;
        added = true;
        break;
      }
    }
    if (!added) break;
  }

  // Final repair: flip non-ridge creases to clear any remaining Maekawa
  // conflicts (degree-8 flap-center 3-1 splits, and crossings the hinge repair
  // could not resolve to the edge).
  repairByFlipping(edges, sheet);

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
  const byKey = new Map<string, AssignedEdge>();
  for (const e of edges) byKey.set(segKey(e.a, e.b), e);

  for (const hinge of m.hinges) {
    // The hinge originates where it was created (hinge.a, the failing junction).
    const origin = hinge.a;
    const units = unitsAlong(hinge.a, hinge.b).filter((u) => byKey.get(segKey(u.a, u.b))?.type === "hinge");
    units.sort((u, v) => distTo(midpoint(u), origin) - distTo(midpoint(v), origin));

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

    for (const u of units) {
      const e = byKey.get(segKey(u.a, u.b));
      if (e) e.mv = color;
      color = color === "M" ? "V" : "M";
    }
  }
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

function unitsAlong(a: GridPoint, b: GridPoint): OriSegment[] {
  const sd = { x: Math.sign(b.x - a.x), y: Math.sign(b.y - a.y) };
  const n = Math.round(Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)));
  const out: OriSegment[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ a: { x: a.x + sd.x * i, y: a.y + sd.y * i }, b: { x: a.x + sd.x * (i + 1), y: a.y + sd.y * (i + 1) } });
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
