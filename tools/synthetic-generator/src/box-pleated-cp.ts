// Build a full crease pattern (molecule) from a generated BP Studio packing and
// gate it on validity.
//
// Pulls together the pieces that were previously assembled ad hoc:
//   - ridges from EVERY flap, river, and stretch device (a packing can carry
//     several stretch devices - using only the first drops their ridges/arms and
//     leaves the molecule incomplete), plus the gap-fill filler flaps.
//   - axial seeds from each flap's ridge convergence points (ridgeJunctions),
//     including the filler flaps.
//   - axials -> axial+n pleats -> hinges.
//
// Validity gate: the packing is valid only when it consumes all paper (gap fill
// complete, ODS rule #4) AND every crease junction lands on the grid. The
// Pythagorean stretch edges are non-45-degree, so they frequently cross the
// orthogonal pleat grid between lattice points; such a packing cannot be a unit
// box-pleat pattern and is rejected (offGridJunctions non-empty).

import type { BoxPleatedPacking, BoxPleatedPackingConfig } from "./box-pleated-packing.ts";
import { fillPackingGaps, generateBoxPleatedPacking } from "./box-pleated-packing.ts";
import { repairFlapRidgeHole } from "./box-pleated-gap-fill.ts";
import {
  propagateAxials,
  propagateAllAxialOffsets,
  propagateAxialFamilyWithLevels,
  planarize,
  failingJunctions,
  offGridJunctions,
  ridgeJunctions,
  segmentKey,
} from "./box-pleated-molecule.ts";
import {
  assignMolecule,
  assignCreases,
  routeHinges,
  type AssignedEdge,
  type Assignment,
  type CreaseType,
  type BoxPleatedMolecule,
} from "./box-pleated-assignment.ts";
import { axialChainColors } from "./box-pleated-axial-coloring.ts";
import type { GridPoint, OriSegment } from "./ori-parser.ts";

const EPS = 1e-9;

export interface PackingCP {
  sheet: { width: number; height: number };
  /** Flap, river, stretch, and filler ridges, clipped to the paper. */
  ridges: OriSegment[];
  axials: OriSegment[];
  edgeAxials: OriSegment[];
  pleats: OriSegment[];
  hinges: OriSegment[];
  /** Flap convergence points the axials were seeded from. */
  seeds: GridPoint[];
  /** Crease junctions that miss the grid (a non-empty list rejects the packing). */
  offGrid: GridPoint[];
  /** Interior junctions failing Kawasaki/even-degree after hinge selection. */
  failing: GridPoint[];
  /** True when the packing consumes all paper (ODS rule #4). */
  complete: boolean;
  /** True when complete AND no off-grid crease junction. */
  valid: boolean;
  /** M/V-assigned unit edges (the same geometry, coloured mountain/valley/border). */
  assignedEdges: AssignedEdge[];
  /** Interior vertices still failing Maekawa after assignment + repair. */
  mvConflicts: GridPoint[];
}

interface PackingGeometry {
  molecule: BoxPleatedMolecule;
  gap: ReturnType<typeof fillPackingGaps>;
  offGrid: GridPoint[];
  axials: OriSegment[];
  edgeAxials: OriSegment[];
  pleats: OriSegment[];
  seeds: GridPoint[];
  W: number;
  H: number;
}

/** The axial+ridge molecule (no hinges yet) plus the intermediates cp needs. */
/** A named group of ridges by their source object (for generation tracing). */
export interface RidgeGroup {
  name: string;
  ridges: OriSegment[];
}

/**
 * Ridges from every flap/river, every stretch device, and the gap-fill fillers,
 * grouped by source object in generation order. Shared by packingGeometry and the
 * generation trace so the two can never drift.
 */
export function collectRidges(
  packing: BoxPleatedPacking,
  gap: ReturnType<typeof fillPackingGaps>,
  W: number,
  H: number,
): { ridges: OriSegment[]; groups: RidgeGroup[] } {
  const seg = (a: GridPoint, b: GridPoint): OriSegment => ({ a, b });
  const bounds = { width: W, height: H };
  const ridges: OriSegment[] = [];
  const groups: RidgeGroup[] = [];
  const counts: Record<string, number> = {};
  // Keep every ridge segment, clipped to the paper. Axis-aligned (0/90-degree)
  // segments are legitimate straight-skeleton ridges too (e.g. a non-square flap's
  // spine), so we do NOT filter them - the ring-hole repair already drops BP
  // Studio's spurious ring sides at their source.
  const pushRidge = (g: OriSegment[], a: GridPoint, b: GridPoint): void => {
    pushClipped(g, a, b, W, H);
  };
  for (const object of packing.layout.objects) {
    if (object.kind === "root" || object.kind === "stretch-device") continue;
    const objRidges = object.ridges.map((line) => seg(line[0], line[1]));
    const repaired = object.kind === "flap" ? repairFlapRidgeHole(objRidges, bounds) : objRidges;
    const g: OriSegment[] = [];
    for (const r of repaired) pushRidge(g, r.a, r.b);
    ridges.push(...g);
    counts[object.kind] = (counts[object.kind] ?? 0) + 1;
    groups.push({ name: `${object.kind} ${counts[object.kind]}`, ridges: g });
  }
  for (const object of packing.layout.objects) {
    if (object.kind !== "stretch-device") continue;
    const g: OriSegment[] = [];
    for (const line of object.ridges) pushRidge(g, line[0], line[1]);
    for (const e of sharedContourEdges(object.contours)) pushRidge(g, e.a, e.b);
    ridges.push(...g);
    counts.stretch = (counts.stretch ?? 0) + 1;
    groups.push({ name: `stretch ${counts.stretch}`, ridges: g });
  }
  const filler: OriSegment[] = [];
  for (const r of gap.ridges) pushRidge(filler, r.a, r.b);
  if (filler.length) {
    ridges.push(...filler);
    groups.push({ name: "gap fillers", ridges: filler });
  }
  // A filler polygon (added after BP packed) can have a bisector that dead-ends in
  // the paper interior right beside a Pythagorean stretch: the filler did not
  // exist when BP computed the stretch, so the two were never connected. Ridges
  // must never terminate in the interior, so connect each such dangling filler
  // endpoint to the nearest stretch-ridge vertex.
  const stretchRidges = groups.filter((g) => g.name.startsWith("stretch")).flatMap((g) => g.ridges);
  const stitches = stitchFillerDangles(filler, stretchRidges, ridges, W, H);
  if (stitches.length) {
    ridges.push(...stitches);
    groups.push({ name: "stitches", ridges: stitches });
  }
  return { ridges, groups };
}

/**
 * Connect dangling filler ridge endpoints to the stretch structure. A dangling
 * endpoint is an interior (off-paper-edge), degree-1 endpoint of a gap-filler
 * ridge - a bisector tip with nothing on its far side. For each, add a ridge to
 * the nearest stretch-ridge vertex, so no ridge terminates in the paper interior.
 */
function stitchFillerDangles(
  fillerRidges: OriSegment[],
  stretchRidges: OriSegment[],
  allRidges: OriSegment[],
  W: number,
  H: number,
): OriSegment[] {
  if (fillerRidges.length === 0 || stretchRidges.length === 0) return [];
  const E = 1e-6;
  const same = (a: GridPoint, b: GridPoint): boolean => Math.abs(a.x - b.x) < E && Math.abs(a.y - b.y) < E;
  const onEdge = (q: GridPoint): boolean =>
    Math.abs(q.x) < E || Math.abs(q.x - W) < E || Math.abs(q.y) < E || Math.abs(q.y - H) < E;
  const degree = (q: GridPoint): number => {
    let d = 0;
    for (const s of allRidges) {
      if (same(s.a, q)) d++;
      if (same(s.b, q)) d++;
    }
    return d;
  };
  const stretchVerts: GridPoint[] = [];
  for (const s of stretchRidges) for (const q of [s.a, s.b]) if (!stretchVerts.some((v) => same(v, q))) stretchVerts.push(q);

  const stitches: OriSegment[] = [];
  const handled = new Set<string>();
  for (const s of fillerRidges) {
    for (const q of [s.a, s.b]) {
      const key = `${q.x},${q.y}`;
      if (handled.has(key) || onEdge(q) || degree(q) !== 1) continue;
      let best = Infinity;
      let target: GridPoint | null = null;
      for (const v of stretchVerts) {
        const d = Math.hypot(v.x - q.x, v.y - q.y);
        if (d < best) {
          best = d;
          target = v;
        }
      }
      if (!target) continue;
      handled.add(key);
      stitches.push({ a: q, b: target });
    }
  }
  return stitches;
}

function packingGeometry(packing: BoxPleatedPacking): PackingGeometry {
  const sheet = packing.sheet;
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  const seg = (a: GridPoint, b: GridPoint): OriSegment => ({ a, b });

  const gap = fillPackingGaps(packing);

  const { ridges } = collectRidges(packing, gap, W, H);

  // Seeds: the interior convergence points of each polygon's OWN straight
  // skeleton. A valid axial seed is a junction inside one polygon's skeleton; a
  // point where two adjacent polygons merely touch on a shared boundary is
  // interior to neither and must not be seeded. So we run ridgeJunctions per
  // polygon (each real flap, and each filler flap separately) - never on the
  // union, which would fuse neighbours and invent boundary junctions. The filler
  // groups carry the edge-reflected ridges (croppedFlapRidges) the molecule uses,
  // so an edge/corner filler is seeded from the same skeleton it is drawn with.
  const rawSeeds: GridPoint[] = [];
  for (const object of packing.layout.objects) {
    if (object.kind !== "flap") continue;
    // Seed from the repaired skeleton, so a removed ring's spine endpoints are the
    // real convergence points instead of the (now-deleted) ring corners.
    rawSeeds.push(...ridgeJunctions(repairFlapRidgeHole(object.ridges.map((l) => seg(l[0], l[1])), sheet)));
  }
  // Filler flaps: seed from the centers the tiler carried out (the convergence
  // points of each filler's reflected full rectangle). These subsume what
  // ridgeJunctions(ridgesByFlap) used to find - identical for interior fillers,
  // and additionally recovering the on/off-paper center of an edge/corner filler,
  // whose clipped ridges dropped their on-edge spine.
  rawSeeds.push(...gap.centers);
  const seenSeed = new Set<string>();
  const seeds = rawSeeds.filter((s) => {
    const k = `${s.x},${s.y}`;
    if (seenSeed.has(k)) return false;
    seenSeed.add(k);
    return true;
  });

  const rawAx = propagateAxials(ridges, sheet, seeds);
  const rawPleats = propagateAllAxialOffsets(ridges, sheet, rawAx.axials, rawAx.edgeAxials);

  // A corner/edge flap's center can lie outside the paper, so a crease seeded
  // there overhangs the edge; clip every crease to the paper so they all
  // terminate at the boundary.
  const axials = clipAll(rawAx.axials, W, H);
  const edgeAxials = clipAll(rawAx.edgeAxials, W, H);
  const pleats = clipAll(rawPleats, W, H);
  const axialFamily = [...axials, ...edgeAxials, ...pleats];

  const boundary: OriSegment[] = [
    seg({ x: 0, y: 0 }, { x: W, y: 0 }),
    seg({ x: W, y: 0 }, { x: W, y: H }),
    seg({ x: W, y: H }, { x: 0, y: H }),
    seg({ x: 0, y: H }, { x: 0, y: 0 }),
  ];

  const offGrid = offGridJunctions([...ridges, ...axialFamily]);

  // Build the molecule from the CLIPPED geometry so no off-paper crease (e.g. an
  // edge flap's off-paper spine) leaks into the assignment as a dangling stub.
  // Hinges are routed later by assignMolecule (Phase 2), so start with none.
  const clippedFamily = [...axials, ...edgeAxials, ...pleats];
  const molecule: BoxPleatedMolecule = {
    sheet,
    boundary,
    ridges,
    axialFamily: clippedFamily,
    hinges: [],
    centers: seeds,
    ridgeSeeds: seeds,
    axialColorOf: axialChainColors(clippedFamily, ridges, sheet),
  };
  return { molecule, gap, offGrid, axials, edgeAxials, pleats, seeds, W, H };
}

/** Why a packing is not a valid CP candidate, or null if it is valid. */
function rejectionReason(g: PackingGeometry): string | null {
  if (!g.gap.complete) return "incomplete: an interior region cannot be filled (ODS rule #4)";
  if (g.offGrid.length > 0) return "off-grid: a crease junction misses the unit lattice";
  return null;
}

/**
 * The axial+ridge molecule (no hinges) for a VALID packing - the router's input.
 * Throws on a rejected candidate: geometry must never be built from an invalid
 * packing (use generateValidCP, or check buildPackingCP's `valid` flag).
 */
export function buildPackingMolecule(packing: BoxPleatedPacking): BoxPleatedMolecule {
  const g = packingGeometry(packing);
  const reason = rejectionReason(g);
  if (reason) throw new Error(`buildPackingMolecule: rejected packing (${reason})`);
  return g.molecule;
}

/** Build the crease pattern (Phase 1 colors + Phase 2 routed hinges) and its validity. */
export function buildPackingCP(packing: BoxPleatedPacking): PackingCP {
  const g = packingGeometry(packing);
  const { molecule, gap, offGrid, axials, edgeAxials, pleats, seeds, W, H } = g;
  const sheet = molecule.sheet;

  // Reject invalid candidates BEFORE the (expensive) M/V assignment, so no
  // downstream consumer ever receives creases from a packing that would be
  // rejected. `valid` is false and there are no assigned edges.
  if (rejectionReason(g)) {
    return {
      sheet,
      ridges: molecule.ridges,
      axials,
      edgeAxials,
      pleats,
      hinges: [],
      seeds,
      offGrid,
      assignedEdges: [],
      mvConflicts: [],
      failing: [],
      complete: gap.complete,
      valid: false,
    };
  }

  const assignment = assignMolecule(molecule, sheet);
  const hinges = clipAll(assignment.molecule.hinges, W, H);

  const adj = planarize([...molecule.boundary, ...molecule.ridges, ...molecule.axialFamily, ...hinges]);
  const failing = failingJunctions(adj, sheet).map((f) => ({ x: f.x, y: f.y }));

  return {
    sheet,
    ridges: molecule.ridges,
    axials,
    edgeAxials,
    pleats,
    hinges,
    seeds,
    offGrid,
    assignedEdges: assignment.edges,
    mvConflicts: assignment.conflicts,
    failing,
    complete: gap.complete,
    valid: gap.complete && offGrid.length === 0,
  };
}

/**
 * THE canonical entry: generate a packing and build its CP, returning it only if
 * valid, otherwise null. Every consumer that wants a crease pattern (data gen,
 * tests, debug renders) should go through this so a rejected candidate can never
 * leak downstream.
 */
export async function generateValidCP(config: BoxPleatedPackingConfig): Promise<PackingCP | null> {
  const cp = buildPackingCP(await generateBoxPleatedPacking(config));
  return cp.valid ? cp : null;
}

// ---------------------------------------------------------------------------
// Generation trace: an ordered, stage-by-stage record of the whole pipeline, for
// debugging. Replays the SAME pipeline functions the build uses (nothing is
// re-implemented), so what it shows is exactly what is generated.
// ---------------------------------------------------------------------------

export type CreaseKind = "boundary" | "ridge" | "axial" | "edge-axial" | "pleat" | CreaseType;

export interface TracedCrease {
  a: GridPoint;
  b: GridPoint;
  kind: CreaseKind;
  level?: number;
  mv?: Assignment | null;
}

/** One stage of generation: the creases it adds (or, if recolor, the full colored CP). */
export interface GenStage {
  name: string;
  creases: TracedCrease[];
  /** The final assignment stage replaces the whole CP with its M/V-coloured edges. */
  recolor?: boolean;
}

/** Ordered stage-by-stage trace of building this (valid) packing's CP. */
export function traceGeneration(packing: BoxPleatedPacking): { stages: GenStage[]; sheet: { width: number; height: number } } {
  const g = packingGeometry(packing);
  const reason = rejectionReason(g);
  if (reason) throw new Error(`traceGeneration: rejected packing (${reason})`);
  const { molecule, gap, W, H, axials, edgeAxials, seeds } = g;
  const sheet = molecule.sheet;
  const clip = (segs: OriSegment[]): OriSegment[] => {
    const out: OriSegment[] = [];
    for (const s of segs) pushClipped(out, s.a, s.b, W, H);
    return out;
  };

  const stages: GenStage[] = [];
  stages.push({ name: "paper boundary", creases: molecule.boundary.map((s) => ({ a: s.a, b: s.b, kind: "boundary" as const })) });

  // Ridges, grouped by their source object (so an extra crease can be traced to it).
  for (const grp of collectRidges(packing, gap, W, H).groups) {
    stages.push({ name: `ridges: ${grp.name}`, creases: grp.ridges.map((s) => ({ a: s.a, b: s.b, kind: "ridge" as const })) });
  }

  stages.push({ name: "base axials", creases: axials.map((s) => ({ a: s.a, b: s.b, kind: "axial" as const })) });
  stages.push({ name: "edge axials", creases: edgeAxials.map((s) => ({ a: s.a, b: s.b, kind: "edge-axial" as const })) });

  // Pleats, one stage per offset level (levels come from the same offset pass).
  const rawAx = propagateAxials(molecule.ridges, sheet, seeds);
  const fam = propagateAxialFamilyWithLevels(molecule.ridges, sheet, rawAx.axials, rawAx.edgeAxials);
  const levels = [...new Set(fam.pleats.map((p) => fam.level.get(segmentKey(p)) ?? 0))].sort((a, b) => a - b);
  for (const L of levels) {
    const lp = clip(fam.pleats.filter((p) => (fam.level.get(segmentKey(p)) ?? 0) === L));
    if (lp.length) stages.push({ name: `pleats: level ${L}`, creases: lp.map((s) => ({ a: s.a, b: s.b, kind: "pleat" as const, level: L })) });
  }

  // Hinges (Phase 2), then the final M/V assignment (recolours the whole CP).
  const hinges = routeHinges(molecule, sheet);
  if (hinges.length) stages.push({ name: "hinges", creases: clip(hinges).map((s) => ({ a: s.a, b: s.b, kind: "hinge" as const })) });
  const edges = assignCreases({ ...molecule, hinges });
  stages.push({ name: "assignment (M/V)", creases: edges.map((e) => ({ a: e.a, b: e.b, kind: e.type, mv: e.mv })), recolor: true });

  return { stages, sheet };
}

/**
 * Edges that appear on the boundary of two or more of an object's contours - the
 * internal creases where its sub-regions meet. (For a two-contour stretch device,
 * this is the single shared edge BP Studio omits from its ridge list.)
 */
function sharedContourEdges(
  contours: Array<{ outer: GridPoint[] }>,
): OriSegment[] {
  const count = new Map<string, { a: GridPoint; b: GridPoint; n: number }>();
  const round = (v: number): number => Math.round(v * 1e4) / 1e4;
  for (const c of contours) {
    const ring = c.outer;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i];
      const b = ring[(i + 1) % ring.length];
      const ka = `${round(a.x)},${round(a.y)}`;
      const kb = `${round(b.x)},${round(b.y)}`;
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      const entry = count.get(key) ?? { a, b, n: 0 };
      entry.n += 1;
      count.set(key, entry);
    }
  }
  return [...count.values()].filter((e) => e.n >= 2).map((e) => ({ a: e.a, b: e.b }));
}

/** Clip each segment to the paper, dropping any that fall entirely outside. */
function clipAll(segments: OriSegment[], W: number, H: number): OriSegment[] {
  const out: OriSegment[] = [];
  for (const s of segments) pushClipped(out, s.a, s.b, W, H);
  return out;
}

/** Clip a segment to [0,W]x[0,H] (Liang-Barsky) and push it if any part remains. */
function pushClipped(out: OriSegment[], a: GridPoint, b: GridPoint, W: number, H: number): void {
  let t0 = 0;
  let t1 = 1;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [a.x, W - a.x, a.y, H - a.y];
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < EPS) {
      if (q[i] < 0) return;
    } else {
      const r = q[i] / p[i];
      if (p[i] < 0) {
        if (r > t1) return;
        if (r > t0) t0 = r;
      } else {
        if (r < t0) return;
        if (r < t1) t1 = r;
      }
    }
  }
  if (t1 - t0 < EPS) return;
  out.push({
    a: { x: a.x + t0 * dx, y: a.y + t0 * dy },
    b: { x: a.x + t1 * dx, y: a.y + t1 * dy },
  });
}
