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

import type { BoxPleatedPacking } from "./box-pleated-packing.ts";
import { fillPackingGaps } from "./box-pleated-packing.ts";
import { repairFlapRidgeHole } from "./box-pleated-gap-fill.ts";
import {
  propagateAxials,
  propagateAllAxialOffsets,
  planarize,
  failingJunctions,
  offGridJunctions,
  ridgeJunctions,
} from "./box-pleated-molecule.ts";
import {
  assignMolecule,
  type AssignedEdge,
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
function packingGeometry(packing: BoxPleatedPacking): PackingGeometry {
  const sheet = packing.sheet;
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  const seg = (a: GridPoint, b: GridPoint): OriSegment => ({ a, b });

  const gap = fillPackingGaps(packing);

  // Ridges from every flap/river, EVERY stretch device, and the filler flaps.
  // A non-square flap's BP Studio ridges are a hollow rectangular ring (the box
  // diagonals stop at the ring corners). fillRidgeRectHole adds the straight
  // skeleton that fills that interior (spine + corner diagonals) so no empty
  // rectangle is left inside the flap. (We keep the ring sides: the pleats reflect
  // off them and the M/V balance depends on them - deleting them is worse.)
  const ridges: OriSegment[] = [];
  for (const object of packing.layout.objects) {
    if (object.kind === "root" || object.kind === "stretch-device") continue;
    const objRidges = object.ridges.map((line) => seg(line[0], line[1]));
    const repaired = object.kind === "flap" ? repairFlapRidgeHole(objRidges, sheet) : objRidges;
    for (const r of repaired) pushClipped(ridges, r.a, r.b, W, H);
  }
  for (const object of packing.layout.objects) {
    if (object.kind !== "stretch-device") continue;
    for (const line of object.ridges) pushClipped(ridges, line[0], line[1], W, H);
    // A stretch device split into two contours shares an interior edge between
    // them. BP Studio emits each contour's other edges as ridges but drops the
    // shared edge (it is the internal boundary between the two gadget halves),
    // leaving its two endpoints under-creased. That edge is a real ridge, so add
    // it back.
    for (const e of sharedContourEdges(object.contours)) pushClipped(ridges, e.a, e.b, W, H);
  }
  for (const r of gap.ridges) pushClipped(ridges, r.a, r.b, W, H);

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
  for (const group of gap.ridgesByFlap) rawSeeds.push(...ridgeJunctions(group));
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

/** The axial+ridge molecule (no hinges) for a packing - the router's input. */
export function buildPackingMolecule(packing: BoxPleatedPacking): BoxPleatedMolecule {
  return packingGeometry(packing).molecule;
}

/** Build the crease pattern (Phase 1 colors + Phase 2 routed hinges) and its validity. */
export function buildPackingCP(packing: BoxPleatedPacking): PackingCP {
  const g = packingGeometry(packing);
  const { molecule, gap, offGrid, axials, edgeAxials, pleats, seeds, W, H } = g;
  const sheet = molecule.sheet;

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
