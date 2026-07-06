// Stage A of the box-pleated crease construction: axial-contour propagation.
//
// Input: a fixture's paper boundary + ridge creases (blue packing discarded).
// Output: axial creases produced by launching contour lines from the ridge
// skeleton's interior junctions (the polygon "spine"/center) and marching them
// outward, reflecting off ridges per Lang's ODS Fig 13.26/13.27:
//   - 45-degree incidence on a ridge interior -> 90-degree turn (mirror reflect).
//   - reaching a junction where >=2 ridges meet -> terminate the contour.
//   - reaching the paper boundary -> terminate.
// (Y-junction "reflect across all" emission is not yet handled; such hits
// terminate and are reported, so we can see where it matters.)
//
// Axial contours also run ALONG the paper edge (Lang ODS: these edge runs are
// part of the axial contour but are not folds). We surface them separately as
// `edgeAxials` so the axial+n pleat elevation field has its level-0 sources on
// the boundary, not just on interior axials.

import type { GridPoint, OriSegment } from "./ori-parser.ts";

export interface AxialResult {
  /** Interior axial creases (actual folds). */
  axials: OriSegment[];
  /**
   * Axial-contour segments that run ALONG the paper edge. Per Lang ODS these are
   * still part of the axial contour (level 0) but are not folds. We keep them so
   * the pleat (axial+n) elevation field has its level-0 sources on the boundary.
   */
  edgeAxials: OriSegment[];
  seeds: GridPoint[];
  /** Contour hits that terminated at a multi-ridge junction (need Y-handling later). */
  junctionTerminations: GridPoint[];
  /**
   * Reflection/termination points that did NOT land on a grid point. Every axial
   * must reflect or end on an integer grid point; a non-empty list means the
   * packing produced a sub-grid crease (e.g. a Pythagorean stretch edge whose
   * reflection misses the grid) and should be rejected.
   */
  offGrid: GridPoint[];
  reflections: number;
}

const EPS = 1e-7;
// Float-dust tolerance for snapping a reflection/termination point to the grid.
// Far below any genuine sub-grid offset (a half cell, or a Pythagorean fraction
// like 1/3), so it only cleans rounding error and never hides a real off-grid hit.
const GRID_EPS = 1e-6;
// Tolerance for the Kawasaki alternating angle sum (radians). A junction incident
// to an off-grid vertex (where a non-45 stretch ridge crosses an axis crease at a
// fractional point) carries that vertex at a slightly irrational angle, so its
// angle sum misses zero by rounding dust (~6e-5 rad observed). This tolerance
// absorbs that dust while staying far below any genuine Kawasaki violation (which
// is on the order of a radian). The real off-grid vertices are still rejected -
// they are odd-degree, which fails the even-degree condition regardless.
const KAWASAKI_EPS = 1e-3;
const MAX_BOUNCES = 64;
const AXIS_DIRS: GridPoint[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 },
];

/**
 * Stage A: launch axial contours from each flap center in the four axis
 * directions and march them. A contour runs until it hits a ridge (where it
 * reflects across, or terminates at a multi-ridge junction) or runs off the
 * paper edge - hinges do NOT stop an axial. Contours traced from two different
 * flap centers can retrace the same reflected path from opposite ends; they
 * dedupe to one crease. Segments lying along the paper edge are dropped (they
 * are the boundary, not interior creases) - so corner flaps contribute none.
 */
export function propagateAxials(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  seeds: GridPoint[],
): AxialResult {
  const all: OriSegment[] = [];
  const seen = new Set<string>();
  const junctionTerminations: GridPoint[] = [];
  const offGrid: GridPoint[] = [];
  let reflections = 0;

  const insideSheet = (p: GridPoint): boolean =>
    p.x >= -EPS && p.y >= -EPS && p.x <= sheet.width + EPS && p.y <= sheet.height + EPS;

  for (const seed of seeds) {
    // A seed that sits on a stretch corner is the tip of a stretched (non-45)
    // flap whose axis runs into the parallelogram along an OFF-axis direction.
    // marchRay ignores ridges at a ray's own origin (t<=EPS), so an axis-aligned
    // ray fired from the seed can never reflect off the arm it starts on - the
    // stretch axis would be missing. Emit it explicitly: reflect the axis dirs
    // off each incident arm and keep the ones pointing into the stretch sector.
    for (const dir of [...AXIS_DIRS, ...stretchSeedDirs(seed, ridges)]) {
      let from = seed;
      let d = dir;
      for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        const hit = marchRay(from, d, ridges, sheet);
        if (!hit) break;
        // Every reflection/termination must land on a grid point. Snap away float
        // dust; a point not near any grid point is a genuine sub-grid crease - we
        // record it and stop this contour so callers can reject the packing.
        const point = snapToGrid(hit.point);
        if (!point) {
          offGrid.push(hit.point);
          addSegment(all, seen, from, hit.point);
          break;
        }
        // A corner/edge flap's center can lie outside the paper. Its axial is
        // seeded there and marches IN; the paper edge it first meets is an entry,
        // not a termination - drop the off-paper stub and keep marching inside so
        // the crease starts at the paper edge rather than overhanging it.
        if (hit.type === "boundary" && !insideSheet(from)) {
          from = point;
          continue;
        }
        addSegment(all, seen, from, point);
        if (hit.type === "boundary") break;
        if (hit.type === "junction") {
          junctionTerminations.push(point);
          break;
        }
        // ridge interior: reflect across and continue.
        d = reflect(d, hit.ridgeDir!);
        from = point;
        reflections++;
      }
    }
  }

  const edgeAxials = all.filter((s) => onPaperEdge(s, sheet));
  const interior = all.filter((s) => !onPaperEdge(s, sheet));
  // Where an axial runs ALONG an axis-aligned (spine) ridge it overlays the
  // ridge - two creases in one place. We dedupe to a single crease by dropping
  // the overlapping axial portion and keeping the ridge.
  // NOTE: uncertain this is correct. The merged crease is still a crease, but
  // its M/V may need special handling at assignment (it is "both" a ridge and
  // an axial). Revisit when we wire crease assignment.
  const axials = subtractRidgeOverlaps(interior, ridges);
  return { axials, edgeAxials, seeds, junctionTerminations, offGrid, reflections };
}

/**
 * Extra axial launch directions for a seed sitting on a Pythagorean-stretch
 * corner. The stretch flap's axis runs into the parallelogram along an off-axis
 * direction that is the reflection of an axis-aligned ray across one of the two
 * arms meeting at the corner. We reflect all four axis dirs across each incident
 * stretch arm (non-axis AND non-45 ridge) and keep a reflected direction only if
 * it points strictly INTO the sector spanned by the two arms (the parallelogram
 * interior) - that keeps the stretch axis (e.g. (4,3)) and drops the mirror-image
 * directions that reflect back out of the stretch. Returns [] for ordinary seeds.
 */
function stretchSeedDirs(seed: GridPoint, ridges: OriSegment[]): GridPoint[] {
  const arms: GridPoint[] = [];
  for (const r of ridges) {
    let o: GridPoint | null = null;
    if (samePoint(r.a, seed)) o = r.b;
    else if (samePoint(r.b, seed)) o = r.a;
    if (!o) continue;
    const dx = o.x - seed.x;
    const dy = o.y - seed.y;
    const len = Math.hypot(dx, dy);
    if (len < EPS) continue;
    if (Math.abs(dx) < EPS || Math.abs(dy) < EPS) continue; // axis-aligned: not a stretch arm
    if (Math.abs(Math.abs(dx) - Math.abs(dy)) < EPS) continue; // 45-degree: not a stretch arm
    arms.push({ x: dx / len, y: dy / len });
  }
  if (arms.length < 2) return [];
  const ang = (p: GridPoint): number => Math.atan2(p.y, p.x);
  const inSector = (d: GridPoint): boolean => {
    const da = ang(d);
    for (let i = 0; i < arms.length; i++) {
      for (let j = i + 1; j < arms.length; j++) {
        let lo = ang(arms[i]);
        let hi = ang(arms[j]);
        if (lo > hi) [lo, hi] = [hi, lo];
        if (hi - lo >= Math.PI - EPS) continue; // reflex span: interior is the other side
        if (da > lo + EPS && da < hi - EPS) return true;
      }
    }
    return false;
  };
  const out: GridPoint[] = [];
  const seen = new Set<string>();
  for (const arm of arms) {
    for (const ax of AXIS_DIRS) {
      const d = reflect(ax, arm);
      if (Math.abs(d.x) < EPS || Math.abs(d.y) < EPS) continue; // reflected back to an axis dir
      if (!inSector(d)) continue;
      const key = `${Math.round(d.x * 1e5)},${Math.round(d.y * 1e5)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(d);
    }
  }
  return out;
}

/** Snap a point to the integer grid if it is within float-dust tolerance, else null. */
function snapToGrid(p: GridPoint): GridPoint | null {
  const rx = Math.round(p.x);
  const ry = Math.round(p.y);
  return Math.abs(p.x - rx) < GRID_EPS && Math.abs(p.y - ry) < GRID_EPS ? { x: rx, y: ry } : null;
}

/** Remove the portions of each axial that lie along a collinear ridge. */
function subtractRidgeOverlaps(axials: OriSegment[], ridges: OriSegment[]): OriSegment[] {
  const out: OriSegment[] = [];
  for (const a of axials) {
    const dx = a.b.x - a.a.x;
    const dy = a.b.y - a.a.y;
    const len2 = dx * dx + dy * dy;
    const param = (p: GridPoint): number => ((p.x - a.a.x) * dx + (p.y - a.a.y) * dy) / len2;
    const collinear = (p: GridPoint): boolean => Math.abs((p.x - a.a.x) * dy - (p.y - a.a.y) * dx) < 1e-6;
    // Collect [t0,t1] intervals covered by collinear ridges.
    const covered: Array<[number, number]> = [];
    for (const r of ridges) {
      if (!collinear(r.a) || !collinear(r.b)) continue;
      const t0 = Math.max(0, Math.min(param(r.a), param(r.b)));
      const t1 = Math.min(1, Math.max(param(r.a), param(r.b)));
      if (t1 - t0 > 1e-6) covered.push([t0, t1]);
    }
    if (covered.length === 0) {
      out.push(a);
      continue;
    }
    covered.sort((p, q) => p[0] - q[0]);
    // Keep the complement of the covered intervals along [0,1].
    let cursor = 0;
    for (const [t0, t1] of covered) {
      if (t0 - cursor > 1e-6) out.push(segAt(a, cursor, t0));
      cursor = Math.max(cursor, t1);
    }
    if (1 - cursor > 1e-6) out.push(segAt(a, cursor, 1));
  }
  return out;
}

function segAt(s: OriSegment, t0: number, t1: number): OriSegment {
  const dx = s.b.x - s.a.x;
  const dy = s.b.y - s.a.y;
  return {
    a: { x: s.a.x + dx * t0, y: s.a.y + dy * t0 },
    b: { x: s.a.x + dx * t1, y: s.a.y + dy * t1 },
  };
}

/**
 * Stage B: axial+n pleat contours, generated level by level (axials = level 0).
 *
 * Manual construction (the author's process): for each crease at the current
 * level, walk it at unit gridpoints; offset 1 unit PERPENDICULAR to a candidate
 * point p. Keep p only if the unit connecting segment from the walk point to p
 * does NOT intersect or touch any ridge or any already-generated crease (touch
 * = a crossing, a collinear overlap, or p landing on a crease). Then march a new
 * crease from p in both directions: reflect at ridges, and STOP at the paper
 * edge or any existing crease - so a pleat never runs straight through an axial
 * or another pleat. The marched contour is a crease at the next level.
 *
 * `axials` are the interior axials; `edgeAxials` the paper-edge runs.
 */
export function propagateAxialOffsets(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  axials: OriSegment[],
  edgeAxials: OriSegment[],
): OriSegment[] {
  const sources = [...axials, ...edgeAxials];
  return offsetRound(ridges, sheet, sources, sources).pleats;
}

/** One level of offset pleats: offset each source crease, stopping at every crease. */
function offsetRound(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  frontier: OriSegment[],
  existingCreases: OriSegment[],
): { pleats: OriSegment[] } {
  // Stoppers grow as pleats are produced, so a later pleat stops at an earlier
  // one in the same round (matching drawing them one at a time).
  const stoppers = [...existingCreases];
  const produced: OriSegment[] = [];
  const seen = new Set<string>();
  for (const src of frontier) {
    const dir = unit({ x: src.b.x - src.a.x, y: src.b.y - src.a.y });
    // Only axis-parallel sources offset onto the grid; a non-45 stretch crease's
    // pleats live in the rotated grid (not handled yet), so skip it as a source.
    if (Math.abs(dir.x) > EPS && Math.abs(dir.y) > EPS) continue;
    const perp = { x: -dir.y, y: dir.x };
    const steps = Math.round(Math.hypot(src.b.x - src.a.x, src.b.y - src.a.y));
    for (let i = 0; i <= steps; i++) {
      const base = { x: src.a.x + dir.x * i, y: src.a.y + dir.y * i };
      // Do not offset from a walk point that lies on a ridge - that is where the
      // axial terminated against the ridge (e.g. a stretch corner), so a pleat
      // seeded there hangs off the junction. (connectingClear can't catch it: the
      // contact is at the foot, which is exempt as "on the source".)
      if (ridges.some((r) => pointOnSegment(base, r))) continue;
      for (const sign of [1, -1]) {
        const p = { x: base.x + perp.x * sign, y: base.y + perp.y * sign };
        if (p.x < 0 || p.y < 0 || p.x > sheet.width || p.y > sheet.height) continue;
        if (!connectingClear(base, p, ridges, stoppers)) continue;
        for (const seg of marchPleat(p, dir, ridges, stoppers, sheet)) {
          const k = segmentKey(seg);
          if (seen.has(k)) continue;
          seen.add(k);
          produced.push(seg);
          stoppers.push(seg);
        }
      }
    }
  }
  return { pleats: subtractRidgeOverlaps(produced.filter((s) => !onPaperEdge(s, sheet)), ridges) };
}

/**
 * The unit connecting segment from a source walk point `base` to its offset `p`
 * is clear iff it does not touch any ridge or crease except at its foot `base`
 * (which lies on the source). Touch = p on a crease, the segment's midpoint on a
 * crease (collinear overlap), or a proper interior crossing.
 */
function connectingClear(base: GridPoint, p: GridPoint, ridges: OriSegment[], creases: OriSegment[]): boolean {
  const mid = { x: (base.x + p.x) / 2, y: (base.y + p.y) / 2 };
  for (const c of ridges) {
    if (pointOnSegment(p, c) || pointOnSegment(mid, c) || segmentsCross(base, p, c.a, c.b)) return false;
  }
  for (const c of creases) {
    if (pointOnSegment(p, c) || pointOnSegment(mid, c) || segmentsCross(base, p, c.a, c.b)) return false;
  }
  return true;
}

/** March a pleat from `start` along ±dir: reflect at ridges, stop at any crease or the edge. */
function marchPleat(
  start: GridPoint,
  dir: GridPoint,
  ridges: OriSegment[],
  creases: OriSegment[],
  sheet: { width: number; height: number },
): OriSegment[] {
  const out: OriSegment[] = [];
  const seen = new Set<string>();
  for (const d0 of [dir, { x: -dir.x, y: -dir.y }]) {
    let from = start;
    let d = d0;
    for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
      const hit = marchPleatRay(from, d, ridges, creases, sheet);
      if (!hit) break;
      const point = snapToGrid(hit.point);
      if (!point) break; // off-grid reflection: stop rather than emit a sub-grid crease
      addSegment(out, seen, from, point);
      if (hit.type !== "ridge") break; // crease or boundary: stop
      d = reflect(d, hit.ridgeDir!);
      from = point;
    }
  }
  return out;
}

/** Nearest hit for a pleat march: ridge (reflect), crease (stop), or paper edge (stop). */
function marchPleatRay(
  from: GridPoint,
  dir: GridPoint,
  ridges: OriSegment[],
  creases: OriSegment[],
  sheet: { width: number; height: number },
): RayHit | null {
  let best: { t: number; hit: RayHit } | null = null;
  for (const r of ridges) {
    const inter = raySegmentIntersection(from, dir, r.a, r.b);
    if (!inter || inter.t <= EPS || inter.collinear) continue;
    const ridgeDir = straightRidgeDirAt(inter.point, ridges);
    const hit: RayHit = ridgeDir
      ? { point: inter.point, type: "ridge", ridgeDir }
      : { point: inter.point, type: "junction" };
    if (!best || inter.t < best.t - EPS) best = { t: inter.t, hit };
  }
  for (const c of creases) {
    const inter = raySegmentIntersection(from, dir, c.a, c.b);
    if (!inter || inter.t <= EPS || inter.collinear) continue;
    if (!best || inter.t < best.t - EPS) best = { t: inter.t, hit: { point: inter.point, type: "junction" } };
  }
  const boundaryHit = rayBoundaryIntersection(from, dir, sheet);
  if (boundaryHit && (!best || boundaryHit.t < best.t - EPS)) {
    best = { t: boundaryHit.t, hit: { point: boundaryHit.point, type: "boundary" } };
  }
  return best?.hit ?? null;
}

/**
 * Stage B (full): the complete axial+n pleat set. Iterates the axial+1 rule:
 * axial+1 are the pleats one unit off the axials; axial+2 are one unit off the
 * axial+1 family; and so on. We re-run the offset rule with the accumulated
 * axial-family creases as sources until no new creases appear. The seed
 * "skip if already on a crease" guard makes this converge - each round only the
 * frontier of empty paper produces new pleats.
 */
export function propagateAllAxialOffsets(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  axials: OriSegment[],
  edgeAxials: OriSegment[],
): OriSegment[] {
  return propagateAxialFamilyWithLevels(ridges, sheet, axials, edgeAxials).pleats;
}

export interface AxialFamilyLevels {
  /** Pleats (axial+1, +2, ...), excluding the level-0 axials. */
  pleats: OriSegment[];
  /** Map from segment key to pleat level: 0 = base axial, 1 = axial+1, etc. */
  level: Map<string, number>;
  /** The deepest level present. */
  maxLevel: number;
}

/**
 * Like {@link propagateAllAxialOffsets} but also records the pleat level of each
 * axial-family segment (0 = base axial, 1 = axial+1, ...). M/V assignment uses
 * the level parity: outermost (highest level) is mountain, alternating inward.
 */
export function propagateAxialFamilyWithLevels(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  axials: OriSegment[],
  edgeAxials: OriSegment[],
): AxialFamilyLevels {
  const level = new Map<string, number>();
  for (const s of axials) level.set(segmentKey(s), 0);
  // Every crease generated so far stops later pleats; ridges (reflectors) stay
  // separate. Level 0 = the axials and the paper-edge axials.
  const stoppers: OriSegment[] = [...axials, ...edgeAxials];
  const seen = new Set<string>(stoppers.map(segmentKey));
  const pleats: OriSegment[] = [];
  let frontier: OriSegment[] = [...axials, ...edgeAxials];
  let maxLevel = 0;
  for (let round = 1; round < 64; round++) {
    const { pleats: produced } = offsetRound(ridges, sheet, frontier, stoppers);
    const fresh: OriSegment[] = [];
    for (const s of produced) {
      const k = segmentKey(s);
      if (seen.has(k)) continue;
      seen.add(k);
      pleats.push(s);
      stoppers.push(s);
      level.set(k, round);
      fresh.push(s);
    }
    if (fresh.length === 0) break;
    maxLevel = round;
    frontier = fresh;
  }
  return { pleats, level, maxLevel };
}

export function segmentKey(s: OriSegment): string {
  const a = `${s.a.x},${s.a.y}`;
  const b = `${s.b.x},${s.b.y}`;
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

function onPaperEdge(s: OriSegment, sheet: { width: number; height: number }): boolean {
  const onX = (v: number): boolean => Math.abs(v) < EPS || Math.abs(v - sheet.width) < EPS;
  const onY = (v: number): boolean => Math.abs(v) < EPS || Math.abs(v - sheet.height) < EPS;
  const vertical = Math.abs(s.a.x - s.b.x) < EPS;
  const horizontal = Math.abs(s.a.y - s.b.y) < EPS;
  return (vertical && onX(s.a.x)) || (horizontal && onY(s.a.y));
}

/**
 * Seeds = flap centers. Each flap is a (possibly paper-cropped) square; its
 * ridges run from the center out to the flap corners, and every corner lies on
 * a hinge. So a flap center is exactly a ridge endpoint that lies on NO blue
 * hinge segment. Corner/edge flaps have centers on the paper boundary with the
 * square cropped outside the sheet; that is fine - only the center and its
 * inscribed circle need to be on the paper for the flap to be valid. Blue is
 * read only to classify endpoints, never emitted as a crease.
 */
export function findFlapCenters(
  ridges: OriSegment[],
  packing: OriSegment[],
  options?: { boundary: OriSegment[]; sheet: { width: number; height: number } },
): GridPoint[] {
  const onHinge = (p: GridPoint): boolean => packing.some((seg) => pointOnSegment(p, seg));
  // Ridge degree per endpoint - the flap center is the convergence of its ridges.
  const degree = new Map<string, number>();
  for (const r of ridges) {
    for (const p of [r.a, r.b]) degree.set(key(p), (degree.get(key(p)) ?? 0) + 1);
  }
  const deg = (p: GridPoint): number => degree.get(key(p)) ?? 0;

  const centers = new Map<string, GridPoint>();
  for (const r of ridges) {
    const aHinge = onHinge(r.a);
    const bHinge = onHinge(r.b);
    let center: GridPoint | null;
    if (aHinge && !bHinge) center = r.b; // corner -> center
    else if (bHinge && !aHinge) center = r.a;
    else if (!aHinge && !bHinge) center = deg(r.a) >= deg(r.b) ? r.a : r.b; // convergence end
    else center = null; // both on hinges: a ridge between two corners, no center here
    if (center) centers.set(key(center), center);
  }

  // Rivers do not generate axials - they are gaps consumed by axial+n from
  // neighbouring flaps. Drop any candidate center that sits in a river region.
  if (options) {
    const isFlapCell = classifyFlapRegions(packing, options.boundary, options.sheet);
    return [...centers.values()].filter((p) => cellsAround(p).some(([cx, cy]) => isFlapCell(cx, cy)));
  }
  return [...centers.values()];
}

function cellsAround(p: GridPoint): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (const dx of [-1, 0]) {
    for (const dy of [-1, 0]) out.push([Math.round(p.x) + dx, Math.round(p.y) + dy]);
  }
  return out;
}

/**
 * Classify each packing region as flap or river and return a predicate for
 * whether a unit cell belongs to a flap. The packing + paper boundary partition
 * the sheet into regions; a region is a RIVER when it is non-solid (it does not
 * fill its bounding box - an L, ring, or wrapping band, which is how rivers that
 * turn or encircle a flap appear). A solid rectangular region is a FLAP (a
 * possibly paper-edge-cropped square; note the blue packing does not always wall
 * a flap off from a neighbour, so flap regions can be non-square rectangles).
 * Matches the manual rule: rivers are fixed-width gaps that emit no axials.
 *
 * LIMITATION: a perfectly straight river spanning edge-to-edge is a solid
 * rectangle and would be misclassified as a flap. No current fixture exercises
 * that; revisit if one does.
 */
function classifyFlapRegions(
  packing: OriSegment[],
  boundary: OriSegment[],
  sheet: { width: number; height: number },
): (cx: number, cy: number) => boolean {
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  const vWall = new Set<string>();
  const hWall = new Set<string>();
  for (const s of [...packing, ...boundary]) {
    if (Math.abs(s.a.x - s.b.x) < EPS) {
      const x = Math.round(s.a.x);
      for (let y = Math.round(Math.min(s.a.y, s.b.y)); y < Math.round(Math.max(s.a.y, s.b.y)); y++) vWall.add(`${x},${y}`);
    } else if (Math.abs(s.a.y - s.b.y) < EPS) {
      const y = Math.round(s.a.y);
      for (let x = Math.round(Math.min(s.a.x, s.b.x)); x < Math.round(Math.max(s.a.x, s.b.x)); x++) hWall.add(`${x},${y}`);
    }
  }
  const label = new Int32Array(W * H).fill(-1);
  const idx = (x: number, y: number): number => y * W + x;
  const regions: Array<Array<[number, number]>> = [];
  for (let sy = 0; sy < H; sy++) {
    for (let sx = 0; sx < W; sx++) {
      if (label[idx(sx, sy)] !== -1) continue;
      const id = regions.length;
      const cells: Array<[number, number]> = [];
      const stack: Array<[number, number]> = [[sx, sy]];
      label[idx(sx, sy)] = id;
      while (stack.length) {
        const [x, y] = stack.pop()!;
        cells.push([x, y]);
        if (x > 0 && !vWall.has(`${x},${y}`) && label[idx(x - 1, y)] === -1) { label[idx(x - 1, y)] = id; stack.push([x - 1, y]); }
        if (x < W - 1 && !vWall.has(`${x + 1},${y}`) && label[idx(x + 1, y)] === -1) { label[idx(x + 1, y)] = id; stack.push([x + 1, y]); }
        if (y > 0 && !hWall.has(`${x},${y}`) && label[idx(x, y - 1)] === -1) { label[idx(x, y - 1)] = id; stack.push([x, y - 1]); }
        if (y < H - 1 && !hWall.has(`${x},${y + 1}`) && label[idx(x, y + 1)] === -1) { label[idx(x, y + 1)] = id; stack.push([x, y + 1]); }
      }
      regions.push(cells);
    }
  }
  const flap = new Array<boolean>(regions.length).fill(false);
  for (let i = 0; i < regions.length; i++) {
    const cs = regions[i];
    const xs = cs.map((c) => c[0]);
    const ys = cs.map((c) => c[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    // Solid (fills its bbox) -> flap; non-solid (L/ring/wrapping band) -> river.
    flap[i] = cs.length === bw * bh;
  }
  return (cx: number, cy: number): boolean =>
    cx < 0 || cy < 0 || cx >= W || cy >= H ? false : flap[label[idx(cx, cy)]];
}

/**
 * True when segment [p1,p2] crosses segment [p3,p4] at a point strictly inside
 * [p1,p2] (and on [p3,p4], endpoints included). Used to detect a ridge lying
 * between a pleat seed and its source axial.
 */
function segmentsCross(p1: GridPoint, p2: GridPoint, p3: GridPoint, p4: GridPoint): boolean {
  const r = { x: p2.x - p1.x, y: p2.y - p1.y };
  const s = { x: p4.x - p3.x, y: p4.y - p3.y };
  const denom = r.x * s.y - r.y * s.x;
  if (Math.abs(denom) < EPS) return false; // parallel or collinear
  const t = ((p3.x - p1.x) * s.y - (p3.y - p1.y) * s.x) / denom;
  const u = ((p3.x - p1.x) * r.y - (p3.y - p1.y) * r.x) / denom;
  return t > 1e-6 && t < 1 - 1e-6 && u > -1e-6 && u < 1 + 1e-6;
}

function pointOnSegment(p: GridPoint, seg: OriSegment): boolean {
  const cross = (seg.b.x - seg.a.x) * (p.y - seg.a.y) - (seg.b.y - seg.a.y) * (p.x - seg.a.x);
  if (Math.abs(cross) > 1e-6) return false;
  const dot = (p.x - seg.a.x) * (seg.b.x - seg.a.x) + (p.y - seg.a.y) * (seg.b.y - seg.a.y);
  const len2 = (seg.b.x - seg.a.x) ** 2 + (seg.b.y - seg.a.y) ** 2;
  return dot >= -1e-6 && dot <= len2 + 1e-6;
}

interface RayHit {
  point: GridPoint;
  type: "ridge" | "junction" | "boundary";
  ridgeDir?: GridPoint;
}

function marchRay(
  from: GridPoint,
  dir: GridPoint,
  ridges: OriSegment[],
  sheet: { width: number; height: number },
): RayHit | null {
  let best: { t: number; hit: RayHit } | null = null;

  // Ridge intersections. Reflect over a straight ridge (even one stored as
  // several collinear segments); terminate only at a true junction where ridges
  // of different directions meet.
  for (const r of ridges) {
    const inter = raySegmentIntersection(from, dir, r.a, r.b);
    if (!inter || inter.t <= EPS) continue;
    if (inter.collinear) continue; // leaving along this ridge - no crossing.
    const ridgeDir = straightRidgeDirAt(inter.point, ridges);
    const hit: RayHit = ridgeDir
      ? { point: inter.point, type: "ridge", ridgeDir }
      : { point: inter.point, type: "junction" };
    if (!best || inter.t < best.t - EPS) best = { t: inter.t, hit };
  }

  // The paper edge is the only non-ridge terminator.
  const boundaryHit = rayBoundaryIntersection(from, dir, sheet);
  if (boundaryHit && (!best || boundaryHit.t < best.t - EPS)) {
    best = { t: boundaryHit.t, hit: { point: boundaryHit.point, type: "boundary" } };
  }

  return best?.hit ?? null;
}

function reflect(d: GridPoint, ridgeDir: GridPoint): GridPoint {
  // The contour must cross the ridge (Lang ODS Fig 13.26): reflect d across the
  // ridge NORMAL, which preserves the component perpendicular to the ridge so
  // the contour continues to the far side, and flips the component along it.
  // d' = d - 2(d.r)r  (the opposite 90-degree turn from a line-mirror bounce).
  // ridgeDir is unit, so d' is unit too. Return the true reflected direction
  // rather than snapping to the 8 grid directions: over a 45-degree ridge an
  // axis-parallel axial reflects to an exact axis-parallel direction anyway,
  // but over a non-45-degree ridge (a Pythagorean stretch edge) the reflection
  // is genuinely off-axis and must not be collapsed.
  const dot = d.x * ridgeDir.x + d.y * ridgeDir.y;
  let rx = d.x - 2 * dot * ridgeDir.x;
  let ry = d.y - 2 * dot * ridgeDir.y;
  // Clean float dust so the 45-degree case stays exactly axis-parallel: a
  // component within EPS of 0 is exactly 0. Genuinely off-axis components (a
  // non-45-degree stretch reflection) are well above EPS and preserved.
  if (Math.abs(rx) < EPS) rx = 0;
  if (Math.abs(ry) < EPS) ry = 0;
  return unit({ x: rx, y: ry });
}

// ---- geometry helpers ----

function raySegmentIntersection(
  o: GridPoint,
  d: GridPoint,
  a: GridPoint,
  b: GridPoint,
): { t: number; point: GridPoint; collinear: boolean } | null {
  const ex = b.x - a.x;
  const ey = b.y - a.y;
  const denom = d.x * ey - d.y * ex;
  if (Math.abs(denom) < EPS) {
    // Parallel; treat collinear specially so we don't "reflect" along a ridge.
    const cross = (a.x - o.x) * d.y - (a.y - o.y) * d.x;
    return Math.abs(cross) < EPS ? { t: 0, point: o, collinear: true } : null;
  }
  const t = ((a.x - o.x) * ey - (a.y - o.y) * ex) / denom;
  const s = ((a.x - o.x) * d.y - (a.y - o.y) * d.x) / denom;
  if (t < EPS || s < -EPS || s > 1 + EPS) return null;
  return { t, point: { x: o.x + d.x * t, y: o.y + d.y * t }, collinear: false };
}

function rayBoundaryIntersection(
  o: GridPoint,
  d: GridPoint,
  sheet: { width: number; height: number },
): { t: number; point: GridPoint } | null {
  const candidates: Array<{ t: number; point: GridPoint }> = [];
  const consider = (t: number, x: number, y: number): void => {
    if (t > EPS && x >= -EPS && x <= sheet.width + EPS && y >= -EPS && y <= sheet.height + EPS) {
      candidates.push({ t, point: { x, y } });
    }
  };
  if (Math.abs(d.x) > EPS) {
    consider((0 - o.x) / d.x, 0, o.y + d.y * ((0 - o.x) / d.x));
    consider((sheet.width - o.x) / d.x, sheet.width, o.y + d.y * ((sheet.width - o.x) / d.x));
  }
  if (Math.abs(d.y) > EPS) {
    consider((0 - o.y) / d.y, o.x + d.x * ((0 - o.y) / d.y), 0);
    consider((sheet.height - o.y) / d.y, o.x + d.x * ((sheet.height - o.y) / d.y), sheet.height);
  }
  candidates.sort((p, q) => p.t - q.t);
  return candidates[0] ?? null;
}

/**
 * Junction vertices of a ridge set: points where two or more ridges of distinct
 * directions meet. For a single flap's straight skeleton these are its spine
 * convergence points - one for a point/square flap, two (the spine endpoints)
 * for a longer rectangular flap. Axials must be seeded from ALL of them, not
 * just the flap anchor, or a rectangular flap only grows half its molecule.
 */
export function ridgeJunctions(ridges: OriSegment[]): GridPoint[] {
  const byPoint = new Map<string, { point: GridPoint; dirs: GridPoint[] }>();
  for (const r of ridges) {
    for (const [end, other] of [[r.a, r.b], [r.b, r.a]] as const) {
      const dx = other.x - end.x;
      const dy = other.y - end.y;
      const len = Math.hypot(dx, dy);
      if (len < EPS) continue;
      const key = `${Math.round(end.x * 1e6) / 1e6},${Math.round(end.y * 1e6) / 1e6}`;
      const entry = byPoint.get(key) ?? { point: end, dirs: [] };
      entry.dirs.push({ x: dx / len, y: dy / len });
      byPoint.set(key, entry);
    }
  }
  const out: GridPoint[] = [];
  for (const { point, dirs } of byPoint.values()) {
    // A junction has >= 2 distinct (undirected) ridge directions.
    const distinct: GridPoint[] = [];
    for (const d of dirs) {
      if (!distinct.some((e) => Math.abs(e.x * d.y - e.y * d.x) < EPS)) distinct.push(d);
    }
    if (distinct.length >= 2) out.push(point);
  }
  return out;
}

/**
 * If `point` lies on a single straight ridge line - even one BP stored as
 * several collinear segments meeting end-to-end - return that line's direction
 * (the axial reflects over it). Return null at a true junction, where ridges of
 * two or more distinct directions meet (a corner, a flap-center X, a T), so the
 * axial terminates there. This lets an axial reflect at an interior grid point
 * of a straight ridge that merely happens to be split into segments.
 */
function straightRidgeDirAt(point: GridPoint, ridges: OriSegment[]): GridPoint | null {
  const lineDirs: GridPoint[] = [];
  for (const r of ridges) {
    if (!pointOnSegment(point, r)) continue;
    const d = unit({ x: r.b.x - r.a.x, y: r.b.y - r.a.y });
    if (d.x === 0 && d.y === 0) continue;
    // Treat a direction and its opposite as the same (undirected) line.
    if (!lineDirs.some((e) => Math.abs(e.x * d.y - e.y * d.x) < EPS)) lineDirs.push(d);
  }
  return lineDirs.length === 1 ? lineDirs[0] : null;
}

function unit(v: GridPoint): GridPoint {
  const len = Math.hypot(v.x, v.y);
  return len < EPS ? { x: 0, y: 0 } : { x: v.x / len, y: v.y / len };
}

function addSegment(out: OriSegment[], seen: Set<string>, a: GridPoint, b: GridPoint): void {
  if (samePoint(a, b)) return;
  const k = segKey(a, b);
  if (seen.has(k)) return;
  seen.add(k);
  out.push({ a, b });
}

function samePoint(a: GridPoint, b: GridPoint): boolean {
  return Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6;
}

function key(p: GridPoint): string {
  return `${round(p.x)},${round(p.y)}`;
}

function segKey(a: GridPoint, b: GridPoint): string {
  const ka = key(a);
  const kb = key(b);
  return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
}

function round(v: number): number {
  return Math.round(v * 1e4) / 1e4;
}

// ---------------------------------------------------------------------------
// Stage C: hinge creases.
//
// After axials and the axial+n pleats, some interior junctions still fail the
// local flat-foldability conditions (Kawasaki + even degree >= 4). A hinge
// crease resolves such a junction by extending from it until it meets a stop.
//
// Rule (from the manual construction):
//   - Only real junctions are considered (a straight degree-2 pass-through is
//     not a junction).
//   - For each failing junction, try a hinge in each axis direction that is not
//     already occupied by an incident crease. March it until the nearest stop:
//     a ridge, a flap center, another hinge, the start of a collinear
//     axial-family crease (a hinge may cross axials but not lie along one), or
//     the paper edge.
//   - Accept the hinge only if it actually makes that junction valid. Adding one
//     hinge can fix several junctions (e.g. a straight hinge connecting two
//     failing junctions through a non-center ridge crossing), and adding a hinge
//     can break a previously-valid vertex, so we re-evaluate the whole pattern
//     after each hinge and loop until nothing fails or no progress is possible.
// ---------------------------------------------------------------------------

export interface PlanarVertex {
  x: number;
  y: number;
  neighbors: GridPoint[];
}

export interface HingeResult {
  hinges: OriSegment[];
  /** Junctions still failing after hinge propagation (empty when fully solved). */
  unresolved: GridPoint[];
}

const HINGE_DIRS: GridPoint[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 },
];
const MAX_HINGE_ITERS = 2000;

export function propagateHinges(
  ridges: OriSegment[],
  axialFamily: OriSegment[],
  flapCenters: GridPoint[],
  sheet: { width: number; height: number },
  existing: OriSegment[],
): HingeResult {
  const all = [...existing];
  const hinges: OriSegment[] = [];
  const placed = new Set<string>();

  for (let iter = 0; iter < MAX_HINGE_ITERS; iter++) {
    const failing = failingJunctions(planarize(all), sheet);
    if (failing.length === 0) break;

    let progressed = false;
    for (const junction of failing) {
      const armDirs = new Set(
        junction.neighbors.map((n) => `${Math.sign(n.x - junction.x)},${Math.sign(n.y - junction.y)}`),
      );
      // Gather every direction whose hinge resolves this junction, then choose:
      //  1. prefer one that terminates on the paper edge over the interior (an
      //     edge-terminating hinge ends at a boundary vertex, not a new interior
      //     junction to balance), then
      //  2. among those, prefer the hinge that crosses the fewest axial-family
      //     creases - the shorter/cleaner route, which tends to keep the result
      //     globally flat-foldable.
      const candidates: Array<{ end: GridPoint; onEdge: boolean; axialCrossings: number }> = [];
      for (const dir of HINGE_DIRS) {
        if (armDirs.has(`${dir.x},${dir.y}`)) continue;
        const end = hingeEndpoint(
          { x: junction.x, y: junction.y },
          dir,
          ridges,
          flapCenters,
          hinges,
          [...axialFamily, ...hinges],
          sheet,
        );
        if (!end || samePoint(end, junction)) continue;
        if (placed.has(segKey({ x: junction.x, y: junction.y }, end))) continue;
        const candidate: OriSegment = { a: { x: junction.x, y: junction.y }, b: end };
        const probe = planarize([...all, candidate]);
        const refreshed = probe.get(key({ x: junction.x, y: junction.y }));
        if (refreshed && !isFailingJunction(junction.x, junction.y, refreshed, sheet)) {
          candidates.push({
            end,
            onEdge: isOnPaperEdge(end, sheet),
            axialCrossings: countAxialCrossings(candidate, axialFamily),
          });
        }
      }
      if (candidates.length === 0) continue;
      candidates.sort((a, b) => Number(b.onEdge) - Number(a.onEdge) || a.axialCrossings - b.axialCrossings);
      const candidate: OriSegment = { a: { x: junction.x, y: junction.y }, b: candidates[0].end };
      placed.add(segKey(candidate.a, candidate.b));
      hinges.push(candidate);
      all.push(candidate);
      progressed = true;
      break;
    }
    if (!progressed) break;
  }

  const unresolved = failingJunctions(planarize(all), sheet).map((j) => ({ x: j.x, y: j.y }));
  return { hinges, unresolved };
}

/** Planarize a set of segments into a vertex adjacency map (split at every crossing/T-junction). */
export let PLANARIZE_CALLS = 0;
export function resetPlanarizeCalls(): void { PLANARIZE_CALLS = 0; }
export function planarize(segments: OriSegment[]): Map<string, GridPoint[]> {
  PLANARIZE_CALLS++;
  const edges = new Set<string>();
  for (let i = 0; i < segments.length; i++) {
    const a = segments[i].a;
    const b = segments[i].b;
    const points: GridPoint[] = [a, b];
    for (let j = 0; j < segments.length; j++) {
      if (i === j) continue;
      const x = segmentIntersection(a, b, segments[j].a, segments[j].b);
      if (x) points.push(x);
      for (const e of [segments[j].a, segments[j].b]) {
        if (pointOnSegmentStrict(e, a, b)) points.push(e);
      }
    }
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    points.sort((p, q) => (p.x - a.x) * dx + (p.y - a.y) * dy - ((q.x - a.x) * dx + (q.y - a.y) * dy));
    for (let k = 0; k + 1 < points.length; k++) {
      const u = points[k];
      const v = points[k + 1];
      if (key(u) === key(v)) continue;
      edges.add(segKey(u, v));
    }
  }
  const adj = new Map<string, GridPoint[]>();
  for (const edge of edges) {
    const [uk, vk] = edge.split("|");
    const u = parseKey(uk);
    const v = parseKey(vk);
    pushNeighbor(adj, uk, v);
    pushNeighbor(adj, vk, u);
  }
  return adj;
}

function pushNeighbor(adj: Map<string, GridPoint[]>, vk: string, n: GridPoint): void {
  const list = adj.get(vk);
  if (!list) {
    adj.set(vk, [n]);
    return;
  }
  if (!list.some((p) => key(p) === key(n))) list.push(n);
}

/** Interior real junctions that fail Kawasaki or are not even degree >= 4. */
export function failingJunctions(adj: Map<string, GridPoint[]>, sheet: { width: number; height: number }): PlanarVertex[] {
  const out: PlanarVertex[] = [];
  for (const [vk, neighbors] of adj) {
    const v = parseKey(vk);
    if (isFailingJunction(v.x, v.y, neighbors, sheet)) out.push({ x: v.x, y: v.y, neighbors });
  }
  return out;
}

/**
 * Junction vertices that do NOT lie on an integer grid point. A junction is a
 * planar vertex where two or more non-collinear creases meet (a corner, T, or
 * crossing - a straight degree-2 pass-through is not one). Every crease
 * intersection in a box-pleat pattern must land on the grid; a non-empty result
 * means the packing produced a sub-grid crease (typically a non-45 stretch edge
 * or an off-grid river crossing the grid between lattice points) and should be
 * rejected.
 */
export function offGridJunctions(creases: OriSegment[]): GridPoint[] {
  const adj = planarize(creases);
  const out: GridPoint[] = [];
  for (const [vk, neighbors] of adj) {
    const v = parseKey(vk);
    if (Math.abs(v.x - Math.round(v.x)) < GRID_EPS && Math.abs(v.y - Math.round(v.y)) < GRID_EPS) continue;
    // Count distinct undirected edge directions; >= 2 means a real junction.
    const dirs: GridPoint[] = [];
    for (const n of neighbors) {
      const d = unit({ x: n.x - v.x, y: n.y - v.y });
      if (d.x === 0 && d.y === 0) continue;
      if (!dirs.some((e) => Math.abs(e.x * d.y - e.y * d.x) < EPS)) dirs.push(d);
    }
    if (dirs.length >= 2) out.push(v);
  }
  return out;
}

function isFailingJunction(vx: number, vy: number, neighbors: GridPoint[], sheet: { width: number; height: number }): boolean {
  // Boundary vertices are not interior flat-fold vertices.
  if (vx < EPS || vy < EPS || Math.abs(vx - sheet.width) < EPS || Math.abs(vy - sheet.height) < EPS) return false;
  const deg = neighbors.length;
  const angles = neighbors.map((n) => Math.atan2(n.y - vy, n.x - vx)).sort((a, b) => a - b);
  // A straight degree-2 pass-through is not a junction.
  if (deg === 2 && Math.abs(Math.abs(angles[1] - angles[0]) - Math.PI) < 1e-6) return false;
  const sectors: number[] = [];
  for (let i = 0; i < deg; i++) {
    let s = angles[(i + 1) % deg] - angles[i];
    if (s < 0) s += 2 * Math.PI;
    sectors.push(s);
  }
  let alt = 0;
  for (let i = 0; i < sectors.length; i++) alt += (i % 2 === 0 ? 1 : -1) * sectors[i];
  const kawasaki = deg >= 4 && deg % 2 === 0 && Math.abs(alt) < KAWASAKI_EPS;
  return !kawasaki;
}

/** Nearest stop for a hinge marching from `from` along axis `dir`. */
/**
 * March a hinge from `from` along axis `dir` to its nearest stop: a ridge, a
 * flap center, another hinge, the start of a collinear axial-family crease (a
 * hinge may cross axials but not lie along one), or the paper edge. Shared by
 * the geometry-stage hinge propagation and the M/V-stage ridge-crossing repair.
 */
export function hingeEndpoint(
  from: GridPoint,
  dir: GridPoint,
  ridges: OriSegment[],
  flapCenters: GridPoint[],
  hinges: OriSegment[],
  collinearAxials: OriSegment[],
  sheet: { width: number; height: number },
): GridPoint | null {
  let best = Infinity;
  const crossStop = (segs: OriSegment[]): void => {
    for (const r of segs) {
      const inter = raySegmentIntersection(from, dir, r.a, r.b);
      if (inter && !inter.collinear && inter.t > EPS && inter.t < best) best = inter.t;
    }
  };
  // A non-45 (Pythagorean stretch) ridge is off-grid everywhere except where it
  // passes through a lattice point. An axis-aligned hinge that stops on such a
  // ridge lands at a fractional point (an off-grid junction). Stop only at clean
  // 45/90 ridges; reject the whole direction below if it would reach a stretch
  // ridge at an off-grid point.
  const stretchRidges = ridges.filter((r) => !isCleanRidge(r));
  crossStop(ridges.filter(isCleanRidge));
  crossStop(hinges);
  for (const c of flapCenters) {
    const t = (c.x - from.x) * dir.x + (c.y - from.y) * dir.y;
    if (t <= EPS) continue;
    if (Math.abs(from.x + dir.x * t - c.x) < EPS && Math.abs(from.y + dir.y * t - c.y) < EPS && t < best) best = t;
  }
  const edge = rayBoundaryIntersection(from, dir, sheet);
  if (edge && edge.t > EPS && edge.t < best) best = edge.t;
  // A hinge may cross axials but must stop where it would start running along one.
  for (const c of collinearAxials) {
    const ex = c.b.x - c.a.x;
    const ey = c.b.y - c.a.y;
    if (Math.abs(ex * dir.y - ey * dir.x) > EPS) continue;
    if (Math.abs((c.a.x - from.x) * dir.y - (c.a.y - from.y) * dir.x) > EPS) continue;
    const ta = (c.a.x - from.x) * dir.x + (c.a.y - from.y) * dir.y;
    const tb = (c.b.x - from.x) * dir.x + (c.b.y - from.y) * dir.y;
    const hi = Math.max(ta, tb);
    if (hi > EPS) best = Math.min(best, Math.max(Math.min(ta, tb), 0));
  }
  if (!Number.isFinite(best) || best < EPS) return null;
  // Refuse to extend a hinge that would cross or end on a stretch ridge at an
  // off-grid point: the resulting junction would never be a unit box-pleat vertex.
  for (const r of stretchRidges) {
    const inter = raySegmentIntersection(from, dir, r.a, r.b);
    if (!inter || inter.collinear || inter.t <= EPS || inter.t > best + EPS) continue;
    const px = from.x + dir.x * inter.t;
    const py = from.y + dir.y * inter.t;
    if (Math.abs(px - Math.round(px)) > GRID_EPS || Math.abs(py - Math.round(py)) > GRID_EPS) return null;
  }
  return { x: from.x + dir.x * best, y: from.y + dir.y * best };
}

export interface HingeTrace {
  /** The reflected polyline from origin to terminus. */
  path: OriSegment[];
  terminus: GridPoint;
  terminusType: "edge" | "hinge" | "junction";
  /** Total grid distance travelled - its parity decides the terminus colour. */
  steps: number;
}

/**
 * Trace a hinge as a reflecting ray from `from` in `dir`. Like an axial it
 * reflects off ridges (mirror) and continues, crossing axials freely; it
 * terminates at the paper edge, on another hinge, or at a ridge junction (where
 * ridges of different directions meet and it cannot cleanly reflect). Returns
 * null if any reflection/termination lands off the integer grid (a non-45 stretch
 * reflection can never be a unit box-pleat vertex).
 */
export function traceHingeRay(
  from: GridPoint,
  dir: GridPoint,
  ridges: OriSegment[],
  hinges: OriSegment[],
  axials: OriSegment[],
  sheet: { width: number; height: number },
  opts: { allowOffGrid?: boolean } = {},
): HingeTrace | null {
  const path: OriSegment[] = [];
  let cur = from;
  let d = dir;
  let steps = 0;
  const stepLen = (a: GridPoint, b: GridPoint): number =>
    Math.round(Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)));
  // A hinge is perpendicular to axials and can never run COINCIDENT with one. The
  // test is per produced SEGMENT (cur -> pt), not the infinite forward ray: a segment
  // is invalid only if it actually overlaps an axial along its own extent. This lets a
  // hinge cross an empty gap toward a collinear axial and reflect off a ridge BEFORE
  // ever reaching it (touching an axial endpoint is a zero-length overlap, allowed),
  // while still forbidding a segment that truly runs down an axial.
  const segOverlapsAxial = (a: GridPoint, bpt: GridPoint): boolean => {
    const sx = bpt.x - a.x;
    const sy = bpt.y - a.y;
    const len2 = sx * sx + sy * sy;
    if (len2 < EPS) return false;
    const proj = (p: GridPoint): number => ((p.x - a.x) * sx + (p.y - a.y) * sy) / len2;
    return axials.some((c) => {
      const ex = c.b.x - c.a.x;
      const ey = c.b.y - c.a.y;
      if (Math.abs(ex * sy - ey * sx) > EPS) return false; // axial not parallel to segment
      if (Math.abs((c.a.x - a.x) * sy - (c.a.y - a.y) * sx) > EPS) return false; // axial not on the segment's line
      const u0 = proj(c.a);
      const u1 = proj(c.b);
      const lo = Math.max(0, Math.min(u0, u1));
      const hi = Math.min(1, Math.max(u0, u1));
      return hi - lo > EPS; // overlapping sub-interval of positive length
    });
  };

  for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
    const ride = marchRay(cur, d, ridges, sheet);
    let hinge: { t: number; point: GridPoint } | null = null;
    for (const h of hinges) {
      const inter = raySegmentIntersection(cur, d, h.a, h.b);
      if (inter && !inter.collinear && inter.t > EPS && (!hinge || inter.t < hinge.t)) {
        hinge = { t: inter.t, point: inter.point };
      }
    }
    const rideDist = ride ? Math.hypot(ride.point.x - cur.x, ride.point.y - cur.y) : Infinity;
    const hingeDist = hinge ? Math.hypot(hinge.point.x - cur.x, hinge.point.y - cur.y) : Infinity;

    if (hinge && hingeDist < rideDist - EPS) {
      const snapped = snapToGrid(hinge.point);
      if (!snapped && !opts.allowOffGrid) return null;
      const pt = snapped ?? hinge.point;
      if (segOverlapsAxial(cur, pt)) return null; // this segment would run along an axial
      path.push({ a: cur, b: pt });
      return { path, terminus: pt, terminusType: "hinge", steps: steps + stepLen(cur, pt) };
    }
    if (!ride) return null;
    if (ride.type === "boundary") {
      if (segOverlapsAxial(cur, ride.point)) return null;
      path.push({ a: cur, b: ride.point });
      return { path, terminus: ride.point, terminusType: "edge", steps: steps + stepLen(cur, ride.point) };
    }
    // A reflection off a non-45 Pythagorean stretch ridge can land sub-grid; such a
    // hinge is still valid (it reflects over the stretch), so allowOffGrid keeps the
    // raw point and marches on instead of discarding the whole ray.
    const snapped = snapToGrid(ride.point);
    if (!snapped && !opts.allowOffGrid) return null; // off-grid reflection or junction
    const pt = snapped ?? ride.point;
    if (segOverlapsAxial(cur, pt)) return null;
    path.push({ a: cur, b: pt });
    steps += stepLen(cur, pt);
    if (ride.type === "junction") return { path, terminus: pt, terminusType: "junction", steps };
    d = reflect(d, ride.ridgeDir!);
    cur = pt;
  }
  return null;
}

/** A 45/90 (axis-aligned or diagonal) ridge, vs a non-45 Pythagorean stretch ridge. */
function isCleanRidge(s: OriSegment): boolean {
  const dx = s.b.x - s.a.x;
  const dy = s.b.y - s.a.y;
  return Math.abs(dx) < EPS || Math.abs(dy) < EPS || Math.abs(Math.abs(dx) - Math.abs(dy)) < EPS;
}

function segmentIntersection(p1: GridPoint, p2: GridPoint, p3: GridPoint, p4: GridPoint): GridPoint | null {
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

function pointOnSegmentStrict(p: GridPoint, a: GridPoint, b: GridPoint): boolean {
  const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
  if (Math.abs(cross) > 1e-7) return false;
  const dot = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);
  const len2 = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
  return dot > -1e-7 && dot < len2 + 1e-7;
}

function parseKey(k: string): GridPoint {
  const [x, y] = k.split(",").map(Number);
  return { x, y };
}

function isOnPaperEdge(p: GridPoint, sheet: { width: number; height: number }): boolean {
  return Math.abs(p.x) < EPS || Math.abs(p.y) < EPS || Math.abs(p.x - sheet.width) < EPS || Math.abs(p.y - sheet.height) < EPS;
}

/** Count axial-family creases a hinge segment properly crosses (not just touches an endpoint). */
function countAxialCrossings(hinge: OriSegment, axialFamily: OriSegment[]): number {
  let count = 0;
  for (const a of axialFamily) {
    const inter = raySegmentIntersection(hinge.a, { x: hinge.b.x - hinge.a.x, y: hinge.b.y - hinge.a.y }, a.a, a.b);
    if (!inter || inter.collinear) continue;
    // Proper crossing strictly between the hinge endpoints.
    if (inter.t > EPS && inter.t < 1 - EPS) count++;
  }
  return count;
}
