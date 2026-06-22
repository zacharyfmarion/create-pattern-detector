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
  reflections: number;
}

const EPS = 1e-7;
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
  let reflections = 0;

  for (const seed of seeds) {
    for (const dir of AXIS_DIRS) {
      let from = seed;
      let d = dir;
      for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        const hit = marchRay(from, d, ridges, sheet);
        if (!hit) break;
        addSegment(all, seen, from, hit.point);
        if (hit.type === "boundary") break;
        if (hit.type === "junction") {
          junctionTerminations.push(hit.point);
          break;
        }
        // ridge interior: reflect across and continue.
        d = reflect(d, hit.ridgeDir!);
        from = hit.point;
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
  return { axials, edgeAxials, seeds, junctionTerminations, reflections };
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
 * Stage B: axial+1 pleat contours.
 *
 * Rule (from the manual construction): take every point that is exactly 1 unit
 * perpendicular-offset from an axial crease and is not already a crease, then
 * march a contour through it in both directions following the SAME rules as
 * axial generation (reflect across ridges, terminate at a multi-ridge junction
 * or the paper edge). Both the interior axials and the paper-edge axials are
 * valid sources to offset from.
 *
 * `axials` should be the interior axials; `edgeAxials` the paper-edge runs.
 */
export function propagateAxialOffsets(
  ridges: OriSegment[],
  sheet: { width: number; height: number },
  axials: OriSegment[],
  edgeAxials: OriSegment[],
): OriSegment[] {
  const out: OriSegment[] = [];
  const seen = new Set<string>();
  const march = (start: GridPoint, dir: GridPoint): void => {
    let from = start;
    let d = dir;
    for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
      const hit = marchRay(from, d, ridges, sheet);
      if (!hit) break;
      addSegment(out, seen, from, hit.point);
      if (hit.type !== "ridge") break;
      d = reflect(d, hit.ridgeDir!);
      from = hit.point;
    }
  };

  // Existing creases. A candidate seed point that already lies on any of these
  // is skipped - we only seed in still-empty paper, then let the ridge
  // reflection rules trace the contour.
  const creases = [...axials, ...edgeAxials, ...ridges];
  const onCrease = (p: GridPoint): boolean => creases.some((c) => pointOnSegment(p, c));

  // Seeds: every interior integer point one unit perpendicular off an axial
  // line, paired with the axial's direction. Offsetting from interior axials and
  // edge-axials alike. Dedupe identical (point, dir) seeds.
  const seedKeys = new Set<string>();
  for (const a of [...axials, ...edgeAxials]) {
    const dir = unit({ x: a.b.x - a.a.x, y: a.b.y - a.a.y });
    const perp = { x: -dir.y, y: dir.x };
    const steps = Math.round(Math.hypot(a.b.x - a.a.x, a.b.y - a.a.y));
    for (let i = 0; i <= steps; i++) {
      const base = { x: a.a.x + dir.x * i, y: a.a.y + dir.y * i };
      for (const sign of [1, -1]) {
        const p = { x: base.x + perp.x * sign, y: base.y + perp.y * sign };
        if (p.x < 0 || p.y < 0 || p.x > sheet.width || p.y > sheet.height) continue;
        if (onCrease(p)) continue; // only seed in empty paper
        const k = `${p.x},${p.y}:${dir.x},${dir.y}`;
        if (seedKeys.has(k)) continue;
        seedKeys.add(k);
        march(p, dir);
        march(p, { x: -dir.x, y: -dir.y });
      }
    }
  }
  return subtractRidgeOverlaps(out.filter((s) => !onPaperEdge(s, sheet)), ridges);
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
  const family: OriSegment[] = [...axials];
  const seen = new Set<string>(family.map(segmentKey));
  const level = new Map<string, number>();
  for (const s of axials) level.set(segmentKey(s), 0);
  let maxLevel = 0;
  for (let round = 1; round < 64; round++) {
    const next = propagateAxialOffsets(ridges, sheet, family, edgeAxials);
    let added = 0;
    for (const s of next) {
      const k = segmentKey(s);
      if (seen.has(k)) continue;
      seen.add(k);
      family.push(s);
      level.set(k, round);
      maxLevel = round;
      added++;
    }
    if (added === 0) break;
  }
  const axialKeys = new Set(axials.map(segmentKey));
  return { pleats: family.filter((s) => !axialKeys.has(segmentKey(s))), level, maxLevel };
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
export function planarize(segments: OriSegment[]): Map<string, GridPoint[]> {
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
  const kawasaki = deg >= 4 && deg % 2 === 0 && Math.abs(alt) < 1e-6;
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
  crossStop(ridges);
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
  return { x: from.x + dir.x * best, y: from.y + dir.y * best };
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
