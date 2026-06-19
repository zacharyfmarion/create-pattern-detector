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
  const family: OriSegment[] = [...axials];
  const seen = new Set<string>(family.map(segmentKey));
  for (let round = 0; round < 64; round++) {
    const next = propagateAxialOffsets(ridges, sheet, family, edgeAxials);
    let added = 0;
    for (const s of next) {
      const k = segmentKey(s);
      if (seen.has(k)) continue;
      seen.add(k);
      family.push(s);
      added++;
    }
    if (added === 0) break;
  }
  // Return only the pleats (everything beyond the level-0 axials).
  const axialKeys = new Set(axials.map(segmentKey));
  return family.filter((s) => !axialKeys.has(segmentKey(s)));
}

function segmentKey(s: OriSegment): string {
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

  // Ridge intersections (reflect, or terminate at a multi-ridge junction).
  for (const r of ridges) {
    const inter = raySegmentIntersection(from, dir, r.a, r.b);
    if (!inter || inter.t <= EPS) continue;
    if (inter.collinear) continue; // leaving along this ridge - no crossing.
    const atJunction = isRidgeEndpoint(inter.point, ridges);
    const hit: RayHit = atJunction
      ? { point: inter.point, type: "junction" }
      : { point: inter.point, type: "ridge", ridgeDir: unit({ x: r.b.x - r.a.x, y: r.b.y - r.a.y }) };
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
  const dot = d.x * ridgeDir.x + d.y * ridgeDir.y;
  const rx = d.x - 2 * dot * ridgeDir.x;
  const ry = d.y - 2 * dot * ridgeDir.y;
  return snapDir({ x: rx, y: ry });
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

function isRidgeEndpoint(point: GridPoint, ridges: OriSegment[]): boolean {
  for (const r of ridges) {
    if (samePoint(point, r.a) || samePoint(point, r.b)) return true;
  }
  return false;
}

function unit(v: GridPoint): GridPoint {
  const len = Math.hypot(v.x, v.y);
  return len < EPS ? { x: 0, y: 0 } : { x: v.x / len, y: v.y / len };
}

function snapDir(v: GridPoint): GridPoint {
  // Axis-parallel contours stay axis-parallel after a 90-degree reflection.
  const x = Math.abs(v.x) < EPS ? 0 : Math.sign(v.x);
  const y = Math.abs(v.y) < EPS ? 0 : Math.sign(v.y);
  return { x, y };
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
