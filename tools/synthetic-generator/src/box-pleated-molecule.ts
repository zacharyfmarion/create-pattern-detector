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

import type { GridPoint, OriSegment } from "./ori-parser.ts";

export interface AxialResult {
  axials: OriSegment[];
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

export function propagateAxials(
  ridges: OriSegment[],
  terminators: OriSegment[],
  sheet: { width: number; height: number },
  seeds: GridPoint[],
): AxialResult {
  const axials: OriSegment[] = [];
  const seen = new Set<string>();
  const junctionTerminations: GridPoint[] = [];
  let reflections = 0;

  for (const seed of seeds) {
    for (const dir of AXIS_DIRS) {
      let from = seed;
      let d = dir;
      for (let bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        const hit = marchRay(from, d, ridges, terminators, sheet);
        if (!hit) break;
        addSegment(axials, seen, from, hit.point);
        if (hit.type === "boundary") break;
        if (hit.type === "junction") {
          junctionTerminations.push(hit.point);
          break;
        }
        // ridge interior: reflect and continue.
        d = reflect(d, hit.ridgeDir!);
        from = hit.point;
        reflections++;
      }
    }
  }

  return { axials, seeds, junctionTerminations, reflections };
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
export function findFlapCenters(ridges: OriSegment[], packing: OriSegment[]): GridPoint[] {
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
  return [...centers.values()];
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
  terminators: OriSegment[],
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

  // Terminators: the packing (hinge) outlines + paper boundary. A contour stops
  // when it reaches its polygon's edge rather than leaking into a neighbour.
  for (const seg of terminators) {
    const inter = raySegmentIntersection(from, dir, seg.a, seg.b);
    if (!inter || inter.t <= EPS || inter.collinear) continue;
    if (!best || inter.t < best.t - EPS) best = { t: inter.t, hit: { point: inter.point, type: "boundary" } };
  }

  // Sheet rectangle as a guaranteed fallback terminator.
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
