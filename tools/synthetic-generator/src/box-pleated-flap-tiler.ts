// Flap-tiling solver for the gap-filling packing problem.
//
// Tile a W x H region (each side labeled paper-boundary or interior) exactly
// with valid flap rectangles. A flap is valid when the straight skeleton of its
// "full" rectangle - the flap reflected across its boundary-coincident sides -
// has its nodes on integer grid points, i.e. min(fullW, fullH) is even. Placing
// flaps directly in the region (rather than reflecting and slicing) keeps every
// flap's center on the paper by construction.
//
// Regions are small, so an exact top-left-anchored backtracking tiler suffices.

import type { OriSegment } from "./ori-parser.ts";
import type { RegionEdges } from "./box-pleated-region-fixtures.ts";
import { flapRidges, type GapRect } from "./box-pleated-gap-fill.ts";

export interface TilingResult {
  flaps: GapRect[];
  ridges: OriSegment[];
  solved: boolean;
}

const EPS = 1e-9;

export function tileRegion(W: number, H: number, edges: RegionEdges): TilingResult {
  const occupied: boolean[][] = Array.from({ length: H }, () => new Array<boolean>(W).fill(false));
  const flaps = solve(occupied, W, H, edges);
  if (!flaps) return { flaps: [], ridges: [], solved: false };
  const ridges = flaps.flatMap((f) => croppedFlapRidges(f, W, H, edges));
  return { flaps, ridges, solved: true };
}

/** Backtracking exact tiling. Returns the flap list, or null if untileable. */
function solve(occupied: boolean[][], W: number, H: number, edges: RegionEdges): GapRect[] | null {
  const anchor = firstEmpty(occupied, W, H);
  if (!anchor) return [];
  const [cx, cy] = anchor;
  // Try larger flaps first (fewer pieces). The top-left-most empty cell must be
  // the top-left corner of some flap, so anchor there.
  for (let x1 = W; x1 > cx; x1--) {
    for (let y1 = H; y1 > cy; y1--) {
      const rect: GapRect = { x0: cx, y0: cy, x1, y1 };
      if (!rectEmpty(occupied, rect)) continue;
      if (!flapValid(rect, W, H, edges)) continue;
      setRect(occupied, rect, true);
      const rest = solve(occupied, W, H, edges);
      if (rest) return [rect, ...rest];
      setRect(occupied, rect, false);
    }
  }
  return null;
}

/** A flap is valid iff its reflected "full" rectangle has an even shorter side. */
export function flapValid(r: GapRect, W: number, H: number, edges: RegionEdges): boolean {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const free = freeSides(r, W, H, edges);
  const fullW = free.left || free.right ? 2 * w : w;
  const fullH = free.top || free.bottom ? 2 * h : h;
  return Math.min(fullW, fullH) % 2 === 0;
}

function freeSides(r: GapRect, W: number, H: number, edges: RegionEdges) {
  return {
    left: r.x0 === 0 && edges.left,
    right: r.x1 === W && edges.right,
    top: r.y0 === 0 && edges.top,
    bottom: r.y1 === H && edges.bottom,
  };
}

/** Ridge creases of a placed flap: skeleton of the reflected full rect, cropped to the flap. */
function croppedFlapRidges(r: GapRect, W: number, H: number, edges: RegionEdges): OriSegment[] {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const free = freeSides(r, W, H, edges);
  const full: GapRect = {
    x0: free.left ? r.x0 - w : r.x0,
    x1: free.right ? r.x1 + w : r.x1,
    y0: free.top ? r.y0 - h : r.y0,
    y1: free.bottom ? r.y1 + h : r.y1,
  };
  const out: OriSegment[] = [];
  for (const ridge of flapRidges(full)) {
    const seg = clipToRect(ridge, r);
    if (seg && !onFreeBoundary(seg, r, free)) out.push(seg);
  }
  return out;
}

// ---------------------------------------------------------------------------

function firstEmpty(occupied: boolean[][], W: number, H: number): [number, number] | null {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!occupied[y][x]) return [x, y];
    }
  }
  return null;
}

function rectEmpty(occupied: boolean[][], r: GapRect): boolean {
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) {
      if (occupied[y][x]) return false;
    }
  }
  return true;
}

function setRect(occupied: boolean[][], r: GapRect, value: boolean): void {
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) occupied[y][x] = value;
  }
}

/** Clip a segment to the rectangle (Liang-Barsky), or null if outside. */
function clipToRect(seg: OriSegment, r: GapRect): OriSegment | null {
  let t0 = 0;
  let t1 = 1;
  const dx = seg.b.x - seg.a.x;
  const dy = seg.b.y - seg.a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [seg.a.x - r.x0, r.x1 - seg.a.x, seg.a.y - r.y0, r.y1 - seg.a.y];
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < EPS) {
      if (q[i] < 0) return null;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) {
        if (t > t1) return null;
        if (t > t0) t0 = t;
      } else {
        if (t < t0) return null;
        if (t < t1) t1 = t;
      }
    }
  }
  if (t1 - t0 < EPS) return null;
  return {
    a: { x: seg.a.x + t0 * dx, y: seg.a.y + t0 * dy },
    b: { x: seg.a.x + t1 * dx, y: seg.a.y + t1 * dy },
  };
}

/** A ridge piece lying along a free (paper-boundary) side is the on-edge spine - not a fold. */
function onFreeBoundary(
  seg: OriSegment,
  r: GapRect,
  free: { left: boolean; right: boolean; top: boolean; bottom: boolean },
): boolean {
  const vertical = Math.abs(seg.a.x - seg.b.x) < EPS;
  const horizontal = Math.abs(seg.a.y - seg.b.y) < EPS;
  if (vertical && free.left && Math.abs(seg.a.x - r.x0) < EPS) return true;
  if (vertical && free.right && Math.abs(seg.a.x - r.x1) < EPS) return true;
  if (horizontal && free.top && Math.abs(seg.a.y - r.y0) < EPS) return true;
  if (horizontal && free.bottom && Math.abs(seg.a.y - r.y1) < EPS) return true;
  return false;
}
