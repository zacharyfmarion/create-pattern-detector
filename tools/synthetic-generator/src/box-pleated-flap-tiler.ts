// Flap-tiling solver for the gap-filling packing problem.
//
// Tile an empty region exactly with valid flap rectangles. A flap is valid when
// the straight skeleton of its "full" rectangle - the flap reflected across its
// paper-boundary-coincident sides - has its nodes on integer grid points, i.e.
// min(fullW, fullH) is even. Placing flaps directly in the region (rather than
// reflecting and slicing) keeps every flap's center on the paper.
//
// Two entry points share one backtracker:
//   - tileRegion: a W x H rectangle with each side labeled paper-boundary or
//     interior (used by the region fixtures/tests).
//   - tileEmptyCells: an arbitrary cell-set within a sheet, where any flap side
//     coinciding with the sheet edge is a paper boundary (used on real packings,
//     handling non-rectangular empty regions directly).
//
// Regions are small, so an exact top-left-anchored backtracking tiler suffices.

import type { OriSegment } from "./ori-parser.ts";
import type { RegionEdges } from "./box-pleated-region-fixtures.ts";
import { flapRidges, type GapRect } from "./box-pleated-gap-fill.ts";

export interface TilingResult {
  flaps: GapRect[];
  ridges: OriSegment[];
  /**
   * Ridges grouped per filler flap (parallel to `flaps`), so axial seeds can be
   * taken from each flap's OWN straight skeleton. Seeding from the flattened
   * `ridges` would fuse adjacent flaps and invent junctions at their shared
   * boundary (a point interior to neither skeleton).
   */
  ridgesByFlap: OriSegment[][];
  solved: boolean;
}

interface FreeSides {
  left: boolean;
  right: boolean;
  top: boolean;
  bottom: boolean;
}

/** Which sides of a flap rectangle coincide with the paper boundary. */
type FreeSidesFn = (r: GapRect) => FreeSides;

const EPS = 1e-9;

export function tileRegion(W: number, H: number, edges: RegionEdges): TilingResult {
  const occupied: boolean[][] = Array.from({ length: H }, () => new Array<boolean>(W).fill(false));
  const freeFn: FreeSidesFn = (r) => ({
    left: r.x0 === 0 && edges.left,
    right: r.x1 === W && edges.right,
    top: r.y0 === 0 && edges.top,
    bottom: r.y1 === H && edges.bottom,
  });
  return tile(occupied, W, H, freeFn);
}

/**
 * Tile the empty cells of a sheet (cells marked false in `occupied`) with valid
 * flaps. Handles non-rectangular empty regions directly. A flap side on the
 * sheet edge is a paper boundary (free); a side abutting an occupied cell is
 * interior.
 */
// Proving a region untileable can require exhausting the backtracker's whole
// search space, so we cap the number of placements explored. Hitting the cap
// reports the region as unsolved (the packing is then rejected) rather than
// hanging. Solvable regions are found well within this budget.
const MAX_NODES = 200_000;

export function tileEmptyCells(occupied: boolean[][], W: number, H: number): TilingResult {
  const grid = occupied.map((row) => row.slice());
  const freeFn: FreeSidesFn = (r) => ({
    left: r.x0 === 0,
    right: r.x1 === W,
    top: r.y0 === 0,
    bottom: r.y1 === H,
  });

  // Necessary condition: a fully-interior region (no cell on a sheet edge) can
  // only be tiled by even-area flaps, so an odd total area is untileable. Cheap
  // reject avoids exhausting the backtracker on the common odd-interior void.
  if (!touchesSheetEdge(grid, W, H) && emptyArea(grid, W, H) % 2 === 1) {
    return { flaps: [], ridges: [], ridgesByFlap: [], solved: false };
  }
  return tile(grid, W, H, freeFn);
}

function tile(occupied: boolean[][], W: number, H: number, freeFn: FreeSidesFn): TilingResult {
  const flaps = solve(occupied, W, H, freeFn, { nodes: 0 });
  if (!flaps) return { flaps: [], ridges: [], ridgesByFlap: [], solved: false };
  const ridgesByFlap = flaps.map((f) => croppedFlapRidges(f, freeFn));
  return { flaps, ridges: ridgesByFlap.flat(), ridgesByFlap, solved: true };
}

/** Backtracking exact tiling. Returns the flap list, or null if untileable / budget exhausted. */
function solve(occupied: boolean[][], W: number, H: number, freeFn: FreeSidesFn, budget: { nodes: number }): GapRect[] | null {
  const anchor = firstEmpty(occupied, W, H);
  if (!anchor) return [];
  const [cx, cy] = anchor;
  // The top-left-most empty cell must be the top-left corner of some flap, so
  // anchor there. Try larger flaps first (fewer pieces).
  for (let x1 = W; x1 > cx; x1--) {
    for (let y1 = H; y1 > cy; y1--) {
      const rect: GapRect = { x0: cx, y0: cy, x1, y1 };
      if (!rectEmpty(occupied, rect)) continue;
      if (!flapValid(rect, freeFn)) continue;
      if (++budget.nodes > MAX_NODES) return null;
      setRect(occupied, rect, true);
      const rest = solve(occupied, W, H, freeFn, budget);
      if (rest) return [rect, ...rest];
      setRect(occupied, rect, false);
    }
  }
  return null;
}

function emptyArea(occupied: boolean[][], W: number, H: number): number {
  let n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (!occupied[y][x]) n++;
  return n;
}

function touchesSheetEdge(occupied: boolean[][], W: number, H: number): boolean {
  for (let x = 0; x < W; x++) if (!occupied[0][x] || !occupied[H - 1][x]) return true;
  for (let y = 0; y < H; y++) if (!occupied[y][0] || !occupied[y][W - 1]) return true;
  return false;
}

/** A flap is valid iff its reflected "full" rectangle has an even shorter side. */
function flapValid(r: GapRect, freeFn: FreeSidesFn): boolean {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const free = freeFn(r);
  const fullW = free.left || free.right ? 2 * w : w;
  const fullH = free.top || free.bottom ? 2 * h : h;
  return Math.min(fullW, fullH) % 2 === 0;
}

/** Ridge creases of a placed flap: skeleton of the reflected full rect, cropped to the flap. */
function croppedFlapRidges(r: GapRect, freeFn: FreeSidesFn): OriSegment[] {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const free = freeFn(r);
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
