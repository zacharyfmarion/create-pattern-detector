// Gap filling for box-pleated packings.
//
// A valid box-pleated packing must consume every bit of paper (Origami Design
// Secrets polygon-packing rule #4): all area is in a flap or river. BP Studio's
// optimizer leaves rectangular voids between the packed flaps/rivers. This module
// finds those empty regions and fills them with new flap rectangles, subject to
// the box-pleat grid constraint: a flap's straight-skeleton ridges must converge
// on integer grid points, which requires the flap's SHORTER side to be even.
//
// Decomposition of an empty axis-aligned rectangle (w x h):
//   - shorter side even            -> a single flap.
//   - shorter odd, longer even     -> slice the even (longer) dimension into
//                                     width-2 strips, each a valid even flap
//                                     (e.g. 7x10 -> five 7x2 flaps).
//   - both sides odd               -> not resolvable by interior flaps. (A flap
//                                     touching the paper edge can have its center
//                                     on the edge to absorb odd parity - the
//                                     edge-crop case - handled separately.)
//
// Regions we cannot resolve are reported in `unresolved`, and `resolved` is
// false; callers reject such packings.

import type { GridPoint, OriSegment } from "./ori-parser.ts";
import { tileEmptyCells } from "./box-pleated-flap-tiler.ts";

export interface GapRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface OccupiedPolygon {
  outer: GridPoint[];
  inner?: GridPoint[][];
}

export interface GapFillResult {
  /** New flap rectangles that fill the empty regions. */
  flaps: GapRect[];
  /** Straight-skeleton ridge creases for every new flap. */
  ridges: OriSegment[];
  /** True when every empty region was filled. */
  resolved: boolean;
  /** Bounding rectangles of empty regions that could not be filled. */
  unresolved: GapRect[];
}

const EPS = 1e-9;

export function fillBoxPleatedGaps(
  sheet: { width: number; height: number },
  occupied: OccupiedPolygon[],
): GapFillResult {
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  return fillEmptyGrid(W, H, emptyGrid(W, H, occupied));
}

/** Tile a pre-computed empty grid (true = empty cell) with filler flaps. */
export function fillBoxPleatedGapsFromGrid(
  sheet: { width: number; height: number },
  empty: boolean[][],
): GapFillResult {
  return fillEmptyGrid(Math.round(sheet.width), Math.round(sheet.height), empty);
}

function fillEmptyGrid(W: number, H: number, empty: boolean[][]): GapFillResult {
  const regions = connectedRegions(empty, W, H);

  const flaps: GapRect[] = [];
  const ridges: OriSegment[] = [];
  const unresolved: GapRect[] = [];
  for (const region of regions) {
    // Tile this region's actual cells (in sheet coordinates), so non-rectangular
    // empty regions are handled directly and flap free-sides come from the sheet
    // edge. Mark everything occupied except this region's cells.
    const grid: boolean[][] = Array.from({ length: H }, () => new Array<boolean>(W).fill(true));
    for (const [x, y] of region) grid[y][x] = false;
    const tiling = tileEmptyCells(grid, W, H);
    if (tiling.solved) {
      flaps.push(...tiling.flaps);
      ridges.push(...tiling.ridges);
    } else {
      unresolved.push(boundingRect(region));
    }
  }

  return { flaps, ridges, resolved: unresolved.length === 0, unresolved };
}

/** Straight-skeleton ridge creases of a rectangular flap. */
export function flapRidges(r: GapRect): OriSegment[] {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const corners = [
    { x: r.x0, y: r.y0 },
    { x: r.x1, y: r.y0 },
    { x: r.x1, y: r.y1 },
    { x: r.x0, y: r.y1 },
  ];
  if (w === h) {
    const c = { x: r.x0 + w / 2, y: r.y0 + h / 2 };
    return corners.map((p) => ({ a: p, b: c }));
  }
  if (w > h) {
    const s = h / 2;
    const left = { x: r.x0 + s, y: r.y0 + s };
    const right = { x: r.x1 - s, y: r.y0 + s };
    return [
      { a: { x: r.x0, y: r.y0 }, b: left },
      { a: { x: r.x0, y: r.y1 }, b: left },
      { a: { x: r.x1, y: r.y0 }, b: right },
      { a: { x: r.x1, y: r.y1 }, b: right },
      { a: left, b: right },
    ];
  }
  const s = w / 2;
  const top = { x: r.x0 + s, y: r.y0 + s };
  const bottom = { x: r.x0 + s, y: r.y1 - s };
  return [
    { a: { x: r.x0, y: r.y0 }, b: top },
    { a: { x: r.x1, y: r.y0 }, b: top },
    { a: { x: r.x0, y: r.y1 }, b: bottom },
    { a: { x: r.x1, y: r.y1 }, b: bottom },
    { a: top, b: bottom },
  ];
}

const RIDGE_EPS = 1e-6;

/**
 * BP Studio emits a non-square flap's ridges as a rectangular "ring" (four
 * axis-aligned sides) with the box's four 45-degree diagonals terminating on the
 * ring's corners - leaving the ring's interior un-creased: a rectangular donut
 * hole. The true straight skeleton continues into it - extend the diagonals
 * inward until they meet, then join the two meeting points with the spine
 * segment. This returns those missing interior creases (the straight skeleton of
 * the ring rectangle), or an empty list when the ridges have no 2D hole (a flap
 * whose skeleton already collapses to a point or segment).
 */
export function fillRidgeRectHole(ridges: OriSegment[]): OriSegment[] {
  const horizontal = ridges.filter((s) => Math.abs(s.a.y - s.b.y) < RIDGE_EPS);
  const vertical = ridges.filter((s) => Math.abs(s.a.x - s.b.x) < RIDGE_EPS);
  if (horizontal.length === 0 || vertical.length === 0) return [];

  // The ring is the bounding rectangle of the axis-aligned ridges.
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const s of [...horizontal, ...vertical]) {
    x0 = Math.min(x0, s.a.x, s.b.x);
    y0 = Math.min(y0, s.a.y, s.b.y);
    x1 = Math.max(x1, s.a.x, s.b.x);
    y1 = Math.max(y1, s.a.y, s.b.y);
  }
  // No 2D interior (a 1D spine, already creased) - nothing to fill.
  if (x1 - x0 < 1 - RIDGE_EPS || y1 - y0 < 1 - RIDGE_EPS) return [];

  // Confirm it is a closed rectangular ring: each of the four sides is covered by
  // an axis-aligned ridge spanning it. Otherwise these axis-aligned ridges are
  // not a flap's straight-skeleton core and we leave them alone.
  const spansH = (y: number): boolean =>
    horizontal.some(
      (s) =>
        Math.abs(s.a.y - y) < RIDGE_EPS &&
        Math.min(s.a.x, s.b.x) <= x0 + RIDGE_EPS &&
        Math.max(s.a.x, s.b.x) >= x1 - RIDGE_EPS,
    );
  const spansV = (x: number): boolean =>
    vertical.some(
      (s) =>
        Math.abs(s.a.x - x) < RIDGE_EPS &&
        Math.min(s.a.y, s.b.y) <= y0 + RIDGE_EPS &&
        Math.max(s.a.y, s.b.y) >= y1 - RIDGE_EPS,
    );
  if (!spansH(y0) || !spansH(y1) || !spansV(x0) || !spansV(x1)) return [];

  return flapRidges({ x0, y0, x1, y1 });
}

// ---------------------------------------------------------------------------

function emptyGrid(W: number, H: number, occupied: OccupiedPolygon[]): boolean[][] {
  const grid: boolean[][] = Array.from({ length: H }, () => new Array<boolean>(W).fill(true));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const center = { x: x + 0.5, y: y + 0.5 };
      if (occupied.some((poly) => insidePolygon(center, poly))) grid[y][x] = false;
    }
  }
  return grid;
}

function connectedRegions(empty: boolean[][], W: number, H: number): Array<Array<[number, number]>> {
  const seen = Array.from({ length: H }, () => new Array<boolean>(W).fill(false));
  const regions: Array<Array<[number, number]>> = [];
  for (let sy = 0; sy < H; sy++) {
    for (let sx = 0; sx < W; sx++) {
      if (!empty[sy][sx] || seen[sy][sx]) continue;
      const cells: Array<[number, number]> = [];
      const stack: Array<[number, number]> = [[sx, sy]];
      seen[sy][sx] = true;
      while (stack.length) {
        const [x, y] = stack.pop()!;
        cells.push([x, y]);
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < W && ny < H && empty[ny][nx] && !seen[ny][nx]) {
            seen[ny][nx] = true;
            stack.push([nx, ny]);
          }
        }
      }
      regions.push(cells);
    }
  }
  return regions;
}

function boundingRect(cells: Array<[number, number]>): GapRect {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of cells) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x + 1);
    y1 = Math.max(y1, y + 1);
  }
  return { x0, y0, x1, y1 };
}

export function insidePolygon(p: GridPoint, poly: OccupiedPolygon): boolean {
  if (!pointInRing(p, poly.outer)) return false;
  for (const hole of poly.inner ?? []) {
    if (pointInRing(p, hole)) return false;
  }
  return true;
}

function pointInRing(p: GridPoint, ring: GridPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].x;
    const yi = ring[i].y;
    const xj = ring[j].x;
    const yj = ring[j].y;
    if ((yi > p.y) !== (yj > p.y) && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi + EPS) + xi) {
      inside = !inside;
    }
  }
  return inside;
}
