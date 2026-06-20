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

export interface GapFillOptions {
  /**
   * Allow flaps whose center rests on the paper edge to absorb odd parity (the
   * edge-crop case): an odd region touching the boundary is reflected across the
   * edge into an even rectangle, decomposed, and cropped back to the paper.
   */
  edgeCrop?: boolean;
}

export function fillBoxPleatedGaps(
  sheet: { width: number; height: number },
  occupied: OccupiedPolygon[],
  options: GapFillOptions = {},
): GapFillResult {
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  const empty = emptyGrid(W, H, occupied);
  const regions = connectedRegions(empty, W, H);

  const flaps: GapRect[] = [];
  const ridges: OriSegment[] = [];
  const unresolved: GapRect[] = [];
  for (const region of regions) {
    const bbox = boundingRect(region);
    // Only solid rectangular regions are fillable by axis-aligned flaps.
    const solid = region.length === (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
    const filled = solid ? fillRect(bbox, W, H, options.edgeCrop ?? false) : null;
    if (filled) {
      flaps.push(...filled.flaps);
      ridges.push(...filled.ridges);
    } else {
      unresolved.push(bbox);
    }
  }

  return { flaps, ridges, resolved: unresolved.length === 0, unresolved };
}

/** Fill one rectangular gap. Tries interior flaps, then edge-crop if allowed. */
function fillRect(r: GapRect, W: number, H: number, edgeCrop: boolean): { flaps: GapRect[]; ridges: OriSegment[] } | null {
  const interior = decomposeRect(r);
  if (interior) return { flaps: interior, ridges: interior.flatMap(flapRidges) };
  if (!edgeCrop) return null;

  // Reflect across a touched edge when the perpendicular dimension is odd, so
  // that dimension doubles to an even length. A dimension that stays odd is fine
  // as long as the other becomes even - decomposeRect then resolves it by
  // slicing. We only fail if both dimensions remain odd.
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  let ext = { ...r };
  if (w % 2 === 1) {
    if (r.x0 === 0) ext = { ...ext, x0: -r.x1 };
    else if (r.x1 === W) ext = { ...ext, x1: 2 * W - r.x0 };
  }
  if (h % 2 === 1) {
    if (r.y0 === 0) ext = { ...ext, y0: -r.y1 };
    else if (r.y1 === H) ext = { ...ext, y1: 2 * H - r.y0 };
  }

  const pieces = decomposeRect(ext);
  if (!pieces) return null;

  // Crop each (possibly out-of-paper) flap and its ridges back to the sheet.
  const flaps: GapRect[] = [];
  const ridges: OriSegment[] = [];
  for (const piece of pieces) {
    const clipped = clampRect(piece, W, H);
    if (!clipped) continue;
    flaps.push(clipped);
    for (const ridge of flapRidges(piece)) {
      const seg = clipToSheet(ridge, W, H);
      // Drop ridge pieces that lie on the paper boundary (non-folds).
      if (seg && !onSheetBoundary(seg, W, H)) ridges.push(seg);
    }
  }
  return { flaps, ridges };
}

/** Decompose an empty rectangle into valid (even-shorter-side) flaps, or null. */
function decomposeRect(r: GapRect): GapRect[] | null {
  const w = r.x1 - r.x0;
  const h = r.y1 - r.y0;
  const shorter = Math.min(w, h);
  const longer = Math.max(w, h);
  if (shorter % 2 === 0) return [r]; // already a valid flap
  if (longer % 2 === 0) {
    // Slice the even (longer) dimension into width-2 strips; each strip has an
    // even shorter side and is a valid flap.
    const strips: GapRect[] = [];
    if (w === longer) {
      for (let x = r.x0; x < r.x1; x += 2) strips.push({ x0: x, y0: r.y0, x1: x + 2, y1: r.y1 });
    } else {
      for (let y = r.y0; y < r.y1; y += 2) strips.push({ x0: r.x0, y0: y, x1: r.x1, y1: y + 2 });
    }
    return strips;
  }
  return null; // both odd - interior-unsolvable
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

// ---------------------------------------------------------------------------

function clampRect(r: GapRect, W: number, H: number): GapRect | null {
  const x0 = Math.max(r.x0, 0);
  const y0 = Math.max(r.y0, 0);
  const x1 = Math.min(r.x1, W);
  const y1 = Math.min(r.y1, H);
  if (x1 - x0 <= 0 || y1 - y0 <= 0) return null;
  return { x0, y0, x1, y1 };
}

/** Clip a segment to the sheet rectangle (Liang-Barsky), or null if outside. */
function clipToSheet(seg: OriSegment, W: number, H: number): OriSegment | null {
  let t0 = 0;
  let t1 = 1;
  const dx = seg.b.x - seg.a.x;
  const dy = seg.b.y - seg.a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [seg.a.x - 0, W - seg.a.x, seg.a.y - 0, H - seg.a.y];
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < EPS) {
      if (q[i] < 0) return null;
    } else {
      const r = q[i] / p[i];
      if (p[i] < 0) {
        if (r > t1) return null;
        if (r > t0) t0 = r;
      } else {
        if (r < t0) return null;
        if (r < t1) t1 = r;
      }
    }
  }
  if (t1 - t0 < EPS) return null;
  return {
    a: { x: seg.a.x + t0 * dx, y: seg.a.y + t0 * dy },
    b: { x: seg.a.x + t1 * dx, y: seg.a.y + t1 * dy },
  };
}

function onSheetBoundary(seg: OriSegment, W: number, H: number): boolean {
  const onX = (v: number): boolean => Math.abs(v) < EPS || Math.abs(v - W) < EPS;
  const onY = (v: number): boolean => Math.abs(v) < EPS || Math.abs(v - H) < EPS;
  const vertical = Math.abs(seg.a.x - seg.b.x) < EPS;
  const horizontal = Math.abs(seg.a.y - seg.b.y) < EPS;
  return (vertical && onX(seg.a.x)) || (horizontal && onY(seg.a.y));
}

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

function insidePolygon(p: GridPoint, poly: OccupiedPolygon): boolean {
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
