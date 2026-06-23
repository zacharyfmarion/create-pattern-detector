// Reusable debug view for a BP Studio packing: colours every river (tree node)
// distinctly with yellow boundaries between adjacent rivers, fills flaps grey,
// stretch devices green, and holes magenta, and overlays the straight-skeleton
// ridges. Use renderPackingDebugSvg(packing) to get an SVG string, or run
// scripts/debug-packing.ts <seed...> to render packings by seed.

import type { BoxPleatedPacking } from "./box-pleated-packing.ts";
import { packingEmptyGrid, packingRiverCells } from "./box-pleated-packing.ts";
import type { GridPoint } from "./ori-parser.ts";

const RIVER_PALETTE = ["#0e7490", "#a855f7", "#3b82f6", "#06b6d4", "#6366f1", "#64748b", "#0d9488", "#8b5cf6"];
const EPS = 1e-9;

interface DebugRenderOptions {
  /** Pixels per grid unit (default 36). */
  cellSize?: number;
}

function insidePolygon(p: GridPoint, poly: GridPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    if (yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}

/** Liang-Barsky clip of segment a-b to [0,W]x[0,H]; null if fully outside. */
function clip(a: GridPoint, b: GridPoint, W: number, H: number): [GridPoint, GridPoint] | null {
  let t0 = 0;
  let t1 = 1;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [a.x, W - a.x, a.y, H - a.y];
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
  return [
    { x: a.x + t0 * dx, y: a.y + t0 * dy },
    { x: a.x + t1 * dx, y: a.y + t1 * dy },
  ];
}

/** Render a labelled debug SVG of a packing (rivers, holes, flaps, stretch, ridges). */
export function renderPackingDebugSvg(packing: BoxPleatedPacking, options: DebugRenderOptions = {}): string {
  const W = Math.round(packing.sheet.width);
  const H = Math.round(packing.sheet.height);
  const cell = options.cellSize ?? 36;
  const m = 12;
  const X = (x: number): number => x * cell + m;
  const Yf = (y: number): number => (H - y) * cell + m;
  const line = (a: GridPoint, b: GridPoint, col: string, w: number): string =>
    `<line x1="${X(a.x).toFixed(1)}" y1="${Yf(a.y).toFixed(1)}" x2="${X(b.x).toFixed(1)}" y2="${Yf(b.y).toFixed(1)}" stroke="${col}" stroke-width="${w}"/>`;

  const empty = packingEmptyGrid(packing);
  const owner = packingRiverCells(packing); // "x,y" -> node id

  // Stable colour per river node.
  const nodeColor = new Map<number, string>();
  let next = 0;
  for (const node of [...new Set(owner.values())].sort((a, b) => a - b)) {
    nodeColor.set(node, RIVER_PALETTE[next++ % RIVER_PALETTE.length]);
  }

  const covers: Array<{ outer: GridPoint[]; inner?: GridPoint[][]; kind: string }> = [];
  for (const o of packing.layout.objects) {
    if (o.kind !== "flap" && o.kind !== "stretch-device") continue;
    for (const c of o.contours) covers.push({ outer: c.outer, inner: c.inner, kind: o.kind });
  }
  const covered = (cx: number, cy: number): boolean =>
    covers.some((pg) => {
      if (!insidePolygon({ x: cx, y: cy }, pg.outer)) return false;
      for (const h of pg.inner ?? []) if (insidePolygon({ x: cx, y: cy }, h)) return false;
      return true;
    });
  const inStretch = (cx: number, cy: number): boolean =>
    covers.some((pg) => pg.kind === "stretch-device" && insidePolygon({ x: cx, y: cy }, pg.outer));

  // "H" hole, "S" stretch, "F" flap, "R<node>" river, "." uncovered-untyped.
  const typeOf = (x: number, y: number): string => {
    if (empty[y][x]) return "H";
    const cx = x + 0.5;
    const cy = y + 0.5;
    if (inStretch(cx, cy)) return "S";
    const own = owner.get(`${x},${y}`);
    if (own !== undefined && !covered(cx, cy)) return `R${own}`;
    if (covered(cx, cy)) return "F";
    return ".";
  };

  let body = "";
  // Cell fills.
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const t = typeOf(x, y);
      let col = "#0e7490";
      if (t === "H") col = "#ff00aa";
      else if (t === "S") col = "#15803d";
      else if (t === "F") col = "#3a3a3a";
      else if (t[0] === "R") col = nodeColor.get(Number(t.slice(1))) ?? "#0e7490";
      body += `<rect x="${X(x)}" y="${Yf(y + 1)}" width="${cell}" height="${cell}" fill="${col}" fill-opacity="0.8"/>`;
    }
  }
  // Grid lines.
  for (let x = 0; x <= W; x++) body += line({ x, y: 0 }, { x, y: H }, "#ffffff0c", 0.3);
  for (let y = 0; y <= H; y++) body += line({ x: 0, y }, { x: W, y }, "#ffffff0c", 0.3);
  for (let x = 0; x <= W; x += 2) body += `<text x="${X(x) - 3}" y="${Yf(0) + 12}" font-size="9" fill="#aaa">${x}</text>`;
  for (let y = 0; y <= H; y += 2) body += `<text x="${X(0) - 11}" y="${Yf(y) + 3}" font-size="9" fill="#aaa">${y}</text>`;

  // Yellow boundaries between different rivers.
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const t = typeOf(x, y);
      if (t[0] !== "R") continue;
      if (x + 1 < W && typeOf(x + 1, y)[0] === "R" && typeOf(x + 1, y) !== t)
        body += line({ x: x + 1, y }, { x: x + 1, y: y + 1 }, "#fde047", 2.2);
      if (y + 1 < H && typeOf(x, y + 1)[0] === "R" && typeOf(x, y + 1) !== t)
        body += line({ x, y: y + 1 }, { x: x + 1, y: y + 1 }, "#fde047", 2.2);
    }
  }

  // Flap / stretch contour edges.
  for (const o of packing.layout.objects) {
    const col = o.kind === "flap" ? "#fb923c" : o.kind === "stretch-device" ? "#22c55e" : null;
    if (!col) continue;
    for (const c of o.contours) {
      for (const ring of [c.outer, ...(c.inner ?? [])]) {
        for (let i = 0; i < ring.length; i++) {
          const cl = clip(ring[i], ring[(i + 1) % ring.length], W, H);
          if (cl) body += line(cl[0], cl[1], col, 1.3);
        }
      }
    }
  }
  // Ridges (straight-skeleton creases).
  for (const o of packing.layout.objects) {
    for (const rg of o.ridges) {
      const cl = clip(rg[0], rg[1], W, H);
      if (cl) body += line(cl[0], cl[1], "#ff3657", 1);
    }
  }
  // Paper border.
  const corners: GridPoint[] = [
    { x: 0, y: 0 },
    { x: W, y: 0 },
    { x: W, y: H },
    { x: 0, y: H },
  ];
  for (let i = 0; i < 4; i++) body += line(corners[i], corners[(i + 1) % 4], "#fff", 1.6);

  let holeCells = 0;
  for (const row of empty) for (const v of row) if (v) holeCells++;
  const title = `seed ${packing.seed}  ${W}x${H}  rivers=${nodeColor.size}  holeCells=${holeCells}`;
  const legend =
    "each river = its own colour, YELLOW = boundary between rivers; magenta=hole, green=stretch, grey=flap, red=ridge";
  const width = W * cell + m * 2;
  const height = H * cell + m * 2 + 30;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
    `<rect width="100%" height="100%" fill="#0d0d0d"/>` +
    `<text x="${m}" y="13" font-family="Helvetica" font-size="12" fill="#fff" font-weight="700">${title}</text>` +
    `<text x="${m}" y="${height - 6}" font-family="Helvetica" font-size="10" fill="#fde047">${legend}</text>` +
    `<g transform="translate(0 18)">${body}</g>` +
    `</svg>`
  );
}
