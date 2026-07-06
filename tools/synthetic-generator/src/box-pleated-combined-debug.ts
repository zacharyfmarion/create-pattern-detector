#!/usr/bin/env bun
// Two-panel debug view of a packing in a single SVG:
//   LEFT  - the packing as a clean line drawing: paper border, flap/stretch
//           contours, river-region outlines, the straight-skeleton ridges, and
//           the hinge creases. No filled regions - boundary lines only.
//   RIGHT - the axial-family crease pattern with every X-crossing (red) and
//           T-junction (orange) marked.
// The two panels are embedded in one canvas that is padded to a square, so
// macOS `qlmanage` (which square-crops) renders the whole thing without clipping.
//
//   bun run src/box-pleated-combined-debug.ts <seed...> [--leaves N] [--out DIR] [--cell PX]

import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { generateBoxPleatedPacking, packingRiverCells } from "./box-pleated-packing.ts";
import type { BoxPleatedPacking } from "./box-pleated-packing.ts";
import { buildPackingCP } from "./box-pleated-cp.ts";
import { computeCrossings, renderCrossingsBody } from "./box-pleated-crossing-debug.ts";
import type { GridPoint } from "./ori-parser.ts";

const EPS = 1e-9;

/** Liang-Barsky clip of a-b to [0,W]x[0,H]; null if fully outside. */
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

/**
 * Left panel: the packing drawn as boundary lines only (no fills). Shows the
 * paper border, flap/stretch contours, river-region outlines (an edge wherever
 * two grid cells belong to different rivers, or a river abuts a non-river cell),
 * the straight-skeleton ridges, and the hinge creases.
 */
function renderPackingOutlineBody(
  packing: BoxPleatedPacking,
  cp: ReturnType<typeof buildPackingCP>,
  cell: number,
): { body: string; width: number; height: number } {
  const W = Math.round(packing.sheet.width);
  const H = Math.round(packing.sheet.height);
  const m = 16;
  const X = (x: number): number => x * cell + m;
  const Yf = (y: number): number => (H - y) * cell + m;
  const ln = (a: GridPoint, b: GridPoint, col: string, w: number): string =>
    `<line x1="${X(a.x).toFixed(1)}" y1="${Yf(a.y).toFixed(1)}" x2="${X(b.x).toFixed(1)}" y2="${Yf(b.y).toFixed(1)}" stroke="${col}" stroke-width="${w}"/>`;

  let body = "";
  // Faint grid + axis labels for orientation.
  for (let x = 0; x <= W; x++) body += ln({ x, y: 0 }, { x, y: H }, "#ffffff10", 0.3);
  for (let y = 0; y <= H; y++) body += ln({ x: 0, y }, { x: W, y }, "#ffffff10", 0.3);
  for (let x = 0; x <= W; x += 2) body += `<text x="${X(x) - 3}" y="${Yf(0) + 12}" font-size="8" fill="#888">${x}</text>`;
  for (let y = 0; y <= H; y += 2) body += `<text x="${X(0) - 13}" y="${Yf(y) + 3}" font-size="8" fill="#888">${y}</text>`;

  // River-region outlines: -1 = not a river cell.
  const owner = packingRiverCells(packing); // "x,y" -> node id
  const at = (x: number, y: number): number =>
    x < 0 || y < 0 || x >= W || y >= H ? -1 : owner.get(`${x},${y}`) ?? -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const here = at(x, y);
      if (here < 0) continue;
      if (at(x + 1, y) !== here) body += ln({ x: x + 1, y }, { x: x + 1, y: y + 1 }, "#38bdf8", 1.6);
      if (at(x - 1, y) !== here) body += ln({ x, y }, { x, y: y + 1 }, "#38bdf8", 1.6);
      if (at(x, y + 1) !== here) body += ln({ x, y: y + 1 }, { x: x + 1, y: y + 1 }, "#38bdf8", 1.6);
      if (at(x, y - 1) !== here) body += ln({ x, y }, { x: x + 1, y }, "#38bdf8", 1.6);
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
          if (cl) body += ln(cl[0], cl[1], col, 1.3);
        }
      }
    }
  }
  // Ridges (straight-skeleton) and hinges, from the built CP (gaps already filled).
  for (const s of cp.ridges) body += ln(s.a, s.b, "#ff3657", 1);
  for (const s of cp.hinges) body += ln(s.a, s.b, "#1e3a8a", 0.9);
  // Paper border.
  const corners: GridPoint[] = [
    { x: 0, y: 0 },
    { x: W, y: 0 },
    { x: W, y: H },
    { x: 0, y: H },
  ];
  for (let i = 0; i < 4; i++) body += ln(corners[i], corners[(i + 1) % 4], "#fff", 1.6);

  const title = `seed ${packing.seed}  ${W}x${H}  packing outline + ridges + hinges`;
  const legend = "sky=river boundary, orange=flap, green=stretch, red=ridge, blue=hinge";
  const width = W * cell + m * 2;
  const height = H * cell + m * 2 + 26;
  const inner =
    `<text x="${m}" y="13" font-family="Helvetica" font-size="11" fill="#fff" font-weight="700">${title}</text>` +
    `<text x="${m}" y="${height - 6}" font-family="Helvetica" font-size="9" fill="#9ca3af">${legend}</text>` +
    `<g transform="translate(0 18)">${body}</g>`;
  return { body: inner, width, height };
}

function parseArgs(argv: string[]): { seeds: number[]; leaves?: number; out: string; cell: number } {
  const seeds: number[] = [];
  let leaves: number | undefined;
  let out = "/tmp/bp-combined";
  let cell = 30;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--leaves") leaves = Number(argv[++i]);
    else if (a === "--out") out = argv[++i];
    else if (a === "--cell") cell = Number(argv[++i]);
    else if (/^\d+$/.test(a)) seeds.push(Number(a));
  }
  if (seeds.length === 0) throw new Error("usage: box-pleated-combined-debug.ts <seed...> [--leaves N] [--out DIR]");
  return { seeds, leaves, out, cell };
}

async function main(): Promise<void> {
  const { seeds, leaves, out, cell } = parseArgs(Bun.argv.slice(2));
  await mkdir(out, { recursive: true });
  for (const seed of seeds) {
    const lv = leaves ?? [4, 5, 6][seed % 3];
    const packing = await generateBoxPleatedPacking({
      id: `combined-${seed}`,
      seed,
      numCreases: 300,
      bucket: "s",
      symmetry: "none",
      targetLeafCount: lv,
      tight: true,
      tightRestarts: 14,
    });
    const cp = buildPackingCP(packing);
    const crossings = computeCrossings(cp);

    const left = renderPackingOutlineBody(packing, cp, cell);
    const right = renderCrossingsBody(seed, cp, crossings, cell);

    const gap = 40; // gutter between panels
    const pad = 24; // outer padding so nothing touches the edge
    const contentW = left.width + gap + right.width;
    const contentH = Math.max(left.height, right.height);
    // Pad to a square canvas; qlmanage square-crops, so a square is never clipped.
    const side = Math.max(contentW, contentH) + pad * 2;
    const ox = pad + (side - pad * 2 - contentW) / 2;
    const oy = pad + (side - pad * 2 - contentH) / 2;

    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}" viewBox="0 0 ${side} ${side}">` +
      `<rect width="100%" height="100%" fill="#080808"/>` +
      `<g transform="translate(${ox.toFixed(1)} ${oy.toFixed(1)})">` +
      `<svg x="0" y="0" width="${left.width}" height="${left.height}" overflow="visible">` +
      `<rect width="${left.width}" height="${left.height}" fill="#0a0a0a"/>${left.body}</svg>` +
      `<svg x="${left.width + gap}" y="0" width="${right.width}" height="${right.height}" overflow="visible">` +
      `<rect width="${right.width}" height="${right.height}" fill="#0a0a0a"/>${right.body}</svg>` +
      `</g></svg>`;

    const file = join(out, `${seed}.svg`);
    await Bun.write(file, svg);
    const nX = crossings.filter((c) => c.kind === "X").length;
    const nT = crossings.filter((c) => c.kind === "T").length;
    console.log(`${seed} (lv${lv}): canvas ${side}x${side}  X=${nX} T=${nT}  ${file}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
