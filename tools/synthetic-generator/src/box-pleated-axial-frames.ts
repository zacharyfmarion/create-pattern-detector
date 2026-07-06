#!/usr/bin/env bun
// Frame-by-frame visualisation of axial-family generation for a packing. Each
// frame draws one more crease (in generation order: edge axials, then base
// axials, then pleats), highlighting the newest crease and circling every
// axial-by-axial X-crossing among the creases drawn so far. The frame where a
// new red X-crossing first appears is the crease whose march failed to terminate
// (e.g. overran a Y-junction). Writes one SVG per frame; assemble into a GIF with
//   qlmanage -t -s 700 -o <dir> <dir>/frame_*.svg
//   magick -delay 35 -loop 0 <dir>/frame_*.svg.png <dir>/axials.gif
//
//   bun run src/box-pleated-axial-frames.ts <seed> [--leaves N] [--out DIR] [--cell PX]

import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { generateBoxPleatedPacking } from "./box-pleated-packing.ts";
import { buildPackingCP } from "./box-pleated-cp.ts";
import type { GridPoint, OriSegment } from "./ori-parser.ts";

const EPS = 1e-6;
type Pt = { x: number; y: number };
const sub = (a: Pt, b: Pt): Pt => ({ x: a.x - b.x, y: a.y - b.y });
const cross = (a: Pt, b: Pt): number => a.x * b.y - a.y * b.x;
const dot = (a: Pt, b: Pt): number => a.x * b.x + a.y * b.y;
const near = (a: Pt, b: Pt): boolean => Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS;

// Interior-interior crossing point of two segments, or null. (Shared endpoints
// and T-junctions are legal and ignored - we only flag true X-crossings.)
function xCross(s1: OriSegment, s2: OriSegment): Pt | null {
  const a = s1.a as Pt, b = s1.b as Pt, c = s2.a as Pt, d = s2.b as Pt;
  const r = sub(b, a), s = sub(d, c);
  const denom = cross(r, s);
  if (Math.abs(denom) < EPS) return null; // parallel/collinear: not an X
  const t = cross(sub(c, a), s) / denom;
  const u = cross(sub(c, a), r) / denom;
  if (t < EPS || t > 1 - EPS || u < EPS || u > 1 - EPS) return null; // need both interiors
  return { x: a.x + t * r.x, y: a.y + t * r.y };
}

interface Crease {
  seg: OriSegment;
  kind: "edge" | "axial" | "pleat";
}

function parseArgs(argv: string[]): { seed: number; leaves?: number; out: string; cell: number } {
  let seed: number | undefined;
  let leaves: number | undefined;
  let out = "/tmp/bp-axial-frames";
  let cell = 18;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--leaves") leaves = Number(argv[++i]);
    else if (a === "--out") out = argv[++i];
    else if (a === "--cell") cell = Number(argv[++i]);
    else if (/^\d+$/.test(a) && seed === undefined) seed = Number(a);
  }
  if (seed === undefined) throw new Error("usage: box-pleated-axial-frames.ts <seed> [--leaves N] [--out DIR]");
  return { seed, leaves, out, cell };
}

async function main(): Promise<void> {
  const { seed, leaves, out, cell } = parseArgs(Bun.argv.slice(2));
  await mkdir(out, { recursive: true });
  const lv = leaves ?? [4, 5, 6][seed % 3];
  const packing = await generateBoxPleatedPacking({
    id: `axframes-${seed}`, seed, numCreases: 300, bucket: "s",
    symmetry: "none", targetLeafCount: lv, tight: true, tightRestarts: 14,
  });
  const cp = buildPackingCP(packing);
  const W = Math.round(cp.sheet.width), H = Math.round(cp.sheet.height);

  // Generation order: edge axials, then base axials, then pleats.
  const order: Crease[] = [
    ...cp.edgeAxials.map((seg) => ({ seg, kind: "edge" as const })),
    ...cp.axials.map((seg) => ({ seg, kind: "axial" as const })),
    ...cp.pleats.map((seg) => ({ seg, kind: "pleat" as const })),
  ];

  const m = 16;
  const X = (x: number) => x * cell + m;
  const Yf = (y: number) => (H - y) * cell + m;
  const ln = (a: GridPoint, b: GridPoint, col: string, w: number) =>
    `<line x1="${X(a.x).toFixed(1)}" y1="${Yf(a.y).toFixed(1)}" x2="${X(b.x).toFixed(1)}" y2="${Yf(b.y).toFixed(1)}" stroke="${col}" stroke-width="${w}"/>`;
  const colorOf = (k: Crease["kind"]) => (k === "axial" ? "#22d3ee" : k === "edge" ? "#a78bfa" : "#34d399");

  const width = W * cell + m * 2;
  const height = H * cell + m * 2 + 24;
  let prevCrossings = 0;
  let firstCulprit = -1;

  for (let i = 1; i <= order.length; i++) {
    const drawn = order.slice(0, i);
    const newest = drawn[i - 1];

    // X-crossings among all drawn creases (dedup by rounded point).
    const seenPt = new Set<string>();
    const crossings: Pt[] = [];
    for (let a = 0; a < drawn.length; a++)
      for (let b = a + 1; b < drawn.length; b++) {
        const p = xCross(drawn[a].seg, drawn[b].seg);
        if (!p) continue;
        const key = `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
        if (seenPt.has(key)) continue;
        seenPt.add(key);
        crossings.push(p);
      }
    const isNewCrossing = crossings.length > prevCrossings;
    if (isNewCrossing && firstCulprit < 0) firstCulprit = i;
    prevCrossings = crossings.length;

    let body = "";
    for (let x = 0; x <= W; x++) body += ln({ x, y: 0 }, { x, y: H }, "#ffffff10", 0.3);
    for (let y = 0; y <= H; y++) body += ln({ x: 0, y }, { x: W, y }, "#ffffff10", 0.3);
    for (let x = 0; x <= W; x += 2) body += `<text x="${X(x) - 3}" y="${Yf(0) + 11}" font-size="7" fill="#777">${x}</text>`;
    for (let y = 0; y <= H; y += 2) body += `<text x="${X(0) - 12}" y="${Yf(y) + 3}" font-size="7" fill="#777">${y}</text>`;
    // ridges faint
    for (const s of cp.ridges) body += ln(s.a, s.b, "#5b2330", 1);
    // previously drawn creases dim
    for (let k = 0; k < i - 1; k++) body += ln(drawn[k].seg.a, drawn[k].seg.b, "#5f6b70", 1);
    // newest crease bright
    body += ln(newest.seg.a, newest.seg.b, colorOf(newest.kind), 2.6);
    // crossings
    for (const p of crossings)
      body += `<circle cx="${X(p.x).toFixed(1)}" cy="${Yf(p.y).toFixed(1)}" r="5" fill="none" stroke="#ef4444" stroke-width="2"/>`;

    const flag = isNewCrossing ? "   *** NEW X-CROSSING ***" : "";
    const title = `seed ${seed}  frame ${i}/${order.length}  +${newest.kind} (${newest.seg.a.x},${newest.seg.a.y})-(${newest.seg.b.x},${newest.seg.b.y})  Xings=${crossings.length}${flag}`;
    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
      `<rect width="100%" height="100%" fill="#0a0a0a"/>` +
      `<text x="${m}" y="12" font-family="Helvetica" font-size="10" fill="${isNewCrossing ? "#fca5a5" : "#fff"}" font-weight="700">${title}</text>` +
      `<g transform="translate(0 18)">${body}</g></svg>`;
    await Bun.write(join(out, `frame_${String(i).padStart(3, "0")}.svg`), svg);
  }

  console.log(`wrote ${order.length} frames to ${out}`);
  console.log(`first crease that introduces an X-crossing: frame ${firstCulprit}`);
  if (firstCulprit > 0) {
    const c = order[firstCulprit - 1];
    console.log(`  -> +${c.kind} (${c.seg.a.x},${c.seg.a.y})-(${c.seg.b.x},${c.seg.b.y})`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
