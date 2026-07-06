#!/usr/bin/env bun
// Debug axial-family crease intersections for a packing. Every axial, edge-axial,
// and pleat is an axial crease; the only legal way two of them can meet is at a
// shared vertex (an endpoint of both). This script classifies every other
// intersection:
//   X       - both creases pass through the point in their interior (two creases
//             genuinely crossing with no vertex). This must never happen.
//   T       - the point is an interior point of one crease and an endpoint of the
//             other (one crease ends on another). Usually a legal degree-3 vertex.
//   OVERLAP - the creases are collinear and share a span.
// Renders an SVG with X marked red, T orange, OVERLAP yellow, over the grid.
//
//   bun run src/box-pleated-crossing-debug.ts <seed...> [--leaves N] [--out DIR]

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

// Is p on segment a-b (inclusive of endpoints)?
function onSeg(a: Pt, b: Pt, p: Pt): boolean {
  const ab = sub(b, a);
  const ap = sub(p, a);
  if (Math.abs(cross(ab, ap)) > EPS * Math.hypot(ab.x, ab.y)) return false;
  const t = dot(ap, ab) / dot(ab, ab);
  return t >= -EPS && t <= 1 + EPS;
}
const isEndpoint = (a: Pt, b: Pt, p: Pt): boolean => near(a, p) || near(b, p);
// p on segment a-b strictly between the endpoints
const isInterior = (a: Pt, b: Pt, p: Pt): boolean => onSeg(a, b, p) && !isEndpoint(a, b, p);

export interface Crossing {
  kind: "X" | "T" | "OVERLAP";
  at: Pt;
}

/** Classify every pair of axial-family creases (axials, edge-axials, pleats). */
export function computeCrossings(cp: ReturnType<typeof buildPackingCP>): Crossing[] {
  const fam = [...cp.axials, ...cp.edgeAxials, ...cp.pleats];
  const W = Math.round(cp.sheet.width);
  const H = Math.round(cp.sheet.height);
  // Two creases meeting on the paper boundary is fine (axials run along and end
  // on the edge); only flag intersections in the interior of the paper.
  const onBoundary = (p: Pt): boolean =>
    Math.abs(p.x) < EPS || Math.abs(p.x - W) < EPS || Math.abs(p.y) < EPS || Math.abs(p.y - H) < EPS;
  const out: Crossing[] = [];
  for (let i = 0; i < fam.length; i++) {
    for (let j = i + 1; j < fam.length; j++) {
      const c = classify(fam[i], fam[j]);
      if (c && !onBoundary(c.at)) out.push(c);
    }
  }
  return out;
}

// Classify how two axial-family segments meet. Returns null if they only touch at
// a shared endpoint (a legal vertex) or do not meet at all.
function classify(s1: OriSegment, s2: OriSegment): Crossing | null {
  const a = s1.a as Pt;
  const b = s1.b as Pt;
  const c = s2.a as Pt;
  const d = s2.b as Pt;
  const r = sub(b, a);
  const s = sub(d, c);
  const denom = cross(r, s);

  if (Math.abs(denom) < EPS) {
    // Parallel. Collinear overlap?
    if (Math.abs(cross(sub(c, a), r)) > EPS) return null; // parallel, not collinear
    const rr = dot(r, r);
    const t0 = dot(sub(c, a), r) / rr;
    const t1 = dot(sub(d, a), r) / rr;
    const lo = Math.max(0, Math.min(t0, t1));
    const hi = Math.min(1, Math.max(t0, t1));
    if (hi - lo > EPS) {
      const at = { x: a.x + ((lo + hi) / 2) * r.x, y: a.y + ((lo + hi) / 2) * r.y };
      return { kind: "OVERLAP", at };
    }
    return null;
  }

  const t = cross(sub(c, a), s) / denom;
  const u = cross(sub(c, a), r) / denom;
  if (t < -EPS || t > 1 + EPS || u < -EPS || u > 1 + EPS) return null;
  const at = { x: a.x + t * r.x, y: a.y + t * r.y };

  const int1 = isInterior(a, b, at);
  const int2 = isInterior(c, d, at);
  if (int1 && int2) return { kind: "X", at };
  if (int1 || int2) return { kind: "T", at };
  return null; // shared endpoint - legal vertex
}

function parseArgs(argv: string[]): { seeds: number[]; leaves?: number; out: string } {
  const seeds: number[] = [];
  let leaves: number | undefined;
  let out = "/tmp/bp-crossings";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--leaves") leaves = Number(argv[++i]);
    else if (a === "--out") out = argv[++i];
    else if (/^\d+$/.test(a)) seeds.push(Number(a));
  }
  return { seeds, leaves, out };
}

/** Inner SVG markup (no <svg> wrapper) plus its pixel size, for embedding. */
export function renderCrossingsBody(
  seed: number,
  cp: ReturnType<typeof buildPackingCP>,
  crossings: Crossing[],
  cell = 30,
): { body: string; width: number; height: number } {
  const W = Math.round(cp.sheet.width);
  const H = Math.round(cp.sheet.height);
  const m = 16;
  const X = (x: number): number => x * cell + m;
  const Yf = (y: number): number => (H - y) * cell + m;
  const ln = (a: GridPoint, b: GridPoint, col: string, w: number): string =>
    `<line x1="${X(a.x).toFixed(1)}" y1="${Yf(a.y).toFixed(1)}" x2="${X(b.x).toFixed(1)}" y2="${Yf(b.y).toFixed(1)}" stroke="${col}" stroke-width="${w}"/>`;
  let body = "";
  for (let x = 0; x <= W; x++) body += ln({ x, y: 0 }, { x, y: H }, "#ffffff14", 0.4);
  for (let y = 0; y <= H; y++) body += ln({ x: 0, y }, { x: W, y }, "#ffffff14", 0.4);
  for (let x = 0; x <= W; x += 2) body += `<text x="${X(x) - 3}" y="${Yf(0) + 12}" font-size="8" fill="#888">${x}</text>`;
  for (let y = 0; y <= H; y += 2) body += `<text x="${X(0) - 13}" y="${Yf(y) + 3}" font-size="8" fill="#888">${y}</text>`;
  for (const s of cp.ridges) body += ln(s.a, s.b, "#7f1d1d", 1);
  for (const s of cp.hinges) body += ln(s.a, s.b, "#1e3a8a", 0.7);
  for (const s of cp.axials) body += ln(s.a, s.b, "#22d3ee", 1.4);
  for (const s of cp.edgeAxials) body += ln(s.a, s.b, "#a78bfa", 1.4);
  for (const s of cp.pleats) body += ln(s.a, s.b, "#34d399", 1.1);
  for (const c of crossings) {
    const col = c.kind === "X" ? "#ef4444" : c.kind === "T" ? "#f59e0b" : "#facc15";
    const rad = c.kind === "X" ? 6 : 4;
    body += `<circle cx="${X(c.at.x).toFixed(1)}" cy="${Yf(c.at.y).toFixed(1)}" r="${rad}" fill="none" stroke="${col}" stroke-width="2"/>`;
  }
  const nX = crossings.filter((c) => c.kind === "X").length;
  const nT = crossings.filter((c) => c.kind === "T").length;
  const nO = crossings.filter((c) => c.kind === "OVERLAP").length;
  const title = `seed ${seed}  ${W}x${H}  X(bad)=${nX} red, T=${nT} orange, OVERLAP=${nO} yellow`;
  const legend = "cyan=axial  purple=edge-axial  green=pleat  dark-red=ridge  blue=hinge";
  const width = W * cell + m * 2;
  const height = H * cell + m * 2 + 26;
  const inner =
    `<text x="${m}" y="13" font-family="Helvetica" font-size="11" fill="#fff" font-weight="700">${title}</text>` +
    `<text x="${m}" y="${height - 6}" font-family="Helvetica" font-size="9" fill="#9ca3af">${legend}</text>` +
    `<g transform="translate(0 18)">${body}</g>`;
  return { body: inner, width, height };
}

function render(seed: number, cp: ReturnType<typeof buildPackingCP>, crossings: Crossing[]): string {
  const { body, width, height } = renderCrossingsBody(seed, cp, crossings);
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
    `<rect width="100%" height="100%" fill="#0a0a0a"/>${body}</svg>`
  );
}

async function main(): Promise<void> {
  const { seeds, leaves, out } = parseArgs(Bun.argv.slice(2));
  await mkdir(out, { recursive: true });
  for (const seed of seeds) {
    const lv = leaves ?? [4, 5, 6][seed % 3];
    const packing = await generateBoxPleatedPacking({
      id: `xdebug-${seed}`,
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
    const svg = render(seed, cp, crossings);
    const file = join(out, `${seed}.svg`);
    await Bun.write(file, svg);
    const nX = crossings.filter((c) => c.kind === "X").length;
    const nT = crossings.filter((c) => c.kind === "T").length;
    const nO = crossings.filter((c) => c.kind === "OVERLAP").length;
    console.log(`${seed} (lv${lv}): X=${nX} T=${nT} OVERLAP=${nO}  ${file}`);
    for (const c of crossings.filter((x) => x.kind === "X"))
      console.log(`    X at (${c.at.x.toFixed(2)}, ${c.at.y.toFixed(2)})`);
  }
}

if (import.meta.main)
  main().catch((e) => {
  console.error(e);
  process.exit(1);
});
