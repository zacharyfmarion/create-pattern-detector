// Reusable renderer for a generation trace (see traceGeneration in box-pleated-cp).
// Turns the ordered GenStage[] into a cumulative SVG frame sequence: each geometry
// stage adds its creases (kind-coloured, the newest stage highlighted), and the
// final assignment stage shows the M/V-coloured CP. Consumers do their own PNG
// rasterisation (e.g. qlmanage) - this stays a pure string producer.

import type { GenStage, TracedCrease } from "./box-pleated-cp.ts";
import { packingRiverCells, type BoxPleatedPacking } from "./box-pleated-packing.ts";

const KIND_COLOR: Record<string, string> = {
  boundary: "#777",
  ridge: "#a16207",
  axial: "#38bdf8",
  "edge-axial": "#0ea5e9",
  pleat: "#2dd4bf",
  hinge: "#e879f9",
};

export interface TraceFrame {
  svg: string;
  label: string;
}

/**
 * Render a generation trace to one cumulative SVG frame per stage. The stage's own
 * creases are highlighted green; everything before is kind-coloured. The final
 * (recolor) stage renders the whole CP in M/V.
 */
export function renderGenerationFrames(
  trace: { stages: GenStage[]; sheet: { width: number; height: number } },
  packing: BoxPleatedPacking,
  opts: { pixels?: number } = {},
): TraceFrame[] {
  const { stages, sheet } = trace;
  const W = Math.round(sheet.width);
  const H = Math.round(sheet.height);
  const cell = Math.max(6, Math.round((opts.pixels ?? 760) / Math.max(W, H)));
  const m = 16;
  const side = Math.max(W, H) * cell + 2 * m + 22;
  const X = (x: number): number => x * cell + m;
  const Y = (y: number): number => (H - y) * cell + m;
  const line = (c: TracedCrease | { a: { x: number; y: number }; b: { x: number; y: number } }, col: string, w: number): string =>
    `<line x1="${X(c.a.x).toFixed(1)}" y1="${Y(c.a.y).toFixed(1)}" x2="${X(c.b.x).toFixed(1)}" y2="${Y(c.b.y).toFixed(1)}" stroke="${col}" stroke-width="${w}"/>`;

  const riv = packingRiverCells(packing);
  const context = ((): string => {
    let b = "";
    for (const o of packing.layout.objects) {
      const fill = o.kind === "flap" ? "#5b6472" : o.kind === "stretch-device" ? "#166534" : null;
      if (!fill) continue;
      for (const c of o.contours)
        b += `<polygon points="${c.outer.map((q: { x: number; y: number }) => `${X(q.x).toFixed(1)},${Y(q.y).toFixed(1)}`).join(" ")}" fill="${fill}" fill-opacity="0.12"/>`;
    }
    for (const k of riv.keys()) {
      const [x, y] = k.split(",").map(Number);
      b += `<rect x="${X(x)}" y="${Y(y + 1)}" width="${cell}" height="${cell}" fill="#1e3a8a" fill-opacity="0.10"/>`;
    }
    const cor = [{ x: 0, y: 0 }, { x: W, y: 0 }, { x: W, y: H }, { x: 0, y: H }];
    for (let i = 0; i < 4; i++) b += line({ a: cor[i], b: cor[(i + 1) % 4] }, "#fff", 1.3);
    return b;
  })();

  const frames: TraceFrame[] = [];
  const cumulative: TracedCrease[] = [];
  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i];
    let body = context;
    if (stage.recolor) {
      for (const c of stage.creases) {
        const col = c.kind === "boundary" ? "#666" : c.mv === "M" ? "#ef4444" : c.mv === "V" ? "#3b82f6" : "#eab308";
        body += line(c, col, c.kind === "ridge" ? 2.2 : 1.2);
      }
    } else {
      for (const c of cumulative) body += line(c, KIND_COLOR[c.kind] ?? "#38bdf8", c.kind === "ridge" ? 1.4 : 1.1);
      for (const c of stage.creases) body += line(c, "#22ff88", 3.0);
      cumulative.push(...stage.creases);
    }
    const label = `step ${i}/${stages.length - 1}: ${stage.name} (+${stage.creases.length})`;
    frames.push({
      svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}"><rect width="100%" height="100%" fill="#0a0a0a"/><text x="${m}" y="12" font-family="Helvetica" font-size="10" fill="#fff">${label}</text><g transform="translate(0 15)">${body}</g></svg>`,
      label,
    });
  }
  return frames;
}
