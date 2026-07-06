// Reusable renderer for the hinge router's backtracking search (see routeHinges'
// optional onEvent callback in box-pleated-assignment). Turns the recorded
// HingeTraceEvent[] into one SVG frame per event, each showing the FULL M/V
// assignment at that search state (committed hinges + the placement being tried),
// so the coloring can be debugged step by step. The tried/undone placement is
// drawn as a green (try) / red (backtrack) halo behind its M/V edges. Pure string
// producer; consumers rasterise (e.g. qlmanage).

import { assignCreases, type HingeTraceEvent, type BoxPleatedMolecule } from "./box-pleated-assignment.ts";
import type { OriSegment } from "./ori-parser.ts";

export interface HingeTraceFrame {
  svg: string;
  label: string;
}

// Only backtracks get a halo (red, on the placement being undone). "try" frames are
// dropped entirely - the committed state they preview is shown by the next "enter".
const HALO: Partial<Record<HingeTraceEvent["kind"], string>> = {
  backtrack: "#ff2d2d",
};

/** Render a hinge-router event trace to one M/V-coloured SVG frame per event. */
export function renderHingeTraceFrames(
  events: HingeTraceEvent[],
  m: BoxPleatedMolecule,
  opts: { pixels?: number } = {},
): HingeTraceFrame[] {
  const W = Math.round(m.sheet.width);
  const H = Math.round(m.sheet.height);
  const cell = Math.max(8, Math.round((opts.pixels ?? 820) / Math.max(W, H)));
  const mgn = 16;
  const side = Math.max(W, H) * cell + 2 * mgn + 26;
  const X = (x: number): number => x * cell + mgn;
  const Y = (y: number): number => (H - y) * cell + mgn + 26;
  const ln = (s: OriSegment, c: string, w: number): string =>
    `<line x1="${X(s.a.x).toFixed(1)}" y1="${Y(s.a.y).toFixed(1)}" x2="${X(s.b.x).toFixed(1)}" y2="${Y(s.b.y).toFixed(1)}" stroke="${c}" stroke-width="${w}"/>`;
  const dot = (p: { x: number; y: number }, c: string, r: number): string =>
    `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${r}" fill="${c}"/>`;
  const ring = (p: { x: number; y: number }, c: string, r: number, w: number): string =>
    `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${r}" fill="none" stroke="${c}" stroke-width="${w}"/>`;
  const mvColor = (e: { type: string; mv: string | null }): string =>
    e.type === "boundary" ? "#888" : e.mv === "M" ? "#ef4444" : e.mv === "V" ? "#3b82f6" : "#eab308";

  const grid = ((): string => {
    let b = "";
    for (let x = 0; x <= W; x++) b += ln({ a: { x, y: 0 }, b: { x, y: H } }, "#141414", 0.6);
    for (let y = 0; y <= H; y++) b += ln({ a: { x: 0, y }, b: { x: W, y } }, "#141414", 0.6);
    return b;
  })();

  return events.filter((e) => e.kind !== "try").map((e, i) => {
    // Committed rays, plus the placement being tried/undone as a ray from its origin,
    // so the M/V colouring in the frame matches the router's per-ray origin walk.
    const rays = e.vertex && e.placement.length ? [...e.hingeRays, { origin: e.vertex, path: e.placement }] : e.hingeRays;
    const hinges = rays.flatMap((r) => r.path);
    const edges = assignCreases({ ...m, hinges }, rays);
    const halo = HALO[e.kind];
    let body = grid;
    // Placement halo (behind), so its edges read against the M/V colors on top.
    if (halo) for (const s of e.placement) body += ln(s, halo, cell * 0.55);
    // Full M/V assignment for this state.
    for (const ed of edges) body += ln(ed, mvColor(ed), ed.type === "ridge" ? 1.5 : 1.2);
    // Frontier vertices (yellow rings) and the worked vertex (white).
    for (const v of e.frontier) body += ring(v, "#ffe14d", Math.max(4, cell * 0.42), 2);
    if (e.vertex) body += dot(e.vertex, "#ffffff", Math.max(3.5, cell * 0.32));

    const nU = edges.filter((ed) => ed.type !== "boundary" && ed.mv !== "M" && ed.mv !== "V").length;
    const label = `#${i} d${e.depth} ${e.kind.toUpperCase()} budget=${e.budget} frontier=${e.frontier.length}` +
      (e.vertex ? ` v=(${e.vertex.x},${e.vertex.y})` : "") + ` hinges=${hinges.length} U=${nU}`;
    const accent = halo ?? (e.kind === "solved" ? "#22d3ee" : e.kind === "exhausted" ? "#f59e0b" : "#94a3b8");
    return {
      svg:
        `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}">` +
        `<rect width="100%" height="100%" fill="#0a0a0a"/>` +
        `<text x="${mgn}" y="12" font-family="Helvetica" font-size="11" fill="${accent}">${label}</text>` +
        `<text x="${mgn}" y="23" font-family="Helvetica" font-size="9" fill="#888">M=red V=blue border=grey U=yellow · halo: green=try red=backtrack · ring=frontier · white=worked vertex</text>` +
        body +
        `</svg>`,
      label,
    };
  });
}
