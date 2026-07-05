// Reusable renderer for the hinge router's backtracking search (see routeHinges'
// optional onEvent callback in box-pleated-assignment). Turns the recorded
// HingeTraceEvent[] into one SVG frame per event so the full DFS path - descending
// (try), dead-ending (backtrack), and finishing (solved) - can be watched as an
// image sequence. Pure string producer; consumers rasterise (e.g. qlmanage).

import type { HingeTraceEvent } from "./box-pleated-assignment.ts";
import type { OriSegment } from "./ori-parser.ts";

interface MoleculeContext {
  sheet: { width: number; height: number };
  boundary: OriSegment[];
  ridges: OriSegment[];
  axialFamily: OriSegment[];
}

export interface HingeTraceFrame {
  svg: string;
  label: string;
}

const KIND_ACCENT: Record<HingeTraceEvent["kind"], string> = {
  enter: "#94a3b8", // grey - just observing this node
  try: "#22ff88", // green - committing a placement, going deeper
  backtrack: "#ff2d2d", // red - placement failed, undoing
  solved: "#22d3ee", // cyan - frontier cleared
  exhausted: "#f59e0b", // orange - budget hit
};

/** Render a hinge-router event trace to one cumulative SVG frame per event. */
export function renderHingeTraceFrames(
  events: HingeTraceEvent[],
  m: MoleculeContext,
  opts: { pixels?: number } = {},
): HingeTraceFrame[] {
  const W = Math.round(m.sheet.width);
  const H = Math.round(m.sheet.height);
  const cell = Math.max(8, Math.round((opts.pixels ?? 820) / Math.max(W, H)));
  const mgn = 16;
  const side = Math.max(W, H) * cell + 2 * mgn + 24;
  const X = (x: number): number => x * cell + mgn;
  const Y = (y: number): number => (H - y) * cell + mgn + 24;
  const ln = (s: OriSegment, c: string, w: number): string =>
    `<line x1="${X(s.a.x).toFixed(1)}" y1="${Y(s.a.y).toFixed(1)}" x2="${X(s.b.x).toFixed(1)}" y2="${Y(s.b.y).toFixed(1)}" stroke="${c}" stroke-width="${w}"/>`;
  const dot = (p: { x: number; y: number }, c: string, r: number): string =>
    `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${r}" fill="${c}"/>`;
  const ring = (p: { x: number; y: number }, c: string, r: number, w: number): string =>
    `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${r}" fill="none" stroke="${c}" stroke-width="${w}"/>`;

  // Static context: paper grid + boundary + ridges + axial family (all faint).
  const context = ((): string => {
    let b = "";
    for (let x = 0; x <= W; x++) b += ln({ a: { x, y: 0 }, b: { x, y: H } }, "#141414", 0.6);
    for (let y = 0; y <= H; y++) b += ln({ a: { x: 0, y }, b: { x: W, y } }, "#141414", 0.6);
    for (const s of m.axialFamily) b += ln(s, "#1e355e", 1.0);
    for (const s of m.ridges) b += ln(s, "#5a4410", 1.1);
    for (const s of m.boundary) b += ln(s, "#555", 1.4);
    return b;
  })();

  const frames: HingeTraceFrame[] = [];
  events.forEach((e, i) => {
    const accent = KIND_ACCENT[e.kind];
    let body = context;
    // committed hinges (magenta)
    for (const s of e.hinges) body += ln(s, "#c026d3", 2.0);
    // the placement being tried / undone, in the event accent (green try / red backtrack)
    if (e.placement.length) for (const s of e.placement) body += ln(s, accent, 3.2);
    // frontier vertices (yellow rings), and the worked vertex filled white
    for (const v of e.frontier) body += ring(v, "#ffe14d", Math.max(4, cell * 0.4), 2);
    if (e.vertex) body += dot(e.vertex, "#ffffff", Math.max(3.5, cell * 0.32));
    const label = `#${i} d${e.depth} ${e.kind.toUpperCase()} budget=${e.budget} frontier=${e.frontier.length}` +
      (e.vertex ? ` v=(${e.vertex.x},${e.vertex.y})` : "") + ` hinges=${e.hinges.length}`;
    frames.push({
      svg:
        `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}">` +
        `<rect width="100%" height="100%" fill="#0a0a0a"/>` +
        `<text x="${mgn}" y="12" font-family="Helvetica" font-size="11" fill="${accent}">${label}</text>` +
        `<text x="${mgn}" y="23" font-family="Helvetica" font-size="9" fill="#888">magenta=committed hinges · green=try · red=backtrack · yellow=frontier · white=worked vertex</text>` +
        body +
        `</svg>`,
      label,
    });
  });
  return frames;
}
