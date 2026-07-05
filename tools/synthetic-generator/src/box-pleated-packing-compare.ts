// Reusable two-panel comparison for a box-pleat CP: the original BP packing (left)
// vs the gap-filled packing with its creases (right). Each panel is clipped to its
// own box so overhanging flap contours never bleed into the other panel, and the
// canvas is padded to a square (qlmanage square-crops non-square SVGs).
//
// CLI: bun run src/box-pleated-packing-compare.ts <seed> [pixels]
//   -> /tmp/packing-compare-<seed>.png

import { generateBoxPleatedPacking, packingRiverCells, fillPackingGaps, type BoxPleatedPacking } from "./box-pleated-packing.ts";
import { buildPackingCP } from "./box-pleated-cp.ts";
import type { AssignedEdge } from "./box-pleated-assignment.ts";

/** SVG string comparing the original packing (left) with gap-filled + creases (right). */
export function packingComparisonSvg(packing: BoxPleatedPacking, edges: AssignedEdge[]): string {
  const W = Math.round(packing.sheet.width);
  const H = Math.round(packing.sheet.height);
  const cell = Math.max(6, Math.round(700 / Math.max(W, H)));
  const panelW = W * cell;
  const panelH = H * cell;
  const pad = 32;
  const titleH = 26;
  const gapx = Math.max(240, Math.round(panelW * 0.45)); // generous gap so boxes never touch
  const axO = pad;
  const bxO = pad + panelW + gapx;
  const oy = pad + titleH;
  const contentW = pad + panelW + gapx + panelW + pad;
  const contentH = oy + panelH + pad;
  const side = Math.max(contentW, contentH); // square canvas

  const gap = fillPackingGaps(packing);
  const riv = packingRiverCells(packing);
  const line = (X: (x: number) => number, Y: (y: number) => number, a: { x: number; y: number }, b: { x: number; y: number }, c: string, w: number, dash = ""): string =>
    `<line x1="${X(a.x).toFixed(1)}" y1="${Y(a.y).toFixed(1)}" x2="${X(b.x).toFixed(1)}" y2="${Y(b.y).toFixed(1)}" stroke="${c}" stroke-width="${w}" ${dash}/>`;
  const poly = (X: (x: number) => number, Y: (y: number) => number, pts: { x: number; y: number }[], fill: string): string =>
    `<polygon points="${pts.map((q) => `${X(q.x).toFixed(1)},${Y(q.y).toFixed(1)}`).join(" ")}" fill="${fill}" fill-opacity="0.16" stroke="${fill}" stroke-width="1.1"/>`;

  const objects = (X: (x: number) => number, Y: (y: number) => number): string => {
    let b = "";
    for (const k of riv.keys()) {
      const [x, y] = k.split(",").map(Number);
      b += `<rect x="${X(x)}" y="${Y(y + 1)}" width="${cell}" height="${cell}" fill="#3b82f6" fill-opacity="0.14"/>`;
    }
    for (const o of packing.layout.objects) {
      const f = o.kind === "flap" ? "#94a3b8" : o.kind === "stretch-device" ? "#22c55e" : o.kind === "river" ? "#3b82f6" : null;
      if (!f) continue;
      for (const c of o.contours) b += poly(X, Y, c.outer, f);
    }
    const cor = [{ x: 0, y: 0 }, { x: W, y: 0 }, { x: W, y: H }, { x: 0, y: H }];
    for (let i = 0; i < 4; i++) b += line(X, Y, cor[i], cor[(i + 1) % 4], "#fff", 1.5);
    return b;
  };

  const XA = (x: number): number => axO + x * cell;
  const XB = (x: number): number => bxO + x * cell;
  const Y = (y: number): number => oy + (H - y) * cell;

  let bBody = objects(XB, Y);
  for (const f of gap.flaps) {
    const r = [{ x: f.x0, y: f.y0 }, { x: f.x1, y: f.y0 }, { x: f.x1, y: f.y1 }, { x: f.x0, y: f.y1 }];
    bBody += poly(XB, Y, r, "#eab308");
  }
  for (const e of edges) {
    const c = e.type === "boundary" ? "#555" : e.mv === "M" ? "#ef4444" : e.mv === "V" ? "#3b82f6" : "#f59e0b";
    bBody += line(XB, Y, e.a, e.b, c, e.type === "ridge" ? 2.0 : 1.1);
  }
  for (const f of gap.flaps) {
    const r = [{ x: f.x0, y: f.y0 }, { x: f.x1, y: f.y0 }, { x: f.x1, y: f.y1 }, { x: f.x0, y: f.y1 }];
    for (let i = 0; i < 4; i++) bBody += line(XB, Y, r[i], r[(i + 1) % 4], "#eab308", 2.6, 'stroke-dasharray="5 3"');
  }

  const clip = (id: string, ox: number): string =>
    `<clipPath id="${id}"><rect x="${ox - 4}" y="${oy - 4}" width="${panelW + 8}" height="${panelH + 8}"/></clipPath>`;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}">` +
    `<defs>${clip("pa", axO)}${clip("pb", bxO)}</defs>` +
    `<rect width="100%" height="100%" fill="#0a0a0a"/>` +
    `<text x="${axO}" y="${pad + 12}" font-family="Helvetica" font-size="13" fill="#fff">A: original BP packing (grey=flap blue=river green=stretch; black=gaps)</text>` +
    `<text x="${bxO}" y="${pad + 12}" font-family="Helvetica" font-size="13" fill="#fff">B: gap-filled + creases (yellow dashed = filler polygon boundary)</text>` +
    `<g clip-path="url(#pa)">${objects(XA, Y)}</g>` +
    `<g clip-path="url(#pb)">${bBody}</g>` +
    `</svg>`
  );
}

if (import.meta.main) {
  const seed = Number(Bun.argv[2] ?? 60045);
  const pixels = Number(Bun.argv[3] ?? 2200);
  const packing = await generateBoxPleatedPacking({
    id: `s${seed}`,
    seed,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: [4, 5, 6][seed % 3],
    tight: true,
    tightRestarts: 14,
  });
  const cp = buildPackingCP(packing);
  if (!cp.valid) {
    console.log(`seed ${seed}: INVALID packing (${!cp.complete ? "incomplete" : "off-grid"}) - nothing to compare.`);
  } else {
    const out = `/tmp/packing-compare-${seed}.svg`;
    await Bun.write(out, packingComparisonSvg(packing, cp.assignedEdges));
    await Bun.$`qlmanage -t -s ${pixels} -o /tmp ${out}`.quiet().catch(() => {});
    await Bun.$`mv ${out}.png /tmp/packing-compare-${seed}.png`.quiet().catch(() => {});
    await Bun.$`rm -f ${out}`.quiet().catch(() => {});
    console.log(`seed ${seed}: ${gapFlaps(packing)} filler flap(s) -> /tmp/packing-compare-${seed}.png`);
  }
}

function gapFlaps(packing: BoxPleatedPacking): number {
  return fillPackingGaps(packing).flaps.length;
}
