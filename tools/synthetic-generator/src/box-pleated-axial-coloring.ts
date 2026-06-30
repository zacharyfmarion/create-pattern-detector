// Axial M/V coloring by the manual box-pleat method (the author's process):
//
//   1. A crease is the full path that reflects off ridge creases until it
//      terminates at the paper edge on both sides. That entire reflected path is
//      ONE fold and therefore ONE color along its whole length. We recover it by
//      chaining the axial-family segments that meet at a collinear pass-through
//      or a ridge reflection.
//   2. Start with the crease one unit from the paper edge and call it Mountain.
//      The next parallel crease one unit over is the opposite (Valley), and so on
//      inward. Formally: 2-color the graph whose nodes are chains and whose edges
//      connect two chains that have parallel segments exactly one unit apart
//      (with overlapping projection). Anchor the color so the chain closest to a
//      parallel paper edge is Mountain.
//
// This replaces the old pleat-level-parity coloring, which colored each segment
// by its offset level and so (a) flipped a single crease mid-length where its
// collinear pieces had different levels and (b) painted two parallel creases the
// same color where the leveler wrongly gave them the same level. The chain +
// alternation method has neither failure: a crease is one variable, and adjacency
// is purely geometric.

import type { GridPoint, OriSegment } from "./ori-parser.ts";
import { segmentKey } from "./box-pleated-molecule.ts";

export type AxialColor = "M" | "V";

const EPS = 1e-6;

function dir(s: OriSegment): GridPoint {
  const dx = s.b.x - s.a.x;
  const dy = s.b.y - s.a.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: dx / len, y: dy / len };
}

/** Does the segment lie entirely along one of the paper's four edges? */
function onPaperEdge(s: OriSegment, sheet: { width: number; height: number }): boolean {
  const onX = (x: number): boolean => Math.abs(s.a.x - x) < EPS && Math.abs(s.b.x - x) < EPS;
  const onY = (y: number): boolean => Math.abs(s.a.y - y) < EPS && Math.abs(s.b.y - y) < EPS;
  return onX(0) || onX(sheet.width) || onY(0) || onY(sheet.height);
}

/**
 * Color every interior axial-family segment Mountain/Valley by the chain +
 * parallel-alternation method. Returns a lookup keyed by the segment; paper-edge
 * runs (which are border, not folds) and anything unrecognized return null.
 */
export function axialChainColors(
  axialFamily: OriSegment[],
  ridges: OriSegment[],
  sheet: { width: number; height: number },
): (seg: OriSegment) => AxialColor | null {
  // Interior axial-family segments only; the paper-edge runs are border.
  const fam = axialFamily.filter((s) => !onPaperEdge(s, sheet));
  const n = fam.length;

  // --- Step 1: chain segments into reflected creases (union-find). ---
  const parent = fam.map((_, i) => i);
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const union = (a: number, b: number): void => {
    parent[find(a)] = find(b);
  };

  const onRidge = (p: GridPoint): boolean =>
    ridges.some((r) => {
      const dx = r.b.x - r.a.x;
      const dy = r.b.y - r.a.y;
      const cross = (p.x - r.a.x) * dy - (p.y - r.a.y) * dx;
      if (Math.abs(cross) > 1e-4 * Math.hypot(dx, dy)) return false;
      const t = ((p.x - r.a.x) * dx + (p.y - r.a.y) * dy) / (dx * dx + dy * dy);
      return t > -EPS && t < 1 + EPS;
    });

  const pkey = (p: GridPoint): string => `${Math.round(p.x * 100) / 100},${Math.round(p.y * 100) / 100}`;
  const incident = new Map<string, number[]>();
  fam.forEach((s, i) => {
    for (const e of [s.a, s.b]) {
      const k = pkey(e);
      (incident.get(k) ?? incident.set(k, []).get(k)!).push(i);
    }
  });
  for (const [k, ids] of incident) {
    const uniq = [...new Set(ids)];
    if (uniq.length !== 2) continue; // a real junction (3+ creases) does not chain
    const [px, py] = k.split(",").map(Number);
    const d0 = dir(fam[uniq[0]]);
    const d1 = dir(fam[uniq[1]]);
    const collinear = Math.abs(d0.x * d1.y - d0.y * d1.x) < EPS;
    if (collinear || onRidge({ x: px, y: py })) union(uniq[0], uniq[1]);
  }

  // --- Step 2: adjacency between chains that are parallel and 1 unit apart. ---
  const adj = new Map<number, Set<number>>();
  const link = (a: number, b: number): void => {
    if (a === b) return;
    (adj.get(a) ?? adj.set(a, new Set()).get(a)!).add(b);
    (adj.get(b) ?? adj.set(b, new Set()).get(b)!).add(a);
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const di = dir(fam[i]);
      const dj = dir(fam[j]);
      if (Math.abs(di.x * dj.y - di.y * dj.x) > EPS) continue; // not parallel
      const delta = { x: fam[j].a.x - fam[i].a.x, y: fam[j].a.y - fam[i].a.y };
      const perp = Math.abs(delta.x * di.y - delta.y * di.x);
      if (Math.abs(perp - 1) > 1e-3) continue; // not exactly one unit apart
      // Overlap of the two segments' projections onto the shared direction.
      const proj = (p: GridPoint): number => (p.x - fam[i].a.x) * di.x + (p.y - fam[i].a.y) * di.y;
      const lo = Math.max(Math.min(proj(fam[i].a), proj(fam[i].b)), Math.min(proj(fam[j].a), proj(fam[j].b)));
      const hi = Math.min(Math.max(proj(fam[i].a), proj(fam[i].b)), Math.max(proj(fam[j].a), proj(fam[j].b)));
      if (hi - lo > EPS) link(find(i), find(j));
    }
  }

  // Distance of a chain to the nearest paper edge parallel to its segments (so
  // the outermost crease, one unit in, is the Mountain anchor). Axis-aligned legs
  // give a finite distance; pure-diagonal stretch chains fall back to Infinity.
  const edgeDist = (root: number): number => {
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (find(i) !== root) continue;
      const s = fam[i];
      const d = dir(s);
      const mx = (s.a.x + s.b.x) / 2;
      const my = (s.a.y + s.b.y) / 2;
      if (Math.abs(d.x) < EPS) best = Math.min(best, mx, sheet.width - mx); // vertical: left/right edges
      else if (Math.abs(d.y) < EPS) best = Math.min(best, my, sheet.height - my); // horizontal: top/bottom
    }
    return best;
  };

  // --- Step 3: 2-color each component; anchor Mountain at the edge-closest chain. ---
  const colorIdx = new Map<number, number>();
  const chainColor = new Map<number, AxialColor>();
  const roots = [...new Set(fam.map((_, i) => find(i)))];
  for (const r of roots) {
    if (colorIdx.has(r)) continue;
    const comp: number[] = [];
    const queue = [r];
    colorIdx.set(r, 0);
    while (queue.length) {
      const u = queue.shift()!;
      comp.push(u);
      for (const v of adj.get(u) ?? []) {
        if (!colorIdx.has(v)) {
          colorIdx.set(v, colorIdx.get(u)! ^ 1);
          queue.push(v);
        }
      }
    }
    let anchor = comp[0];
    let anchorDist = Infinity;
    for (const c of comp) {
      const d = edgeDist(c);
      if (d < anchorDist) {
        anchorDist = d;
        anchor = c;
      }
    }
    const mIdx = colorIdx.get(anchor)!;
    for (const c of comp) chainColor.set(c, colorIdx.get(c)! === mIdx ? "M" : "V");
  }

  const byKey = new Map<string, AxialColor>();
  fam.forEach((s, i) => byKey.set(segmentKey(s), chainColor.get(find(i))!));
  return (seg: OriSegment): AxialColor | null => byKey.get(segmentKey(seg)) ?? null;
}
