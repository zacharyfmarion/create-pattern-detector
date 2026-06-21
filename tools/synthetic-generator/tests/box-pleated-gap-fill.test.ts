import { expect, test } from "bun:test";
import { fillBoxPleatedGaps, flapRidges, type GapRect, type OccupiedPolygon } from "../src/box-pleated-gap-fill.ts";

const rectPoly = (x0: number, y0: number, x1: number, y1: number): OccupiedPolygon => ({
  outer: [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x1, y: y1 },
    { x: x0, y: y1 },
  ],
});

const area = (rects: GapRect[]): number => rects.reduce((sum, r) => sum + (r.x1 - r.x0) * (r.y1 - r.y0), 0);

test("an even-shorter-side gap becomes a single flap", () => {
  // 8x8 sheet, occupied top and bottom bands, leaving an 8x4 middle gap.
  const res = fillBoxPleatedGaps({ width: 8, height: 8 }, [rectPoly(0, 0, 8, 2), rectPoly(0, 6, 8, 8)]);
  expect(res.resolved).toBe(true);
  expect(res.flaps).toHaveLength(1);
  expect(res.flaps[0]).toEqual({ x0: 0, y0: 2, x1: 8, y1: 6 });
});

// Build a sheet with an occupied frame so the central WxH gap has all-interior
// sides (no side coincides with the paper edge).
function interiorGap(w: number, h: number): { sheet: { width: number; height: number }; occupied: OccupiedPolygon[]; gap: GapRect } {
  const pad = 2;
  const W = w + 2 * pad;
  const H = h + 2 * pad;
  const occupied = [
    rectPoly(0, 0, W, pad),
    rectPoly(0, H - pad, W, H),
    rectPoly(0, 0, pad, H),
    rectPoly(W - pad, 0, W, H),
  ];
  return { sheet: { width: W, height: H }, occupied, gap: { x0: pad, y0: pad, x1: pad + w, y1: pad + h } };
}

test("an interior odd-by-even gap tiles into even-shorter-side flaps", () => {
  // Interior 7x10: shorter=7 (odd), longer=10 (even). The tiler splits the even
  // dimension so every flap has an even shorter side, fully covering the gap.
  const { sheet, occupied } = interiorGap(7, 10);
  const res = fillBoxPleatedGaps(sheet, occupied);
  expect(res.resolved).toBe(true);
  for (const f of res.flaps) {
    const shorter = Math.min(f.x1 - f.x0, f.y1 - f.y0);
    expect(shorter % 2).toBe(0);
  }
  expect(area(res.flaps)).toBe(70); // fully covered, no overlap
});

test("a both-odd interior gap is reported as unresolved", () => {
  // Interior 5x5: both sides odd and no boundary to crop against -> unsolvable.
  const { sheet, occupied, gap } = interiorGap(5, 5);
  const res = fillBoxPleatedGaps(sheet, occupied);
  expect(res.resolved).toBe(false);
  expect(res.unresolved).toEqual([gap]);
  expect(res.flaps).toHaveLength(0);
});

test("a both-odd gap in a paper corner is solved by a corner flap", () => {
  // 5x5 occupying the whole sheet: all sides are paper boundary -> corner flap.
  const res = fillBoxPleatedGaps({ width: 5, height: 5 }, []);
  expect(res.resolved).toBe(true);
});

test("multiple disjoint gaps are each filled", () => {
  // 12x4 sheet, occupy a 4x4 block in the middle, leaving two 4x4 gaps.
  const res = fillBoxPleatedGaps({ width: 12, height: 4 }, [rectPoly(4, 0, 8, 4)]);
  expect(res.resolved).toBe(true);
  expect(res.flaps).toHaveLength(2);
  expect(area(res.flaps)).toBe(32);
});

test("flap ridges of a square converge on its (grid) center", () => {
  const ridges = flapRidges({ x0: 0, y0: 0, x1: 4, y1: 4 });
  // Four arms to the center (2,2).
  expect(ridges).toHaveLength(4);
  for (const r of ridges) {
    expect(r.b).toEqual({ x: 2, y: 2 });
  }
});

test("flap ridges of an even-shorter rectangle land on grid (spine nodes integer)", () => {
  // 8x2 flap: spine nodes at (1,1) and (7,1) - both integer.
  const ridges = flapRidges({ x0: 0, y0: 0, x1: 8, y1: 2 });
  const nodes = ridges.flatMap((r) => [r.a, r.b]);
  for (const n of nodes) {
    expect(Number.isInteger(n.x)).toBe(true);
    expect(Number.isInteger(n.y)).toBe(true);
  }
});
