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

test("an odd-by-even gap is sliced into even width-2 flaps", () => {
  // Whole 7x10 sheet empty: shorter=7 (odd), longer=10 (even) -> five 7x2 flaps.
  const res = fillBoxPleatedGaps({ width: 7, height: 10 }, []);
  expect(res.resolved).toBe(true);
  expect(res.flaps).toHaveLength(5);
  for (const f of res.flaps) {
    expect(f.x1 - f.x0).toBe(7);
    expect(f.y1 - f.y0).toBe(2); // even shorter side
  }
  expect(area(res.flaps)).toBe(70); // fully covered
});

test("a both-odd interior gap is reported as unresolved", () => {
  // 5x5 empty: both sides odd -> not fillable by interior flaps.
  const res = fillBoxPleatedGaps({ width: 5, height: 5 }, []);
  expect(res.resolved).toBe(false);
  expect(res.unresolved).toHaveLength(1);
  expect(res.unresolved[0]).toEqual({ x0: 0, y0: 0, x1: 5, y1: 5 });
  expect(res.flaps).toHaveLength(0);
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
