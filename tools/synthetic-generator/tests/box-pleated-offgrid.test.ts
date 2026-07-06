import { expect, test } from "bun:test";
import { offGridJunctions } from "../src/box-pleated-molecule.ts";
import type { OriSegment } from "../src/ori-parser.ts";

const seg = (ax: number, ay: number, bx: number, by: number): OriSegment => ({ a: { x: ax, y: ay }, b: { x: bx, y: by } });

test("a crossing on a grid point is accepted", () => {
  // Two creases crossing at (2,2) - on the grid.
  expect(offGridJunctions([seg(0, 2, 4, 2), seg(2, 0, 2, 4)])).toEqual([]);
});

test("a crossing off the grid is flagged", () => {
  // A 45-degree crease crosses a vertical at (2, 1.5) - off the grid.
  const off = offGridJunctions([seg(2, 0, 2, 4), seg(0, 3.5, 4, -0.5)]);
  expect(off).toHaveLength(1);
  expect(off[0].x).toBeCloseTo(2);
  expect(off[0].y).toBeCloseTo(1.5);
});

test("a straight collinear pass-through is not a junction", () => {
  // (0,0)-(2,2)-(4,4) split at an off-grid-ish midpoint is still one line: no junction.
  expect(offGridJunctions([seg(0, 0, 2, 2), seg(2, 2, 4, 4)])).toEqual([]);
});

test("an off-grid contour corner (two non-collinear edges) is flagged", () => {
  // A bent path 0,0 -> 1.5,1 -> 3,0: the bend at (1.5,1) is an off-grid junction.
  const off = offGridJunctions([seg(0, 0, 1.5, 1), seg(1.5, 1, 3, 0)]);
  expect(off).toHaveLength(1);
  expect(off[0].x).toBeCloseTo(1.5);
});
