import { expect, test } from "bun:test";
import { ridgeJunctions } from "../src/box-pleated-molecule.ts";
import type { OriSegment } from "../src/ori-parser.ts";

const seg = (ax: number, ay: number, bx: number, by: number): OriSegment => ({ a: { x: ax, y: ay }, b: { x: bx, y: by } });
const key = (p: { x: number; y: number }) => `${p.x},${p.y}`;

test("a square flap skeleton has one convergence point (its center)", () => {
  // Straight skeleton of a 4x4 square: four corner miters to the center (2,2).
  const ridges = [seg(0, 0, 2, 2), seg(4, 0, 2, 2), seg(4, 4, 2, 2), seg(0, 4, 2, 2)];
  const j = ridgeJunctions(ridges);
  expect(j.map(key)).toEqual(["2,2"]);
});

test("a rectangular flap skeleton has two convergence points (the spine ends)", () => {
  // Straight skeleton of a 6x2 rectangle: spine from (1,1) to (5,1), miters to each.
  const ridges = [
    seg(0, 0, 1, 1),
    seg(0, 2, 1, 1),
    seg(6, 0, 5, 1),
    seg(6, 2, 5, 1),
    seg(1, 1, 5, 1), // spine
  ];
  const j = ridgeJunctions(ridges).map(key).sort();
  expect(j).toEqual(["1,1", "5,1"]);
});

test("a straight ridge split into collinear segments yields no junction", () => {
  // (0,0)-(2,2)-(4,4) is one line in two pieces: the split point is not a junction.
  const ridges = [seg(0, 0, 2, 2), seg(2, 2, 4, 4)];
  expect(ridgeJunctions(ridges)).toEqual([]);
});
