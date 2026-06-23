import { expect, test } from "bun:test";
import { generateBoxPleatedPacking } from "../src/box-pleated-packing.ts";
import { buildPackingCP } from "../src/box-pleated-cp.ts";
import { fillRidgeRectHole } from "../src/box-pleated-gap-fill.ts";
import type { OriSegment } from "../src/ori-parser.ts";

// BP Studio emits a non-square flap's straight-skeleton ridges as a hollow
// rectangular ring: the box's four 45-degree diagonals stop at the ring corners,
// leaving the ring's interior un-creased (a rectangular "donut hole"). The CP
// build fills that hole with the ring rectangle's true straight skeleton -
// extend the diagonals inward until they meet, then join the two meeting points
// with the spine segment.
//
// Seed 64040 (30x30, 5 leaves) has two such holes:
//   node 8 (top-left flap):     ring (0,25)-(2,28), spine (1,26)-(1,27)
//   node 6 (bottom-middle flap): ring (13,0)-(15,3), spine (14,1)-(14,2)

const EPS = 1e-6;
const sameSeg = (s: OriSegment, ax: number, ay: number, bx: number, by: number): boolean => {
  const fwd = Math.abs(s.a.x - ax) < EPS && Math.abs(s.a.y - ay) < EPS && Math.abs(s.b.x - bx) < EPS && Math.abs(s.b.y - by) < EPS;
  const rev = Math.abs(s.a.x - bx) < EPS && Math.abs(s.a.y - by) < EPS && Math.abs(s.b.x - ax) < EPS && Math.abs(s.b.y - ay) < EPS;
  return fwd || rev;
};
const has = (segs: OriSegment[], ax: number, ay: number, bx: number, by: number): boolean =>
  segs.some((s) => sameSeg(s, ax, ay, bx, by));

test("seed 64040 fills both flap donut holes with straight-skeleton ridges", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "ridge-holes-64040",
    seed: 64040,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: 5,
    tight: true,
    tightRestarts: 14,
  });

  // The fill helper produces the full straight skeleton of each ring (spine +
  // four corner diagonals) for both flaps.
  const fillFor = (nodeId: number): OriSegment[] => {
    const flap = packing.layout.objects.find((o) => o.kind === "flap" && o.nodeId === nodeId)!;
    return fillRidgeRectHole(flap.ridges.map((l) => ({ a: l[0], b: l[1] })));
  };

  const node8 = fillFor(8);
  expect(node8.length).toBe(5);
  expect(has(node8, 1, 26, 1, 27)).toBe(true); // spine
  expect(has(node8, 0, 25, 1, 26)).toBe(true);
  expect(has(node8, 2, 25, 1, 26)).toBe(true);
  expect(has(node8, 0, 28, 1, 27)).toBe(true);
  expect(has(node8, 2, 28, 1, 27)).toBe(true);

  const node6 = fillFor(6);
  expect(node6.length).toBe(5);
  expect(has(node6, 14, 1, 14, 2)).toBe(true); // spine
  expect(has(node6, 13, 0, 14, 1)).toBe(true);
  expect(has(node6, 15, 0, 14, 1)).toBe(true);
  expect(has(node6, 13, 3, 14, 2)).toBe(true);
  expect(has(node6, 15, 3, 14, 2)).toBe(true);

  // The built CP carries both filled spines (no donut hole survives into the CP).
  const cp = buildPackingCP(packing);
  expect(has(cp.ridges, 1, 26, 1, 27)).toBe(true);
  expect(has(cp.ridges, 14, 1, 14, 2)).toBe(true);
});

test("a square flap (skeleton collapses to a point) is left untouched", () => {
  // 4x4 box: straight skeleton meets at the center, no axis-aligned ring.
  const square: OriSegment[] = [
    { a: { x: 0, y: 0 }, b: { x: 2, y: 2 } },
    { a: { x: 4, y: 0 }, b: { x: 2, y: 2 } },
    { a: { x: 4, y: 4 }, b: { x: 2, y: 2 } },
    { a: { x: 0, y: 4 }, b: { x: 2, y: 2 } },
  ];
  expect(fillRidgeRectHole(square)).toEqual([]);
});

test("a flap whose skeleton is a 1D spine segment is left untouched", () => {
  // 6x4 box: skeleton is the horizontal segment (2,2)-(4,2), already creased.
  const segmentSkeleton: OriSegment[] = [
    { a: { x: 0, y: 0 }, b: { x: 2, y: 2 } },
    { a: { x: 0, y: 4 }, b: { x: 2, y: 2 } },
    { a: { x: 6, y: 0 }, b: { x: 4, y: 2 } },
    { a: { x: 6, y: 4 }, b: { x: 4, y: 2 } },
    { a: { x: 2, y: 2 }, b: { x: 4, y: 2 } },
  ];
  expect(fillRidgeRectHole(segmentSkeleton)).toEqual([]);
});
