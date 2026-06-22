import { expect, test } from "bun:test";
import { generateBoxPleatedPacking, findPackingHoles } from "../src/box-pleated-packing.ts";

// A packing's holes are the gaps between facing flaps that exceed their river
// width. BP Studio's river contours over-cover these gaps (they claim the whole
// inter-flap region as river), so occupancy by contour alone reports no holes.
// findPackingHoles instead models each flap as an L-infinity square and finds the
// axis-aligned slab between facing flaps, minus the river width.
//
// Seed 50444 (18x18) has two sibling-pair holes: f5/f6 leave a 3x5 gap on the
// left, f2/f3 leave a 5x2 gap on the right (no river between siblings).
test("seed 50444 has the two sibling-pair holes", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "holes-50444",
    seed: 50444,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: 4,
    tight: true,
    tightRestarts: 14,
  });

  const holes = findPackingHoles(packing)
    .map((r) => ({ x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1 }))
    .sort((a, b) => a.x0 - b.x0);

  expect(holes).toEqual([
    { x0: 0, y0: 8, x1: 3, y1: 13 }, // 3 wide x 5 tall, between f5 and f6
    { x0: 13, y0: 7, x1: 18, y1: 9 }, // 5 wide x 2 tall, between f2 and f3
  ]);
});
