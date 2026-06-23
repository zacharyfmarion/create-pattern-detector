import { expect, test } from "bun:test";
import { generateBoxPleatedPacking, findPackingHoles } from "../src/box-pleated-packing.ts";

// A packing's holes are the paper cells covered by no flap, stretch, or river.
// BP Studio's river contours over-cover (they grow each subtree by the edge length
// in every direction, filling interior gaps), so occupancy by contour reports no
// holes. findPackingHoles instead keeps only the OUTWARD river ring: a cell next to
// a flap-group is river unless that group sandwiches it on opposite sides, which
// marks it as an enclosed interior hole.
//
// Seed 50444 (18x18) has two such holes: f5/f6 sandwich a 3x5 gap on the left,
// f2/f3 sandwich a 5x2 gap on the right.
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
