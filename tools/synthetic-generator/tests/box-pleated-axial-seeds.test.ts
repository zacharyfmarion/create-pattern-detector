import { expect, test } from "bun:test";
import { generateBoxPleatedPacking } from "../src/box-pleated-packing.ts";
import { buildPackingCP } from "../src/box-pleated-cp.ts";
import { computeCrossings } from "../src/box-pleated-crossing-debug.ts";

// A valid axial seed is an interior convergence of ONE polygon's straight
// skeleton. Seeds are taken per filler flap (gap.ridgesByFlap), never from the
// fused union of all filler ridges - unioning two adjacent fillers invents a
// junction at the point where they touch on their shared boundary, which is
// interior to neither skeleton.
//
// Seed 64040 (30x30, 5 leaves): the central 4x6 hole is tiled by two filler
// flaps whose top corners both land on (16,18) on the hole's boundary. Seeding
// from the union wrongly launched an axial (16,18)-(18,18) that crossed its
// neighbour. Per-polygon seeding drops it (it is a degree-1 box corner within
// each filler, not a junction), leaving no interior axial X-crossings.
test("seed 64040 seeds axials per polygon (no boundary junction, no X-crossings)", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "axial-seeds-64040",
    seed: 64040,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: 5,
    tight: true,
    tightRestarts: 14,
  });
  const cp = buildPackingCP(packing);

  // (16,18) is a shared-boundary corner of two fillers, never an axial seed.
  expect(cp.seeds.some((s) => s.x === 16 && s.y === 18)).toBe(false);
  // ...so the bad axial it used to launch does not exist.
  const hasBadAxial = cp.axials.some(
    (a) =>
      (a.a.x === 16 && a.a.y === 18 && a.b.x === 18 && a.b.y === 18) ||
      (a.b.x === 16 && a.b.y === 18 && a.a.x === 18 && a.a.y === 18),
  );
  expect(hasBadAxial).toBe(false);

  // No axial-family crease crosses another in the interior of the paper.
  const xCrossings = computeCrossings(cp).filter((c) => c.kind === "X");
  expect(xCrossings).toEqual([]);

  // The packing is still fully resolved.
  expect(cp.valid).toBe(true);
});
