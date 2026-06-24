import { expect, test } from "bun:test";
import { generateBoxPleatedPacking } from "../src/box-pleated-packing.ts";
import { buildPackingCP } from "../src/box-pleated-cp.ts";

// A stretch device split into two contours shares an internal edge between its
// two gadget halves. BP Studio emits each contour's OTHER edges as ridges but
// drops the shared edge (it treats it as the internal boundary), leaving its two
// endpoints under-creased and failing even-degree/Kawasaki. buildPackingCP
// recovers any edge appearing on >= 2 of an object's contours and adds it as a
// ridge.
//
// Seed 60023 (30x30, 6 leaves): stretch device "s2,8,9.0" has contours
//   (11,18)(14,19)(11,13)(7,10) and (7,10)(11,13)(26,18)(25,16)
// sharing the edge (7,10)-(11,13). Restoring that ridge resolves the two failing
// junctions at its endpoints.
test("seed 60023 recovers the shared stretch-contour ridge (7,10)-(11,13)", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "shared-ridge-60023",
    seed: 60023,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: 6,
    tight: true,
    tightRestarts: 14,
  });
  const cp = buildPackingCP(packing);

  const hasShared = cp.ridges.some(
    (r) =>
      (r.a.x === 7 && r.a.y === 10 && r.b.x === 11 && r.b.y === 13) ||
      (r.b.x === 7 && r.b.y === 10 && r.a.x === 11 && r.a.y === 13),
  );
  expect(hasShared).toBe(true);

  // The two junctions that were under-creased no longer fail.
  expect(cp.failing.some((f) => f.x === 7 && f.y === 10)).toBe(false);
  expect(cp.failing.some((f) => f.x === 11 && f.y === 13)).toBe(false);
});
