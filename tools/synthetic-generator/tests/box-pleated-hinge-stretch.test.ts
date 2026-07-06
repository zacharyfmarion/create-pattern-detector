import { expect, test } from "bun:test";
import { generateBoxPleatedPacking } from "../src/box-pleated-packing.ts";
import { buildPackingCP } from "../src/box-pleated-cp.ts";

// A non-45 (Pythagorean stretch) ridge is off-grid everywhere except its lattice
// points. Axials, pleats and hinges REFLECT over such an arm and land on it at a
// fractional point before returning to the grid on the far side - a legitimate
// reflection vertex, not a lattice break. offGridJunctions accepts an off-grid
// vertex that lies on a stretch arm, so these packings are valid.
//
// Seeds 60001 and 60016 both have a hinge that reflects off a stretch arm at a
// fractional point (e.g. (19.333,21), (8.333,10)); this locks in that (a) the
// reflection happens and (b) it does not make the packing invalid. (Earlier the
// hinge router refused to reach a stretch ridge, dangling instead of reflecting.)

const onGrid = (v: number): boolean => Math.abs(v - Math.round(v)) < 1e-6;

for (const [seed, leaves] of [
  [60001, 5],
  [60016, 5],
] as const) {
  test(`seed ${seed} reflects hinges over stretch arms and stays valid`, async () => {
    const packing = await generateBoxPleatedPacking({
      id: `hinge-stretch-${seed}`,
      seed,
      numCreases: 300,
      bucket: "s",
      symmetry: "none",
      targetLeafCount: leaves,
      tight: true,
      tightRestarts: 14,
    });
    const cp = buildPackingCP(packing);

    // At least one hinge reflects over a stretch arm and ends off-grid.
    const offGridHinge = cp.hinges.some(
      (h) => !onGrid(h.a.x) || !onGrid(h.a.y) || !onGrid(h.b.x) || !onGrid(h.b.y),
    );
    expect(offGridHinge).toBe(true);

    // Those reflection points are accepted (they lie on a stretch arm), so no
    // off-grid junction is rejected and the packing is valid.
    expect(cp.offGrid).toEqual([]);
    expect(cp.valid).toBe(true);
  });
}
