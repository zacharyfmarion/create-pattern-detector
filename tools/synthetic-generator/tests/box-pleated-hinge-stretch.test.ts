import { expect, test } from "bun:test";
import { generateBoxPleatedPacking } from "../src/box-pleated-packing.ts";
import { buildPackingCP } from "../src/box-pleated-cp.ts";

// A non-45 (Pythagorean stretch) ridge is off-grid everywhere except where it
// passes through a lattice point. An axis-aligned hinge that marches into one
// stops on it at a fractional point, creating an off-grid junction - and that
// junction's grid neighbours then fail Kawasaki by a rounding hair. hingeEndpoint
// refuses to extend a hinge that would reach a stretch ridge at an off-grid
// point, so neither the off-grid junction nor its collateral failures appear.
//
// Seeds 60001 and 60016 both exhibited this: a clean hinge ended on a stretch
// ridge at a 1/3 point (e.g. (19.333,21), (8.333,10)), yielding 2 off-grid +
// 4 collateral failures each.

const onGrid = (v: number): boolean => Math.abs(v - Math.round(v)) < 1e-6;

for (const [seed, leaves] of [
  [60001, 5],
  [60016, 5],
] as const) {
  test(`seed ${seed} extends no hinge onto a stretch ridge (no off-grid junctions)`, async () => {
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

    // No hinge ends at an off-grid (fractional) point.
    const offGridHinge = cp.hinges.find(
      (h) => !onGrid(h.a.x) || !onGrid(h.a.y) || !onGrid(h.b.x) || !onGrid(h.b.y),
    );
    expect(offGridHinge).toBeUndefined();

    // Every interior junction passes the non-coloring checks (even degree +
    // Kawasaki within tolerance); no off-grid or collateral failures remain.
    expect(cp.failing).toEqual([]);
  });
}
