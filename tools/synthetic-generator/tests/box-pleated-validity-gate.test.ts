import { expect, test } from "bun:test";
import { generateBoxPleatedPacking, type BoxPleatedPackingConfig } from "../src/box-pleated-packing.ts";
import { buildPackingCP, buildPackingMolecule, generateValidCP } from "../src/box-pleated-cp.ts";

// Every path to a crease pattern must go through the same validity gate: a packing
// that is incomplete (an interior region cannot be filled) or off-grid is a
// rejected candidate and must never yield geometry or M/V. This locks that in so a
// debug/test script can't accidentally operate on an invalid CP (as happened once
// by calling buildPackingMolecule without checking validity).

const cfg = (seed: number): BoxPleatedPackingConfig => ({
  id: `gate-${seed}`,
  seed,
  numCreases: 300,
  bucket: "s",
  symmetry: "none",
  targetLeafCount: [4, 5, 6][seed % 3],
  tight: true,
  tightRestarts: 14,
});

// Deterministic seeds: 60109 fails the completeness check, 60007 is valid.
test("a rejected candidate yields no geometry and no M/V", async () => {
  const packing = await generateBoxPleatedPacking(cfg(60109));
  const cp = buildPackingCP(packing);

  expect(cp.valid).toBe(false);
  expect(cp.complete).toBe(false);
  expect(cp.assignedEdges).toEqual([]); // rejected before the M/V stage
  expect(() => buildPackingMolecule(packing)).toThrow(); // geometry cannot be built from it
  expect(await generateValidCP(cfg(60109))).toBeNull(); // canonical entry returns null
});

test("a valid packing passes every entry point", async () => {
  const packing = await generateBoxPleatedPacking(cfg(60007));
  const cp = buildPackingCP(packing);

  expect(cp.valid).toBe(true);
  expect(cp.assignedEdges.length).toBeGreaterThan(0);
  expect(() => buildPackingMolecule(packing)).not.toThrow();
  expect(await generateValidCP(cfg(60007))).not.toBeNull();
});
