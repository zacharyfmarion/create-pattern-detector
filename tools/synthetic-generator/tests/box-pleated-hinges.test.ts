import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type OriFixture, type OriSegment } from "../src/ori-parser.ts";
import {
  findFlapCenters,
  propagateAxials,
  propagateAllAxialOffsets,
  propagateHinges,
  planarize,
  failingJunctions,
} from "../src/box-pleated-molecule.ts";

const gt = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_gt.ori")).json());
const rivers = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_river_fixtures_stable.ori")).json());

function scale(f: OriFixture, k: number): OriFixture {
  const s = (seg: OriSegment): OriSegment => ({ a: { x: seg.a.x * k, y: seg.a.y * k }, b: { x: seg.b.x * k, y: seg.b.y * k } });
  return { index: f.index, sheet: { width: f.sheet.width * k, height: f.sheet.height * k }, boundary: f.boundary.map(s), ridges: f.ridges.map(s), packing: f.packing.map(s) };
}

function buildMolecule(f: OriFixture) {
  const centers = findFlapCenters(f.ridges, f.packing, { boundary: f.boundary, sheet: f.sheet });
  const { axials, edgeAxials } = propagateAxials(f.ridges, f.sheet, centers);
  const pleats = propagateAllAxialOffsets(f.ridges, f.sheet, axials, edgeAxials);
  const axialFamily = [...axials, ...edgeAxials, ...pleats];
  const base = [...f.boundary, ...f.ridges, ...axialFamily];
  const { hinges, unresolved } = propagateHinges(f.ridges, axialFamily, centers, f.sheet, base);
  return { centers, axialFamily, base, hinges, unresolved };
}

test("hinge propagation resolves every interior junction (valid flap fixtures, 2x)", () => {
  for (let idx = 12; idx <= 17; idx++) {
    const f = scale(gt[idx], 2);
    const { base, hinges, unresolved } = buildMolecule(f);
    expect(unresolved).toEqual([]);
    // The full crease set has no failing interior junction.
    const remaining = failingJunctions(planarize([...base, ...hinges]), f.sheet);
    expect(remaining).toEqual([]);
  }
});

test("hinge propagation resolves every interior junction (river fixtures)", () => {
  for (const base of rivers) {
    for (const k of [1, 2]) {
      const f = scale(base, k);
      const m = buildMolecule(f);
      expect(m.unresolved).toEqual([]);
      const remaining = failingJunctions(planarize([...m.base, ...m.hinges]), f.sheet);
      expect(remaining).toEqual([]);
    }
  }
});

test("the pinwheel river (river1) is solved by straight hinges through the center", () => {
  // river1's center (4,4) is a ridge crossing, not a flap center, so a hinge
  // passes straight through it to connect opposite failing junctions. This was
  // the case that exposed the need to loop until all constraints hold.
  const f = rivers[1];
  const { hinges, unresolved } = buildMolecule(f);
  expect(unresolved).toEqual([]);
  expect(hinges.length).toBeGreaterThan(0);
});

test("a fully flat-foldable molecule needs no hinges (river0)", () => {
  const { hinges, unresolved } = buildMolecule(rivers[0]);
  expect(unresolved).toEqual([]);
  expect(hinges.length).toBe(0);
});
