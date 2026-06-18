import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type GridPoint } from "../src/ori-parser.ts";
import { findFlapCenters, propagateAxials } from "../src/box-pleated-molecule.ts";

// Frozen snapshot for stable test indices; the working .ori is a scratchpad.
const ORI_PATH = join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_stable.ori");
const doc = await Bun.file(ORI_PATH).json();
const fixtures = parseOriFixtures(doc);

const sortKeys = (pts: GridPoint[]): string[] => pts.map((p) => `${p.x},${p.y}`).sort();

test("flap centers: single full-paper flap seeds at its center", () => {
  // #0 is a 2x2 single flap with an X of ridges; the only center is the X cross.
  expect(sortKeys(findFlapCenters(fixtures[0].ridges, fixtures[0].packing))).toEqual(["1,1"]);
});

test("flap centers: #1 matches the hand-verified ground truth", () => {
  // Author-confirmed: corner/edge flap centers, some on the paper boundary.
  expect(sortKeys(findFlapCenters(fixtures[1].ridges, fixtures[1].packing))).toEqual([
    "0,0",
    "1,3",
    "3,1",
    "3,3",
  ]);
});

test("flap centers are ridge endpoints that lie on no hinge (or the convergence end)", () => {
  // An interior flap (#3) seeds at its true center (1,1), never at the corner it
  // shares with the paper boundary.
  const centers = sortKeys(findFlapCenters(fixtures[3].ridges, fixtures[3].packing));
  expect(centers).toContain("1,1");
  expect(centers).not.toContain("0,0");
});

test("every fixture yields at least one flap center", () => {
  for (const f of fixtures) {
    expect(findFlapCenters(f.ridges, f.packing).length).toBeGreaterThan(0);
  }
});

const segKey = (s: { a: GridPoint; b: GridPoint }): string => {
  const a = `${s.a.x},${s.a.y}`;
  const b = `${s.b.x},${s.b.y}`;
  return a < b ? `${a}|${b}` : `${b}|${a}`;
};

test("axials reflect across ridges and the two ends dedupe to one path (#1)", () => {
  const f = fixtures[1];
  const seeds = findFlapCenters(f.ridges, f.packing);
  const { axials } = propagateAxials(f.ridges, f.sheet, seeds);
  // The contour from (3,1) reflects at the ridge (1,1) down to (1,3); the
  // contour from (1,3) retraces it - both collapse to these two segments.
  const keys = axials.map(segKey).sort();
  expect(keys).toEqual(["1,1|3,1", "1,1|1,3"].map((k) => k).sort());
  // No paper-edge segments leak through, and no duplicates.
  expect(new Set(keys).size).toBe(keys.length);
});

test("a single full-paper flap produces the waterbomb '+' (#0)", () => {
  const f = fixtures[0];
  const seeds = findFlapCenters(f.ridges, f.packing);
  const { axials } = propagateAxials(f.ridges, f.sheet, seeds);
  // Four axials from the center (1,1) to the four edge midpoints.
  expect(axials.length).toBe(4);
});
