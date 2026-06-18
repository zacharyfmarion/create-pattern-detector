import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures } from "../src/ori-parser.ts";
import { findFlapCenters, type GridPoint } from "../src/box-pleated-molecule.ts";

const ORI_PATH = join(import.meta.dir, "..", "fixtures", "box_pleating_packing_fixtures.ori");
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
