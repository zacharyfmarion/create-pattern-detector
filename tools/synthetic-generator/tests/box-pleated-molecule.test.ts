import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type GridPoint, type OriFixture, type OriSegment } from "../src/ori-parser.ts";
import { findFlapCenters, propagateAxials, propagateAxialOffsets } from "../src/box-pleated-molecule.ts";

// Frozen snapshot for stable test indices; the working .ori is a scratchpad.
const ORI_PATH = join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_stable.ori");
const doc = await Bun.file(ORI_PATH).json();
const fixtures = parseOriFixtures(doc);

// Frozen snapshot carrying the hand-drawn axial+1 ground truth (fixture #21 is a
// 10x10 whose blue/packing layer is the author's axial+1 for the #14 packing).
const GT_DOC = await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_gt.ori")).json();
const gtFixtures = parseOriFixtures(GT_DOC);

const sortKeys = (pts: GridPoint[]): string[] => pts.map((p) => `${p.x},${p.y}`).sort();

function unitSegs(segs: OriSegment[]): Set<string> {
  const set = new Set<string>();
  for (const e of segs) {
    const dx = Math.sign(e.b.x - e.a.x);
    const dy = Math.sign(e.b.y - e.a.y);
    const n = Math.round(Math.max(Math.abs(e.b.x - e.a.x), Math.abs(e.b.y - e.a.y)));
    for (let i = 0; i < n; i++) {
      const k = `${e.a.x + dx * i},${e.a.y + dy * i}|${e.a.x + dx * (i + 1)},${e.a.y + dy * (i + 1)}`;
      set.add(k.split("|").sort().join("|"));
    }
  }
  return set;
}

function scale(f: OriFixture, k: number): OriFixture {
  const s = (seg: OriSegment): OriSegment => ({ a: { x: seg.a.x * k, y: seg.a.y * k }, b: { x: seg.b.x * k, y: seg.b.y * k } });
  return {
    index: f.index,
    sheet: { width: f.sheet.width * k, height: f.sheet.height * k },
    boundary: f.boundary.map(s),
    ridges: f.ridges.map(s),
    packing: f.packing.map(s),
  };
}

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

test("axial+1 pleats exactly reproduce the hand-drawn ground truth (GT-B, #21)", () => {
  // The author drew axial+1 for the #14 packing on a 2x-scaled (10x10) sheet,
  // stored as fixture #21's blue layer. Generating axial+1 from #14's hinges ->
  // centers -> axials must reproduce it exactly (set equality, no extras).
  const f14 = scale(gtFixtures[14], 2);
  const seeds = findFlapCenters(f14.ridges, f14.packing);
  const { axials, edgeAxials } = propagateAxials(f14.ridges, f14.sheet, seeds);
  const generated = unitSegs(propagateAxialOffsets(f14.ridges, f14.sheet, axials, edgeAxials));
  const groundTruth = unitSegs(gtFixtures[21].packing);

  expect(groundTruth.size).toBeGreaterThan(0);
  expect([...generated].sort()).toEqual([...groundTruth].sort());
});

test("axial+1 pleats exactly reproduce the hand-drawn ground truth (GT-A, #12)", () => {
  // GT-A: the author's axial+1 for the #12 packing, 2x-scaled. Given as explicit
  // coordinates (a connected spiral plus a corner rectangle).
  const path = (pts: number[][]): OriSegment[] =>
    pts.slice(1).map((p, i) => ({ a: { x: pts[i][0], y: pts[i][1] }, b: { x: p[0], y: p[1] } }));
  const gtA = [
    ...path([[0, 1], [1, 1], [1, 3], [0, 3]]),
    ...path([[0, 5], [1, 5], [1, 9], [5, 9], [5, 7], [3, 7], [3, 5], [5, 5], [5, 3], [3, 3], [3, 1], [7, 1], [7, 9], [9, 9], [9, 1], [10, 1]]),
  ];
  const f12 = scale(gtFixtures[12], 2);
  const seeds = findFlapCenters(f12.ridges, f12.packing);
  const { axials, edgeAxials } = propagateAxials(f12.ridges, f12.sheet, seeds);
  const generated = unitSegs(propagateAxialOffsets(f12.ridges, f12.sheet, axials, edgeAxials));

  expect([...generated].sort()).toEqual([...unitSegs(gtA)].sort());
});
