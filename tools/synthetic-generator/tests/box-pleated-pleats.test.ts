import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type OriFixture, type OriSegment } from "../src/ori-parser.ts";
import { findFlapCenters, propagateAxials, propagateAllAxialOffsets } from "../src/box-pleated-molecule.ts";

// GT snapshot: #15 is the river-containing 5x5 whose pleating we verified.
const gt = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_gt.ori")).json());
const rivers = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_river_fixtures_stable.ori")).json());

function scale(f: OriFixture, k: number): OriFixture {
  const s = (seg: OriSegment): OriSegment => ({ a: { x: seg.a.x * k, y: seg.a.y * k }, b: { x: seg.b.x * k, y: seg.b.y * k } });
  return { index: f.index, sheet: { width: f.sheet.width * k, height: f.sheet.height * k }, boundary: f.boundary.map(s), ridges: f.ridges.map(s), packing: f.packing.map(s) };
}

function ridgeContains(p: { x: number; y: number }, ridges: OriSegment[]): boolean {
  return ridges.some((r) => {
    const cross = (r.b.x - r.a.x) * (p.y - r.a.y) - (r.b.y - r.a.y) * (p.x - r.a.x);
    if (Math.abs(cross) > 1e-6) return false;
    const dot = (p.x - r.a.x) * (r.b.x - r.a.x) + (p.y - r.a.y) * (r.b.y - r.a.y);
    const len2 = (r.b.x - r.a.x) ** 2 + (r.b.y - r.a.y) ** 2;
    return dot >= -1e-6 && dot <= len2 + 1e-6;
  });
}

/**
 * Count forbidden axial crossings: lattice points where axial-family creases
 * pass straight through in BOTH x and y with no ridge to justify it. Axials may
 * never intersect, so a correct pleating has zero of these.
 */
function forbiddenCrossings(creases: OriSegment[], ridges: OriSegment[]): number {
  const dirs = new Map<string, Set<string>>();
  const add = (p: { x: number; y: number }, d: string): void => {
    const k = `${p.x},${p.y}`;
    if (!dirs.has(k)) dirs.set(k, new Set());
    dirs.get(k)!.add(d);
  };
  for (const e of creases) {
    const dx = Math.sign(e.b.x - e.a.x);
    const dy = Math.sign(e.b.y - e.a.y);
    const n = Math.round(Math.max(Math.abs(e.b.x - e.a.x), Math.abs(e.b.y - e.a.y)));
    for (let i = 0; i < n; i++) {
      const a = { x: e.a.x + dx * i, y: e.a.y + dy * i };
      const b = { x: e.a.x + dx * (i + 1), y: e.a.y + dy * (i + 1) };
      const dn = dx > 0 ? "E" : dx < 0 ? "W" : dy > 0 ? "S" : "N";
      const rev = dn === "E" ? "W" : dn === "W" ? "E" : dn === "S" ? "N" : "S";
      add(a, dn);
      add(b, rev);
    }
  }
  let count = 0;
  for (const [k, ds] of dirs) {
    if (ds.has("E") && ds.has("W") && ds.has("N") && ds.has("S")) {
      const [x, y] = k.split(",").map(Number);
      if (!ridgeContains({ x, y }, ridges)) count++;
    }
  }
  return count;
}

function fullPleating(f: OriFixture): { axials: OriSegment[]; pleats: OriSegment[] } {
  const centers = findFlapCenters(f.ridges, f.packing, { boundary: f.boundary, sheet: f.sheet });
  const { axials, edgeAxials } = propagateAxials(f.ridges, f.sheet, centers);
  const pleats = propagateAllAxialOffsets(f.ridges, f.sheet, axials, edgeAxials);
  return { axials, pleats };
}

test("river centers are excluded from axial seeding (#15 drops the river point (0,0))", () => {
  const f15 = gt[15];
  const withRivers = findFlapCenters(f15.ridges, f15.packing).map((p) => `${p.x},${p.y}`);
  const flapsOnly = findFlapCenters(f15.ridges, f15.packing, { boundary: f15.boundary, sheet: f15.sheet }).map((p) => `${p.x},${p.y}`);
  expect(withRivers).toContain("0,0");
  expect(flapsOnly).not.toContain("0,0");
  // No real flap centers lost.
  expect(flapsOnly.length).toBe(withRivers.length - 1);
});

test("full axial+n pleating never makes axials cross (valid packing fixtures, 4x)", () => {
  // Scaled to expose deep pleats. Excluded as non-valid inputs:
  //   #0  - invalid packing: two 4x4 flaps overlap, and black is not the paper edge.
  //   #9  - invalid packing: an unassigned/unpacked interior region.
  //   #18-22 - manual solves, the axial+1 GT (#21), and a malformed fixture (#22).
  // Axials may never cross, so every valid packing must yield zero crossings.
  const broken = new Set([0, 9]);
  for (let idx = 1; idx < 18; idx++) {
    if (broken.has(idx)) continue;
    const f = scale(gt[idx], 4);
    const { axials, pleats } = fullPleating(f);
    expect(forbiddenCrossings([...axials, ...pleats], f.ridges)).toBe(0);
  }
});

test("full axial+n pleating never makes axials cross (river fixtures)", () => {
  expect(rivers.length).toBe(3);
  for (const base of rivers) {
    for (const k of [1, 2]) {
      const f = scale(base, k);
      const { axials, pleats } = fullPleating(f);
      expect(forbiddenCrossings([...axials, ...pleats], f.ridges)).toBe(0);
      expect(pleats.length).toBeGreaterThan(0);
    }
  }
});
