import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type OriFixture, type OriSegment } from "../src/ori-parser.ts";
import { buildMolecule, assignCreases, maekawaConflicts, assignBoxPleated } from "../src/box-pleated-assignment.ts";

const gt = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_gt.ori")).json());
const rivers = parseOriFixtures(await Bun.file(join(import.meta.dir, "..", "fixtures", "box_pleating_river_fixtures_stable.ori")).json());

function scale(f: OriFixture, k: number): OriFixture {
  const s = (seg: OriSegment): OriSegment => ({ a: { x: seg.a.x * k, y: seg.a.y * k }, b: { x: seg.b.x * k, y: seg.b.y * k } });
  return { index: f.index, sheet: { width: f.sheet.width * k, height: f.sheet.height * k }, boundary: f.boundary.map(s), ridges: f.ridges.map(s), packing: f.packing.map(s) };
}

function assign(f: OriFixture) {
  const m = buildMolecule(f);
  const edges = assignCreases(m);
  return { m, edges, conflicts: maekawaConflicts(edges, f.sheet) };
}

test("every crease receives a mountain/valley/boundary label", () => {
  for (let idx = 12; idx <= 17; idx++) {
    const { edges } = assign(scale(gt[idx], 2));
    expect(edges.length).toBeGreaterThan(0);
    expect(edges.filter((e) => e.mv === null)).toEqual([]);
    // Every crease is exactly one of M/V/B.
    for (const e of edges) expect(["M", "V", "B"]).toContain(e.mv as string);
  }
  for (const f of rivers) {
    const { edges } = assign(f);
    expect(edges.filter((e) => e.mv === null)).toEqual([]);
  }
});

test("the outermost axial crease is a mountain fold (box-pleat convention)", () => {
  // The axial-family crease whose midpoint is nearest the paper edge is M.
  for (const f of [scale(gt[13], 2), scale(gt[16], 2), rivers[0]]) {
    const { edges } = assign(f);
    const axials = edges.filter((e) => e.type === "axial");
    let nearest = axials[0];
    let best = Infinity;
    for (const e of axials) {
      const mx = (e.a.x + e.b.x) / 2;
      const my = (e.a.y + e.b.y) / 2;
      const d = Math.min(mx, my, f.sheet.width - mx, f.sheet.height - my);
      if (d < best) {
        best = d;
        nearest = e;
      }
    }
    expect(nearest.mv).toBe("M");
  }
});

test("ridges through multiple flap centers seed from the nearest center (river0 diagonal)", () => {
  // The main diagonal of river0 passes through centers (0,0),(4,4),(8,8).
  // Each sub-span must alternate from its own center: the segment from (0,0)
  // crosses an M axial at (1,1) -> M, and the segment into center (4,4) crosses
  // an M axial at (3,3) -> M. Seeding the whole diagonal from one end would
  // flip the parity and color these V.
  const { edges } = assign(rivers[0]);
  const colorOf = (a: number[], b: number[]): string | null => {
    const e = edges.find(
      (e) =>
        (e.a.x === a[0] && e.a.y === a[1] && e.b.x === b[0] && e.b.y === b[1]) ||
        (e.a.x === b[0] && e.a.y === b[1] && e.b.x === a[0] && e.b.y === a[1]),
    );
    return e ? e.mv : null;
  };
  expect(colorOf([0, 0], [1, 1])).toBe("M");
  expect(colorOf([3, 3], [4, 4])).toBe("M");
});

test("fully flat-foldable molecules assign with zero Maekawa conflicts", () => {
  // gt16 and river2 are satisfied by the deterministic pass alone.
  expect(assign(scale(gt[16], 2)).conflicts).toEqual([]);
  expect(assign(rivers[2]).conflicts).toEqual([]);
});

test("deterministic-pass conflict counts (before ridge-crossing repair)", () => {
  // The deterministic assignment (no repair) leaves Maekawa conflicts at
  // degree-8 flap centers (the 3-1 split) and degree-4 ridge crossings. Locked
  // so the repair stage's improvement stays measurable.
  const counts: Record<string, number> = {};
  for (let idx = 12; idx <= 17; idx++) counts[`gt${idx}`] = assign(scale(gt[idx], 2)).conflicts.length;
  rivers.forEach((f, i) => (counts[`river${i}`] = assign(f).conflicts.length));
  expect(counts).toEqual({
    gt12: 3,
    gt13: 2,
    gt14: 1,
    gt15: 3,
    gt16: 0,
    gt17: 1,
    river0: 5,
    river1: 9,
    river2: 0,
  });
});

test("ridge-crossing repair adds the expected hinges (river0)", () => {
  // assignBoxPleated resolves degree-4 ridge crossings by adding two edge-biased
  // hinge arms. river0 gains eight hinges (two per crossing).
  expect(assignBoxPleated(rivers[0]).molecule.hinges.length).toBe(8);
});

test("full-pipeline Maekawa-conflict baseline (post ridge-crossing repair)", () => {
  // The goal is zero conflicts everywhere; the full pipeline gets gt16/gt17/river2
  // there today and knocks river0 down from 5 to 1, but several fixtures still
  // carry residual conflicts (the hinge router does not fully resolve degree-8
  // centers / coupled ridge crossings yet). Locked as a baseline so the remaining
  // reductions stay measurable and regressions are caught.
  const counts: Record<string, number> = {};
  for (let idx = 12; idx <= 17; idx++) counts[`gt${idx}`] = assignBoxPleated(scale(gt[idx], 2)).conflicts.length;
  rivers.forEach((f, i) => (counts[`river${i}`] = assignBoxPleated(f).conflicts.length));
  expect(counts).toEqual({
    gt12: 3,
    gt13: 2,
    gt14: 2,
    gt15: 3,
    gt16: 0,
    gt17: 0,
    river0: 1,
    river1: 7,
    river2: 0,
  });
});

test("river0 ridge crossings get the two edge-going hinge arms (hand-verified)", () => {
  const { molecule } = assignBoxPleated(rivers[0]);
  const hingeKeys = molecule.hinges
    .map((h) => `${h.a.x},${h.a.y}->${h.b.x},${h.b.y}`)
    .sort();
  expect(hingeKeys).toEqual(
    [
      "2,2->0,2", "2,2->2,0",
      "2,6->0,6", "2,6->2,8",
      "6,2->8,2", "6,2->6,0",
      "6,6->8,6", "6,6->6,8",
    ].sort(),
  );
});
