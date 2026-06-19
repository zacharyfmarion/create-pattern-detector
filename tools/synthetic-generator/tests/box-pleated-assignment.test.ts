import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type OriFixture, type OriSegment } from "../src/ori-parser.ts";
import { buildMolecule, assignCreases, maekawaConflicts } from "../src/box-pleated-assignment.ts";

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

test("fully flat-foldable molecules assign with zero Maekawa conflicts", () => {
  // gt16 and river2 are satisfied by the deterministic pass alone.
  expect(assign(scale(gt[16], 2)).conflicts).toEqual([]);
  expect(assign(rivers[2]).conflicts).toEqual([]);
});

test("deterministic-pass conflict counts (baseline for the upcoming repair stage)", () => {
  // The deterministic assignment leaves a small, stable set of Maekawa
  // conflicts at degree-8 flap centers (the 3-1 split) and pure ridge-ridge
  // crossings, to be resolved by the crease-flip repair stage. This locks the
  // current behavior so the repair stage's improvement is measurable.
  const counts: Record<string, number> = {};
  for (let idx = 12; idx <= 17; idx++) counts[`gt${idx}`] = assign(scale(gt[idx], 2)).conflicts.length;
  rivers.forEach((f, i) => (counts[`river${i}`] = assign(f).conflicts.length));
  expect(counts).toEqual({
    gt12: 2,
    gt13: 2,
    gt14: 2,
    gt15: 2,
    gt16: 0,
    gt17: 1,
    river0: 4,
    river1: 7,
    river2: 0,
  });
});
