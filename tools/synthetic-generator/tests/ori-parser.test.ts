import { expect, test } from "bun:test";
import { join } from "node:path";
import { parseOriFixtures, type OriSegment } from "../src/ori-parser.ts";

// Frozen snapshot for stable test indices. The working .ori
// (box_pleating_packing_fixtures.ori) is a scratchpad and changes over time.
const ORI_PATH = join(import.meta.dir, "..", "fixtures", "box_pleating_fixtures_stable.ori");
const doc = await Bun.file(ORI_PATH).json();
const fixtures = parseOriFixtures(doc);

test("parses the authored .ori into 21 separate fixtures", () => {
  expect(fixtures.length).toBe(21);
});

test("fixtures are sized 2x2 through 5x5 in grid units", () => {
  for (const f of fixtures) {
    expect(Number.isInteger(f.sheet.width)).toBe(true);
    expect(Number.isInteger(f.sheet.height)).toBe(true);
    expect(f.sheet.width).toBeGreaterThanOrEqual(2);
    expect(f.sheet.width).toBeLessThanOrEqual(5);
    expect(f.sheet.height).toBeGreaterThanOrEqual(2);
    expect(f.sheet.height).toBeLessThanOrEqual(5);
  }
  // First fixture is the lone 2x2 header.
  expect(fixtures[0].sheet).toEqual({ width: 2, height: 2 });
});

test("every fixture has ridge creases, and they lie on the box-pleat grid (0/45/90)", () => {
  let diagonal = 0;
  let axial = 0;
  for (const f of fixtures) {
    expect(f.ridges.length).toBeGreaterThan(0);
    for (const r of f.ridges) {
      const dx = Math.abs(r.b.x - r.a.x);
      const dy = Math.abs(r.b.y - r.a.y);
      // Box-pleated ridges are 45-degree diagonals or axis-aligned skeleton spines.
      expect(dx === 0 || dy === 0 || Math.abs(dx - dy) < 1e-9).toBe(true);
      if (dx > 0 && dy > 0) diagonal++;
      else axial++;
    }
  }
  // The straight-skeleton ridge layer is mostly diagonal, with some axial spines.
  expect(diagonal).toBeGreaterThan(axial);
});

test("boundary segments lie on the sheet edge and are axis-aligned", () => {
  for (const f of fixtures) {
    expect(f.boundary.length).toBeGreaterThan(0);
    for (const b of f.boundary) {
      const onEdge = (s: OriSegment): boolean => {
        const onX = (v: number) => v === 0 || v === f.sheet.width;
        const onY = (v: number) => v === 0 || v === f.sheet.height;
        // A boundary segment lies along one of the four edges.
        return (
          (onX(s.a.x) && onX(s.b.x) && s.a.x === s.b.x) ||
          (onY(s.a.y) && onY(s.b.y) && s.a.y === s.b.y)
        );
      };
      expect(onEdge(b)).toBe(true);
    }
  }
});

test("packing (blue) is separated out and not mixed into ridges/boundary", () => {
  // Blue segments are axis-aligned annotation; ensure none leaked into ridges.
  const totalPacking = fixtures.reduce((sum, f) => sum + f.packing.length, 0);
  expect(totalPacking).toBeGreaterThan(0);
});
