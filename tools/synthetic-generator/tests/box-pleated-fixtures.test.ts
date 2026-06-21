import { expect, test } from "bun:test";
import {
  allFixtures,
  renderPolygonLayoutSvg,
  validatePolygonLayout,
} from "../src/box-pleated-fixtures.ts";

test("every hand-authored fixture tiles its sheet exactly (polygon-packing rule #4)", () => {
  for (const layout of allFixtures()) {
    const metrics = validatePolygonLayout(layout);
    expect(metrics.errors).toEqual([]);
    expect(metrics.tiles).toBe(true);
    // Exact area accounting: polygons sum to the sheet with no slack.
    expect(metrics.polygonAreaSum).toBe(metrics.sheetArea);
    // No sampled gap or overlap anywhere.
    expect(metrics.gapPct).toBe(0);
    expect(metrics.overlapPct).toBe(0);
    expect(metrics.coveragePct).toBe(100);
    // Box-pleated grid discipline.
    expect(metrics.offGridVertices).toBe(0);
    expect(metrics.nonBoxPleatEdges).toBe(0);
  }
});

test("a layout with a deliberate gap fails the tiling check", () => {
  const broken = {
    id: "broken-gap",
    description: "leaves a 4x4 hole",
    sheet: { width: 8, height: 8 },
    polygons: [
      { id: "f0", kind: "flap" as const, vertices: [
        { x: 0, y: 0 }, { x: 8, y: 0 }, { x: 8, y: 4 }, { x: 0, y: 4 },
      ] },
      { id: "f1", kind: "flap" as const, vertices: [
        { x: 0, y: 4 }, { x: 4, y: 4 }, { x: 4, y: 8 }, { x: 0, y: 8 },
      ] },
    ],
  };
  const metrics = validatePolygonLayout(broken);
  expect(metrics.tiles).toBe(false);
  expect(metrics.gapPct).toBeGreaterThan(0);
});

test("a layout with overlapping flaps fails the tiling check", () => {
  const overlapping = {
    id: "broken-overlap",
    description: "two flaps overlap",
    sheet: { width: 8, height: 4 },
    polygons: [
      { id: "f0", kind: "flap" as const, vertices: [
        { x: 0, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 4 }, { x: 0, y: 4 },
      ] },
      { id: "f1", kind: "flap" as const, vertices: [
        { x: 3, y: 0 }, { x: 8, y: 0 }, { x: 8, y: 4 }, { x: 3, y: 4 },
      ] },
    ],
  };
  const metrics = validatePolygonLayout(overlapping);
  expect(metrics.tiles).toBe(false);
  expect(metrics.overlapPct).toBeGreaterThan(0);
});

test("fixtures render to SVG", () => {
  const svg = renderPolygonLayoutSvg(allFixtures()[2]);
  expect(svg).toContain("<svg");
  expect(svg).toContain("tiles=true");
});
