import { expect, test } from "bun:test";
import {
  compileRegionCandidate,
  fixtureRegionLayout,
  regionLayoutFromCompletionLayout,
  regionCandidateToSvg,
} from "../src/bp-region-compiler.ts";
import type { CompletionLayout, RegionLayout } from "../src/bp-completion-contracts.ts";

test("region compiler builds fixture layouts with local pleat-strip regions", () => {
  const layout = fixtureRegionLayout("insect-lite");
  expect(layout.sourceLayoutId).toContain("insect-lite");
  expect(layout.bodies.length).toBeGreaterThan(0);
  expect(layout.flaps.length).toBeGreaterThan(4);
  expect(layout.pleatStrips.length).toBe(3);
  expect(layout.boundaryPorts.length).toBe(layout.pleatStrips.length * 2);
});

test("region compiler emits alternating M/V pleat strips on the compiler grid", () => {
  const candidate = compileRegionCandidate(fixtureRegionLayout("two-flap-stretch"));
  const pleats = candidate.segments.filter((segment) => segment.kind === "strip-pleat");
  expect(candidate.validity).toBe("candidate-complete");
  expect(candidate.rejectionReasons).toHaveLength(0);
  expect(pleats.length).toBeGreaterThan(8);
  expect(pleats.some((segment) => segment.assignment === "M")).toBe(true);
  expect(pleats.some((segment) => segment.assignment === "V")).toBe(true);

  const gridSize = candidate.layout.gridSize;
  const byRegion = new Map<string, typeof pleats>();
  for (const pleat of pleats) {
    const group = byRegion.get(pleat.regionId) ?? [];
    group.push(pleat);
    byRegion.set(pleat.regionId, group);
    expect(pleat.p1[0] === 0 || pleat.p1[0] === 1 || pleat.p2[0] === 0 || pleat.p2[0] === 1).toBe(false);
    expect(pleat.p1[1] === 0 || pleat.p1[1] === 1 || pleat.p2[1] === 0 || pleat.p2[1] === 1).toBe(false);
    expect([...pleat.p1, ...pleat.p2].every((value) => isOnGrid(value, gridSize))).toBe(true);
  }

  for (const group of byRegion.values()) {
    const sorted = [...group].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
    for (let index = 1; index < sorted.length; index += 1) {
      expect(sorted[index].assignment).not.toBe(sorted[index - 1].assignment);
    }
  }
});

test("region compiler snaps optimized scaffold rectangles to visible pleat grid", () => {
  const layout: CompletionLayout = {
    id: "optimized-half-grid-fixture",
    source: "bp-studio-optimized-layout",
    gridSize: 128,
    axis: "horizontal",
    spineCoordinate: 0.5,
    regions: [{
      id: "body",
      kind: "body",
      x1: 57 / 128,
      y1: 55 / 128,
      x2: 80 / 128,
      y2: 165 / 256,
    }],
    terminals: [{
      id: "leg",
      nodeId: "leg",
      x: 90 / 256,
      y: 67 / 256,
      side: "left",
      width: 8 / 128,
      height: 31 / 256,
      priority: 1,
    }],
    corridors: [{
      id: "leg-body",
      from: "leg",
      to: "body",
      orientation: "horizontal",
      coordinate: 91 / 256,
      width: 2 / 64,
    }],
    scaffoldSummary: {
      adapterLineCount: 0,
      adapterVertexCount: 0,
      adapterEdgeCount: 0,
      optimizedFlapCount: 1,
      optimizedTreeEdgeCount: 1,
    },
  };

  const candidate = compileRegionCandidate(regionLayoutFromCompletionLayout(layout));
  expect(candidate.validity).toBe("candidate-complete");
  expect(candidate.rejectionReasons).toHaveLength(0);
  const pitch = 1 / 32;
  for (const segment of candidate.segments) {
    expect([...segment.p1, ...segment.p2].every((value) => isOnStep(value, pitch))).toBe(true);
  }
});


test("region debug SVG exposes BP steering layers", () => {
  const candidate = compileRegionCandidate(fixtureRegionLayout("two-flap-stretch"));
  expect(candidate.stairBoundaries.every((boundary) => boundary.lines.length > 1)).toBe(true);
  const svg = regionCandidateToSvg(candidate, 320);
  expect(svg).toContain("<svg");
  expect(svg).toContain("#ff1f1f");
  expect(svg).toContain("#0057ff");
  expect(svg).toContain("Debug legend");
  expect(svg).toContain("Pleat corridor");
  expect(svg).toContain("Flap target");
  expect(svg).toContain("stroke-dasharray");
});

test("region compiler rejects accidental pleat-strip overlaps outside body regions", () => {
  const layout: RegionLayout = {
    id: "overlap-fixture",
    sourceLayoutId: "overlap-fixture",
    gridSize: 16,
    axis: "horizontal",
    bodies: [{
      id: "body",
      rect: { x1: 0.75, y1: 0.75, x2: 0.875, y2: 0.875 },
      center: { x: 0.8125, y: 0.8125 },
    }],
    flaps: [],
    boundaryPorts: [],
    pleatStrips: [
      {
        id: "strip-a",
        from: "a",
        to: "body",
        rect: { x1: 0.125, y1: 0.125, x2: 0.5, y2: 0.25 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "M",
      },
      {
        id: "strip-b",
        from: "b",
        to: "body",
        rect: { x1: 0.25, y1: 0.1875, x2: 0.625, y2: 0.3125 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "V",
      },
    ],
  };

  const candidate = compileRegionCandidate(layout);
  expect(candidate.validity).toBe("rejected");
  expect(candidate.rejectionReasons.some((reason) => reason.startsWith("pleat-strip-overlap"))).toBe(true);
});

function isOnGrid(value: number, gridSize: number): boolean {
  return isOnStep(value, 1 / Math.min(gridSize, 32));
}

function isOnStep(value: number, step: number): boolean {
  return Math.abs(value / step - Math.round(value / step)) < 1e-9;
}
