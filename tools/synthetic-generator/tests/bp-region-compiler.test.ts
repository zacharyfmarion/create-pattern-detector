import { expect, test } from "bun:test";
import {
  compileRegionCandidate,
  fixtureRegionLayout,
  regionLayoutFromCompletionLayout,
  regionCandidateToSvg,
  regionPhaseProblem,
  solveRegionPleatStripPhases,
} from "../src/bp-region-compiler.ts";
import { sequenceToString } from "../src/bp-port-assignment-solver.ts";
import { completeRegionCandidateBySheetSweep } from "../src/bp-region-sheet-sweep.ts";
import { validateFold } from "../src/validate.ts";
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
  expect(candidate.validity).toBe("layout-valid");
  expect(candidate.rejectionReasons).toHaveLength(0);
  expect(candidate.localProbe?.locallyFlatFoldable).toBe(false);
  expect(candidate.localProbe?.badVertices).toBeGreaterThan(0);
  expect(candidate.localProbe?.failureReasons["dangling-active-endpoint"]).toBeGreaterThan(0);
  expect(candidate.localProbe?.kawasakiBad).toBe(0);
  expect(pleats.length).toBeGreaterThan(4);
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
  expect(candidate.validity).toBe("layout-valid");
  expect(candidate.rejectionReasons).toHaveLength(0);
  const pitch = 1 / 32;
  for (const segment of candidate.segments) {
    expect([...segment.p1, ...segment.p2].every((value) => isOnStep(value, pitch))).toBe(true);
  }
});


test("region debug SVG exposes BP steering layers", () => {
  const candidate = compileRegionCandidate(fixtureRegionLayout("two-flap-stretch"));
  expect(candidate.stairBoundaries).toHaveLength(0);
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

test("region compiler rejects candidate regions that invade flap allocation circles", () => {
  const layout: RegionLayout = {
    id: "flap-allocation-overlap-fixture",
    sourceLayoutId: "flap-allocation-overlap-fixture",
    gridSize: 28,
    axis: "horizontal",
    bodies: [{
      id: "hub",
      rect: { x1: 0.32, y1: 0.45, x2: 0.39, y2: 0.52 },
      center: { x: 0.355, y: 0.485 },
    }],
    flaps: [{
      id: "flap-head",
      terminalId: "head",
      nodeId: "2",
      side: "top",
      center: { x: 4 / 7, y: 6 / 7 },
      allocationRadius: 1 / 7,
      rect: { x1: 4 / 7, y1: 6 / 7, x2: 4 / 7, y2: 6 / 7 },
    }],
    boundaryPorts: [],
    pleatStrips: [{
      id: "strip-through-head-circle",
      from: "hub",
      to: "head",
      rect: { x1: 0.5, y1: 0.75, x2: 0.65, y2: 0.82 },
      orientation: "vertical",
      pitch: 1 / 28,
      phase: 0,
      startAssignment: "M",
    }],
  };

  const candidate = compileRegionCandidate(layout);
  expect(candidate.validity).toBe("rejected");
  expect(candidate.rejectionReasons).toContain("flap-allocation-overlap:strip-through-head-circle:head");
});

test("region compiler allows candidate regions to touch a flap allocation circle boundary", () => {
  const layout: RegionLayout = {
    id: "flap-allocation-tangent-fixture",
    sourceLayoutId: "flap-allocation-tangent-fixture",
    gridSize: 28,
    axis: "horizontal",
    bodies: [],
    flaps: [{
      id: "flap-head",
      terminalId: "head",
      nodeId: "2",
      side: "top",
      center: { x: 4 / 7, y: 6 / 7 },
      allocationRadius: 1 / 7,
      rect: { x1: 4 / 7, y1: 6 / 7, x2: 4 / 7, y2: 6 / 7 },
    }],
    boundaryPorts: [],
    pleatStrips: [{
      id: "strip-touching-head-circle",
      from: "hub",
      to: "head",
      rect: { x1: 4 / 7 - 1 / 28, y1: 5 / 7, x2: 4 / 7 + 1 / 28, y2: 5 / 7 },
      orientation: "vertical",
      pitch: 1 / 28,
      phase: 0,
      startAssignment: "M",
    }],
  };

  const candidate = compileRegionCandidate(layout);
  expect(candidate.rejectionReasons).not.toContain("flap-allocation-overlap:strip-touching-head-circle:head");
});

test("region compiler solves constrained pleat-strip phases before emitting creases", () => {
  const layout = constrainedPhaseLayout();
  const raw = compileRegionCandidate(layout, { solvePortPhases: false });
  const solved = compileRegionCandidate(layout);

  expect(firstPleat(raw, "strip-b")?.assignment).toBe("M");
  expect(solved.validity).toBe("layout-valid");
  expect(solved.layout.pleatStrips.find((strip) => strip.id === "strip-b")?.phase).toBe(1);
  expect(solved.layout.pleatStrips.find((strip) => strip.id === "strip-b")?.startAssignment).toBe("V");
  expect(firstPleat(solved, "strip-b")?.assignment).toBe("V");
});

test("region phase problem exposes real strip port sequences", () => {
  const problem = regionPhaseProblem(constrainedPhaseLayout());
  const stripA = problem.regions.find((region) => region.id === "strip-a");
  const stripB = problem.regions.find((region) => region.id === "strip-b");
  expect(stripA?.states.map((state) => sequenceToString(state.ports[0].sequence))).toEqual(["VMVMV", "MVMVM"]);
  expect(stripB?.states.map((state) => sequenceToString(state.ports[0].sequence))).toEqual(["MVMVM", "VMVMV"]);
  expect(problem.constraints).toHaveLength(1);
});

test("region compiler rejects unsatisfied strip port constraints", () => {
  const layout = constrainedPhaseLayout({
    stripB: {
      rect: { x1: 9 / 16, y1: 5 / 16, x2: 13 / 16, y2: 13 / 16 },
    },
  });
  const phase = solveRegionPleatStripPhases(layout);
  expect(phase.ok).toBe(false);
  expect(phase.errors).toContain("port-phase:port-solver-unsat");
  const candidate = compileRegionCandidate(layout);
  expect(candidate.validity).toBe("rejected");
  expect(candidate.rejectionReasons).toContain("port-phase:port-solver-unsat");
});

test("region compiler closes collinear lane gaps through body panels", () => {
  const candidate = compileRegionCandidate(hubContinuationLayout());
  const closures = candidate.segments.filter((segment) => segment.kind === "hub-closure");

  expect(candidate.rejectionReasons).toHaveLength(0);
  expect(closures).toHaveLength(1);
  expect(closures[0].p1).toEqual([7 / 16, 5 / 16]);
  expect(closures[0].p2).toEqual([9 / 16, 5 / 16]);
  expect(candidate.localProbe?.kawasakiBad).toBe(0);
});

test("region compiler extends terminal lanes only inside flap allocation circles", () => {
  const candidate = compileRegionCandidate(terminalContinuationLayout());
  const closures = candidate.segments.filter((segment) => segment.kind === "terminal-closure");

  expect(candidate.rejectionReasons).toHaveLength(0);
  expect(closures).toHaveLength(1);
  expect(closures[0].p1).toEqual([14 / 16, 5 / 16]);
  expect(closures[0].p2).toEqual([1, 5 / 16]);
  expect(candidate.localProbe?.kawasakiBad).toBe(0);
});

test("sheet-sweep lab completion can make long-axis corridor lanes strict", async () => {
  const candidate = compileRegionCandidate(fixtureRegionLayout("two-flap-stretch"));
  const completion = completeRegionCandidateBySheetSweep(candidate);
  expect(completion.ok).toBe(true);
  expect(completion.fold?.label_policy?.trainingEligible).toBe(false);
  expect(completion.assignmentSteps).toBeGreaterThan(0);

  const validation = await validateFold(completion.fold!, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-9,
    maxVertices: 1000,
    maxEdges: 1000,
  });
  expect(validation.valid).toBe(true);
  expect(validation.passed).toContain("rabbit-ear-solver");
});

function firstPleat(candidate: ReturnType<typeof compileRegionCandidate>, regionId: string) {
  return candidate.segments
    .filter((segment) => segment.kind === "strip-pleat" && segment.regionId === regionId)
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))[0];
}

function constrainedPhaseLayout(overrides: Partial<{ stripB: Partial<RegionLayout["pleatStrips"][number]> }> = {}): RegionLayout {
  return {
    id: "constrained-phase-layout",
    sourceLayoutId: "constrained-phase-layout",
    gridSize: 16,
    axis: "horizontal",
    bodies: [{
      id: "body",
      rect: { x1: 0.4375, y1: 0.375, x2: 0.5625, y2: 0.625 },
      center: { x: 0.5, y: 0.5 },
    }],
    flaps: [],
    boundaryPorts: [],
    pleatStrips: [
      {
        id: "strip-a",
        from: "flap-a",
        to: "body",
        rect: { x1: 2 / 16, y1: 5 / 16, x2: 8 / 16, y2: 13 / 16 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "V",
      },
      {
        id: "strip-b",
        from: "body",
        to: "flap-b",
        rect: { x1: 8 / 16, y1: 5 / 16, x2: 14 / 16, y2: 13 / 16 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "M",
        ...overrides.stripB,
      },
    ],
    portConstraints: [{
      id: "body-port-phase",
      aStripId: "strip-a",
      aSide: "end",
      bStripId: "strip-b",
      bSide: "start",
    }],
  };
}

function hubContinuationLayout(): RegionLayout {
  return {
    id: "hub-continuation-layout",
    sourceLayoutId: "hub-continuation-layout",
    gridSize: 16,
    axis: "horizontal",
    bodies: [{
      id: "body",
      rect: { x1: 7 / 16, y1: 4 / 16, x2: 9 / 16, y2: 6 / 16 },
      center: { x: 0.5, y: 5 / 16 },
    }],
    flaps: [],
    boundaryPorts: [],
    pleatStrips: [
      {
        id: "left-strip",
        from: "left",
        to: "body",
        rect: { x1: 2 / 16, y1: 4 / 16, x2: 7 / 16, y2: 6 / 16 },
        orientation: "horizontal",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "M",
      },
      {
        id: "right-strip",
        from: "body",
        to: "right",
        rect: { x1: 9 / 16, y1: 4 / 16, x2: 14 / 16, y2: 6 / 16 },
        orientation: "horizontal",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "M",
      },
    ],
  };
}

function terminalContinuationLayout(): RegionLayout {
  return {
    id: "terminal-continuation-layout",
    sourceLayoutId: "terminal-continuation-layout",
    gridSize: 16,
    axis: "horizontal",
    bodies: [],
    flaps: [{
      id: "flap-right",
      terminalId: "right-terminal",
      nodeId: "right-terminal",
      side: "right",
      center: { x: 1, y: 5 / 16 },
      allocationRadius: 2 / 16,
      rect: { x1: 1, y1: 5 / 16, x2: 1, y2: 5 / 16 },
    }],
    boundaryPorts: [],
    pleatStrips: [{
      id: "right-terminal-strip",
      from: "body",
      to: "right-terminal",
      rect: { x1: 8 / 16, y1: 4 / 16, x2: 14 / 16, y2: 6 / 16 },
      orientation: "horizontal",
      pitch: 1 / 16,
      phase: 0,
      startAssignment: "M",
    }],
  };
}

function isOnGrid(value: number, gridSize: number): boolean {
  return isOnStep(value, 1 / Math.min(gridSize, 32));
}

function isOnStep(value: number, step: number): boolean {
  return Math.abs(value / step - Math.round(value / step)) < 1e-9;
}
