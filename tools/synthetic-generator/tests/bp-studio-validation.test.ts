import { expect, test } from "bun:test";
import {
  mapBPStudioAssignments,
  normalizeBPStudioFold,
  prepareBPStudioFoldForStrictValidation,
  summarizeBPStudioExport,
  type BPStudioLineExport,
} from "../src/bp-studio-validation.ts";
import type { FOLDFormat, ValidationConfig } from "../src/types.ts";

test("BP Studio F auxiliaries become valleys in strict canonical output", () => {
  expect(mapBPStudioAssignments(["B", "M", "V", "F", "auxiliary", "U"])).toEqual(["B", "M", "V", "V", "V", "U"]);

  const normalized = normalizeBPStudioFold(bpStudioFoldFixture());
  expect(normalized.edges_assignment).not.toContain("F");
  expect(normalized.edges_assignment.filter((assignment) => assignment === "V")).toHaveLength(4);
  expect(normalized.edges_bpRole).toEqual(expect.arrayContaining(["border", "axis", "hinge"]));
  expect(normalized.edges_bpStudioSource).toHaveLength(normalized.edges_vertices.length);
  expect(normalized.edges_bpStudioSource?.some((source) => source.kind === "device-draw-ridge")).toBe(true);
});

test("normalization retains and splits borders while removing duplicate segment geometry", () => {
  const normalized = normalizeBPStudioFold(bpStudioFoldFixture());
  const summary = normalized.bp_studio_metadata as { normalization: ReturnType<typeof summarizeBPStudioExport> };

  expect(normalized.vertices_coords).toContainEqual([0.5, 0.5]);
  expect(normalized.edges_assignment.filter((assignment) => assignment === "B")).toHaveLength(8);
  expect(normalized.edges_vertices).toHaveLength(14);
  expect(summary.normalization.duplicateSegments).toBe(1);
  expect(summary.normalization.splitIntersections).toBe(1);
  expect(summary.normalization.borderEdgesAfter).toBe(8);
});

test("summary reports BP Studio coordinate scaling and assignment metrics", () => {
  const summary = summarizeBPStudioExport(bpStudioFoldFixture());

  expect(summary.originalVertices).toBe(8);
  expect(summary.originalEdges).toBe(8);
  expect(summary.normalizedVertices).toBe(9);
  expect(summary.normalizedEdges).toBe(14);
  expect(summary.auxiliaryLines).toBe(2);
  expect(summary.assignmentsBefore.F).toBe(2);
  expect(summary.assignmentsAfter.V).toBe(4);
  expect(summary.originalBounds).toEqual({ minX: 0, minY: 0, maxX: 400, maxY: 400 });
  expect(summary.normalizedBounds).toEqual({ minX: 0, minY: 0, maxX: 1, maxY: 1 });
  expect(summary.coordinateScale).toBe(400);
});

test("line exports normalize to canonical FOLD with preserved metadata", () => {
  const lineExport: BPStudioLineExport = {
    metadata: { source: "bp-studio-fixture" },
    lines: [
      { p1: [10, 10], p2: [30, 10], assignment: "B", role: "border" },
      { p1: [30, 10], p2: [30, 30], assignment: "B", role: "border" },
      { p1: [30, 30], p2: [10, 30], assignment: "B", role: "border" },
      { p1: [10, 30], p2: [10, 10], assignment: "B", role: "border" },
      { p1: [20, 10], p2: [20, 30], assignment: "F", role: "axis" },
    ],
  };

  const normalized = normalizeBPStudioFold(lineExport);
  expect(normalized.vertices_coords).toContainEqual([0.5, 0]);
  expect(normalized.edges_assignment).not.toContain("F");
  expect(normalized.edges_assignment.filter((assignment) => assignment === "B")).toHaveLength(6);
  expect(normalized.bp_studio_export_metadata).toEqual({ source: "bp-studio-fixture" });
});

test("prepare helper can optionally run the existing validator", async () => {
  const validationConfig: ValidationConfig = {
    strictGlobal: false,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 32,
    maxEdges: 64,
    requireBoxPleat: true,
  };

  const prepared = await prepareBPStudioFoldForStrictValidation(bpStudioFoldFixture(), { validationConfig });
  expect(prepared.summary.auxiliaryLines).toBe(2);
  expect(prepared.validation?.valid).toBe(true);
});

function bpStudioFoldFixture(): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: "bp-studio-test",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: [
      [0, 0],
      [400, 0],
      [400, 400],
      [0, 400],
      [200, 0],
      [200, 400],
      [0, 200],
      [400, 200],
    ],
    edges_vertices: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 4],
      [6, 7],
      [0, 2],
    ],
    edges_assignment: ["B", "B", "B", "B", "F", "F", "V", "M"],
    edges_bpRole: ["border", "border", "border", "border", "axis", "axis", "hinge", "ridge"],
    edges_bpStudioSource: [
      { kind: "sheet-border", mandatory: true },
      { kind: "sheet-border", mandatory: true },
      { kind: "sheet-border", mandatory: true },
      { kind: "sheet-border", mandatory: true },
      { kind: "node-contour", mandatory: true },
      { kind: "node-contour", mandatory: true },
      { kind: "device-axis-parallel", mandatory: true },
      { kind: "device-draw-ridge", mandatory: true },
    ],
    bp_metadata: {
      gridSize: 2,
      bpSubfamily: "two-flap-stretch",
      flapCount: 2,
      gadgetCount: 1,
      ridgeCount: 1,
      hingeCount: 1,
      axisCount: 1,
    },
  };
}
