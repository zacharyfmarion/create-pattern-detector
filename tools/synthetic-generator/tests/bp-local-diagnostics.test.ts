import { expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  runBPLocalDiagnostics,
  runBPLocalDiagnosticsFile,
  type BPLocalDiagnosticsReport,
} from "../src/bp-local-diagnostics.ts";
import type { FOLDFormat } from "../src/types.ts";

test("local diagnostics explains bad BP vertices by source role and assignment", () => {
  const report = runBPLocalDiagnostics(extraCreaseFixture());

  expect(report.summary.badVertices).toBeGreaterThan(0);
  expect(report.summary.maekawaBadVertices).toBeGreaterThan(0);
  expect(report.badVertexIncidentCounts.bySourceKind["node-contour"]).toBeGreaterThan(0);
  expect(report.badVertexIncidentCounts.byRole.hinge).toBeGreaterThan(0);
  expect(report.badVertexIncidentCounts.byAssignment.V).toBeGreaterThan(0);
  expect(report.badVertexIncidentCounts.bySourceKindRoleAssignment["node-contour:hinge:V"]).toBeGreaterThan(0);
  expect(report.summary.badDegreeHistogram["5"]).toBeGreaterThan(0);
  expect(report.badVertices.samples[0]?.incidentEdges.length).toBeGreaterThan(0);
});

test("local diagnostics compares auxiliary policies without changing production generation", () => {
  const report = runBPLocalDiagnostics(extraCreaseFixture(), { auxiliaryPolicy: "valley" });
  const unassigned = report.alternatePolicies.find((policy) => policy.auxiliaryPolicy === "unassigned");

  expect(unassigned).toBeDefined();
  expect(report.summary.assignments.V).toBeGreaterThan(unassigned?.assignments.V ?? 0);
  expect(unassigned?.assignments.U).toBeGreaterThan(0);
});

test("local diagnostics CLI emits JSON", async () => {
  const dir = await mkdtemp(join(tmpdir(), "bp-local-diagnostics-"));
  const foldPath = join(dir, "bad.fold");
  await Bun.write(foldPath, JSON.stringify(extraCreaseFixture(), null, 2));

  const proc = Bun.spawn({
    cmd: ["bun", "run", "src/bp-local-diagnostics.ts", "--fold", foldPath, "--json", "--max-vertex-samples", "2"],
    cwd: join(import.meta.dir, ".."),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  expect(stderr).toBe("");
  expect(exitCode).toBe(0);
  const report = JSON.parse(stdout) as BPLocalDiagnosticsReport;
  expect(report.badVertices.samples.length).toBeLessThanOrEqual(2);
  expect(report.summary.badVertices).toBeGreaterThan(0);
});

test("local diagnostics file helper reads FOLD exports", async () => {
  const dir = await mkdtemp(join(tmpdir(), "bp-local-diagnostics-file-"));
  const foldPath = join(dir, "bad.fold");
  await Bun.write(foldPath, JSON.stringify(extraCreaseFixture(), null, 2));

  const report = await runBPLocalDiagnosticsFile(foldPath);
  expect(report.summary.edges).toBeGreaterThan(0);
  expect(report.badVertexIncidentCounts.uniqueEdges).toBeGreaterThan(0);
});

function extraCreaseFixture(): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: "bp-local-diagnostics-test",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0.5, 0.5],
      [0.5, 0],
      [1, 0.5],
      [0.5, 1],
      [0, 0.5],
      [1, 1],
    ],
    edges_vertices: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [4, 6],
      [4, 7],
      [4, 8],
      [4, 9],
    ],
    edges_assignment: ["B", "B", "B", "B", "M", "M", "M", "V", "F"],
    edges_bpRole: ["border", "border", "border", "border", "ridge", "ridge", "ridge", "axis", "hinge"],
    edges_bpStudioSource: [
      { kind: "sheet-border", creaseType: 0, mandatory: true },
      { kind: "sheet-border", creaseType: 0, mandatory: true },
      { kind: "sheet-border", creaseType: 0, mandatory: true },
      { kind: "sheet-border", creaseType: 0, mandatory: true },
      { kind: "node-ridge", creaseType: 1, mandatory: true },
      { kind: "device-draw-ridge", creaseType: 1, mandatory: true },
      { kind: "device-draw-ridge", creaseType: 1, mandatory: true },
      { kind: "device-axis-parallel", creaseType: 2, mandatory: true },
      { kind: "node-contour", creaseType: 3, mandatory: false },
    ],
  };
}
