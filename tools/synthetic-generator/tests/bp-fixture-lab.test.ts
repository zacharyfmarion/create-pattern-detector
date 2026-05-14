import { expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { diagnoseBPFixture, type BPFixtureLabReport } from "../src/bp-fixture-lab.ts";
import type { FOLDFormat } from "../src/types.ts";

test("fixture lab reports local Kawasaki and Maekawa details per bad vertex", async () => {
  const report = await diagnoseBPFixture(badThreeCreaseFixture());
  const center = findCenterDiagnostic(report);

  expect(report.summary.localFlatFoldable).toBe(false);
  expect(report.summary.kawasakiBad).toBe(1);
  expect(report.summary.maekawaBad).toBe(1);
  expect(report.validation.failed).toContain("local-flat-foldability");

  expect(center.degree).toBe(3);
  expect(center.foldedDegree).toBe(3);
  expect(center.assignmentCounts).toEqual({ M: 1, V: 2 });
  expect(center.kawasaki.failed).toBe(true);
  expect(center.maekawa.failed).toBe(true);
  expect(center.maekawa.mountain).toBe(1);
  expect(center.maekawa.valley).toBe(2);
  expect(center.maekawa.absoluteDifference).toBe(1);
  expect(center.foldedSectorsDegrees).toEqual([90, 90, 180]);
  expect(center.kawasaki.differenceDegrees).toBe(180);
});

test("fixture lab normalizes BP Studio-style auxiliary assignments before diagnostics", async () => {
  const fixture = badThreeCreaseFixture();
  fixture.edges_assignment[4] = "F";

  const report = await diagnoseBPFixture(fixture);
  const center = findCenterDiagnostic(report);

  expect(report.normalized.edges_assignment).not.toContain("F");
  expect(center.assignmentCounts).toEqual({ V: 3 });
  expect(center.maekawa.signedDifference).toBe(3);
});

test("fixture lab CLI emits JSON diagnostics for --fold input", async () => {
  const dir = await mkdtemp(join(tmpdir(), "bp-fixture-lab-"));
  const foldPath = join(dir, "bad.fold");
  await Bun.write(foldPath, JSON.stringify(badThreeCreaseFixture(), null, 2));

  const proc = Bun.spawn({
    cmd: ["bun", "run", "src/bp-fixture-lab.ts", "--fold", foldPath, "--json"],
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
  const report = JSON.parse(stdout) as BPFixtureLabReport;
  expect(report.summary.badVertices).toBe(1);
  expect(report.summary.localFlatFoldable).toBe(false);
  expect(findCenterDiagnostic(report).foldedSectorsDegrees).toEqual([90, 90, 180]);
});

function findCenterDiagnostic(report: BPFixtureLabReport): BPFixtureLabReport["badVertices"][number] {
  const center = report.badVertices.find(({ coord }) => coord[0] === 0.5 && coord[1] === 0.5);
  expect(center).toBeDefined();
  return center!;
}

function badThreeCreaseFixture(): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: "bp-fixture-lab-test",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0.5, 0.5],
      [0.5, 1],
      [0, 0.5],
      [1, 0.5],
    ],
    edges_vertices: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [4, 6],
      [4, 7],
    ],
    edges_assignment: ["B", "B", "B", "B", "M", "V", "V"],
    edges_bpRole: ["border", "border", "border", "border", "ridge", "hinge", "axis"],
  };
}
