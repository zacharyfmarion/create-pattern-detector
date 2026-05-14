import { expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  runBPStrictRepairLab,
  runBPStrictRepairLabFile,
  type BPStrictRepairLabReport,
} from "../src/bp-strict-repair-lab.ts";
import type { BPRole, FOLDFormat } from "../src/types.ts";

test("strict repair lab unassigns the extra folded edge and preserves a useful local core", async () => {
  const report = await runBPStrictRepairLab(repairableExtraCreaseFixture());

  expect(report.summary.mode).toBe("unassign");
  expect(report.summary.localFlatFoldableBefore).toBe(false);
  expect(report.summary.localFlatFoldableAfter).toBe(true);
  expect(report.summary.badVerticesBefore).toBeGreaterThan(0);
  expect(report.summary.badVerticesAfter).toBe(0);
  expect(report.summary.repairSteps).toBe(1);
  expect(report.summary.foldedBefore.total).toBe(5);
  expect(report.summary.foldedAfter.total).toBe(4);
  expect(report.summary.foldedAfter.byAssignment).toEqual({ M: 3, V: 1 });
  expect(report.summary.assignmentsAfter.U).toBe(1);
  expect(report.summary.validationStrictGlobalFalseValid).toBe(false);
  expect(report.validation.failed).toContain("box-pleat-structure");
  expect(report.validation.failed).not.toContain("local-flat-foldability");
});

test("strict repair lab delete mode removes the selected edge instead of relabeling it", async () => {
  const report = await runBPStrictRepairLab(repairableExtraCreaseFixture(), { mode: "delete" });

  expect(report.summary.mode).toBe("delete");
  expect(report.summary.localFlatFoldableAfter).toBe(true);
  expect(report.summary.repairSteps).toBe(1);
  expect(report.summary.repairedEdges).toBe(report.summary.normalizedEdges - 1);
  expect(report.summary.foldedAfter.total).toBe(4);
  expect(report.summary.foldedAfter.byAssignment).toEqual({ M: 3, V: 1 });
  expect(report.summary.validationStrictGlobalFalseValid).toBe(false);
  expect(report.validation.failed).toContain("box-pleat-structure");
  expect(report.validation.failed).not.toContain("local-flat-foldability");
});

test("strict repair lab reports retained folded labels by BP role", async () => {
  const report = await runBPStrictRepairLab(repairableExtraCreaseFixture([
    "border",
    "border",
    "border",
    "border",
    "ridge",
    "ridge",
    "ridge",
    "axis",
    "hinge",
  ]));

  expect(report.summary.foldedBefore.byRole).toEqual({ axis: 1, hinge: 1, ridge: 3 });
  expect(report.summary.foldedAfter.byRole).toEqual({ axis: 1, ridge: 3 });
  expect(report.summary.foldedAfter.byAssignmentRole).toEqual({
    "M:ridge": 3,
    "V:axis": 1,
  });
  expect(report.summary.foldedRemoved).toBe(1);
});

test("strict repair lab CLI emits JSON and can write the repaired FOLD", async () => {
  const dir = await mkdtemp(join(tmpdir(), "bp-strict-repair-lab-"));
  const foldPath = join(dir, "bad.fold");
  const repairedPath = join(dir, "repaired.fold");
  await Bun.write(foldPath, JSON.stringify(repairableExtraCreaseFixture(), null, 2));

  const proc = Bun.spawn({
    cmd: ["bun", "run", "src/bp-strict-repair-lab.ts", "--fold", foldPath, "--out", repairedPath, "--json"],
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
  const report = JSON.parse(stdout) as BPStrictRepairLabReport;
  expect(report.summary.localFlatFoldableAfter).toBe(true);
  expect(report.summary.foldedAfter.total).toBe(4);

  const repaired = await Bun.file(repairedPath).json() as FOLDFormat;
  expect(repaired.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V")).toHaveLength(4);
});

test("strict repair lab can inspect the adapter two-flap fixture when integration is enabled", async () => {
  if (process.env.BP_STRICT_REPAIR_LAB_INTEGRATION !== "1") return;

  const foldPath = "/tmp/bps-two-flap.fold";
  const metadataPath = "/tmp/bps-two-flap.meta.json";
  const proc = Bun.spawn({
    cmd: [
      "bun",
      "run",
      "generate",
      "--",
      "--spec",
      "fixtures/two-flap.json",
      "--out",
      foldPath,
      "--metadata",
      metadataPath,
    ],
    cwd: join(import.meta.dir, "../../bp-studio-adapter"),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stderr, exitCode] = await Promise.all([
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  expect(exitCode, stderr).toBe(0);

  const report = await runBPStrictRepairLabFile(foldPath);
  expect(report.summary.normalizedEdges).toBeGreaterThan(0);
  expect(report.summary.foldedAfter.total).toBeLessThanOrEqual(report.summary.foldedBefore.total);
  expect(typeof report.summary.validationStrictGlobalFalseValid).toBe("boolean");
});

function repairableExtraCreaseFixture(edges_bpRole?: BPRole[]): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: "bp-strict-repair-lab-test",
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
    edges_assignment: ["B", "B", "B", "B", "M", "M", "M", "V", "V"],
    ...(edges_bpRole ? { edges_bpRole } : {}),
  };
}
