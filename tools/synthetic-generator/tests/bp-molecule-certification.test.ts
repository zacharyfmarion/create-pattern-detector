import { expect, test } from "bun:test";
import {
  certifyCompletionFixture,
  COMPLETION_FIXTURE_LADDER,
  countInteriorDanglingActiveEndpoints,
} from "../src/bp-molecule-certification.ts";
import {
  createMoleculeInstance,
  instantiatePatchSegments,
  moleculePatchLibrary,
} from "../src/bp-molecule-patches.ts";
import { completeBoxPleatLayout, fixtureCompletionLayout } from "../src/bp-completion.ts";

test("molecule transforms preserve grid geometry, assignments, roles, and ports", () => {
  const patch = moleculePatchLibrary().find((item) => item.kind === "body-panel");
  expect(patch).toBeDefined();

  const instance = createMoleculeInstance("rotated-body", patch!, {
    translate: { x: 0.5, y: 0.5 },
    rotateQuarterTurns: 1,
    mirrorX: true,
    scale: 1 / 32,
  });
  const segments = instantiatePatchSegments(instance);

  expect(instance.ports.length).toBe(patch!.ports.length);
  expect(segments.length).toBe(patch!.segments.length);
  expect(segments.every((segment) => segment.assignment === "M" || segment.assignment === "V")).toBe(true);
  expect(segments.every((segment) => segment.role !== "border")).toBe(true);
  expect(segments.every((segment) =>
    [...segment.p1, ...segment.p2].every((coordinate) => Math.abs(coordinate * 64 - Math.round(coordinate * 64)) < 1e-9)
  )).toBe(true);
});

test("fixture ladder certifies every incremental composition", async () => {
  for (const fixture of COMPLETION_FIXTURE_LADDER) {
    const report = await certifyCompletionFixture(fixture.id);
    expect(report.ok, report.errors.join("\n")).toBe(true);
    expect(report.checkedPortJoins).toBeGreaterThan(0);
    expect(report.rejectedPortJoins).toBe(0);
    expect(report.danglingEndpointCount).toBe(0);
    expect(report.validationPassed).toContain("rabbit-ear-solver");
  }
});

test("compiled molecules have no active dangling endpoints", () => {
  const completion = completeBoxPleatLayout(fixtureCompletionLayout("insect-lite"));
  expect(completion.ok).toBe(true);
  expect(countInteriorDanglingActiveEndpoints(completion.fold!)).toBe(0);
});

test("eligible fixtures receive certified pleat-strip staircase cells", () => {
  for (const fixture of ["two-flap-stretch", "five-flap-uniaxial", "insect-lite"] as const) {
    const completion = completeBoxPleatLayout(fixtureCompletionLayout(fixture));
    expect(completion.ok).toBe(true);
    expect(completion.fold?.molecule_metadata?.molecules["diagonal-staircase"]).toBeGreaterThanOrEqual(4);
    expect(completion.fold?.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length).toBeGreaterThan(150);
  }
});

test("three-flap relay receives certified body relay hubs instead of pleat strips", () => {
  const completion = completeBoxPleatLayout(fixtureCompletionLayout("three-flap-relay"));
  expect(completion.ok).toBe(true);
  expect(completion.fold?.molecule_metadata?.molecules["body-panel"]).toBeGreaterThanOrEqual(3);
  expect(completion.fold?.molecule_metadata?.molecules["diagonal-staircase"] ?? 0).toBe(0);
  expect(completion.fold?.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length).toBeGreaterThanOrEqual(150);
});
