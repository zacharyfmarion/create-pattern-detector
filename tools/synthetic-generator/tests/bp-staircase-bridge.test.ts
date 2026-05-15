import { expect, test } from "bun:test";
import {
  buildStaircaseBridgePrimitive,
  staircaseBridgePortProfile,
} from "../src/bp-staircase-bridge.ts";
import { compatiblePorts, port } from "../src/bp-port-assignment-solver.ts";
import { makeFlatFoldedPreview } from "../src/folded-preview.ts";
import { validateFold } from "../src/validate.ts";

const strictDenseBPValidation = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver" as const,
  minVertexDistance: 1e-9,
  maxVertices: 2000,
  maxEdges: 2000,
  requireBoxPleat: true,
  boxPleatMode: "dense" as const,
};

test("staircase bridge composes two caps and solves final M/V assignments", async () => {
  const result = buildStaircaseBridgePrimitive({
    laneCount: 5,
    orientation: "diagonal-positive",
  });

  expect(result.ok, result.errors.join("\n")).toBe(true);
  expect(result.fold).toBeDefined();
  expect(result.fold?.label_policy?.trainingEligible).toBe(false);
  expect(result.fold?.label_policy?.notes.join(" ")).toContain("overlapping flap allocation");
  expect(result.assignmentSteps).toBeGreaterThan(0);
  expect(result.fold?.bp_metadata?.bpSubfamily).toBe("staircase-bridge-primitive");
  expect(result.fold?.edges_assignment.filter((assignment) => assignment === "M").length).toBeGreaterThan(0);
  expect(result.fold?.edges_assignment.filter((assignment) => assignment === "V").length).toBeGreaterThan(0);

  const validation = await validateFold(result.fold!, strictDenseBPValidation);
  expect(validation.valid, validation.errors.join("\n")).toBe(true);
  expect(validation.passed).toContain("rabbit-ear-solver");

  const preview = makeFlatFoldedPreview(result.fold!);
  expect(preview.foldedFold.vertices_coords).toHaveLength(result.fold!.vertices_coords.length);
  expect(preview.faces).toBeGreaterThan(1);
});

test("staircase bridge certifies both diagonal orientations", async () => {
  for (const orientation of ["diagonal-positive", "diagonal-negative"] as const) {
    const result = buildStaircaseBridgePrimitive({ laneCount: 5, orientation });
    expect(result.ok, `${orientation}: ${result.errors.join("\n")}`).toBe(true);
    const validation = await validateFold(result.fold!, strictDenseBPValidation);
    expect(validation.valid, `${orientation}: ${validation.errors.join("\n")}`).toBe(true);
  }
});

test("staircase bridge is deterministic", () => {
  const a = buildStaircaseBridgePrimitive({ laneCount: 5, orientation: "diagonal-negative" });
  const b = buildStaircaseBridgePrimitive({ laneCount: 5, orientation: "diagonal-negative" });
  expect(a).toEqual(b);
});

test("staircase bridge exposes boundary port sequences for solver integration", () => {
  const profile = staircaseBridgePortProfile({ laneCount: 5, orientation: "diagonal-positive" });

  expect(profile.ports.top.sequence).toHaveLength(5);
  expect(profile.ports.right.sequence).toHaveLength(5);
  expect(profile.ports.bottom.sequence).toHaveLength(5);
  expect(profile.ports.left.sequence).toHaveLength(5);

  const matching = port("matching-top", profile.ports.top.sequence, {
    orientation: profile.ports.top.orientation,
    side: "bottom",
    width: profile.ports.top.width,
    parity: profile.ports.top.parity,
    role: profile.ports.top.role,
  });
  const mismatched = port("mismatched-top", profile.ports.right.sequence, {
    orientation: profile.ports.top.orientation,
    side: "bottom",
    width: profile.ports.top.width,
    parity: profile.ports.top.parity,
    role: profile.ports.top.role,
  });
  expect(compatiblePorts(profile.ports.top, matching).ok).toBe(true);
  expect(compatiblePorts(profile.ports.top, mismatched).ok).toBe(false);
});
