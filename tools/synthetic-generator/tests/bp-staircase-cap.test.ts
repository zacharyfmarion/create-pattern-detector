import { expect, test } from "bun:test";
import { buildDiagonalStaircaseCapPrimitive } from "../src/bp-staircase-cap.ts";
import { makeFlatFoldedPreview } from "../src/folded-preview.ts";
import { validateFold } from "../src/validate.ts";
import type { StaircaseCapCorner } from "../src/bp-staircase-cap.ts";

const strictDenseBPValidation = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver" as const,
  minVertexDistance: 1e-9,
  maxVertices: 1000,
  maxEdges: 1000,
  requireBoxPleat: true,
  boxPleatMode: "dense" as const,
};

test("diagonal staircase cap is locally and globally flat-foldable", async () => {
  const fold = buildDiagonalStaircaseCapPrimitive({
    laneCount: 7,
    startAxisAssignment: "V",
  });

  expect(fold.bp_metadata?.bpSubfamily).toBe("diagonal-staircase-cap-primitive");
  expect(fold.bp_metadata?.gridSize).toBe(10);
  expect(fold.edges_bpRole?.filter((role) => role === "ridge").length).toBeGreaterThan(1);
  expect(fold.edges_bpRole?.filter((role) => role === "axis").length).toBe(7);
  expect(fold.edges_bpRole?.filter((role) => role === "hinge").length).toBe(7);

  const validation = await validateFold(fold, strictDenseBPValidation);
  expect(validation.valid, validation.errors.join("\n")).toBe(true);
  expect(validation.passed).toContain("rabbit-ear-solver");

  const preview = makeFlatFoldedPreview(fold);
  expect(preview.foldedFold.vertices_coords).toHaveLength(fold.vertices_coords.length);
  expect(preview.faces).toBeGreaterThan(1);
});

test("diagonal staircase cap certifies every sheet-corner orientation", async () => {
  for (const corner of ["bottom-left", "bottom-right", "top-left", "top-right"] as StaircaseCapCorner[]) {
    const fold = buildDiagonalStaircaseCapPrimitive({
      laneCount: 5,
      startAxisAssignment: "V",
      corner,
    });
    const validation = await validateFold(fold, strictDenseBPValidation);
    expect(validation.valid, `${corner}: ${validation.errors.join("\n")}`).toBe(true);
  }
});

test("diagonal staircase cap is deterministic", () => {
  const a = buildDiagonalStaircaseCapPrimitive({ laneCount: 5, startAxisAssignment: "M", corner: "top-right" });
  const b = buildDiagonalStaircaseCapPrimitive({ laneCount: 5, startAxisAssignment: "M", corner: "top-right" });
  expect(a).toEqual(b);
});
