import { expect, test } from "bun:test";
import { buildSheetPleatPrimitive } from "../src/bp-pleat-primitive.ts";
import { makeFlatFoldedPreview } from "../src/folded-preview.ts";
import { validateFold } from "../src/validate.ts";

const strictValidation = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver" as const,
  minVertexDistance: 1e-9,
  maxVertices: 1000,
  maxEdges: 1000,
};

test("sheet pleat primitive is locally and globally flat-foldable", async () => {
  const fold = buildSheetPleatPrimitive({
    laneCount: 9,
    orientation: "vertical",
    startAssignment: "M",
  });

  expect(fold.edges_assignment.filter((assignment) => assignment === "M")).toHaveLength(5);
  expect(fold.edges_assignment.filter((assignment) => assignment === "V")).toHaveLength(4);
  expect(fold.edges_bpRole).toBeUndefined();
  expect(fold.bp_metadata).toBeUndefined();

  const validation = await validateFold(fold, strictValidation);
  expect(validation.valid, validation.errors.join("\n")).toBe(true);
  expect(validation.passed).toContain("rabbit-ear-solver");

  const preview = makeFlatFoldedPreview(fold);
  expect(preview.foldedFold.vertices_coords).toHaveLength(fold.vertices_coords.length);
  expect(preview.faces).toBeGreaterThan(1);
});

test("sheet pleat primitive is deterministic", () => {
  const a = buildSheetPleatPrimitive({ laneCount: 7, orientation: "horizontal", startAssignment: "V" });
  const b = buildSheetPleatPrimitive({ laneCount: 7, orientation: "horizontal", startAssignment: "V" });
  expect(a).toEqual(b);
});
