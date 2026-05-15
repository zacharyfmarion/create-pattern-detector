import { expect, test } from "bun:test";
import { generateBPStudioCompletedFold } from "../src/bp-studio-completed.ts";
import { validateFold } from "../src/validate.ts";

test("BP Studio completed generator consumes optimized scaffold and emits strict FOLD", async () => {
  const fold = generateBPStudioCompletedFold({
    id: "completed-smoke",
    family: "bp-studio-completed",
    seed: 12345,
    numCreases: 16,
    bucket: "small",
  });

  expect(fold.completion_metadata?.source).toBe("bp-studio-optimized-layout");
  expect(fold.completion_metadata?.scaffoldSummary.optimizedFlapCount).toBeGreaterThan(0);
  expect(fold.label_policy).toMatchObject({
    labelSource: "compiler",
    geometrySource: "compiler",
    assignmentSource: "compiler",
    trainingEligible: true,
  });
  expect(fold.bp_studio_summary?.optimizedFlapCount).toBeGreaterThan(0);
  expect(fold.edges_bpStudioSource).toBeUndefined();
  expect(fold.edges_compilerSource).toHaveLength(fold.edges_vertices.length);
  expect(fold.bp_studio_metadata).toBeDefined();
  expect(fold.bp_metadata?.bpSubfamily).toBe("bp-studio-completed-uniaxial");
  expect(fold.edges_bpRole).toContain("ridge");
  expect(fold.edges_bpRole).toContain("hinge");
  expect(fold.edges_bpRole).toContain("axis");

  const validation = await validateFold(fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 4000,
    maxEdges: 4000,
    requireBoxPleat: true,
    boxPleatMode: "dense",
    requireDense: false,
    requireRealistic: false,
  });
  expect(validation.valid).toBe(true);
  expect(validation.passed).toContain("rabbit-ear-solver");
});
