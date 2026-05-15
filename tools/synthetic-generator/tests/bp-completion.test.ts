import { expect, test } from "bun:test";
import { completeBoxPleatLayout, fixtureCompletionLayout } from "../src/bp-completion.ts";
import { validateFold } from "../src/validate.ts";

const strictBPValidation = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver" as const,
  minVertexDistance: 1e-6,
  maxVertices: 4000,
  maxEdges: 4000,
  requireBoxPleat: true,
  boxPleatMode: "dense" as const,
  requireDense: false,
  requireRealistic: false,
};

for (const fixture of ["two-flap-stretch", "three-flap-relay", "five-flap-uniaxial", "insect-lite"] as const) {
  test(`completion fixture ${fixture} compiles to strict BP FOLD`, async () => {
    const result = completeBoxPleatLayout(fixtureCompletionLayout(fixture), { maxFoldLines: 28 });
    expect(result.ok).toBe(true);
    expect(result.fold).toBeDefined();
    expect(result.rejected).toHaveLength(0);
    expect(result.portJoins.length).toBeGreaterThan(0);
    expect(result.portJoins.every((join) => join.accepted)).toBe(true);
    expect(result.molecules.length).toBeGreaterThan(3);
    expect(result.foldLines).toHaveLength(0);
    expect(result.segments?.length).toBeGreaterThan(12);
    expect(result.moleculeInstances?.length).toBeGreaterThan(3);
    expect(result.fold?.completion_metadata?.engine).toBe("strict-box-pleat-completion");
    expect(result.fold?.completion_metadata?.source).toBe("fixture");
    expect(result.fold?.completion_metadata?.compilerSteps).toContain("emit-local-molecule-segments");
    expect(result.fold?.label_policy?.labelSource).toBe("compiler");
    expect(result.fold?.label_policy?.trainingEligible).toBe(true);
    expect(result.fold?.edges_bpStudioSource).toBeUndefined();
    expect(result.fold?.edges_compilerSource).toHaveLength(result.fold?.edges_vertices.length ?? 0);
    expect(result.fold?.molecule_metadata?.molecules["corner-fan"]).toBeGreaterThan(0);
    expect(result.fold?.molecule_metadata?.portChecks.rejected).toBe(0);

    const validation = await validateFold(result.fold!, strictBPValidation);
    expect(validation.valid, validation.errors.join("\n")).toBe(true);
    expect(validation.passed).toContain("rabbit-ear-solver");
  });
}

test("completion fixtures are not scaffold-only placeholder stars", () => {
  const result = completeBoxPleatLayout(fixtureCompletionLayout("insect-lite"), { maxFoldLines: 28 });
  expect(result.ok).toBe(true);
  expect(result.fold?.edges_vertices.length).toBeGreaterThan(24);
  expect(result.fold?.bp_metadata?.ridgeCount).toBeGreaterThan(4);
  expect(result.fold?.layout_metadata?.bodyRegions.length).toBeGreaterThan(1);
  expect(result.fold?.layout_metadata?.flapTerminals.length).toBeGreaterThan(4);
});
