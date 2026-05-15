import { expect, test } from "bun:test";
import ear from "rabbit-ear";
import { generateBPStudioSpec, validateBPStudioSpec } from "../src/bp-studio-sampler.ts";
import { availableFamilies, generateFold } from "../src/generators.ts";
import { splitForIndex } from "../src/fold-utils.ts";
import { loadRecipe } from "../src/recipe.ts";

test("strict validation APIs are available", () => {
  expect(typeof ear.graph.populate).toBe("function");
  expect(typeof ear.singleVertex.validateKawasaki).toBe("function");
  expect(typeof ear.singleVertex.validateMaekawa).toBe("function");
  expect(typeof ear.graph.makeVerticesCoordsFlatFolded).toBe("function");
  expect(typeof ear.layer.solver).toBe("function");
});

test("synthetic generation is BP-Studio-backed-only", () => {
  expect(availableFamilies()).toEqual(["bp-studio-realistic", "bp-studio-completed"]);
  expect(() =>
    generateFold({
      id: "legacy",
      family: "box-pleat" as "bp-studio-realistic",
      seed: 1,
      numCreases: 80,
      bucket: "small",
    }),
  ).toThrow(/BP-Studio-backed-only/);
});

test("BP Studio raw diagnostic recipe loads through Bun YAML parser", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/bp_studio_realistic_v1.yaml");
  expect(recipe.name).toBe("bp_studio_realistic_v1");
  expect(recipe.families).toEqual({ "bp-studio-realistic": 1, "bp-studio-completed": 0 });
  expect(recipe.validation).toMatchObject({
    strictGlobal: true,
    requireBoxPleat: true,
    requireDense: true,
    requireRealistic: true,
  });
});

test("default recipe uses compiler-backed labels", async () => {
  const recipe = await loadRecipe();
  expect(recipe.name).toBe("bp_completed_uniaxial_v1");
  expect(recipe.families).toEqual({ "bp-studio-realistic": 0, "bp-studio-completed": 1 });
  expect(recipe.validation.requireRealistic).toBe(false);
});

test("BP Studio completed recipe loads as the strict e2e smoke path", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/bp_completed_uniaxial_v1.yaml");
  expect(recipe.name).toBe("bp_completed_uniaxial_v1");
  expect(recipe.families).toEqual({ "bp-studio-realistic": 0, "bp-studio-completed": 1 });
  expect(recipe.validation).toMatchObject({
    strictGlobal: true,
    requireBoxPleat: true,
    requireDense: true,
    requireRealistic: false,
  });
});

test("deterministic split helper preserves recipe ratios for smoke counts", () => {
  const counts = { train: 0, val: 0, test: 0 };
  for (let index = 0; index < 32; index++) {
    counts[splitForIndex(index, { train: 0.85, val: 0.1, test: 0.05 })] += 1;
  }
  expect(counts).toEqual({ train: 27, val: 4, test: 1 });
});

test("BP Studio tree/layout specs are deterministic and schema-valid", () => {
  const config = { seed: 12345, id: "spec", bucket: "small" as const, archetype: "insect" as const };
  const first = generateBPStudioSpec(config);
  const second = generateBPStudioSpec(config);
  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(validateBPStudioSpec(first)).toEqual([]);
  expect(first.layout.flaps.length).toBeGreaterThanOrEqual(first.expectedComplexity.targetFlaps[0]);
  expect(first.layout.rivers.length).toBeGreaterThan(0);
});
