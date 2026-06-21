import { expect, test } from "bun:test";
import ear from "rabbit-ear";
import { availableFamilies, generateFold } from "../src/generators.ts";
import { splitForIndex } from "../src/fold-utils.ts";
import { loadRecipe } from "../src/recipe.ts";
import { validateFold } from "../src/validate.ts";

test("strict validation APIs are available", () => {
  expect(typeof ear.graph.square).toBe("function");
  expect(typeof ear.graph.flatFold).toBe("function");
  expect(typeof ear.axiom.axiom1).toBe("function");
  expect(typeof ear.axiom.axiom2).toBe("function");
  expect(typeof ear.axiom.axiom3).toBe("function");
  expect(typeof ear.axiom.axiom4).toBe("function");
  expect(typeof ear.axiom.axiom7).toBe("function");
  expect(typeof ear.graph.populate).toBe("function");
  expect(typeof ear.singleVertex.validateKawasaki).toBe("function");
  expect(typeof ear.singleVertex.validateMaekawa).toBe("function");
  expect(typeof ear.graph.makeVerticesCoordsFlatFolded).toBe("function");
  expect(typeof ear.layer.solver).toBe("function");
});

test("synthetic generation exposes maintained fold-only families", () => {
  expect(availableFamilies()).toEqual(["treemaker-tree", "rabbit-ear-fold-program", "tessellation-fold-program"]);
  for (const family of ["grid", "classic", "single-vertex", "dense-lattice"] as const) {
    expect(() =>
      generateFold({
        id: `legacy-${family}`,
        family: family as "treemaker-tree",
        seed: 1,
        numCreases: 80,
        bucket: "small",
      }),
    ).toThrow(/Unsupported synthetic generator family/);
  }
});

test("default recipe uses TreeMaker tree labels", async () => {
  const recipe = await loadRecipe();
  expect(recipe.name).toBe("treemaker_tree_v1");
  expect(recipe.families).toEqual({ "treemaker-tree": 1, "rabbit-ear-fold-program": 0, "tessellation-fold-program": 0 });
  expect(recipe.validation.requireTreeMaker).toBe(true);
});

test("Rabbit Ear fold-program recipe loads as a supplemental strict family", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/rabbit_ear_fold_program_v1.yaml");
  expect(recipe.name).toBe("rabbit_ear_fold_program_v1");
  expect(recipe.families).toEqual({ "treemaker-tree": 0, "rabbit-ear-fold-program": 1, "tessellation-fold-program": 0 });
  expect(recipe.validation).toMatchObject({
    strictGlobal: true,
    requireRabbitEarFoldProgram: true,
    requireTreeMaker: false,
  });
});

test("Tessellation fold-program recipe loads as a dense orthogonal supplement", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/tessellation_fold_program_v1.yaml");
  expect(recipe.name).toBe("tessellation_fold_program_v1");
  expect(recipe.families).toEqual({ "treemaker-tree": 0, "rabbit-ear-fold-program": 0, "tessellation-fold-program": 1 });
  expect(recipe.validation).toMatchObject({
    strictGlobal: true,
    requireTessellationFoldProgram: true,
    requireLocalFlatFoldability: true,
    requireTreeMaker: false,
  });
  expect(recipe.tessellationSampler?.subfamilyWeights).toEqual({ "orthogonal-bp-grid": 1 });
  expect(recipe.tessellationSampler?.gridSizes).toContain(10);
  expect(recipe.tessellationSampler?.gridSizes).toContain(15);
  expect(recipe.tessellationSampler?.pleatIntervalPairs).toContainEqual({ horizontal: 3, vertical: 1, weight: 0.9 });
});

test("Rabbit Ear fold-program generation is deterministic by seed", () => {
  const config = {
    id: "rabbit-ear-unit",
    family: "rabbit-ear-fold-program" as const,
    seed: 24680,
    numCreases: 40,
    maxCreases: 90,
    bucket: "small",
  };
  const first = generateFold(config);
  const second = generateFold(config);
  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(first.rabbit_ear_metadata?.generator).toBe("rabbit-ear-fold-program");
  expect(first.rabbit_ear_metadata?.appliedFoldCount).toBeGreaterThan(0);
  expect(first.rabbit_ear_metadata?.activeCreaseCount).toBeGreaterThanOrEqual(40);
  expect(first.label_policy?.labelSource).toBe("rabbit-ear-fold-program");
});

test("Tessellation fold-program generation is deterministic and vertical-heavy", () => {
  const config = {
    id: "tessellation-unit",
    family: "tessellation-fold-program" as const,
    seed: 24680,
    numCreases: 180,
    maxCreases: 360,
    bucket: "small",
    tessellationSampler: {
      subfamilyWeights: { "orthogonal-bp-grid": 1 },
      verticalBiasProbability: 1,
      minRepeats: 6,
      maxRepeats: 24,
    },
  };
  const first = generateFold(config);
  const second = generateFold(config);
  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(first.tessellation_metadata?.generator).toBe("tessellation-fold-program");
  expect(first.tessellation_metadata?.subfamily).toBe("orthogonal-bp-grid");
  expect(first.tessellation_metadata?.coordinateMode).toBe("regular-grid-intervals");
  expect(first.tessellation_metadata?.gridSizeX).toBeGreaterThanOrEqual(first.tessellation_metadata?.repeatX ?? 0);
  expect(first.tessellation_metadata?.gridSizeY).toBeGreaterThanOrEqual(first.tessellation_metadata?.repeatY ?? 0);
  expect(first.tessellation_metadata?.gridSizeX).toBe((first.tessellation_metadata?.repeatX ?? 0) * (first.tessellation_metadata?.verticalPleatInterval ?? 0));
  expect(first.tessellation_metadata?.gridSizeY).toBe((first.tessellation_metadata?.repeatY ?? 0) * (first.tessellation_metadata?.horizontalPleatInterval ?? 0));
  expect(first.tessellation_metadata?.assignmentMode).toBe("vertical-line-alternating");
  expect(first.tessellation_metadata?.activeCreaseCount).toBeGreaterThanOrEqual(180);
  expect(first.tessellation_metadata?.verticalCreaseLengthFraction).toBeGreaterThanOrEqual(0.58);
  expect(first.label_policy?.labelSource).toBe("tessellation-fold-program");
});

test("Tessellation fold-program generation passes Rabbit Ear flat-foldability checks", async () => {
  const fold = generateFold({
    id: "tessellation-flat-foldable-unit",
    family: "tessellation-fold-program" as const,
    seed: 13579,
    numCreases: 120,
    maxCreases: 220,
    bucket: "small",
    tessellationSampler: {
      subfamilyWeights: { "orthogonal-bp-grid": 1 },
      verticalBiasProbability: 1,
      minRepeats: 6,
      maxRepeats: 18,
    },
  });
  const result = await validateFold(fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 512,
    maxEdges: 1024,
    requireTessellationFoldProgram: true,
    requireLocalFlatFoldability: true,
  });
  expect(result.valid).toBe(true);
  expect(result.passed).toContain("local-flat-foldability");
  expect(result.passed).toContain("rabbit-ear-solver");
});

test("Miura tessellation generation varies density and passes Rabbit Ear flat-foldability checks", async () => {
  const fold = generateFold({
    id: "miura-flat-foldable-unit",
    family: "tessellation-fold-program" as const,
    seed: 97531,
    numCreases: 120,
    maxCreases: 220,
    bucket: "small",
    tessellationSampler: {
      subfamilyWeights: { "miura-ori": 1 },
      minRepeats: 6,
      maxRepeats: 18,
      miuraCols: [8, 10, 12],
      miuraRows: [8, 10, 12],
      miuraSkewFactors: [0.33, 0.58],
    },
  });
  expect(fold.tessellation_metadata?.subfamily).toBe("miura-ori");
  expect(fold.tessellation_metadata?.coordinateMode).toBe("miura-square-zigzag-grid");
  expect(fold.tessellation_metadata?.assignmentMode).toBe("miura-column-alternating");
  expect(fold.tessellation_metadata?.diagonalCreaseLengthFraction).toBeGreaterThan(0);
  expect(fold.tessellation_metadata?.verticalCreaseLengthFraction).toBe(0);
  expect(fold.vertices_coords.some(([x]) => x === 0)).toBe(true);
  expect(fold.vertices_coords.some(([x]) => x === 1)).toBe(true);
  expect(fold.vertices_coords.some(([, y]) => y === 0)).toBe(true);
  expect(fold.vertices_coords.some(([, y]) => y === 1)).toBe(true);

  const result = await validateFold(fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 512,
    maxEdges: 1024,
    requireTessellationFoldProgram: true,
    requireLocalFlatFoldability: true,
  });
  expect(result.valid).toBe(true);
  expect(result.passed).toContain("local-flat-foldability");
  expect(result.passed).toContain("rabbit-ear-solver");
});

test("deterministic split helper preserves recipe ratios for smoke counts", () => {
  const counts = { train: 0, val: 0, test: 0 };
  for (let index = 0; index < 32; index++) {
    counts[splitForIndex(index, { train: 0.85, val: 0.1, test: 0.05 })] += 1;
  }
  expect(counts).toEqual({ train: 27, val: 4, test: 1 });
});
