import { expect, test } from "bun:test";
import ear from "rabbit-ear";
import { availableFamilies, generateFold } from "../src/generators.ts";
import { splitForIndex } from "../src/fold-utils.ts";
import { loadRecipe } from "../src/recipe.ts";

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

test("synthetic generation exposes only TreeMaker and Rabbit Ear fold-program families", () => {
  expect(availableFamilies()).toEqual(["treemaker-tree", "rabbit-ear-fold-program"]);
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
  expect(recipe.families).toEqual({ "treemaker-tree": 1, "rabbit-ear-fold-program": 0 });
  expect(recipe.validation.requireTreeMaker).toBe(true);
});

test("Rabbit Ear fold-program recipe loads as a supplemental strict family", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/rabbit_ear_fold_program_v1.yaml");
  expect(recipe.name).toBe("rabbit_ear_fold_program_v1");
  expect(recipe.families).toEqual({ "treemaker-tree": 0, "rabbit-ear-fold-program": 1 });
  expect(recipe.validation).toMatchObject({
    strictGlobal: true,
    requireRabbitEarFoldProgram: true,
    requireTreeMaker: false,
  });
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

test("deterministic split helper preserves recipe ratios for smoke counts", () => {
  const counts = { train: 0, val: 0, test: 0 };
  for (let index = 0; index < 32; index++) {
    counts[splitForIndex(index, { train: 0.85, val: 0.1, test: 0.05 })] += 1;
  }
  expect(counts).toEqual({ train: 27, val: 4, test: 1 });
});
