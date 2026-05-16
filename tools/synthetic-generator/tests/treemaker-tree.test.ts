import { expect, test } from "bun:test";
import { generateFold } from "../src/generators.ts";
import { loadRecipe } from "../src/recipe.ts";
import { generateTreeMakerSpec, validateTreeMakerSpec } from "../src/treemaker-sampler.ts";
import { runTreeMakerAdapter, treeMakerOutputToFold, type TreeMakerExternalOutput } from "../src/treemaker-adapter.ts";
import { validateFold } from "../src/validate.ts";

test("TreeMaker sampler is deterministic and preserves mirrored lengths", () => {
  const config = {
    id: "tree",
    family: "treemaker-tree" as const,
    seed: 12345,
    numCreases: 320,
    bucket: "medium",
    treeMakerSampler: {
      symmetryWeights: { diagonal: 1, "middle-axis": 0, asymmetric: 0 },
      diagonalWeights: { "main-diagonal": 1, "anti-diagonal": 0 },
      archetypeWeights: { insect: 1 },
    },
  };
  const first = generateTreeMakerSpec(config);
  const second = generateTreeMakerSpec(config);
  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(validateTreeMakerSpec(first)).toEqual([]);
  expect(first.symmetryClass).toBe("diagonal");
  expect(first.symmetryVariant).toBe("main-diagonal");
});

test("TreeMaker symmetry sampler follows the configured 85 percent symmetric distribution", () => {
  const counts = { diagonal: 0, "middle-axis": 0, asymmetric: 0 };
  for (let seed = 1; seed <= 2000; seed++) {
    const spec = generateTreeMakerSpec({
      id: `tree-${seed}`,
      family: "treemaker-tree",
      seed,
      numCreases: 200,
      bucket: "small",
    });
    counts[spec.symmetryClass] += 1;
  }
  expect(counts.diagonal / 2000).toBeGreaterThan(0.38);
  expect(counts.diagonal / 2000).toBeLessThan(0.47);
  expect(counts["middle-axis"] / 2000).toBeGreaterThan(0.38);
  expect(counts["middle-axis"] / 2000).toBeLessThan(0.47);
  expect(counts.asymmetric / 2000).toBeGreaterThan(0.11);
  expect(counts.asymmetric / 2000).toBeLessThan(0.19);
});

test("TreeMaker adapter fails clearly when no external CLI is configured", () => {
  const spec = generateTreeMakerSpec({
    id: "missing-cli",
    family: "treemaker-tree",
    seed: 1,
    numCreases: 120,
    bucket: "small",
  });
  expect(() => runTreeMakerAdapter(spec, { command: [] })).toThrow(/TREEMAKER_CLI is required/);
});

test("TreeMaker converter preserves crease kinds and full CP assignments", async () => {
  const spec = generateTreeMakerSpec({
    id: "converter-fixture",
    family: "treemaker-tree",
    seed: 777,
    numCreases: 220,
    bucket: "medium",
  });
  const output: TreeMakerExternalOutput = {
    ok: true,
    toolVersion: "unit-fixture",
    optimization: { success: true },
    foldedForm: { success: true },
    creases: [
      { p1: [0, 0], p2: [1, 0], assignment: "B", kind: "AXIAL" },
      { p1: [1, 0], p2: [1, 1], assignment: "B", kind: "AXIAL" },
      { p1: [1, 1], p2: [0, 1], assignment: "B", kind: "AXIAL" },
      { p1: [0, 1], p2: [0, 0], assignment: "B", kind: "AXIAL" },
      { p1: [0, 0], p2: [1, 1], assignment: "M", kind: "RIDGE" },
      { p1: [0, 1], p2: [1, 0], assignment: "V", kind: "RIDGE" },
      { p1: [0, 0.5], p2: [1, 0.5], assignment: "F", kind: "UNFOLDED_HINGE" },
    ],
  };
  const fold = treeMakerOutputToFold(spec, output);
  expect(fold.tree_metadata?.generator).toBe("treemaker-tree");
  expect(fold.treemaker_metadata?.optimizationSuccess).toBe(true);
  expect(fold.edges_treemakerKind).toHaveLength(fold.edges_vertices.length);
  expect(fold.edges_assignment).toContain("M");
  expect(fold.edges_assignment).toContain("V");
  expect(fold.edges_assignment.some((assignment) => assignment === "U" || assignment === "F")).toBe(true);

  const validation = await validateFold(fold, {
    strictGlobal: false,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 4000,
    maxEdges: 4000,
    requireDense: false,
    requireTreeMaker: true,
    requireLocalFlatFoldability: false,
  });
  expect(validation.valid).toBe(true);
  expect(validation.passed).toContain("treemaker-structure");
});

test("TreeMaker converter keeps the full square sheet when TreeMaker emits a useful polygon border", async () => {
  const spec = generateTreeMakerSpec({
    id: "useful-polygon-fixture",
    family: "treemaker-tree",
    seed: 778,
    numCreases: 220,
    bucket: "medium",
  });
  const output: TreeMakerExternalOutput = {
    ok: true,
    toolVersion: "unit-fixture",
    optimization: { success: true },
    foldedForm: { success: true },
    creases: [
      { p1: [0.2, 0.0000004], p2: [0.8, 0.0000003], assignment: "B", kind: "AXIAL" },
      { p1: [0.8, 0], p2: [1, 0.3], assignment: "B", kind: "AXIAL" },
      { p1: [1, 0.3], p2: [0.55, 0.95], assignment: "B", kind: "AXIAL" },
      { p1: [0.55, 0.95], p2: [0.05, 0.6], assignment: "B", kind: "AXIAL" },
      { p1: [0.05, 0.6], p2: [0.2, 0], assignment: "B", kind: "AXIAL" },
      { p1: [0.2, 0], p2: [0.55, 0.95], assignment: "M", kind: "RIDGE" },
      { p1: [0.05, 0.6], p2: [1, 0.3], assignment: "V", kind: "RIDGE" },
      { p1: [0.2, 0.5], p2: [0.8, 0.5], assignment: "F", kind: "UNFOLDED_HINGE" },
    ],
  };

  const fold = treeMakerOutputToFold(spec, output);
  const corners = new Set(fold.vertices_coords.map(([x, y]) => `${x},${y}`));
  expect(corners.has("0,0")).toBe(true);
  expect(corners.has("1,0")).toBe(true);
  expect(corners.has("1,1")).toBe(true);
  expect(corners.has("0,1")).toBe(true);

  const isBoundary = ([x, y]: [number, number]) => x === 0 || x === 1 || y === 0 || y === 1;
  const borderEdges = fold.edges_vertices
    .map(([a, b], index) => ({ a: fold.vertices_coords[a], b: fold.vertices_coords[b], assignment: fold.edges_assignment[index] }))
    .filter((edge) => edge.assignment === "B");
  expect(borderEdges.length).toBeGreaterThanOrEqual(4);
  expect(borderEdges.every((edge) => isBoundary(edge.a) && isBoundary(edge.b))).toBe(true);
  expect(fold.edges_assignment).toContain("F");

  const validation = await validateFold(fold, {
    strictGlobal: false,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 4000,
    maxEdges: 4000,
    requireDense: false,
    requireTreeMaker: true,
    requireLocalFlatFoldability: false,
  });
  expect(validation.valid).toBe(true);
});

test("treemaker-tree recipe loads and requires a real external CLI for generation", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/treemaker_tree_v1.yaml");
  expect(recipe.families).toEqual({ "bp-studio-realistic": 0, "bp-studio-completed": 0, "treemaker-tree": 1 });
  expect(recipe.treeMakerSampler?.symmetryWeights).toMatchObject({
    diagonal: 0.425,
    "middle-axis": 0.425,
    asymmetric: 0.15,
  });
  const previousCli = process.env.TREEMAKER_CLI;
  delete process.env.TREEMAKER_CLI;
  try {
    expect(() => generateFold({
      id: "tree-generator",
      family: "treemaker-tree",
      seed: 991,
      numCreases: 180,
      bucket: "small",
      treeMakerSampler: recipe.treeMakerSampler,
    })).toThrow(/TREEMAKER_CLI is required/);
  } finally {
    if (previousCli !== undefined) process.env.TREEMAKER_CLI = previousCli;
  }
});

if (process.env.TREEMAKER_CLI) {
  test("treemaker-tree generator runs through the configured real CLI", async () => {
    const recipe = await loadRecipe("../../recipes/synthetic/treemaker_tree_v1.yaml");
    const fold = generateFold({
      id: "real-tree-generator",
      family: "treemaker-tree",
      seed: 917001,
      numCreases: 180,
      bucket: "small",
      treeMakerSampler: recipe.treeMakerSampler,
    });
    expect(fold.edges_vertices.length).toBeGreaterThan(100);
    expect(fold.treemaker_metadata?.foldedFormSuccess).toBe(true);
    expect(fold.label_policy?.geometrySource).toBe("treemaker-external");
  });
}
