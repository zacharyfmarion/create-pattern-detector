import { expect, test } from "bun:test";
import ear from "rabbit-ear";
import { findGopsPiece } from "../src/box-pleat.ts";
import { makeFlatFoldedPreview } from "../src/folded-preview.ts";
import { splitForIndex } from "../src/fold-utils.ts";
import { generateFold } from "../src/generators.ts";
import { arrangeSegments } from "../src/line-arrangement.ts";
import { loadRecipe } from "../src/recipe.ts";
import type { GeneratorFamily, ValidationConfig } from "../src/types.ts";
import { preflightValidation, validateFold } from "../src/validate.ts";

const validation: ValidationConfig = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-5,
  maxVertices: 700,
  maxEdges: 1400,
};

test("rabbit-ear exposes every API used by the generator", () => {
  expect(typeof ear.graph.square).toBe("function");
  expect(typeof ear.graph.kite).toBe("function");
  expect(typeof ear.graph.flatFold).toBe("function");
  expect(typeof ear.axiom.axiom1).toBe("function");
  expect(typeof ear.axiom.axiom2).toBe("function");
  expect(typeof ear.axiom.axiom3).toBe("function");
  expect(typeof ear.graph.populate).toBe("function");
  expect(typeof ear.singleVertex.validateKawasaki).toBe("function");
  expect(typeof ear.singleVertex.validateMaekawa).toBe("function");
  expect(typeof ear.graph.makeVerticesCoordsFlatFolded).toBe("function");
  expect(typeof ear.layer.solver).toBe("function");
});

test("fixed seed produces identical FOLD output", () => {
  const config = { id: "fixed", family: "axiom" as const, seed: 12345, numCreases: 32, bucket: "test" };
  expect(JSON.stringify(generateFold(config))).toEqual(JSON.stringify(generateFold(config)));
});

test("recipe file loads through Bun YAML parser", async () => {
  const recipe = await loadRecipe("../../recipes/synthetic/clean_cp_v1.yaml");
  expect(recipe.name).toBe("clean_cp_v1");
  expect(recipe.validation.globalBackend).toBe("rabbit-ear-solver");
  expect(recipe.renderVariants.length).toBe(3);
});

test("deterministic split helper preserves recipe ratios for smoke counts", () => {
  const counts = { train: 0, val: 0, test: 0 };
  for (let index = 0; index < 32; index++) {
    counts[splitForIndex(index, { train: 0.85, val: 0.1, test: 0.05 })] += 1;
  }
  expect(counts).toEqual({ train: 27, val: 4, test: 1 });
});

test("axiom, classic, single-vertex, box-pleat, and baseline grid families produce accepted graphs", async () => {
  const families: GeneratorFamily[] = ["axiom", "classic", "single-vertex", "box-pleat", "grid-baseline"];
  for (const family of families) {
    const fold = generateFold({ id: family, family, seed: 12345, numCreases: 32, bucket: "test" });
    const result = await validateFold(fold, { ...validation, requireBoxPleat: family === "box-pleat" });
    expect(result, `${family}: ${result.errors.join("; ")}`).toMatchObject({ valid: true });
  }
});

test("fixed seed produces identical box-pleat FOLD output", () => {
  const config = { id: "bp-fixed", family: "box-pleat" as const, seed: 8675309, numCreases: 40, bucket: "test" };
  expect(JSON.stringify(generateFold(config))).toEqual(JSON.stringify(generateFold(config)));
});

test("GOPS factor search rejects odd-overlap cases", () => {
  expect(findGopsPiece(3, 5)).toBeNull();
  expect(findGopsPiece(4, 5)).toMatchObject({ ox: 4, oy: 5 });
});

test("line arrangement splits intersections and preserves BP roles", () => {
  const arranged = arrangeSegments([
    { p1: [0, 0.5], p2: [1, 0.5], assignment: "V", role: "hinge" },
    { p1: [0.5, 0], p2: [0.5, 1], assignment: "M", role: "ridge" },
  ]);
  expect(arranged.vertices_coords).toContainEqual([0.5, 0.5]);
  expect(arranged.edges_vertices.length).toBe(4);
  expect(arranged.edges_bpRole).toEqual(expect.arrayContaining(["hinge", "ridge"]));
});

test("box-pleat samples carry BP roles and are not full-sheet alternating grids", async () => {
  const fold = generateFold({ id: "bp", family: "box-pleat", seed: 24680, numCreases: 40, bucket: "test" });
  const result = await validateFold(fold, { ...validation, maxVertices: 900, maxEdges: 1800, requireBoxPleat: true });
  expect(result, result.errors.join("; ")).toMatchObject({ valid: true });
  expect(fold.bp_metadata?.bpSubfamily).toMatch(/two-flap-stretch|uniaxial-chain|symmetric-insect-lite/);
  expect(fold.edges_bpRole).toContain("ridge");
  expect(fold.edges_bpRole).toContain("hinge");

  const roles = fold.edges_bpRole ?? [];
  const interior = roles.filter((role) => role !== "border").length;
  const ridges = roles.filter((role) => role === "ridge").length;
  const axisLike = roles.filter((role) => role === "axis" || role === "stretch" || role === "hinge").length;
  expect(ridges / Math.max(1, interior)).toBeGreaterThanOrEqual(0.08);
  expect(axisLike).toBeLessThan(fold.edges_vertices.length);
});

test("folded preview computes finite flat-folded coordinates after solver pass", () => {
  const fold = generateFold({ id: "preview", family: "box-pleat", seed: 24680, numCreases: 40, bucket: "test" });
  const preview = makeFlatFoldedPreview(fold);
  expect(preview.faces).toBeGreaterThan(1);
  expect(preview.solverRootOrders + preview.solverSolutionCount).toBeGreaterThanOrEqual(0);
  expect(preview.foldedFold.frame_classes).toEqual(["foldedForm"]);
  expect(preview.foldedFold.vertices_coords).toHaveLength(fold.vertices_coords.length);
  for (const coord of preview.foldedFold.vertices_coords) {
    expect(coord.every(Number.isFinite)).toBe(true);
  }
});

test("strict validation preflight uses explicit backends only", () => {
  expect(() => preflightValidation(validation)).not.toThrow();
  expect(() =>
    preflightValidation({
      ...validation,
      globalBackend: "fold-cli",
      foldCliCommand: "/usr/bin/fold",
    }),
  ).toThrow(/system text-wrapping/);
});
