import { expect, test } from "bun:test";
import ear from "rabbit-ear";
import { findGopsPiece } from "../src/box-pleat.ts";
import { makeDenseBoxPleatTessellation, portsCompatible } from "../src/dense-box-pleat.ts";
import { makeFlatFoldedPreview } from "../src/folded-preview.ts";
import { splitForIndex } from "../src/fold-utils.ts";
import { generateFold } from "../src/generators.ts";
import { arrangeSegments } from "../src/line-arrangement.ts";
import { scoreFoldRealism } from "../src/realistic-box-pleat.ts";
import { loadRecipe } from "../src/recipe.ts";
import type { GeneratorFamily, RealisticBPArchetype, ValidationConfig } from "../src/types.ts";
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

test("axiom, classic, single-vertex, box-pleat, realistic BP, dense non-BP, and baseline grid families produce accepted graphs", async () => {
  const families: GeneratorFamily[] = ["axiom", "classic", "single-vertex", "box-pleat", "realistic-box-pleat", "dense-non-bp", "grid-baseline"];
  for (const family of families) {
    const dense = family === "dense-non-bp" || family === "realistic-box-pleat";
    const fold = generateFold({ id: family, family, seed: 12345, numCreases: dense ? 220 : 32, bucket: dense ? "small" : "test", dense });
    const result = await validateFold(fold, {
      ...validation,
      maxVertices: 1200,
      maxEdges: 3000,
      requireBoxPleat: family === "box-pleat" || family === "realistic-box-pleat",
      boxPleatMode: family === "realistic-box-pleat" ? "dense" : "simple",
      requireDense: dense,
      requireRealistic: family === "realistic-box-pleat",
      minRealismScore: family === "realistic-box-pleat" ? 0.35 : undefined,
    });
    expect(result, `${family}: ${result.errors.join("; ")}`).toMatchObject({ valid: true });
  }
});

test("fixed seed produces identical box-pleat FOLD output", () => {
  const config = { id: "bp-fixed", family: "box-pleat" as const, seed: 8675309, numCreases: 40, bucket: "test" };
  expect(JSON.stringify(generateFold(config))).toEqual(JSON.stringify(generateFold(config)));
});

test("fixed seed produces identical realistic box-pleat FOLD output", () => {
  const config = {
    id: "realistic-bp-fixed",
    family: "realistic-box-pleat" as const,
    seed: 13579,
    numCreases: 640,
    bucket: "medium",
    realisticArchetype: "insect" as const,
  };
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

test("dense box-pleat samples carry density metadata and pass dense BP validation", async () => {
  const fold = generateFold({ id: "dense-bp", family: "box-pleat", seed: 97531, numCreases: 600, bucket: "dense", dense: true });
  const result = await validateFold(fold, {
    ...validation,
    maxVertices: 3000,
    maxEdges: 4000,
    requireBoxPleat: true,
    boxPleatMode: "dense",
    requireDense: true,
  });
  expect(result, result.errors.join("; ")).toMatchObject({ valid: true });
  expect(fold.bp_metadata?.bpSubfamily).toBe("dense-molecule-tessellation");
  expect(fold.density_metadata).toMatchObject({ densityBucket: "dense", subfamily: "dense-molecule-tessellation" });
  expect(fold.edges_vertices.length).toBeGreaterThanOrEqual(fold.density_metadata!.targetEdgeRange[0]);
});

test("dense BP molecule geometry remains grid or 45-degree after construction", () => {
  const fold = makeDenseBoxPleatTessellation(7, "small", [80, 350]);
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    const dx = Math.abs(p1[0] - p2[0]);
    const dy = Math.abs(p1[1] - p2[1]);
    const role = fold.edges_bpRole?.[edgeIndex];
    expect(dx < 1e-8 || dy < 1e-8 || Math.abs(dx - dy) < 1e-8, `edge ${edgeIndex}`).toBe(true);
    if (role === "ridge") expect(Math.abs(dx - dy)).toBeLessThan(1e-8);
  }
});

test("each realistic BP archetype produces a strict accepted sample", async () => {
  const archetypes: RealisticBPArchetype[] = ["insect", "quadruped", "bird", "object", "abstract"];
  for (const realisticArchetype of archetypes) {
    const fold = generateFold({
      id: realisticArchetype,
      family: "realistic-box-pleat",
      seed: 86420 + realisticArchetype.length,
      numCreases: 640,
      bucket: "medium",
      realisticArchetype,
    });
    const result = await validateFold(fold, {
      ...validation,
      maxVertices: 2400,
      maxEdges: 4000,
      requireBoxPleat: true,
      boxPleatMode: "dense",
      requireDense: true,
      requireRealistic: true,
      minRealismScore: 0.35,
    });
    expect(result, `${realisticArchetype}: ${result.errors.join("; ")}`).toMatchObject({ valid: true });
    expect(fold.design_tree?.archetype).toBe(realisticArchetype);
    expect(fold.realism_metadata?.gates.hasDensityVariation).toBe(true);
  }
});

test("realism gates penalize uniform dense lattices", () => {
  const realistic = generateFold({
    id: "realistic-reference",
    family: "realistic-box-pleat",
    seed: 42,
    numCreases: 260,
    bucket: "small",
    realisticArchetype: "insect",
  });
  const uniform = makeDenseBoxPleatTessellation(10, "small", [80, 350], { seed: 1, pattern: "uniform-twist" });
  const uniformRealism = scoreFoldRealism(uniform, realistic.layout_metadata);
  expect(realistic.realism_metadata?.gates.hasDensityVariation).toBe(true);
  expect(uniformRealism.gates.hasDensityVariation).toBe(false);
});

test("dense tile port compatibility rejects mismatched ports", () => {
  expect(
    portsCompatible(
      { side: "right", offsets: [0.25, 0.75], assignments: ["M", "V"] },
      { side: "left", offsets: [0.25, 0.75], assignments: ["M", "V"] },
    ),
  ).toBe(true);
  expect(
    portsCompatible(
      { side: "right", offsets: [0.25, 0.75], assignments: ["M", "V"] },
      { side: "left", offsets: [0.25, 0.75], assignments: ["V", "M"] },
    ),
  ).toBe(false);
});

test("each dense non-BP subfamily produces a strict accepted sample", async () => {
  const subfamilies = ["recursive-axiom", "expanded-classic", "radial-multi-vertex", "tessellation-like"];
  for (const denseSubfamily of subfamilies) {
    const fold = generateFold({
      id: denseSubfamily,
      family: "dense-non-bp",
      seed: 12345,
      numCreases: 200,
      bucket: "small",
      dense: true,
      denseSubfamily,
    });
    const result = await validateFold(fold, { ...validation, maxVertices: 3000, maxEdges: 4000, requireDense: true });
    expect(result, `${denseSubfamily}: ${result.errors.join("; ")}`).toMatchObject({ valid: true });
    expect(fold.density_metadata?.subfamily).toBe(denseSubfamily);
  }
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
