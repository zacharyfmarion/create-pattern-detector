import { GENERATOR_FAMILIES } from "./types.ts";
import type { GeneratorFamily, SyntheticRecipe } from "./types.ts";

const DEFAULT_RECIPE: SyntheticRecipe = {
  name: "treemaker_tree_v1",
  seed: 9170,
  imageSize: 768,
  padding: 42,
  splits: { train: 0.85, val: 0.1, test: 0.05 },
  families: {
    "treemaker-tree": 1,
    "rabbit-ear-fold-program": 0,
  },
  complexityBuckets: [
    { name: "small", minCreases: 80, maxCreases: 180, weight: 0.35 },
    { name: "medium", minCreases: 180, maxCreases: 500, weight: 0.4 },
    { name: "dense", minCreases: 500, maxCreases: 1200, weight: 0.2 },
    { name: "superdense", minCreases: 1200, maxCreases: 2400, weight: 0.05 },
  ],
  validation: {
    strictGlobal: false,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 4000,
    maxEdges: 4000,
    requireDense: true,
    requireTreeMaker: true,
    requireLocalFlatFoldability: false,
  },
  renderVariants: [
    { name: "treemaker_full_cp", assignmentVisibility: "visible", count: 1 },
    { name: "treemaker_active_only", assignmentVisibility: "active-only", count: 1 },
    { name: "monochrome_ink", assignmentVisibility: "hidden", count: 1 },
  ],
  treeMakerSampler: {
    symmetryWeights: {
      diagonal: 0.425,
      "middle-axis": 0.425,
      asymmetric: 0.15,
    },
    middleAxisWeights: { vertical: 1, horizontal: 1 },
    diagonalWeights: { "main-diagonal": 1, "anti-diagonal": 1 },
  },
};

export async function loadRecipe(path?: string): Promise<SyntheticRecipe> {
  if (!path) return DEFAULT_RECIPE;
  const text = await Bun.file(path).text();
  const parsed = parseRecipeText(text, path);
  return mergeRecipe(DEFAULT_RECIPE, parsed);
}

function parseRecipeText(text: string, path: string): Partial<SyntheticRecipe> {
  if (path.endsWith(".yaml") || path.endsWith(".yml")) {
    return Bun.YAML.parse(text) as Partial<SyntheticRecipe>;
  }
  return JSON.parse(text) as Partial<SyntheticRecipe>;
}

function mergeRecipe(base: SyntheticRecipe, overrides: Partial<SyntheticRecipe>): SyntheticRecipe {
  return {
    ...base,
    ...overrides,
    splits: { ...base.splits, ...(overrides.splits ?? {}) },
    families: mergeFamilies(base.families, overrides.families),
    validation: { ...base.validation, ...(overrides.validation ?? {}) },
    complexityBuckets: overrides.complexityBuckets ?? base.complexityBuckets,
    renderVariants: overrides.renderVariants ?? base.renderVariants,
    treeMakerSampler: { ...(base.treeMakerSampler ?? {}), ...(overrides.treeMakerSampler ?? {}) },
  };
}

function mergeFamilies(
  base: Record<GeneratorFamily, number>,
  overrides?: Partial<Record<GeneratorFamily, number>>,
): Record<GeneratorFamily, number> {
  if (!overrides) return base;
  const families = Object.fromEntries(GENERATOR_FAMILIES.map((family) => [family, 0])) as Record<GeneratorFamily, number>;
  for (const family of GENERATOR_FAMILIES) families[family] = overrides[family] ?? 0;
  return families;
}
