import { GENERATOR_FAMILIES } from "./types.ts";
import type { GeneratorFamily, SyntheticRecipe } from "./types.ts";

const DEFAULT_RECIPE: SyntheticRecipe = {
  name: "clean_cp_v1",
  seed: 9170,
  imageSize: 512,
  padding: 32,
  splits: { train: 0.85, val: 0.1, test: 0.05 },
  families: {
    axiom: 0.35,
    classic: 0.2,
    "single-vertex": 0.25,
    "box-pleat": 0.2,
    "realistic-box-pleat": 0,
    "bp-studio-realistic": 0,
    "dense-non-bp": 0,
    "grid-baseline": 0,
  },
  complexityBuckets: [
    { name: "small", minCreases: 8, maxCreases: 24, weight: 0.4 },
    { name: "medium", minCreases: 25, maxCreases: 80, weight: 0.45 },
    { name: "large", minCreases: 81, maxCreases: 180, weight: 0.15 },
  ],
  validation: {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 700,
    maxEdges: 1400,
    requireBoxPleat: false,
    boxPleatMode: "simple",
    requireDense: false,
    requireRealistic: false,
    minRealismScore: 0,
  },
  renderVariants: [
    { name: "mv_color", assignmentVisibility: "visible", count: 1 },
    { name: "monochrome_black", assignmentVisibility: "hidden", count: 1 },
  ],
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
