import { GENERATOR_FAMILIES } from "./types.ts";
import type { GeneratorFamily, SyntheticRecipe } from "./types.ts";

const DEFAULT_RECIPE: SyntheticRecipe = {
  name: "bp_studio_realistic_v1",
  seed: 9170,
  imageSize: 768,
  padding: 42,
  splits: { train: 0.85, val: 0.1, test: 0.05 },
  families: {
    "bp-studio-realistic": 1,
  },
  complexityBuckets: [
    { name: "small", minCreases: 80, maxCreases: 300, weight: 0.2 },
    { name: "medium", minCreases: 300, maxCreases: 900, weight: 0.4 },
    { name: "dense", minCreases: 900, maxCreases: 2500, weight: 0.3 },
    { name: "superdense", minCreases: 2500, maxCreases: 6000, weight: 0.1 },
  ],
  validation: {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-5,
    maxVertices: 12000,
    maxEdges: 12000,
    requireBoxPleat: true,
    boxPleatMode: "dense",
    requireDense: true,
    requireRealistic: true,
    minRealismScore: 0.45,
  },
  renderVariants: [
    { name: "bp_assignment_clean", assignmentVisibility: "visible", count: 1 },
    { name: "monochrome_ink", assignmentVisibility: "hidden", count: 1 },
    { name: "faint_blueprint", assignmentVisibility: "hidden", count: 1 },
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
