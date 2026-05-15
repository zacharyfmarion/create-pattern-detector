import { generateBPStudioCompletedFold } from "./bp-studio-completed.ts";
import { generateBPStudioRealisticFold } from "./bp-studio-realistic.ts";
import { GENERATOR_FAMILIES } from "./types.ts";
import type { FOLDFormat, GenerationConfig, GeneratorFamily } from "./types.ts";

export function generateFold(config: GenerationConfig): FOLDFormat {
  if (config.family === "bp-studio-realistic") return generateBPStudioRealisticFold(config);
  if (config.family === "bp-studio-completed") return generateBPStudioCompletedFold(config);
  throw new Error(`Synthetic generation is BP-Studio-backed-only. Unsupported generator family: ${String(config.family)}`);
}

export function availableFamilies(): GeneratorFamily[] {
  return [...GENERATOR_FAMILIES];
}
