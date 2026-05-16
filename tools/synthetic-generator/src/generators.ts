import { generateBPStudioCompletedFold } from "./bp-studio-completed.ts";
import { generateBPStudioRealisticFold } from "./bp-studio-realistic.ts";
import { generateRabbitEarFoldProgram } from "./rabbit-ear-fold-program.ts";
import { generateTreeMakerFold } from "./treemaker-adapter.ts";
import { GENERATOR_FAMILIES } from "./types.ts";
import type { FOLDFormat, GenerationConfig, GeneratorFamily } from "./types.ts";

export function generateFold(config: GenerationConfig): FOLDFormat {
  if (config.family === "bp-studio-realistic") return generateBPStudioRealisticFold(config);
  if (config.family === "bp-studio-completed") return generateBPStudioCompletedFold(config);
  if (config.family === "treemaker-tree") return generateTreeMakerFold(config);
  if (config.family === "rabbit-ear-fold-program") return generateRabbitEarFoldProgram(config);
  throw new Error(`Unsupported synthetic generator family: ${String(config.family)}`);
}

export function availableFamilies(): GeneratorFamily[] {
  return [...GENERATOR_FAMILIES];
}
