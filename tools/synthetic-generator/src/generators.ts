import { generateRabbitEarFoldProgram } from "./rabbit-ear-fold-program.ts";
import { generateTreeMakerFold } from "./treemaker-adapter.ts";
import { GENERATOR_FAMILIES } from "./types.ts";
import type { FOLDFormat, GenerationConfig, GeneratorFamily } from "./types.ts";

export function generateFold(config: GenerationConfig): FOLDFormat {
  if (config.family === "treemaker-tree") return generateTreeMakerFold(config);
  if (config.family === "rabbit-ear-fold-program") return generateRabbitEarFoldProgram(config);
  throw new Error(`Unsupported synthetic generator family: ${String(config.family)}`);
}

export function availableFamilies(): GeneratorFamily[] {
  return [...GENERATOR_FAMILIES];
}
