import type {
  TreeMakerArchetype,
  TreeMakerSamplerConfig,
  TreeMakerSymmetryClass,
  TreeMakerTopology,
  TreeMetadata,
} from "./types.ts";

const SYMMETRY_CLASSES: TreeMakerSymmetryClass[] = ["diagonal", "middle-axis", "asymmetric"];
const TOPOLOGIES: TreeMakerTopology[] = ["radial-star", "hubbed-limbs", "spine-chain", "branched-hybrid"];
const ARCHETYPES: TreeMakerArchetype[] = ["insect", "quadruped", "bird", "creature", "object", "abstract"];

export function samplerForAcceptedTreeMakerMix(
  sampler: TreeMakerSamplerConfig | undefined,
  accepted: TreeMetadata[],
  targetCount: number,
): TreeMakerSamplerConfig | undefined {
  if (!sampler?.acceptedMix?.enabled) return sampler;
  const nextSampler: TreeMakerSamplerConfig = { ...sampler };

  const symmetryWeights = sampler.acceptedMix.symmetryWeights ?? sampler.symmetryWeights;
  nextSampler.symmetryWeights = quotaWeights(SYMMETRY_CLASSES, symmetryWeights, countBy(accepted, "symmetryClass"), targetCount);

  const topologyWeights = sampler.acceptedMix.topologyWeights ?? sampler.topologyWeights;
  nextSampler.topologyWeights = quotaWeights(TOPOLOGIES, topologyWeights, countBy(accepted, "topology"), targetCount);

  const archetypeWeights = sampler.acceptedMix.archetypeWeights ?? sampler.archetypeWeights;
  nextSampler.archetypeWeights = quotaWeights(ARCHETYPES, archetypeWeights, countBy(accepted, "archetype"), targetCount);

  return nextSampler;
}

function countBy<K extends keyof TreeMetadata>(items: TreeMetadata[], key: K): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const item of items) {
    const value = item[key];
    if (typeof value !== "string") continue;
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function quotaWeights<T extends string>(
  keys: readonly T[],
  weights: Partial<Record<T, number>> | undefined,
  counts: Record<string, number>,
  targetCount: number,
): Record<T, number> | undefined {
  if (!weights || targetCount <= 0) return undefined;
  const entries = keys
    .map((key) => [key, weights[key] ?? 0] as const)
    .filter(([, weight]) => weight > 0);
  const totalWeight = entries.reduce((sum, [, weight]) => sum + weight, 0);
  if (totalWeight <= 0) return undefined;

  const quotas = new Map<T, number>();
  for (const [key, weight] of entries) {
    const target = (weight / totalWeight) * targetCount;
    const deficit = target - (counts[key] ?? 0);
    quotas.set(key, Math.max(0, deficit));
  }
  const positiveTotal = [...quotas.values()].reduce((sum, value) => sum + value, 0);
  if (positiveTotal <= 0) return Object.fromEntries(keys.map((key) => [key, weights[key] ?? 0])) as Record<T, number>;
  return Object.fromEntries(keys.map((key) => [key, quotas.get(key) ?? 0])) as Record<T, number>;
}
