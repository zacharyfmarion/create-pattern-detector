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
  const symmetry = chooseUnderfilledKey(SYMMETRY_CLASSES, symmetryWeights, countBy(accepted, "symmetryClass"), targetCount);
  if (symmetry) nextSampler.symmetryWeights = oneHot(SYMMETRY_CLASSES, symmetry);

  const topologyWeights = sampler.acceptedMix.topologyWeights ?? sampler.topologyWeights;
  const topology = chooseUnderfilledKey(TOPOLOGIES, topologyWeights, countBy(accepted, "topology"), targetCount);
  if (topology) nextSampler.topologyWeights = oneHot(TOPOLOGIES, topology);

  const archetypeWeights = sampler.acceptedMix.archetypeWeights ?? sampler.archetypeWeights;
  const archetype = chooseUnderfilledKey(ARCHETYPES, archetypeWeights, countBy(accepted, "archetype"), targetCount);
  if (archetype) nextSampler.archetypeWeights = oneHot(ARCHETYPES, archetype);

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

function chooseUnderfilledKey<T extends string>(
  keys: readonly T[],
  weights: Partial<Record<T, number>> | undefined,
  counts: Record<string, number>,
  targetCount: number,
): T | undefined {
  if (!weights || targetCount <= 0) return undefined;
  const entries = keys
    .map((key) => [key, weights[key] ?? 0] as const)
    .filter(([, weight]) => weight > 0);
  const totalWeight = entries.reduce((sum, [, weight]) => sum + weight, 0);
  if (totalWeight <= 0) return undefined;

  let best: { key: T; deficit: number; target: number } | undefined;
  for (const [key, weight] of entries) {
    const target = (weight / totalWeight) * targetCount;
    const deficit = target - (counts[key] ?? 0);
    if (!best || deficit > best.deficit + 1e-9 || (Math.abs(deficit - best.deficit) <= 1e-9 && target > best.target)) {
      best = { key, deficit, target };
    }
  }
  return best && best.deficit > 0 ? best.key : undefined;
}

function oneHot<T extends string>(keys: readonly T[], selected: T): Record<T, number> {
  return Object.fromEntries(keys.map((key) => [key, key === selected ? 1 : 0])) as Record<T, number>;
}
