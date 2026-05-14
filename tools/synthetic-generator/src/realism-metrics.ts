import type { FOLDFormat, LayoutMetadata, RealismMetadata } from "./types.ts";

export function scoreFoldRealism(fold: FOLDFormat, layout?: LayoutMetadata): RealismMetadata {
  const roles = fold.edges_bpRole ?? [];
  const bins = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0));
  const orientationHistogram: Record<string, number> = { horizontal: 0, vertical: 0, diagonalMain: 0, diagonalAnti: 0, other: 0 };
  const degrees = new Map<number, number>();
  let nonBorderEdges = 0;

  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    const assignment = fold.edges_assignment[edgeIndex];
    if (assignment === "B") continue;
    nonBorderEdges += 1;
    degrees.set(a, (degrees.get(a) ?? 0) + 1);
    degrees.set(b, (degrees.get(b) ?? 0) + 1);
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    orientationHistogram[orientationKey(dx, dy)] += 1;
    const mx = (p1[0] + p2[0]) / 2;
    const my = (p1[1] + p2[1]) / 2;
    const bx = Math.max(0, Math.min(7, Math.floor(mx * 8)));
    const by = Math.max(0, Math.min(7, Math.floor(my * 8)));
    const densityWeight = roles[edgeIndex] === "ridge" ? 2 : roles[edgeIndex] === "stretch" ? 0.8 : 0.18;
    bins[by][bx] += densityWeight;
  }

  const binValues = bins.flat();
  const mean = binValues.reduce((sum, value) => sum + value, 0) / binValues.length;
  const variance = binValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / binValues.length;
  const localDensityVariance = variance / (mean * mean + 1);
  const emptySpaceRatio = binValues.filter((value) => value <= Math.max(1, mean * 0.08)).length / binValues.length;
  const degreeHistogram: Record<string, number> = {};
  for (const degree of degrees.values()) degreeHistogram[String(degree)] = (degreeHistogram[String(degree)] ?? 0) + 1;
  const roleTotals = countValues(roles);
  const roleRatios: Record<string, number> = {};
  const roleTotal = Math.max(1, Object.values(roleTotals).reduce((sum, value) => sum + value, 0));
  for (const [role, count] of Object.entries(roleTotals)) roleRatios[role] = count / roleTotal;
  const orientationCounts = Object.values(orientationHistogram);
  const maxOrientationShare = Math.max(...orientationCounts) / Math.max(1, nonBorderEdges);
  const macroRegionDiversity = Math.min(
    1,
    ((layout?.bodyRegions.length ?? 0) * 2 + (layout?.flapTerminals.length ?? 0) + (layout?.corridors.length ?? 0)) / 22,
  );
  const repetitionPenalty = clamp01(
    (maxOrientationShare - 0.44) * 1.4
    + (localDensityVariance < 0.35 ? 0.25 : 0)
    + (emptySpaceRatio < 0.04 ? 0.18 : 0),
  );
  const orientationBalance = clamp01(1 - Math.max(0, maxOrientationShare - 0.35) / 0.45);
  const score = clamp01(
    0.25 * macroRegionDiversity
    + 0.25 * clamp01(localDensityVariance / 1.6)
    + 0.18 * clamp01(emptySpaceRatio / 0.25)
    + 0.17 * orientationBalance
    + 0.15 * (1 - repetitionPenalty),
  );

  return {
    score,
    emptySpaceRatio,
    localDensityVariance,
    repetitionPenalty,
    macroRegionDiversity,
    orientationHistogram,
    degreeHistogram,
    roleRatios,
    gates: {
      hasMacroRegions: macroRegionDiversity >= 0.25,
      hasDensityVariation: localDensityVariance >= 0.25,
      notUniformLattice: repetitionPenalty <= 0.75,
      enoughCreases: nonBorderEdges >= 120,
    },
  };
}

function orientationKey(dx: number, dy: number): string {
  if (Math.abs(dy) < 1e-8) return "horizontal";
  if (Math.abs(dx) < 1e-8) return "vertical";
  if (Math.abs(Math.abs(dx) - Math.abs(dy)) < 1e-8) return dx * dy > 0 ? "diagonalMain" : "diagonalAnti";
  return "other";
}

function countValues(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
