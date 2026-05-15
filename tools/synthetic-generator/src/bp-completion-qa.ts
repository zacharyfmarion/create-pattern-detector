import type { FOLDFormat } from "./types.ts";

type Point = [number, number];

export interface BPCompletionQAReport {
  strictLabelReady: boolean;
  productionDistributionReady: boolean;
  errors: string[];
  warnings: string[];
  metrics: {
    foldedCreases: number;
    moleculeTypeCount: number;
    rejectedPortJoins: number;
    globalLineGroupRatio: number;
    dominantOrientationShare: number;
    maxInteriorVertexDegree: number;
  };
}

export function runBPCompletionQA(fold: FOLDFormat): BPCompletionQAReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  const metrics = {
    foldedCreases: foldedCreaseCount(fold),
    moleculeTypeCount: Object.keys(fold.molecule_metadata?.molecules ?? {}).length,
    rejectedPortJoins: fold.molecule_metadata?.portChecks.rejected ?? 0,
    globalLineGroupRatio: globalLineGroupRatio(fold),
    dominantOrientationShare: dominantOrientationShare(fold),
    maxInteriorVertexDegree: maxInteriorVertexDegree(fold),
  };

  if (fold.label_policy?.labelSource !== "compiler" || fold.label_policy.trainingEligible !== true) {
    errors.push("labels-not-compiler-training-eligible");
  }
  if (!fold.completion_metadata) errors.push("missing-completion-metadata");
  if (metrics.rejectedPortJoins > 0 || (fold.completion_metadata?.rejectedCandidateCount ?? 0) > 0) {
    errors.push("rejected-port-or-candidate-joins");
  }
  if (fold.edges_bpStudioSource?.length) {
    errors.push("compiler-output-uses-bp-studio-edge-source");
  }
  if (!fold.edges_compilerSource?.length) {
    errors.push("missing-compiler-edge-source");
  }

  if (metrics.foldedCreases < 80) warnings.push("too-sparse-for-production-bp");
  if (metrics.moleculeTypeCount < 6) warnings.push("too-few-molecule-types");
  if (metrics.globalLineGroupRatio > 0.55) warnings.push("baseline-global-fold-program-dominance");
  if (metrics.dominantOrientationShare > 0.62) warnings.push("orientation-distribution-too-uniform");
  if (metrics.maxInteriorVertexDegree >= 16) warnings.push("high-degree-global-line-junctions");
  if (fold.completion_metadata?.version.endsWith("/v0.1.0")) warnings.push("baseline-compiler-version-not-production");
  if (
    fold.completion_metadata?.version.endsWith("/v0.2.0") ||
    fold.completion_metadata?.version.endsWith("/v0.3.0") ||
    fold.completion_metadata?.version.endsWith("/v0.4.0") ||
    fold.completion_metadata?.version.endsWith("/v0.5.0")
  ) {
    warnings.push("restricted-pleat-strip-compiler-not-production-distribution");
  }
  if (fold.completion_metadata?.version.endsWith("/v0.6.0")) {
    warnings.push("restricted-clipped-terminal-fan-compiler-not-production-distribution");
  }
  if (fold.completion_metadata?.version.endsWith("/v0.7.0")) {
    warnings.push("restricted-nested-terminal-contour-compiler-not-production-distribution");
  }

  return {
    strictLabelReady: errors.length === 0,
    productionDistributionReady: errors.length === 0 && warnings.length === 0,
    errors,
    warnings,
    metrics,
  };
}

function foldedCreaseCount(fold: FOLDFormat): number {
  return fold.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length;
}

function dominantOrientationShare(fold: FOLDFormat): number {
  const counts: Record<string, number> = {};
  let total = 0;
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    if (fold.edges_assignment[edgeIndex] === "B") continue;
    const key = orientationKey(fold.vertices_coords[a], fold.vertices_coords[b]);
    counts[key] = (counts[key] ?? 0) + 1;
    total += 1;
  }
  return total === 0 ? 0 : Math.max(...Object.values(counts)) / total;
}

function globalLineGroupRatio(fold: FOLDFormat): number {
  const groups = new Map<string, { min: number; max: number }>();
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    if (fold.edges_assignment[edgeIndex] === "B") continue;
    const p = fold.vertices_coords[a];
    const q = fold.vertices_coords[b];
    const signature = lineSignature(p, q);
    if (!signature) continue;
    const [key, min, max] = signature;
    const current = groups.get(key);
    groups.set(key, current ? { min: Math.min(current.min, min), max: Math.max(current.max, max) } : { min, max });
  }
  if (groups.size === 0) return 0;
  const global = [...groups.values()].filter(({ min, max }) => min <= 0.03 && max >= 0.97).length;
  return global / groups.size;
}

function maxInteriorVertexDegree(fold: FOLDFormat): number {
  const borderVertices = new Set<number>();
  const degrees = Array.from({ length: fold.vertices_coords.length }, () => 0);
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    if (fold.edges_assignment[edgeIndex] === "B") {
      borderVertices.add(a);
      borderVertices.add(b);
      continue;
    }
    degrees[a] += 1;
    degrees[b] += 1;
  }
  return Math.max(0, ...degrees.filter((_, vertex) => !borderVertices.has(vertex)));
}

function orientationKey(a: Point, b: Point): string {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (Math.abs(dx) < 1e-8) return "vertical";
  if (Math.abs(dy) < 1e-8) return "horizontal";
  if (Math.abs(Math.abs(dx) - Math.abs(dy)) < 1e-8) return dx * dy > 0 ? "diagonal-positive" : "diagonal-negative";
  return "other";
}

function lineSignature(a: Point, b: Point): [string, number, number] | null {
  const key = orientationKey(a, b);
  if (key === "vertical") return [`v:${a[0].toFixed(6)}`, Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  if (key === "horizontal") return [`h:${a[1].toFixed(6)}`, Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  if (key === "diagonal-positive") {
    return [`dp:${(a[1] - a[0]).toFixed(6)}`, Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  }
  if (key === "diagonal-negative") {
    return [`dn:${(a[1] + a[0]).toFixed(6)}`, Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  }
  return null;
}
