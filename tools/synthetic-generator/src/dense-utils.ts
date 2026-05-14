import type { DensityMetadata, EdgeAssignment, FOLDFormat } from "./types.ts";

export type Point = [number, number];

export interface RawSegment {
  a: Point;
  b: Point;
  assignment: EdgeAssignment;
}

export interface DenseBucketSpec {
  bucket: string;
  gridSize: number;
  targetEdgeRange: [number, number];
}

export interface TilePortSignature {
  side: "top" | "right" | "bottom" | "left";
  offsets: number[];
  assignments: EdgeAssignment[];
}

export function denseBucketSpec(bucket: string, seed: number, family: "bp" | "non-bp"): DenseBucketSpec {
  const normalized = bucket === "dense" || bucket === "medium" || bucket === "small" ? bucket : "medium";
  const bpSizes = {
    small: [5, 7],
    medium: [8, 9],
    dense: [10, 16],
  };
  const nonBPSizes = {
    small: [8, 12],
    medium: [24, 32],
    dense: [40, 48],
  };
  const targets: Record<string, [number, number]> = {
    small: [80, 350],
    medium: [180, 750],
    dense: [500, 1800],
  };
  const choices = family === "bp" ? bpSizes[normalized] : nonBPSizes[normalized];
  return {
    bucket: normalized,
    gridSize: choices[Math.abs(seed) % choices.length],
    targetEdgeRange: targets[normalized],
  };
}

export function portsCompatible(a: TilePortSignature, b: TilePortSignature): boolean {
  if (a.offsets.length !== b.offsets.length || a.assignments.length !== b.assignments.length) return false;
  for (let i = 0; i < a.offsets.length; i++) {
    if (Math.abs(a.offsets[i] - b.offsets[i]) > 1e-9) return false;
    if (a.assignments[i] !== b.assignments[i]) return false;
  }
  return true;
}

export function perimeterPosition([x, y]: Point): number {
  if (Math.abs(y) < 1e-8) return x;
  if (Math.abs(x - 1) < 1e-8) return 1 + y;
  if (Math.abs(y - 1) < 1e-8) return 3 - x;
  return 4 - y;
}

export function addDensityMetadata(fold: FOLDFormat, metadata: DensityMetadata): FOLDFormat {
  fold.density_metadata = metadata;
  return fold;
}

export function assignmentCounts(fold: FOLDFormat): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const assignment of fold.edges_assignment) counts[assignment] = (counts[assignment] ?? 0) + 1;
  return counts;
}

