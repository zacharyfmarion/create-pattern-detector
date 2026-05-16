import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

export function cloneFold(fold: FOLDFormat): FOLDFormat {
  return JSON.parse(JSON.stringify(fold)) as FOLDFormat;
}

export function normalizeFold(fold: FOLDFormat, creator = "cp-synthetic-generator"): FOLDFormat {
  const normalized = cloneFold(fold);
  normalized.file_spec = 1.1;
  normalized.file_creator = creator;
  normalized.file_classes = normalized.file_classes ?? ["singleModel"];
  normalized.frame_classes = normalized.frame_classes ?? ["creasePattern"];
  normalized.vertices_coords = normalized.vertices_coords.map(([x, y]) => [
    roundCoord(x),
    roundCoord(y),
  ]);
  normalized.edges_assignment = normalized.edges_vertices.map((_, i) =>
    normalizeAssignment(normalized.edges_assignment?.[i] ?? "U")
  );
  if (normalized.edges_bpRole) {
    normalized.edges_bpRole = normalized.edges_vertices.map((_, i) =>
      normalizeBPRole(normalized.edges_bpRole?.[i], normalized.edges_assignment[i])
    );
  }
  normalized.edges_foldAngle = normalized.edges_assignment.map(assignmentToFoldAngle);
  return normalized;
}

export function assignmentToFoldAngle(assignment: EdgeAssignment): number {
  if (assignment === "M") return -180;
  if (assignment === "V") return 180;
  return 0;
}

export function normalizeAssignment(assignment: EdgeAssignment): EdgeAssignment {
  if (assignment === "M" || assignment === "V" || assignment === "B" || assignment === "F") return assignment;
  return "U";
}

export function assignmentCounts(fold: FOLDFormat): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const assignment of fold.edges_assignment ?? []) {
    counts[assignment] = (counts[assignment] ?? 0) + 1;
  }
  return counts;
}

export function roleCounts(fold: FOLDFormat): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const role of fold.edges_bpRole ?? []) {
    counts[role] = (counts[role] ?? 0) + 1;
  }
  return counts;
}

export function countCreases(fold: FOLDFormat): number {
  return (fold.edges_assignment ?? []).filter((a) => a === "M" || a === "V" || a === "U" || a === "F").length;
}

export function stableId(prefix: string, seed: number, index: number): string {
  return `${prefix}-${seed.toString(36)}-${index.toString().padStart(6, "0")}`;
}

export function relativePath(path: string, base: string): string {
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return path.startsWith(normalizedBase) ? path.slice(normalizedBase.length) : path;
}

export function splitForIndex(index: number, splits: Record<"train" | "val" | "test", number>): "train" | "val" | "test" {
  const total = splits.train + splits.val + splits.test;
  const position = (index * 0.6180339887498949) % 1;
  const trainCutoff = splits.train / total;
  const valCutoff = (splits.train + splits.val) / total;
  if (position < trainCutoff) return "train";
  if (position < valCutoff) return "val";
  return "test";
}

function roundCoord(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function normalizeBPRole(role: BPRole | undefined, assignment: EdgeAssignment): BPRole {
  if (role === "border" || role === "hinge" || role === "ridge" || role === "axis" || role === "stretch") {
    return role;
  }
  if (assignment === "B") return "border";
  if (assignment === "M") return "ridge";
  return "hinge";
}
