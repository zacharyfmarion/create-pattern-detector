#!/usr/bin/env bun
import ear from "rabbit-ear";
import { normalizeBPStudioFold, type BPStudioExportInput } from "./bp-studio-validation.ts";
import { assignmentCounts, assignmentToFoldAngle, cloneFold, normalizeFold, roleCounts } from "./fold-utils.ts";
import type { BPRole, EdgeAssignment, FOLDFormat, ValidationConfig, ValidationResult } from "./types.ts";
import { validateFold } from "./validate.ts";

export type BPStrictRepairMode = "unassign" | "delete";

export interface BPStrictRepairLabOptions {
  mode?: BPStrictRepairMode;
  maxSteps?: number;
  validationConfig?: ValidationConfig;
}

export interface BPStrictRepairLocalState {
  kawasakiBadVertices: number[];
  maekawaBadVertices: number[];
  badVertices: number[];
}

export interface BPStrictRepairFoldedCounts {
  total: number;
  byAssignment: Record<string, number>;
  byRole: Record<string, number>;
  byAssignmentRole: Record<string, number>;
}

export interface BPStrictRepairStep {
  step: number;
  mode: BPStrictRepairMode;
  edge: number;
  originalEdge: number;
  vertices: [number, number];
  assignment: EdgeAssignment;
  role?: BPRole;
  badVerticesBefore: number[];
  badVerticesAfter: number[];
  foldedBefore: BPStrictRepairFoldedCounts;
  foldedAfter: BPStrictRepairFoldedCounts;
}

export interface BPStrictRepairLabSummary {
  mode: BPStrictRepairMode;
  normalizedVertices: number;
  normalizedEdges: number;
  repairedVertices: number;
  repairedEdges: number;
  assignmentsBefore: Record<string, number>;
  assignmentsAfter: Record<string, number>;
  rolesBefore: Record<string, number>;
  rolesAfter: Record<string, number>;
  foldedBefore: BPStrictRepairFoldedCounts;
  foldedAfter: BPStrictRepairFoldedCounts;
  foldedRemoved: number;
  foldedRetentionRatio: number;
  badVerticesBefore: number;
  badVerticesAfter: number;
  kawasakiBadBefore: number;
  kawasakiBadAfter: number;
  maekawaBadBefore: number;
  maekawaBadAfter: number;
  repairSteps: number;
  localFlatFoldableBefore: boolean;
  localFlatFoldableAfter: boolean;
  validationStrictGlobalFalseValid: boolean;
  validationFailed: string[];
}

export interface BPStrictRepairLabReport {
  normalized: FOLDFormat;
  repaired: FOLDFormat;
  mode: BPStrictRepairMode;
  initial: BPStrictRepairLocalState;
  final: BPStrictRepairLocalState;
  validation: ValidationResult;
  steps: BPStrictRepairStep[];
  summary: BPStrictRepairLabSummary;
}

interface CliArgs {
  fold: string;
  mode: BPStrictRepairMode;
  maxSteps?: number;
  json: boolean;
  out?: string;
}

interface WorkingGraph {
  fold: FOLDFormat;
  originalEdges: number[];
}

interface CandidateScore {
  edge: number;
  state: BPStrictRepairLocalState;
  folded: BPStrictRepairFoldedCounts;
  badCount: number;
  localFailureCount: number;
  foldedTotal: number;
}

const DEFAULT_VALIDATION_CONFIG: ValidationConfig = {
  strictGlobal: false,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-9,
  maxVertices: 100_000,
  maxEdges: 200_000,
};

const FOLDED_ASSIGNMENTS = new Set<EdgeAssignment>(["M", "V"]);

export async function runBPStrictRepairLab(
  input: BPStudioExportInput,
  options: BPStrictRepairLabOptions = {},
): Promise<BPStrictRepairLabReport> {
  const mode = options.mode ?? "unassign";
  const maxSteps = options.maxSteps ?? 10_000;
  const normalized = normalizeBPStudioFold(input, {
    auxiliaryPolicy: "valley",
    creator: "cp-synthetic-generator/bp-strict-repair-lab",
  });
  const working: WorkingGraph = {
    fold: cloneFold(normalized),
    originalEdges: normalized.edges_vertices.map((_, edge) => edge),
  };
  const initial = localState(working.fold);
  const steps: BPStrictRepairStep[] = [];

  for (let step = 0; step < maxSteps; step++) {
    const before = localState(working.fold);
    if (before.badVertices.length === 0) break;
    const selected = chooseRepairCandidate(working.fold, before, mode);
    if (!selected) break;

    const edge = selected.edge;
    const assignment = working.fold.edges_assignment[edge];
    const role = working.fold.edges_bpRole?.[edge];
    const vertices = working.fold.edges_vertices[edge];
    const foldedBefore = foldedCounts(working.fold);
    const originalEdge = working.originalEdges[edge];
    applyRepair(working, edge, mode);
    const after = localState(working.fold);

    steps.push({
      step,
      mode,
      edge,
      originalEdge,
      vertices,
      assignment,
      ...(role ? { role } : {}),
      badVerticesBefore: before.badVertices,
      badVerticesAfter: after.badVertices,
      foldedBefore,
      foldedAfter: foldedCounts(working.fold),
    });
  }

  const final = localState(working.fold);
  const validationConfig = { ...DEFAULT_VALIDATION_CONFIG, ...options.validationConfig, strictGlobal: false };
  const validation = await validateFold(working.fold, validationConfig);
  const foldedBefore = foldedCounts(normalized);
  const foldedAfter = foldedCounts(working.fold);

  return {
    normalized,
    repaired: working.fold,
    mode,
    initial,
    final,
    validation,
    steps,
    summary: {
      mode,
      normalizedVertices: normalized.vertices_coords.length,
      normalizedEdges: normalized.edges_vertices.length,
      repairedVertices: working.fold.vertices_coords.length,
      repairedEdges: working.fold.edges_vertices.length,
      assignmentsBefore: assignmentCounts(normalized),
      assignmentsAfter: assignmentCounts(working.fold),
      rolesBefore: roleCounts(normalized),
      rolesAfter: roleCounts(working.fold),
      foldedBefore,
      foldedAfter,
      foldedRemoved: foldedBefore.total - foldedAfter.total,
      foldedRetentionRatio: foldedBefore.total === 0 ? 1 : foldedAfter.total / foldedBefore.total,
      badVerticesBefore: initial.badVertices.length,
      badVerticesAfter: final.badVertices.length,
      kawasakiBadBefore: initial.kawasakiBadVertices.length,
      kawasakiBadAfter: final.kawasakiBadVertices.length,
      maekawaBadBefore: initial.maekawaBadVertices.length,
      maekawaBadAfter: final.maekawaBadVertices.length,
      repairSteps: steps.length,
      localFlatFoldableBefore: initial.badVertices.length === 0,
      localFlatFoldableAfter: final.badVertices.length === 0,
      validationStrictGlobalFalseValid: validation.valid,
      validationFailed: validation.failed,
    },
  };
}

export async function runBPStrictRepairLabFile(
  path: string,
  options: BPStrictRepairLabOptions = {},
): Promise<BPStrictRepairLabReport> {
  return runBPStrictRepairLab(await readJson<BPStudioExportInput>(path), options);
}

export function formatBPStrictRepairLabReport(report: BPStrictRepairLabReport, label = "BP strict repair lab"): string {
  const lines = [
    label,
    `mode: ${report.mode}`,
    `normalized: vertices=${report.summary.normalizedVertices} edges=${report.summary.normalizedEdges}`,
    `repaired: vertices=${report.summary.repairedVertices} edges=${report.summary.repairedEdges}`,
    `assignments before: ${formatCounts(report.summary.assignmentsBefore)}`,
    `assignments after: ${formatCounts(report.summary.assignmentsAfter)}`,
    `folded before: total=${report.summary.foldedBefore.total} assignments=${formatCounts(report.summary.foldedBefore.byAssignment)} roles=${formatCounts(report.summary.foldedBefore.byRole)} assignmentRoles=${formatCounts(report.summary.foldedBefore.byAssignmentRole)}`,
    `folded after: total=${report.summary.foldedAfter.total} assignments=${formatCounts(report.summary.foldedAfter.byAssignment)} roles=${formatCounts(report.summary.foldedAfter.byRole)} assignmentRoles=${formatCounts(report.summary.foldedAfter.byAssignmentRole)}`,
    `local before: flatFoldable=${report.summary.localFlatFoldableBefore} bad=${report.summary.badVerticesBefore} kawasaki=${report.summary.kawasakiBadBefore} maekawa=${report.summary.maekawaBadBefore}`,
    `local after: flatFoldable=${report.summary.localFlatFoldableAfter} bad=${report.summary.badVerticesAfter} kawasaki=${report.summary.kawasakiBadAfter} maekawa=${report.summary.maekawaBadAfter}`,
    `repair: steps=${report.summary.repairSteps} foldedRemoved=${report.summary.foldedRemoved} retention=${round(report.summary.foldedRetentionRatio)}`,
    `validation strictGlobal=false: valid=${report.validation.valid} failed=${report.validation.failed.length ? report.validation.failed.join(", ") : "none"}`,
  ];

  if (report.steps.length > 0) {
    lines.push("steps:");
    for (const step of report.steps) {
      lines.push(
        `  ${step.step}: ${step.mode} edge=${step.edge} original=${step.originalEdge} assignment=${step.assignment} role=${step.role ?? "unlabeled"} bad=${step.badVerticesBefore.length}->${step.badVerticesAfter.length}`,
      );
    }
  }
  if (report.validation.errors.length > 0) {
    lines.push("validation errors:");
    for (const error of report.validation.errors) lines.push(`  ${error}`);
  }
  return `${lines.join("\n")}\n`;
}

function chooseRepairCandidate(
  fold: FOLDFormat,
  state: BPStrictRepairLocalState,
  mode: BPStrictRepairMode,
): CandidateScore | undefined {
  const badVertices = new Set(state.badVertices);
  const candidateEdges = new Set<number>();
  for (const [edge, vertices] of fold.edges_vertices.entries()) {
    if (!FOLDED_ASSIGNMENTS.has(fold.edges_assignment[edge])) continue;
    if (badVertices.has(vertices[0]) || badVertices.has(vertices[1])) candidateEdges.add(edge);
  }

  let best: CandidateScore | undefined;
  for (const edge of candidateEdges) {
    const candidateFold = cloneFold(fold);
    const candidate: WorkingGraph = {
      fold: candidateFold,
      originalEdges: candidateFold.edges_vertices.map((_, index) => index),
    };
    applyRepair(candidate, edge, mode);
    const candidateState = localState(candidate.fold);
    const candidateFolded = foldedCounts(candidate.fold);
    const score: CandidateScore = {
      edge,
      state: candidateState,
      folded: candidateFolded,
      badCount: candidateState.badVertices.length,
      localFailureCount: candidateState.kawasakiBadVertices.length + candidateState.maekawaBadVertices.length,
      foldedTotal: candidateFolded.total,
    };
    if (!best || compareScore(score, best) < 0) best = score;
  }
  return best;
}

function compareScore(a: CandidateScore, b: CandidateScore): number {
  return (
    a.badCount - b.badCount ||
    a.localFailureCount - b.localFailureCount ||
    b.foldedTotal - a.foldedTotal ||
    a.edge - b.edge
  );
}

function applyRepair(working: WorkingGraph, edge: number, mode: BPStrictRepairMode): void {
  if (mode === "unassign") {
    working.fold.edges_assignment[edge] = "U";
    if (working.fold.edges_foldAngle) working.fold.edges_foldAngle[edge] = assignmentToFoldAngle("U");
    return;
  }

  working.fold.edges_vertices.splice(edge, 1);
  working.fold.edges_assignment.splice(edge, 1);
  working.fold.edges_foldAngle?.splice(edge, 1);
  working.fold.edges_bpRole?.splice(edge, 1);
  working.originalEdges.splice(edge, 1);
}

function localState(fold: FOLDFormat): BPStrictRepairLocalState {
  const graph = normalizeFold(fold, fold.file_creator) as FOLDFormat & {
    vertices_edges?: number[][];
    vertices_vertices?: number[][];
  };
  ear.graph.populate(graph);
  const kawasakiBadVertices = sortedUnique(ear.singleVertex.validateKawasaki(graph) as number[]);
  const maekawaBadVertices = sortedUnique(ear.singleVertex.validateMaekawa(graph) as number[]);
  return {
    kawasakiBadVertices,
    maekawaBadVertices,
    badVertices: sortedUnique([...kawasakiBadVertices, ...maekawaBadVertices]),
  };
}

function foldedCounts(fold: FOLDFormat): BPStrictRepairFoldedCounts {
  const byAssignment: Record<string, number> = {};
  const byRole: Record<string, number> = {};
  const byAssignmentRole: Record<string, number> = {};
  let total = 0;

  for (const [edge, assignment] of fold.edges_assignment.entries()) {
    if (!FOLDED_ASSIGNMENTS.has(assignment)) continue;
    const role = fold.edges_bpRole?.[edge] ?? "unlabeled";
    total += 1;
    byAssignment[assignment] = (byAssignment[assignment] ?? 0) + 1;
    byRole[role] = (byRole[role] ?? 0) + 1;
    const pair = `${assignment}:${role}`;
    byAssignmentRole[pair] = (byAssignmentRole[pair] ?? 0) + 1;
  }

  return { total, byAssignment, byRole, byAssignmentRole };
}

function sortedUnique(values: readonly number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function formatCounts(counts: Record<string, number>): string {
  const keys = Object.keys(counts).sort();
  return keys.length ? keys.map((key) => `${key}=${counts[key]}`).join(" ") : "none";
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { fold: "", mode: "unassign", json: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--fold") args.fold = argv[++i];
    else if (arg === "--mode") args.mode = parseMode(argv[++i]);
    else if (arg === "--max-steps") args.maxSteps = Number.parseInt(argv[++i], 10);
    else if (arg === "--json") args.json = true;
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run src/bp-strict-repair-lab.ts --fold <path> [--mode unassign|delete] [--max-steps <n>] [--out <path>] [--json]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.fold) throw new Error("--fold is required");
  if (args.maxSteps !== undefined && (!Number.isInteger(args.maxSteps) || args.maxSteps < 0)) {
    throw new Error("--max-steps must be a nonnegative integer");
  }
  return args;
}

function parseMode(value: string | undefined): BPStrictRepairMode {
  if (value === "unassign" || value === "delete") return value;
  throw new Error("--mode must be unassign or delete");
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await Bun.file(path).text()) as T;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const report = await runBPStrictRepairLabFile(args.fold, { mode: args.mode, maxSteps: args.maxSteps });
  if (args.out) await Bun.write(args.out, `${JSON.stringify(report.repaired, null, 2)}\n`);
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  process.stdout.write(formatBPStrictRepairLabReport(report, args.fold));
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
