#!/usr/bin/env bun
import ear from "rabbit-ear";
import { normalizeBPStudioFold, type BPStudioCanonicalAuxiliaryPolicy, type BPStudioExportInput } from "./bp-studio-validation.ts";
import { assignmentCounts, normalizeFold, roleCounts } from "./fold-utils.ts";
import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

export interface BPLocalDiagnosticsOptions {
  auxiliaryPolicy?: BPStudioCanonicalAuxiliaryPolicy;
  alternatePolicies?: BPStudioCanonicalAuxiliaryPolicy[];
  maxVertexSamples?: number;
}

export interface BPLocalPolicySummary {
  auxiliaryPolicy: BPStudioCanonicalAuxiliaryPolicy;
  vertices: number;
  edges: number;
  assignments: Record<string, number>;
  roles: Record<string, number>;
  badVertices: number;
  kawasakiBadVertices: number;
  maekawaBadVertices: number;
  badDegreeHistogram: Record<string, number>;
  badFoldedDegreeHistogram: Record<string, number>;
  neutralHingeIncidentRatio: number;
  auxiliaryLikeHingeIncidentRatio: number;
  dominatedByNeutralHinges: boolean;
  dominatedByAuxiliaryLikeHinges: boolean;
}

export interface BPLocalBadVertexSample {
  vertex: number;
  coord: Point;
  degree: number;
  foldedDegree: number;
  assignments: Record<string, number>;
  roles: Record<string, number>;
  sourceKinds: Record<string, number>;
  sourceKindRoleAssignments: Record<string, number>;
  kawasakiBad: boolean;
  maekawaBad: boolean;
  kawasakiDeviation: number | null;
  maekawaDelta: number | null;
  incidentEdges: number[];
}

export interface BPLocalDiagnosticsReport {
  summary: BPLocalPolicySummary;
  degreeHistogram: Record<string, number>;
  foldedDegreeHistogram: Record<string, number>;
  badVertexIncidentCounts: {
    total: number;
    uniqueEdges: number;
    bySourceKind: Record<string, number>;
    byRole: Record<string, number>;
    byAssignment: Record<string, number>;
    bySourceKindRoleAssignment: Record<string, number>;
    byCreaseType: Record<string, number>;
    neutralHingeIncidents: number;
    auxiliaryLikeHingeIncidents: number;
  };
  badVertices: {
    all: number[];
    kawasaki: number[];
    maekawa: number[];
    samples: BPLocalBadVertexSample[];
  };
  alternatePolicies: BPLocalPolicySummary[];
}

interface LocalState {
  kawasakiBadVertices: number[];
  maekawaBadVertices: number[];
  badVertices: number[];
}

interface IncidentEdge {
  edge: number;
  other: number;
  assignment: EdgeAssignment;
  role: BPRole | "unlabeled";
  sourceKind: string;
  creaseType: string;
}

interface CliArgs {
  fold: string;
  auxiliaryPolicy: BPStudioCanonicalAuxiliaryPolicy;
  json: boolean;
  maxVertexSamples: number;
}

const DEFAULT_ALTERNATE_POLICIES: BPStudioCanonicalAuxiliaryPolicy[] = ["valley", "unassigned"];
const FOLDED_ASSIGNMENTS = new Set<EdgeAssignment>(["M", "V"]);
const NEUTRAL_ASSIGNMENTS = new Set<EdgeAssignment>(["U", "F"]);

export async function runBPLocalDiagnosticsFile(
  path: string,
  options: BPLocalDiagnosticsOptions = {},
): Promise<BPLocalDiagnosticsReport> {
  return runBPLocalDiagnostics(await readJson<BPStudioExportInput>(path), options);
}

export function runBPLocalDiagnostics(
  input: BPStudioExportInput,
  options: BPLocalDiagnosticsOptions = {},
): BPLocalDiagnosticsReport {
  const auxiliaryPolicy = options.auxiliaryPolicy ?? "valley";
  const fold = normalizeBPStudioFold(input, {
    auxiliaryPolicy,
    creator: "cp-synthetic-generator/bp-local-diagnostics",
  });
  const state = localState(fold);
  const adjacency = buildAdjacency(fold);
  const badSet = new Set(state.badVertices);
  const badIncidentEdges = new Set<number>();
  const bySourceKind: Record<string, number> = {};
  const byRole: Record<string, number> = {};
  const byAssignment: Record<string, number> = {};
  const bySourceKindRoleAssignment: Record<string, number> = {};
  const byCreaseType: Record<string, number> = {};
  let totalBadIncidents = 0;
  let neutralHingeIncidents = 0;
  let auxiliaryLikeHingeIncidents = 0;

  for (const vertex of state.badVertices) {
    for (const incident of adjacency[vertex] ?? []) {
      totalBadIncidents += 1;
      badIncidentEdges.add(incident.edge);
      increment(bySourceKind, incident.sourceKind);
      increment(byRole, incident.role);
      increment(byAssignment, incident.assignment);
      increment(byCreaseType, incident.creaseType);
      increment(bySourceKindRoleAssignment, `${incident.sourceKind}:${incident.role}:${incident.assignment}`);
      if (isNeutralHinge(incident)) neutralHingeIncidents += 1;
      if (isAuxiliaryLikeHinge(incident)) auxiliaryLikeHingeIncidents += 1;
    }
  }

  const summary = summarizePolicy(fold, state, adjacency, auxiliaryPolicy);
  const alternatePolicies = uniquePolicies(options.alternatePolicies ?? DEFAULT_ALTERNATE_POLICIES)
    .filter((policy) => policy !== auxiliaryPolicy)
    .map((policy) => {
      const alternate = normalizeBPStudioFold(input, {
        auxiliaryPolicy: policy,
        creator: "cp-synthetic-generator/bp-local-diagnostics",
      });
      return summarizePolicy(alternate, localState(alternate), buildAdjacency(alternate), policy);
    });

  return {
    summary,
    degreeHistogram: degreeHistogram(adjacency),
    foldedDegreeHistogram: degreeHistogram(adjacency, true),
    badVertexIncidentCounts: {
      total: totalBadIncidents,
      uniqueEdges: badIncidentEdges.size,
      bySourceKind,
      byRole,
      byAssignment,
      bySourceKindRoleAssignment,
      byCreaseType,
      neutralHingeIncidents,
      auxiliaryLikeHingeIncidents,
    },
    badVertices: {
      all: state.badVertices,
      kawasaki: state.kawasakiBadVertices,
      maekawa: state.maekawaBadVertices,
      samples: state.badVertices
        .slice(0, options.maxVertexSamples ?? 20)
        .map((vertex) => describeBadVertex(fold, adjacency, state, vertex)),
    },
    alternatePolicies,
  };
}

export function formatBPLocalDiagnosticsReport(report: BPLocalDiagnosticsReport, label = "BP local diagnostics"): string {
  const lines = [
    label,
    `policy: ${report.summary.auxiliaryPolicy}`,
    `graph: vertices=${report.summary.vertices} edges=${report.summary.edges}`,
    `assignments: ${formatCounts(report.summary.assignments)}`,
    `roles: ${formatCounts(report.summary.roles)}`,
    `local failures: bad=${report.summary.badVertices} kawasaki=${report.summary.kawasakiBadVertices} maekawa=${report.summary.maekawaBadVertices}`,
    `degree histogram: ${formatCounts(report.degreeHistogram)}`,
    `bad degree histogram: ${formatCounts(report.summary.badDegreeHistogram)}`,
    `bad folded-degree histogram: ${formatCounts(report.summary.badFoldedDegreeHistogram)}`,
    `bad incidents: total=${report.badVertexIncidentCounts.total} uniqueEdges=${report.badVertexIncidentCounts.uniqueEdges}`,
    `bad incidents by source kind: ${formatCounts(report.badVertexIncidentCounts.bySourceKind)}`,
    `bad incidents by role: ${formatCounts(report.badVertexIncidentCounts.byRole)}`,
    `bad incidents by assignment: ${formatCounts(report.badVertexIncidentCounts.byAssignment)}`,
    `bad incidents by source/role/assignment: ${formatCounts(report.badVertexIncidentCounts.bySourceKindRoleAssignment)}`,
    `U/F hinge dominance: neutral=${round(report.summary.neutralHingeIncidentRatio)} auxiliaryLike=${round(report.summary.auxiliaryLikeHingeIncidentRatio)} dominatedNeutral=${report.summary.dominatedByNeutralHinges} dominatedAuxiliaryLike=${report.summary.dominatedByAuxiliaryLikeHinges}`,
  ];

  if (report.alternatePolicies.length > 0) {
    lines.push("alternate auxiliary policies:");
    for (const policy of report.alternatePolicies) {
      lines.push(
        `  ${policy.auxiliaryPolicy}: bad=${policy.badVertices} kawasaki=${policy.kawasakiBadVertices} maekawa=${policy.maekawaBadVertices} assignments=${formatCounts(policy.assignments)} neutralHingeRatio=${round(policy.neutralHingeIncidentRatio)}`,
      );
    }
  }

  if (report.badVertices.samples.length > 0) {
    lines.push("bad vertex samples:");
    for (const sample of report.badVertices.samples) {
      lines.push(
        `  v${sample.vertex} degree=${sample.degree} folded=${sample.foldedDegree} kawasaki=${sample.kawasakiBad} maekawa=${sample.maekawaBad} kawasakiDeviation=${formatNullable(sample.kawasakiDeviation)} maekawaDelta=${formatNullable(sample.maekawaDelta)} assignments=${formatCounts(sample.assignments)} roles=${formatCounts(sample.roles)} sources=${formatCounts(sample.sourceKinds)}`,
      );
    }
  }

  return `${lines.join("\n")}\n`;
}

function summarizePolicy(
  fold: FOLDFormat,
  state: LocalState,
  adjacency: IncidentEdge[][],
  auxiliaryPolicy: BPStudioCanonicalAuxiliaryPolicy,
): BPLocalPolicySummary {
  let badIncidents = 0;
  let neutralHinges = 0;
  let auxiliaryLikeHinges = 0;
  for (const vertex of state.badVertices) {
    for (const incident of adjacency[vertex] ?? []) {
      badIncidents += 1;
      if (isNeutralHinge(incident)) neutralHinges += 1;
      if (isAuxiliaryLikeHinge(incident)) auxiliaryLikeHinges += 1;
    }
  }
  const neutralRatio = badIncidents === 0 ? 0 : neutralHinges / badIncidents;
  const auxiliaryRatio = badIncidents === 0 ? 0 : auxiliaryLikeHinges / badIncidents;

  return {
    auxiliaryPolicy,
    vertices: fold.vertices_coords.length,
    edges: fold.edges_vertices.length,
    assignments: assignmentCounts(fold),
    roles: roleCounts(fold),
    badVertices: state.badVertices.length,
    kawasakiBadVertices: state.kawasakiBadVertices.length,
    maekawaBadVertices: state.maekawaBadVertices.length,
    badDegreeHistogram: degreeHistogram(adjacency, false, state.badVertices),
    badFoldedDegreeHistogram: degreeHistogram(adjacency, true, state.badVertices),
    neutralHingeIncidentRatio: neutralRatio,
    auxiliaryLikeHingeIncidentRatio: auxiliaryRatio,
    dominatedByNeutralHinges: neutralRatio >= 0.5,
    dominatedByAuxiliaryLikeHinges: auxiliaryRatio >= 0.5,
  };
}

function describeBadVertex(
  fold: FOLDFormat,
  adjacency: IncidentEdge[][],
  state: LocalState,
  vertex: number,
): BPLocalBadVertexSample {
  const incidents = adjacency[vertex] ?? [];
  const assignments: Record<string, number> = {};
  const roles: Record<string, number> = {};
  const sourceKinds: Record<string, number> = {};
  const sourceKindRoleAssignments: Record<string, number> = {};

  for (const incident of incidents) {
    increment(assignments, incident.assignment);
    increment(roles, incident.role);
    increment(sourceKinds, incident.sourceKind);
    increment(sourceKindRoleAssignments, `${incident.sourceKind}:${incident.role}:${incident.assignment}`);
  }

  return {
    vertex,
    coord: fold.vertices_coords[vertex],
    degree: incidents.length,
    foldedDegree: incidents.filter((incident) => FOLDED_ASSIGNMENTS.has(incident.assignment)).length,
    assignments,
    roles,
    sourceKinds,
    sourceKindRoleAssignments,
    kawasakiBad: state.kawasakiBadVertices.includes(vertex),
    maekawaBad: state.maekawaBadVertices.includes(vertex),
    kawasakiDeviation: kawasakiDeviation(fold, vertex, incidents),
    maekawaDelta: maekawaDelta(incidents),
    incidentEdges: incidents.map((incident) => incident.edge),
  };
}

function localState(fold: FOLDFormat): LocalState {
  const graph = normalizeFold(fold, fold.file_creator) as FOLDFormat;
  ear.graph.populate(graph);
  const kawasakiBadVertices = sortedUnique(ear.singleVertex.validateKawasaki(graph) as number[]);
  const maekawaBadVertices = sortedUnique(ear.singleVertex.validateMaekawa(graph) as number[]);
  return {
    kawasakiBadVertices,
    maekawaBadVertices,
    badVertices: sortedUnique([...kawasakiBadVertices, ...maekawaBadVertices]),
  };
}

function buildAdjacency(fold: FOLDFormat): IncidentEdge[][] {
  const adjacency = Array.from({ length: fold.vertices_coords.length }, () => [] as IncidentEdge[]);
  for (const [edge, [a, b]] of fold.edges_vertices.entries()) {
    const assignment = fold.edges_assignment[edge] ?? "U";
    const role = fold.edges_bpRole?.[edge] ?? "unlabeled";
    const source = fold.edges_bpStudioSource?.[edge];
    const sourceKind = source?.kind ?? "unknown";
    const creaseType = source?.creaseType === undefined ? "unknown" : String(source.creaseType);
    adjacency[a]?.push({ edge, other: b, assignment, role, sourceKind, creaseType });
    adjacency[b]?.push({ edge, other: a, assignment, role, sourceKind, creaseType });
  }
  return adjacency;
}

function degreeHistogram(adjacency: IncidentEdge[][], foldedOnly = false, vertices?: readonly number[]): Record<string, number> {
  const counts: Record<string, number> = {};
  const selected = vertices ?? adjacency.map((_, vertex) => vertex);
  for (const vertex of selected) {
    const degree = (adjacency[vertex] ?? []).filter((incident) => !foldedOnly || FOLDED_ASSIGNMENTS.has(incident.assignment)).length;
    increment(counts, String(degree));
  }
  return counts;
}

function kawasakiDeviation(fold: FOLDFormat, vertex: number, incidents: readonly IncidentEdge[]): number | null {
  if (incidents.length < 4 || incidents.length % 2 !== 0) return null;
  const origin = fold.vertices_coords[vertex];
  const angles = incidents
    .map((incident) => {
      const other = fold.vertices_coords[incident.other];
      const angle = Math.atan2(other[1] - origin[1], other[0] - origin[0]);
      return angle < 0 ? angle + Math.PI * 2 : angle;
    })
    .sort((a, b) => a - b);
  const sectors = angles.map((angle, index) => {
    const next = angles[(index + 1) % angles.length] + (index === angles.length - 1 ? Math.PI * 2 : 0);
    return next - angle;
  });
  const alternating = sectors.reduce((sum, sector, index) => sum + (index % 2 === 0 ? sector : -sector), 0);
  return Math.abs(alternating);
}

function maekawaDelta(incidents: readonly IncidentEdge[]): number | null {
  const mountain = incidents.filter((incident) => incident.assignment === "M").length;
  const valley = incidents.filter((incident) => incident.assignment === "V").length;
  if (mountain + valley === 0) return null;
  return mountain - valley;
}

function isNeutralHinge(incident: IncidentEdge): boolean {
  return incident.role === "hinge" && NEUTRAL_ASSIGNMENTS.has(incident.assignment);
}

function isAuxiliaryLikeHinge(incident: IncidentEdge): boolean {
  return incident.role === "hinge" && /aux|contour/i.test(incident.sourceKind);
}

function increment(counts: Record<string, number>, key: string): void {
  counts[key] = (counts[key] ?? 0) + 1;
}

function sortedUnique(values: readonly number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function uniquePolicies(policies: readonly BPStudioCanonicalAuxiliaryPolicy[]): BPStudioCanonicalAuxiliaryPolicy[] {
  return [...new Set(policies)];
}

function formatCounts(counts: Record<string, number>): string {
  const keys = Object.keys(counts).sort();
  return keys.length ? keys.map((key) => `${key}=${counts[key]}`).join(" ") : "none";
}

function formatNullable(value: number | null): string {
  return value === null ? "n/a" : String(round(value));
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    fold: "",
    auxiliaryPolicy: "valley",
    json: false,
    maxVertexSamples: 20,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--fold") args.fold = argv[++i];
    else if (arg === "--auxiliary-policy") args.auxiliaryPolicy = parseAuxiliaryPolicy(argv[++i]);
    else if (arg === "--max-vertex-samples") args.maxVertexSamples = Number.parseInt(argv[++i], 10);
    else if (arg === "--json") args.json = true;
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run src/bp-local-diagnostics.ts --fold <path> [--auxiliary-policy valley|unassigned] [--max-vertex-samples <n>] [--json]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.fold) throw new Error("--fold is required");
  if (!Number.isInteger(args.maxVertexSamples) || args.maxVertexSamples < 0) {
    throw new Error("--max-vertex-samples must be a nonnegative integer");
  }
  return args;
}

function parseAuxiliaryPolicy(value: string | undefined): BPStudioCanonicalAuxiliaryPolicy {
  if (value === "valley" || value === "unassigned") return value;
  throw new Error("--auxiliary-policy must be valley or unassigned");
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await Bun.file(path).text()) as T;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const report = await runBPLocalDiagnosticsFile(args.fold, {
    auxiliaryPolicy: args.auxiliaryPolicy,
    maxVertexSamples: args.maxVertexSamples,
  });
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  process.stdout.write(formatBPLocalDiagnosticsReport(report, args.fold));
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
