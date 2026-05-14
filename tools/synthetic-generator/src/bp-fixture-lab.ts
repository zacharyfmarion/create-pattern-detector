#!/usr/bin/env bun
import ear from "rabbit-ear";
import { normalizeBPStudioFold, type BPStudioExportInput, type BPStudioExportSummary } from "./bp-studio-validation.ts";
import { assignmentCounts, normalizeFold, roleCounts } from "./fold-utils.ts";
import type { BPRole, EdgeAssignment, FOLDFormat, ValidationConfig, ValidationResult } from "./types.ts";
import { validateFold } from "./validate.ts";

type Point = [number, number];

export interface BPFixtureLabOptions {
  validationConfig?: ValidationConfig;
}

export interface BPFixtureIncidentEdge {
  edge: number;
  vertices: [number, number];
  otherVertex: number;
  assignment: EdgeAssignment;
  role?: BPRole;
  angleRadians: number;
  angleDegrees: number;
  length: number;
}

export interface BPFixtureVertexDiagnostic {
  vertex: number;
  coord: Point;
  degree: number;
  foldedDegree: number;
  assignments: EdgeAssignment[];
  assignmentCounts: Record<string, number>;
  incidentEdges: BPFixtureIncidentEdge[];
  sectorsRadians: number[];
  sectorsDegrees: number[];
  foldedSectorsRadians: number[];
  foldedSectorsDegrees: number[];
  kawasaki: {
    failed: boolean;
    alternatingSumsRadians: [number, number];
    alternatingSumsDegrees: [number, number];
    differenceRadians: number;
    differenceDegrees: number;
  };
  maekawa: {
    failed: boolean;
    mountain: number;
    valley: number;
    signedDifference: number;
    absoluteDifference: number;
  };
}

export interface BPFixtureLabSummary {
  normalizedVertices: number;
  normalizedEdges: number;
  assignments: Record<string, number>;
  roles: Record<string, number>;
  kawasakiBad: number;
  maekawaBad: number;
  badVertices: number;
  localFlatFoldable: boolean;
  validationValid: boolean;
  validationFailed: string[];
  normalization?: BPStudioExportSummary;
}

export interface BPFixtureLabReport {
  normalized: FOLDFormat;
  validation: ValidationResult;
  summary: BPFixtureLabSummary;
  kawasakiBadVertices: number[];
  maekawaBadVertices: number[];
  badVertices: BPFixtureVertexDiagnostic[];
}

interface CliArgs {
  fold: string;
  json: boolean;
}

const DEFAULT_VALIDATION_CONFIG: ValidationConfig = {
  strictGlobal: false,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-9,
  maxVertices: 100_000,
  maxEdges: 200_000,
};

const FULL_TURN_RADIANS = Math.PI * 2;
const FOLDED_ASSIGNMENTS = new Set<EdgeAssignment>(["M", "V"]);

export async function diagnoseBPFixture(
  input: BPStudioExportInput,
  options: BPFixtureLabOptions = {},
): Promise<BPFixtureLabReport> {
  const normalized = normalizeBPStudioFold(input, { auxiliaryPolicy: "valley", creator: "cp-synthetic-generator/bp-fixture-lab" });
  const graph = normalizeFold(normalized, normalized.file_creator) as FOLDFormat & {
    vertices_edges?: number[][];
    vertices_vertices?: number[][];
  };
  ear.graph.populate(graph);

  const kawasakiBadVertices = sortedUnique(ear.singleVertex.validateKawasaki(graph) as number[]);
  const maekawaBadVertices = sortedUnique(ear.singleVertex.validateMaekawa(graph) as number[]);
  const badVertexSet = new Set([...kawasakiBadVertices, ...maekawaBadVertices]);
  const validation = await validateFold(normalized, options.validationConfig ?? DEFAULT_VALIDATION_CONFIG);

  const badVertices = [...badVertexSet]
    .sort((a, b) => a - b)
    .map((vertex) => describeBadVertex(graph, vertex, {
      kawasaki: kawasakiBadVertices.includes(vertex),
      maekawa: maekawaBadVertices.includes(vertex),
    }));
  const normalization = readNormalizationSummary(normalized);

  return {
    normalized,
    validation,
    summary: {
      normalizedVertices: normalized.vertices_coords.length,
      normalizedEdges: normalized.edges_vertices.length,
      assignments: assignmentCounts(normalized),
      roles: roleCounts(normalized),
      kawasakiBad: kawasakiBadVertices.length,
      maekawaBad: maekawaBadVertices.length,
      badVertices: badVertices.length,
      localFlatFoldable: badVertices.length === 0,
      validationValid: validation.valid,
      validationFailed: validation.failed,
      ...(normalization ? { normalization } : {}),
    },
    kawasakiBadVertices,
    maekawaBadVertices,
    badVertices,
  };
}

export async function diagnoseFoldFile(path: string, options: BPFixtureLabOptions = {}): Promise<BPFixtureLabReport> {
  return diagnoseBPFixture(await readJson<BPStudioExportInput>(path), options);
}

export function formatBPFixtureLabReport(report: BPFixtureLabReport, label = "BP Studio fixture"): string {
  const lines = [
    `${label}`,
    `normalized: vertices=${report.summary.normalizedVertices} edges=${report.summary.normalizedEdges}`,
    `assignments: ${formatCounts(report.summary.assignments)}`,
    `local: flatFoldable=${report.summary.localFlatFoldable} badVertices=${report.summary.badVertices} kawasaki=${report.summary.kawasakiBad} maekawa=${report.summary.maekawaBad}`,
    `validation: valid=${report.validation.valid} failed=${report.validation.failed.length ? report.validation.failed.join(", ") : "none"}`,
  ];

  if (report.badVertices.length > 0) {
    lines.push("bad vertices:");
    for (const vertex of report.badVertices) {
      const failures = [
        vertex.kawasaki.failed ? "Kawasaki" : undefined,
        vertex.maekawa.failed ? "Maekawa" : undefined,
      ].filter(Boolean).join("+");
      lines.push(
        `  v${vertex.vertex} ${failures} degree=${vertex.degree} foldedDegree=${vertex.foldedDegree} coord=${formatPoint(vertex.coord)} assignments=${vertex.assignments.join(",")}`,
      );
      if (vertex.kawasaki.failed) {
        lines.push(
          `    sectors(deg)=${vertex.foldedSectorsDegrees.map(formatNumber).join(",")} alternating=${vertex.kawasaki.alternatingSumsDegrees.map(formatNumber).join("/")} diff=${formatNumber(vertex.kawasaki.differenceDegrees)}`,
        );
      }
      if (vertex.maekawa.failed) {
        lines.push(
          `    maekawa M=${vertex.maekawa.mountain} V=${vertex.maekawa.valley} signed=${vertex.maekawa.signedDifference}`,
        );
      }
    }
  }

  if (report.validation.errors.length > 0) {
    lines.push("validation errors:");
    for (const error of report.validation.errors) lines.push(`  ${error}`);
  }
  return `${lines.join("\n")}\n`;
}

function describeBadVertex(
  graph: FOLDFormat & { vertices_edges?: number[][] },
  vertex: number,
  failures: { kawasaki: boolean; maekawa: boolean },
): BPFixtureVertexDiagnostic {
  const coord = graph.vertices_coords[vertex];
  const incidentEdges = describeIncidentEdges(graph, vertex);
  const assignments = incidentEdges.map((edge) => edge.assignment);
  const foldedEdges = incidentEdges.filter((edge) => FOLDED_ASSIGNMENTS.has(edge.assignment));
  const sectorsRadians = sectorsFromAngles(incidentEdges.map((edge) => edge.angleRadians));
  const foldedSectorsRadians = foldedEdges.length > 1
    ? sectorsFromAngles(foldedEdges.map((edge) => edge.angleRadians))
    : [0, 0];
  const alternatingSumsRadians = alternatingSums(foldedSectorsRadians);
  const mountain = foldedEdges.filter((edge) => edge.assignment === "M").length;
  const valley = foldedEdges.filter((edge) => edge.assignment === "V").length;
  const signedDifference = valley - mountain;

  return {
    vertex,
    coord,
    degree: incidentEdges.length,
    foldedDegree: foldedEdges.length,
    assignments,
    assignmentCounts: countValues(assignments),
    incidentEdges,
    sectorsRadians,
    sectorsDegrees: sectorsRadians.map(radiansToDegrees),
    foldedSectorsRadians,
    foldedSectorsDegrees: foldedSectorsRadians.map(radiansToDegrees),
    kawasaki: {
      failed: failures.kawasaki,
      alternatingSumsRadians,
      alternatingSumsDegrees: alternatingSumsRadians.map(radiansToDegrees) as [number, number],
      differenceRadians: Math.abs(alternatingSumsRadians[0] - alternatingSumsRadians[1]),
      differenceDegrees: radiansToDegrees(Math.abs(alternatingSumsRadians[0] - alternatingSumsRadians[1])),
    },
    maekawa: {
      failed: failures.maekawa,
      mountain,
      valley,
      signedDifference,
      absoluteDifference: Math.abs(signedDifference),
    },
  };
}

function describeIncidentEdges(graph: FOLDFormat & { vertices_edges?: number[][] }, vertex: number): BPFixtureIncidentEdge[] {
  const coord = graph.vertices_coords[vertex];
  const edgeIndices = graph.vertices_edges?.[vertex] ?? incidentEdgesFromEdgesVertices(graph, vertex);
  return edgeIndices
    .map((edge) => {
      const vertices = graph.edges_vertices[edge];
      const otherVertex = vertices[0] === vertex ? vertices[1] : vertices[0];
      const otherCoord = graph.vertices_coords[otherVertex];
      const angleRadians = normalizeAngle(Math.atan2(otherCoord[1] - coord[1], otherCoord[0] - coord[0]));
      return {
        edge,
        vertices,
        otherVertex,
        assignment: graph.edges_assignment[edge] ?? "U",
        role: graph.edges_bpRole?.[edge],
        angleRadians,
        angleDegrees: radiansToDegrees(angleRadians),
        length: distance(coord, otherCoord),
      };
    })
    .sort((a, b) => a.angleRadians - b.angleRadians || a.edge - b.edge);
}

function incidentEdgesFromEdgesVertices(graph: FOLDFormat, vertex: number): number[] {
  const incident: number[] = [];
  for (const [edge, vertices] of graph.edges_vertices.entries()) {
    if (vertices[0] === vertex || vertices[1] === vertex) incident.push(edge);
  }
  return incident;
}

function sectorsFromAngles(angles: readonly number[]): number[] {
  if (angles.length === 0) return [];
  if (angles.length === 1) return [FULL_TURN_RADIANS];
  return [...angles]
    .sort((a, b) => a - b)
    .map((angle, index, sorted) => {
      const next = sorted[(index + 1) % sorted.length] + (index === sorted.length - 1 ? FULL_TURN_RADIANS : 0);
      return next - angle;
    });
}

function alternatingSums(values: readonly number[]): [number, number] {
  return values.reduce<[number, number]>(
    (sums, value, index) => {
      sums[index % 2] += value;
      return sums;
    },
    [0, 0],
  );
}

function countValues(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function sortedUnique(values: readonly number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function readNormalizationSummary(fold: FOLDFormat): BPStudioExportSummary | undefined {
  const metadata = fold.bp_studio_metadata;
  if (!isRecord(metadata) || !isRecord(metadata.normalization)) return undefined;
  return metadata.normalization as unknown as BPStudioExportSummary;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeAngle(angle: number): number {
  return angle < 0 ? angle + FULL_TURN_RADIANS : angle;
}

function radiansToDegrees(radians: number): number {
  return round((radians * 180) / Math.PI);
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function formatCounts(counts: Record<string, number>): string {
  return Object.keys(counts).sort().map((key) => `${key}=${counts[key]}`).join(" ");
}

function formatPoint(point: Point): string {
  return `[${point.map(formatNumber).join(",")}]`;
}

function formatNumber(value: number): string {
  return round(value).toString();
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { fold: "", json: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--fold") args.fold = argv[++i];
    else if (arg === "--json") args.json = true;
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run src/bp-fixture-lab.ts --fold <path> [--json]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.fold) throw new Error("--fold is required");
  return args;
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await Bun.file(path).text()) as T;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const report = await diagnoseFoldFile(args.fold);
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  process.stdout.write(formatBPFixtureLabReport(report, args.fold));
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
