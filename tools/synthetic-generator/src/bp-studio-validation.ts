import { assignmentCounts, roleCounts } from "./fold-utils.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import type { BPRole, EdgeAssignment, FOLDFormat, ValidationConfig, ValidationResult } from "./types.ts";
import { validateFold } from "./validate.ts";

type Point = [number, number];

export type BPStudioAuxiliaryPolicy = "valley" | "unassigned" | "preserve";
export type BPStudioCanonicalAuxiliaryPolicy = Exclude<BPStudioAuxiliaryPolicy, "preserve">;

export interface BPStudioAssignmentMapOptions {
  auxiliaryPolicy?: BPStudioAuxiliaryPolicy;
}

export interface BPStudioNormalizeOptions {
  creator?: string;
  auxiliaryPolicy?: BPStudioCanonicalAuxiliaryPolicy;
  normalizeCoordinates?: boolean;
  coordinateBounds?: CoordinateBounds;
  metadata?: Record<string, unknown>;
}

export interface BPStudioStrictValidationOptions extends BPStudioNormalizeOptions {
  validationConfig?: ValidationConfig;
}

export interface CoordinateBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface BPStudioLine {
  p1?: unknown;
  p2?: unknown;
  start?: unknown;
  end?: unknown;
  from?: unknown;
  to?: unknown;
  vertices?: unknown;
  assignment?: unknown;
  fold_assignment?: unknown;
  foldAngle?: unknown;
  type?: unknown;
  kind?: unknown;
  role?: unknown;
  bpRole?: unknown;
  bpStudioRole?: unknown;
  [key: string]: unknown;
}

export interface BPStudioLineExport {
  lines: readonly BPStudioLine[];
  metadata?: Record<string, unknown>;
  bp_metadata?: FOLDFormat["bp_metadata"];
  [key: string]: unknown;
}

export type BPStudioFoldLike = Omit<Partial<FOLDFormat>, "vertices_coords" | "edges_vertices" | "edges_assignment" | "edges_bpRole"> & {
  vertices_coords: readonly unknown[];
  edges_vertices: readonly unknown[];
  edges_assignment?: readonly unknown[];
  edges_bpRole?: readonly unknown[];
  edges_foldAngle?: readonly unknown[];
  [key: string]: unknown;
};

export type BPStudioExportInput = BPStudioFoldLike | BPStudioLineExport;

export interface BPStudioExportSummary {
  originalVertices: number;
  originalEdges: number;
  normalizedVertices: number;
  normalizedEdges: number;
  auxiliaryLines: number;
  degenerateSegments: number;
  duplicateSegments: number;
  splitIntersections: number;
  borderEdgesBefore: number;
  borderEdgesAfter: number;
  assignmentsBefore: Record<string, number>;
  assignmentsAfter: Record<string, number>;
  rolesBefore: Record<string, number>;
  rolesAfter: Record<string, number>;
  originalBounds: CoordinateBounds;
  normalizedBounds: CoordinateBounds;
  coordinateScale: number;
  auxiliaryPolicy: BPStudioCanonicalAuxiliaryPolicy;
}

export interface BPStudioStrictPreparation {
  fold: FOLDFormat;
  summary: BPStudioExportSummary;
  validation?: ValidationResult;
}

interface RawSegment {
  p1: Point;
  p2: Point;
  assignment: unknown;
  role: unknown;
  sourceIndex: number;
}

interface CanonicalSegment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
}

interface NormalizationBuild {
  rawSegments: RawSegment[];
  arrangedSegments: CanonicalSegment[];
  degenerateSegments: number;
  duplicateSegments: number;
  auxiliaryLines: number;
  originalVertices: number;
  originalBounds: CoordinateBounds;
  normalizedBounds: CoordinateBounds;
  coordinateScale: number;
  assignmentsBefore: Record<string, number>;
  rolesBefore: Record<string, number>;
  borderEdgesBefore: number;
  auxiliaryPolicy: BPStudioCanonicalAuxiliaryPolicy;
}

const DEFAULT_CREATOR = "cp-synthetic-generator/bp-studio-validation";
const EPSILON = 1e-9;
const METADATA_SKIP_KEYS = new Set([
  "file_spec",
  "file_creator",
  "file_classes",
  "frame_classes",
  "vertices_coords",
  "edges_vertices",
  "edges_assignment",
  "edges_foldAngle",
  "edges_bpRole",
  "faces_vertices",
  "faces_edges",
  "lines",
]);

export function normalizeBPStudioFold(input: BPStudioExportInput, options: BPStudioNormalizeOptions = {}): FOLDFormat {
  const creator = options.creator ?? DEFAULT_CREATOR;
  const build = buildNormalization(input, options);
  const normalized = arrangeSegments(build.arrangedSegments, creator, readBPMetadata(input));
  Object.assign(normalized, copyMetadata(input), options.metadata ?? {});

  const summary = summarizeFromBuild(build, normalized);
  const existingBPStudioMetadata = (input as Record<string, unknown>).bp_studio_metadata;
  normalized.bp_studio_metadata = {
    ...(isRecord(existingBPStudioMetadata) ? existingBPStudioMetadata : {}),
    normalization: summary,
  };
  return normalized;
}

export function summarizeBPStudioExport(input: BPStudioExportInput, options: BPStudioNormalizeOptions = {}): BPStudioExportSummary {
  const build = buildNormalization(input, options);
  const normalized = arrangeSegments(build.arrangedSegments, options.creator ?? DEFAULT_CREATOR, readBPMetadata(input));
  return summarizeFromBuild(build, normalized);
}

export async function prepareBPStudioFoldForStrictValidation(
  input: BPStudioExportInput,
  options: BPStudioStrictValidationOptions = {},
): Promise<BPStudioStrictPreparation> {
  const fold = normalizeBPStudioFold(input, { ...options, auxiliaryPolicy: options.auxiliaryPolicy ?? "valley" });
  const summary = (fold.bp_studio_metadata as { normalization?: BPStudioExportSummary } | undefined)?.normalization
    ?? summarizeBPStudioExport(input, options);
  if (!options.validationConfig) return { fold, summary };
  return {
    fold,
    summary,
    validation: await validateFold(fold, options.validationConfig),
  };
}

export function mapBPStudioAssignments(
  assignments: readonly unknown[],
  options: BPStudioAssignmentMapOptions = {},
): EdgeAssignment[] {
  return assignments.map((assignment) => mapBPStudioAssignment(assignment, options));
}

export function mapBPStudioAssignment(assignment: unknown, options: BPStudioAssignmentMapOptions = {}): EdgeAssignment {
  if (typeof assignment === "number") {
    if (assignment < -EPSILON) return "M";
    if (assignment > EPSILON) return "V";
    return "U";
  }

  const token = normalizedToken(assignment);
  if (!token) return "U";
  if (token === "M" || token === "MOUNTAIN") return "M";
  if (token === "V" || token === "VALLEY") return "V";
  if (token === "B" || token === "BORDER" || token === "BOUNDARY") return "B";
  if (token === "U" || token === "UNASSIGNED" || token === "UNKNOWN") return "U";
  if (token === "C" || token === "CUT") return "U";
  if (token === "F" || token === "FLAT" || token === "AUX" || token === "AUXILIARY") {
    return mapAuxiliary(options.auxiliaryPolicy ?? "valley");
  }
  if (token.includes("MOUNTAIN") || token.includes("RIDGE")) return "M";
  if (token.includes("VALLEY") || token.includes("HINGE") || token.includes("CONTOUR") || token.includes("AXIS") || token.includes("STRETCH")) {
    return "V";
  }
  if (token.includes("BORDER") || token.includes("BOUNDARY")) return "B";
  if (token.includes("AUX") || token.includes("FLAT")) return mapAuxiliary(options.auxiliaryPolicy ?? "valley");
  return "U";
}

function buildNormalization(input: BPStudioExportInput, options: BPStudioNormalizeOptions): NormalizationBuild {
  const auxiliaryPolicy = options.auxiliaryPolicy ?? "valley";
  const rawSegments = extractRawSegments(input);
  const originalBounds = options.coordinateBounds ?? deriveBounds(rawSegments);
  const coordinateScale = Math.max(originalBounds.maxX - originalBounds.minX, originalBounds.maxY - originalBounds.minY, 1);
  const normalizeCoordinates = options.normalizeCoordinates ?? true;
  const assignmentsBefore = countRawValues(rawSegments.map((segment) => segment.assignment));
  const rolesBefore = countRawValues(rawSegments.map((segment) => segment.role));
  const borderEdgesBefore = rawSegments.filter((segment) => {
    const assignment = mapBPStudioAssignment(segment.assignment, { auxiliaryPolicy });
    return assignment === "B" || mapBPStudioRole(segment.role, segment.assignment, assignment) === "border";
  }).length;

  let auxiliaryLines = 0;
  let degenerateSegments = 0;
  let duplicateSegments = 0;
  const seenGeometry = new Set<string>();
  const arrangedSegments: CanonicalSegment[] = [];

  for (const raw of rawSegments) {
    if (isAuxiliaryAssignment(raw.assignment)) auxiliaryLines += 1;
    const assignment = mapBPStudioAssignment(raw.assignment, { auxiliaryPolicy });
    const role = mapBPStudioRole(raw.role, raw.assignment, assignment);
    const p1 = normalizeCoordinates ? normalizePoint(raw.p1, originalBounds, coordinateScale) : roundPoint(raw.p1);
    const p2 = normalizeCoordinates ? normalizePoint(raw.p2, originalBounds, coordinateScale) : roundPoint(raw.p2);
    if (distance(p1, p2) < EPSILON) {
      degenerateSegments += 1;
      continue;
    }

    const key = segmentGeometryKey(p1, p2);
    if (seenGeometry.has(key)) duplicateSegments += 1;
    seenGeometry.add(key);
    arrangedSegments.push({ p1, p2, assignment, role });
  }

  return {
    rawSegments,
    arrangedSegments,
    degenerateSegments,
    duplicateSegments,
    auxiliaryLines,
    originalVertices: countOriginalVertices(input, rawSegments),
    originalBounds,
    normalizedBounds: deriveBounds(arrangedSegments),
    coordinateScale,
    assignmentsBefore,
    rolesBefore,
    borderEdgesBefore,
    auxiliaryPolicy,
  };
}

function summarizeFromBuild(build: NormalizationBuild, normalized: FOLDFormat): BPStudioExportSummary {
  const uniqueEndpoints = new Set<string>();
  for (const segment of build.arrangedSegments) {
    uniqueEndpoints.add(pointKey(segment.p1));
    uniqueEndpoints.add(pointKey(segment.p2));
  }
  return {
    originalVertices: build.originalVertices,
    originalEdges: build.rawSegments.length,
    normalizedVertices: normalized.vertices_coords.length,
    normalizedEdges: normalized.edges_vertices.length,
    auxiliaryLines: build.auxiliaryLines,
    degenerateSegments: build.degenerateSegments,
    duplicateSegments: build.duplicateSegments,
    splitIntersections: Math.max(0, normalized.vertices_coords.length - uniqueEndpoints.size),
    borderEdgesBefore: build.borderEdgesBefore,
    borderEdgesAfter: normalized.edges_assignment.filter((assignment) => assignment === "B").length,
    assignmentsBefore: build.assignmentsBefore,
    assignmentsAfter: assignmentCounts(normalized),
    rolesBefore: build.rolesBefore,
    rolesAfter: roleCounts(normalized),
    originalBounds: build.originalBounds,
    normalizedBounds: build.normalizedBounds,
    coordinateScale: build.coordinateScale,
    auxiliaryPolicy: build.auxiliaryPolicy,
  };
}

function extractRawSegments(input: BPStudioExportInput): RawSegment[] {
  if (isLineExport(input)) {
    return input.lines.map((line, sourceIndex) => {
      const [p1, p2] = extractLinePoints(line, sourceIndex);
      return {
        p1,
        p2,
        assignment: line.assignment ?? line.fold_assignment ?? line.foldAngle ?? line.type ?? line.kind,
        role: line.role ?? line.bpRole ?? line.bpStudioRole ?? line.type ?? line.kind,
        sourceIndex,
      };
    });
  }

  const vertices = input.vertices_coords.map((point, index) => coercePoint(point, `vertices_coords[${index}]`));
  return input.edges_vertices.map((edge, sourceIndex) => {
    const [a, b] = coerceEdge(edge, sourceIndex);
    if (!vertices[a] || !vertices[b]) throw new Error(`edges_vertices[${sourceIndex}] references a missing vertex`);
    return {
      p1: vertices[a],
      p2: vertices[b],
      assignment: input.edges_assignment?.[sourceIndex] ?? input.edges_foldAngle?.[sourceIndex] ?? "U",
      role: input.edges_bpRole?.[sourceIndex],
      sourceIndex,
    };
  });
}

function extractLinePoints(line: BPStudioLine, sourceIndex: number): [Point, Point] {
  if (Array.isArray(line.vertices) && line.vertices.length >= 2) {
    return [
      coercePoint(line.vertices[0], `lines[${sourceIndex}].vertices[0]`),
      coercePoint(line.vertices[1], `lines[${sourceIndex}].vertices[1]`),
    ];
  }
  if (line.p1 !== undefined && line.p2 !== undefined) {
    return [coercePoint(line.p1, `lines[${sourceIndex}].p1`), coercePoint(line.p2, `lines[${sourceIndex}].p2`)];
  }
  if (line.start !== undefined && line.end !== undefined) {
    return [coercePoint(line.start, `lines[${sourceIndex}].start`), coercePoint(line.end, `lines[${sourceIndex}].end`)];
  }
  if (line.from !== undefined && line.to !== undefined) {
    return [coercePoint(line.from, `lines[${sourceIndex}].from`), coercePoint(line.to, `lines[${sourceIndex}].to`)];
  }
  if (Array.isArray(line) && line.length >= 4) {
    return [
      coercePoint([line[0], line[1]], `lines[${sourceIndex}][0:2]`),
      coercePoint([line[2], line[3]], `lines[${sourceIndex}][2:4]`),
    ];
  }
  throw new Error(`lines[${sourceIndex}] does not contain recognizable endpoints`);
}

function coercePoint(value: unknown, label: string): Point {
  if (!Array.isArray(value) || value.length < 2) throw new Error(`${label} must be a [x, y] point`);
  const x = Number(value[0]);
  const y = Number(value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`${label} must contain finite coordinates`);
  return [x, y];
}

function coerceEdge(value: unknown, index: number): [number, number] {
  if (!Array.isArray(value) || value.length < 2) throw new Error(`edges_vertices[${index}] must be a vertex pair`);
  const a = Number(value[0]);
  const b = Number(value[1]);
  if (!Number.isInteger(a) || !Number.isInteger(b)) throw new Error(`edges_vertices[${index}] must contain integer vertex indices`);
  return [a, b];
}

function mapAuxiliary(policy: BPStudioAuxiliaryPolicy): EdgeAssignment {
  if (policy === "preserve") return "F";
  if (policy === "unassigned") return "U";
  return "V";
}

function mapBPStudioRole(role: unknown, assignment: unknown, canonicalAssignment: EdgeAssignment): BPRole {
  const token = normalizedToken(role);
  const assignmentToken = normalizedToken(assignment);
  if (token.includes("BORDER") || token.includes("BOUNDARY") || canonicalAssignment === "B") return "border";
  if (token.includes("RIDGE") || token.includes("MOUNTAIN") || canonicalAssignment === "M") return "ridge";
  if (token.includes("AXIS")) return "axis";
  if (token.includes("STRETCH")) return "stretch";
  if (token.includes("HINGE") || token.includes("CONTOUR") || assignmentToken === "F" || assignmentToken.includes("AUX")) return "hinge";
  return "hinge";
}

function deriveBounds(segments: readonly { p1: Point; p2: Point; assignment?: unknown; role?: unknown }[]): CoordinateBounds {
  const borderSegments = segments.filter((segment) => mapBPStudioAssignment(segment.assignment ?? "U") === "B" || mapBPStudioRole(segment.role, segment.assignment, "U") === "border");
  const source = borderSegments.length >= 4 ? borderSegments : segments;
  const points = source.flatMap((segment) => [segment.p1, segment.p2]);
  if (points.length === 0) {
    return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  }
  return {
    minX: Math.min(...points.map((point) => point[0])),
    minY: Math.min(...points.map((point) => point[1])),
    maxX: Math.max(...points.map((point) => point[0])),
    maxY: Math.max(...points.map((point) => point[1])),
  };
}

function normalizePoint(point: Point, bounds: CoordinateBounds, scale: number): Point {
  return roundPoint([(point[0] - bounds.minX) / scale, (point[1] - bounds.minY) / scale]);
}

function roundPoint(point: Point): Point {
  return [round(point[0]), round(point[1])];
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function isAuxiliaryAssignment(value: unknown): boolean {
  const token = normalizedToken(value);
  return token === "F" || token === "FLAT" || token === "AUX" || token === "AUXILIARY" || token.includes("AUX");
}

function countRawValues(values: readonly unknown[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const key = normalizedToken(value) || "UNKNOWN";
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function countOriginalVertices(input: BPStudioExportInput, rawSegments: readonly RawSegment[]): number {
  if (!isLineExport(input)) return input.vertices_coords.length;
  const vertices = new Set<string>();
  for (const segment of rawSegments) {
    vertices.add(pointKey(segment.p1));
    vertices.add(pointKey(segment.p2));
  }
  return vertices.size;
}

function segmentGeometryKey(a: Point, b: Point): string {
  const ak = pointKey(a);
  const bk = pointKey(b);
  return ak < bk ? `${ak}:${bk}` : `${bk}:${ak}`;
}

function pointKey(point: Point): string {
  return `${point[0].toFixed(9)},${point[1].toFixed(9)}`;
}

function normalizedToken(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value).trim().replace(/[\s-]+/g, "_").toUpperCase();
}

function readBPMetadata(input: BPStudioExportInput): FOLDFormat["bp_metadata"] {
  return isRecord((input as Record<string, unknown>).bp_metadata)
    ? ((input as Record<string, unknown>).bp_metadata as FOLDFormat["bp_metadata"])
    : undefined;
}

function copyMetadata(input: BPStudioExportInput): Record<string, unknown> {
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    if (METADATA_SKIP_KEYS.has(key)) continue;
    metadata[key] = cloneMetadata(value);
  }
  if (isLineExport(input) && input.metadata) {
    metadata.bp_studio_export_metadata = cloneMetadata(input.metadata);
  }
  return metadata;
}

function cloneMetadata(value: unknown): unknown {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value)) as unknown;
}

function isLineExport(input: BPStudioExportInput): input is BPStudioLineExport {
  return Array.isArray((input as BPStudioLineExport).lines);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
