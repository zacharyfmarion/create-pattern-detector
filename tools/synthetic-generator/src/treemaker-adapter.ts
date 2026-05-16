import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { normalizeFold } from "./fold-utils.ts";
import { generateTreeMakerSpec, treeMetadataFromSpec, validateTreeMakerSpec } from "./treemaker-sampler.ts";
import type { TreeMakerAdapterSpec } from "./treemaker-sampler.ts";
import type {
  EdgeAssignment,
  FOLDFormat,
  GenerationConfig,
  TreeMakerCreaseKind,
  TreeMakerMetadata,
} from "./types.ts";

type Point = [number, number];

export interface TreeMakerExternalCrease {
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  p1?: [number, number];
  p2?: [number, number];
  fold?: string;
  assignment?: string;
  kind?: string;
}

export interface TreeMakerExternalOutput {
  schemaVersion?: string;
  ok?: boolean;
  toolVersion?: string;
  optimization?: {
    success?: boolean;
    error?: string;
    warnings?: string[];
  };
  foldedForm?: {
    success?: boolean;
    cpStatus?: string;
  };
  stats?: {
    vertices?: number;
    creases?: number;
    facets?: number;
  };
  creases?: TreeMakerExternalCrease[];
  fold?: Partial<FOLDFormat> & {
    edges_treemakerKind?: string[];
  };
  warnings?: string[];
}

export interface TreeMakerAdapterOptions {
  command?: string[];
  keepTemp?: boolean;
}

interface Segment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  kind: TreeMakerCreaseKind;
}

interface ArrangedEdge {
  a: number;
  b: number;
  assignment: EdgeAssignment;
  kind: TreeMakerCreaseKind;
}

const ADAPTER_VERSION = "treemaker-adapter/v0.1.0";
const EPSILON = 1e-9;
const SHEET_EPSILON = 1e-6;

export function generateTreeMakerFold(config: GenerationConfig): FOLDFormat {
  const spec = generateTreeMakerSpec(config);
  const errors = validateTreeMakerSpec(spec);
  if (errors.length) throw new Error(`invalid TreeMaker spec: ${errors.join("; ")}`);
  const output = runTreeMakerAdapter(spec);
  return treeMakerOutputToFold(spec, output);
}

export function runTreeMakerAdapter(
  spec: TreeMakerAdapterSpec,
  options: TreeMakerAdapterOptions = {},
): TreeMakerExternalOutput & { _commandLabel?: string } {
  const command = options.command ?? commandFromEnvironment();
  if (!command.length) {
    throw new Error(
      "TREEMAKER_CLI is required for treemaker-tree generation. Provide an external TreeMaker-compatible CLI; GPL TreeMaker source is not vendored in this repo.",
    );
  }

  const tempRoot = createTempDir();
  const specPath = join(tempRoot, "spec.json");
  const outPath = join(tempRoot, "out.json");
  writeFileSync(specPath, JSON.stringify(spec, null, 2) + "\n");
  const timeoutMs = treeMakerTimeoutMs();

  const proc = Bun.spawnSync({
    cmd: [...command, "--spec", specPath, "--out", outPath],
    stdout: "pipe",
    stderr: "pipe",
    timeout: timeoutMs,
  });
  const stdout = new TextDecoder().decode(proc.stdout);
  const stderr = new TextDecoder().decode(proc.stderr);
  if (proc.signalCode) {
    throw new Error(`TreeMaker CLI terminated by ${proc.signalCode} after ${timeoutMs}ms: ${stderr || stdout}`);
  }
  if (proc.exitCode !== 0) {
    throw new Error(`TreeMaker CLI failed (${proc.exitCode}): ${stderr || stdout}`);
  }

  const outputText = existsSync(outPath) ? readFileSync(outPath, "utf8") : stdout;
  const parsed = JSON.parse(outputText) as TreeMakerExternalOutput;
  if (!options.keepTemp && process.env.TREEMAKER_KEEP_TMP !== "1") {
    rmSync(tempRoot, { recursive: true, force: true });
  }
  return { ...parsed, _commandLabel: command.join(" ") };
}

export function treeMakerOutputToFold(spec: TreeMakerAdapterSpec, output: TreeMakerExternalOutput & { _commandLabel?: string }): FOLDFormat {
  if (output.ok === false) {
    const cpStatus = output.foldedForm?.cpStatus ? ` cpStatus=${output.foldedForm.cpStatus}` : "";
    const optimizationError = output.optimization?.error ? ` optimization=${output.optimization.error}` : "";
    throw new Error(`TreeMaker external output reported ok=false.${cpStatus}${optimizationError}`);
  }
  if (output.optimization?.success === false) {
    throw new Error(`TreeMaker optimization did not succeed${output.optimization.error ? `: ${output.optimization.error}` : ""}`);
  }
  const segments = segmentsFromOutput(output);
  if (segments.length < 4) throw new Error("TreeMaker output did not contain enough crease segments");

  const fold = arrangeTreeMakerSegments(segments, "cp-synthetic-generator/treemaker-tree");
  fold.file_title = `${spec.id} TreeMaker tree CP`;
  fold.file_description = "Synthetic non-box-pleated CP generated from a TreeMaker-style external tree-base solver.";
  fold.tree_metadata = treeMetadataFromSpec(spec);
  fold.treemaker_metadata = treeMakerMetadata(output, segments, output._commandLabel);
  fold.density_metadata = {
    densityBucket: "treemaker-tree",
    gridSize: 0,
    targetEdgeRange: [Math.max(4, Math.floor(spec.targetCreases * 0.4)), Math.max(50, spec.targetCreases * 20)],
    subfamily: "treemaker-tree",
    symmetry: spec.symmetryClass,
    generatorSteps: [
      "sample-tree-spec",
      "external-treemaker-cli",
      "preserve-sheet-coordinates",
      "synthesize-square-border",
      "arrange-split-dedupe",
    ],
    moleculeCounts: {
      terminal: spec.nodes.filter((node) => node.kind === "terminal").length,
      hub: spec.nodes.filter((node) => node.kind === "hub").length,
      sourceCrease: segments.length,
    },
  };
  fold.label_policy = {
    labelSource: "treemaker-external",
    geometrySource: "treemaker-external",
    assignmentSource: "treemaker-external",
    trainingEligible: true,
    notes: [
      "TreeMaker-compatible external solver output is canonical for this non-box-pleated family.",
      "Flat/unfolded hinge lines are preserved as F/U-style supervision with fold angle 0.",
      "The full square paper border is synthesized; TreeMaker useful-polygon border segments away from the sheet edge are preserved as flat construction lines.",
    ],
  };
  return fold;
}

function commandFromEnvironment(): string[] {
  const command = process.env.TREEMAKER_CLI?.trim();
  if (!command) return [];
  const args = process.env.TREEMAKER_CLI_ARGS?.trim();
  return [command, ...(args ? args.split(/\s+/u) : [])];
}

function treeMakerTimeoutMs(): number {
  const raw = process.env.TREEMAKER_TIMEOUT_MS?.trim();
  if (!raw) return 15_000;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`TREEMAKER_TIMEOUT_MS must be a positive number of milliseconds, got ${raw}`);
  }
  return value;
}

function createTempDir(): string {
  return mkdtempSync(join(tmpdir(), "cp-treemaker-"));
}

function segmentsFromOutput(output: TreeMakerExternalOutput): Segment[] {
  if (output.fold?.vertices_coords && output.fold.edges_vertices) {
    const vertices = output.fold.vertices_coords as Point[];
    return output.fold.edges_vertices.map(([a, b], index): Segment => ({
      p1: vertices[a],
      p2: vertices[b],
      assignment: mapTreeMakerAssignment(output.fold?.edges_assignment?.[index]),
      kind: mapTreeMakerKind(output.fold?.edges_treemakerKind?.[index]),
    }));
  }
  return (output.creases ?? []).map((crease): Segment => ({
    p1: pointFromCrease(crease, "p1"),
    p2: pointFromCrease(crease, "p2"),
    assignment: mapTreeMakerAssignment(crease.assignment ?? crease.fold),
    kind: mapTreeMakerKind(crease.kind),
  }));
}

function arrangeTreeMakerSegments(segments: Segment[], creator: string): FOLDFormat {
  const normalizedSegments = prepareSheetSegments(segments).filter((segment) => distance(segment.p1, segment.p2) > EPSILON);
  const splitPoints = normalizedSegments.map((segment) => [segment.p1, segment.p2]);
  for (let i = 0; i < normalizedSegments.length; i++) {
    for (let j = i + 1; j < normalizedSegments.length; j++) {
      for (const point of intersectionPoints(normalizedSegments[i], normalizedSegments[j])) {
        if (onSegment(point, normalizedSegments[i])) splitPoints[i].push(point);
        if (onSegment(point, normalizedSegments[j])) splitPoints[j].push(point);
      }
    }
  }

  const vertices: Point[] = [];
  const vertexKeys = new Map<string, number>();
  const edges = new Map<string, ArrangedEdge>();
  const vertexIndex = (point: Point): number => {
    const rounded = roundPoint(point);
    const key = pointKey(rounded);
    const existing = vertexKeys.get(key);
    if (existing !== undefined) return existing;
    const index = vertices.length;
    vertices.push(rounded);
    vertexKeys.set(key, index);
    return index;
  };

  for (let i = 0; i < normalizedSegments.length; i++) {
    const segment = normalizedSegments[i];
    const points = uniquePoints(splitPoints[i]).sort((a, b) => parameterAlong(segment, a) - parameterAlong(segment, b));
    for (let j = 0; j < points.length - 1; j++) {
      if (distance(points[j], points[j + 1]) < EPSILON) continue;
      const a = vertexIndex(points[j]);
      const b = vertexIndex(points[j + 1]);
      if (a === b) continue;
      const key = edgeKey(a, b);
      const candidate = { a, b, assignment: segment.assignment, kind: segment.kind };
      const existing = edges.get(key);
      edges.set(key, existing ? mergeTreeMakerEdge(existing, candidate) : candidate);
    }
  }

  const arrangedEdges = [...edges.values()].sort((a, b) => Math.min(a.a, a.b) - Math.min(b.a, b.b) || Math.max(a.a, a.b) - Math.max(b.a, b.b));
  const fold = normalizeFold(
    {
      file_spec: 1.1,
      file_creator: creator,
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: arrangedEdges.map((edge) => [edge.a, edge.b]),
      edges_assignment: arrangedEdges.map((edge) => edge.assignment),
      edges_treemakerKind: arrangedEdges.map((edge) => edge.kind),
    },
    creator,
  );
  fold.edges_treemakerKind = arrangedEdges.map((edge) => edge.kind);
  return fold;
}

function treeMakerMetadata(
  output: TreeMakerExternalOutput,
  segments: Segment[],
  commandLabel?: string,
): TreeMakerMetadata {
  return {
    adapterVersion: ADAPTER_VERSION,
    toolVersion: output.toolVersion,
    externalCommand: commandLabel,
    optimizationSuccess: output.optimization?.success !== false,
    foldedFormSuccess: output.foldedForm?.success,
    warnings: [...(output.warnings ?? []), ...(output.optimization?.warnings ?? [])],
    creaseKindCounts: countBy(segments.map((segment) => segment.kind)),
    sourceCreaseCount: segments.length,
  };
}

function prepareSheetSegments(segments: Segment[]): Segment[] {
  const normalized = normalizeTreeMakerSheetCoordinates(segments).map((segment) => {
    if (segment.assignment !== "B") return segment;
    if (isSheetBoundarySegment(segment)) return { ...segment, kind: "BORDER" as const };
    return {
      ...segment,
      assignment: "F" as const,
      kind: segment.kind === "UNKNOWN" ? ("CONSTRUCTION" as const) : segment.kind,
    };
  });
  return [...normalized, ...squareBorderSegments()];
}

function normalizeTreeMakerSheetCoordinates(segments: Segment[]): Segment[] {
  return segments.map((segment) => ({
    ...segment,
    p1: snapPointToSheet(segment.p1),
    p2: snapPointToSheet(segment.p2),
  }));
}

function squareBorderSegments(): Segment[] {
  return [
    { p1: [0, 0], p2: [1, 0], assignment: "B", kind: "BORDER" },
    { p1: [1, 0], p2: [1, 1], assignment: "B", kind: "BORDER" },
    { p1: [1, 1], p2: [0, 1], assignment: "B", kind: "BORDER" },
    { p1: [0, 1], p2: [0, 0], assignment: "B", kind: "BORDER" },
  ];
}

function snapPointToSheet(point: Point): Point {
  return [snapCoordToSheet(point[0]), snapCoordToSheet(point[1])];
}

function snapCoordToSheet(value: number): number {
  if (Math.abs(value) < SHEET_EPSILON) return 0;
  if (Math.abs(value - 1) < SHEET_EPSILON) return 1;
  return round(value);
}

function isSheetBoundarySegment(segment: Segment): boolean {
  return (
    (isSheetMin(segment.p1[0]) && isSheetMin(segment.p2[0])) ||
    (isSheetMax(segment.p1[0]) && isSheetMax(segment.p2[0])) ||
    (isSheetMin(segment.p1[1]) && isSheetMin(segment.p2[1])) ||
    (isSheetMax(segment.p1[1]) && isSheetMax(segment.p2[1]))
  );
}

function isSheetMin(value: number): boolean {
  return Math.abs(value) < SHEET_EPSILON;
}

function isSheetMax(value: number): boolean {
  return Math.abs(value - 1) < SHEET_EPSILON;
}

function pointFromCrease(crease: TreeMakerExternalCrease, key: "p1" | "p2"): Point {
  const point = crease[key];
  if (Array.isArray(point) && point.length >= 2) return [Number(point[0]), Number(point[1])];
  if (key === "p1") return [Number(crease.x1), Number(crease.y1)];
  return [Number(crease.x2), Number(crease.y2)];
}

function mapTreeMakerAssignment(value: unknown): EdgeAssignment {
  const token = String(value ?? "U").trim().toUpperCase();
  if (token === "M" || token === "MOUNTAIN") return "M";
  if (token === "V" || token === "VALLEY") return "V";
  if (token === "B" || token === "BORDER") return "B";
  if (token === "F" || token === "FLAT" || token === "UNFOLDED") return "F";
  return "U";
}

function mapTreeMakerKind(value: unknown): TreeMakerCreaseKind {
  const token = String(value ?? "UNKNOWN").trim().toUpperCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (
    token === "BORDER" ||
    token === "AXIAL" ||
    token === "RIDGE" ||
    token === "GUSSET" ||
    token === "FOLDED_HINGE" ||
    token === "UNFOLDED_HINGE" ||
    token === "PSEUDOHINGE" ||
    token === "CONSTRUCTION"
  ) {
    return token;
  }
  return "UNKNOWN";
}

function mergeTreeMakerEdge(a: ArrangedEdge, b: ArrangedEdge): ArrangedEdge {
  return {
    ...a,
    assignment: assignmentPriority(a.assignment) >= assignmentPriority(b.assignment) ? a.assignment : b.assignment,
    kind: kindPriority(a.kind) >= kindPriority(b.kind) ? a.kind : b.kind,
  };
}

function assignmentPriority(assignment: EdgeAssignment): number {
  return { B: 5, M: 4, V: 3, F: 2, U: 1, C: 0 }[assignment];
}

function kindPriority(kind: TreeMakerCreaseKind): number {
  if (kind === "BORDER") return 50;
  if (kind === "RIDGE" || kind === "AXIAL" || kind === "GUSSET") return 40;
  if (kind === "FOLDED_HINGE" || kind === "PSEUDOHINGE") return 30;
  if (kind === "UNFOLDED_HINGE") return 20;
  if (kind === "CONSTRUCTION") return 10;
  return 0;
}

function intersectionPoints(a: Segment, b: Segment): Point[] {
  const result: Point[] = [];
  for (const point of [a.p1, a.p2]) if (onSegment(point, b)) result.push(point);
  for (const point of [b.p1, b.p2]) if (onSegment(point, a)) result.push(point);
  const r: Point = [a.p2[0] - a.p1[0], a.p2[1] - a.p1[1]];
  const s: Point = [b.p2[0] - b.p1[0], b.p2[1] - b.p1[1]];
  const denom = cross(r, s);
  if (Math.abs(denom) < EPSILON) return uniquePoints(result);
  const qp: Point = [b.p1[0] - a.p1[0], b.p1[1] - a.p1[1]];
  const t = cross(qp, s) / denom;
  const u = cross(qp, r) / denom;
  if (t > -EPSILON && t < 1 + EPSILON && u > -EPSILON && u < 1 + EPSILON) {
    result.push([a.p1[0] + t * r[0], a.p1[1] + t * r[1]]);
  }
  return uniquePoints(result);
}

function onSegment(point: Point, segment: Segment): boolean {
  const minX = Math.min(segment.p1[0], segment.p2[0]) - EPSILON;
  const maxX = Math.max(segment.p1[0], segment.p2[0]) + EPSILON;
  const minY = Math.min(segment.p1[1], segment.p2[1]) - EPSILON;
  const maxY = Math.max(segment.p1[1], segment.p2[1]) + EPSILON;
  if (point[0] < minX || point[0] > maxX || point[1] < minY || point[1] > maxY) return false;
  return Math.abs(cross([segment.p2[0] - segment.p1[0], segment.p2[1] - segment.p1[1]], [point[0] - segment.p1[0], point[1] - segment.p1[1]])) < EPSILON;
}

function uniquePoints(points: Point[]): Point[] {
  const seen = new Set<string>();
  const result: Point[] = [];
  for (const point of points) {
    const rounded = roundPoint(point);
    const key = pointKey(rounded);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(rounded);
  }
  return result;
}

function parameterAlong(segment: Segment, point: Point): number {
  const dx = segment.p2[0] - segment.p1[0];
  const dy = segment.p2[1] - segment.p1[1];
  const denom = dx * dx + dy * dy;
  if (denom < EPSILON) return 0;
  return ((point[0] - segment.p1[0]) * dx + (point[1] - segment.p1[1]) * dy) / denom;
}

function edgeKey(a: number, b: number): string {
  return `${Math.min(a, b)}:${Math.max(a, b)}`;
}

function pointKey(point: Point): string {
  return `${point[0].toFixed(9)},${point[1].toFixed(9)}`;
}

function roundPoint(point: Point): Point {
  return [round(point[0]), round(point[1])];
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function cross(a: Point, b: Point): number {
  return a[0] * b[1] - a[1] * b[0];
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function countBy(values: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}
