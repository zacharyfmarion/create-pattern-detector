import { normalizeFold } from "./fold-utils.ts";
import type { BPRole, BPStudioEdgeSource, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

interface Segment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
  source?: BPStudioEdgeSource;
}

interface ArrangementEdge {
  a: number;
  b: number;
  assignment: EdgeAssignment;
  role: BPRole;
  source?: BPStudioEdgeSource;
}

const EPSILON = 1e-9;

export function arrangeFoldGraph(fold: FOLDFormat, creator = fold.file_creator): FOLDFormat {
  const segments = fold.edges_vertices.map(([a, b], index) => ({
    p1: fold.vertices_coords[a],
    p2: fold.vertices_coords[b],
    assignment: fold.edges_assignment[index] ?? "U",
    role: fold.edges_bpRole?.[index] ?? roleFromAssignment(fold.edges_assignment[index] ?? "U"),
    source: fold.edges_bpStudioSource?.[index],
  }));
  return arrangeSegments(segments, creator, fold.bp_metadata);
}

export function arrangeSegments(
  segments: Segment[],
  creator = "cp-synthetic-generator/arrangement",
  bpMetadata?: FOLDFormat["bp_metadata"],
): FOLDFormat {
  const inputSegments = segments.map((segment) => ({
    ...segment,
    p1: snapPointToMetadataGrid(segment.p1, bpMetadata),
    p2: snapPointToMetadataGrid(segment.p2, bpMetadata),
  }));
  const splitPoints = inputSegments.map((segment) => [segment.p1, segment.p2]);

  for (let i = 0; i < inputSegments.length; i++) {
    for (let j = i + 1; j < inputSegments.length; j++) {
      for (const point of intersectionPoints(inputSegments[i], inputSegments[j])) {
        if (onSegment(point, inputSegments[i])) splitPoints[i].push(snapPointToMetadataGrid(point, bpMetadata));
        if (onSegment(point, inputSegments[j])) splitPoints[j].push(snapPointToMetadataGrid(point, bpMetadata));
      }
    }
  }

  const vertices: Point[] = [];
  const vertexKeys = new Map<string, number>();
  const edges = new Map<string, ArrangementEdge>();

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

  for (let i = 0; i < inputSegments.length; i++) {
    const segment = inputSegments[i];
    const points = uniquePoints(splitPoints[i]).sort((a, b) => parameterAlong(segment, a) - parameterAlong(segment, b));
    for (let j = 0; j < points.length - 1; j++) {
      if (distance(points[j], points[j + 1]) < EPSILON) continue;
      const a = vertexIndex(points[j]);
      const b = vertexIndex(points[j + 1]);
      if (a === b) continue;
      const key = edgeKey(a, b);
      const candidate: ArrangementEdge = { a, b, assignment: segment.assignment, role: segment.role, source: segment.source };
      const existing = edges.get(key);
      edges.set(key, existing ? mergeEdge(existing, candidate) : candidate);
    }
  }

  const arrangedEdges = [...edges.values()].sort((a, b) => Math.min(a.a, a.b) - Math.min(b.a, b.b) || Math.max(a.a, a.b) - Math.max(b.a, b.b));
  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: creator,
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: arrangedEdges.map((edge) => [edge.a, edge.b]),
      edges_assignment: arrangedEdges.map((edge) => edge.assignment),
      edges_bpRole: arrangedEdges.map((edge) => edge.role),
      edges_bpStudioSource: arrangedEdges.map((edge) => edge.source ?? { kind: "unknown-arranged-edge" }),
      bp_metadata: bpMetadata,
    },
    creator,
  );
}

function intersectionPoints(a: Segment, b: Segment): Point[] {
  const result: Point[] = [];
  for (const point of [a.p1, a.p2]) {
    if (onSegment(point, b)) result.push(point);
  }
  for (const point of [b.p1, b.p2]) {
    if (onSegment(point, a)) result.push(point);
  }

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

function mergeEdge(a: ArrangementEdge, b: ArrangementEdge): ArrangementEdge {
  const role = rolePriority(a.role) >= rolePriority(b.role) ? a.role : b.role;
  const assignment = assignmentPriority(a.assignment) >= assignmentPriority(b.assignment) ? a.assignment : b.assignment;
  const source = sourcePriority(a.source) >= sourcePriority(b.source) ? a.source : b.source;
  return { ...a, role, assignment, source };
}

function roleFromAssignment(assignment: EdgeAssignment): BPRole {
  if (assignment === "B") return "border";
  if (assignment === "M") return "ridge";
  return "hinge";
}

function rolePriority(role: BPRole): number {
  return { border: 5, ridge: 4, axis: 3, stretch: 2, hinge: 1 }[role];
}

function assignmentPriority(assignment: EdgeAssignment): number {
  return { B: 5, M: 4, V: 3, F: 2, U: 1, C: 0 }[assignment];
}

function sourcePriority(source: BPStudioEdgeSource | undefined): number {
  if (!source) return 0;
  return (source.mandatory ? 100 : 0) + sourceKindPriority(source.kind);
}

function sourceKindPriority(kind: string): number {
  if (kind.includes("border")) return 50;
  if (kind.includes("ridge")) return 40;
  if (kind.includes("axis")) return 30;
  if (kind.includes("contour")) return 20;
  return 10;
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

function onSegment(point: Point, segment: Segment): boolean {
  const minX = Math.min(segment.p1[0], segment.p2[0]) - EPSILON;
  const maxX = Math.max(segment.p1[0], segment.p2[0]) + EPSILON;
  const minY = Math.min(segment.p1[1], segment.p2[1]) - EPSILON;
  const maxY = Math.max(segment.p1[1], segment.p2[1]) + EPSILON;
  if (point[0] < minX || point[0] > maxX || point[1] < minY || point[1] > maxY) return false;
  return Math.abs(cross([segment.p2[0] - segment.p1[0], segment.p2[1] - segment.p1[1]], [point[0] - segment.p1[0], point[1] - segment.p1[1]])) < EPSILON;
}

function cross(a: Point, b: Point): number {
  return a[0] * b[1] - a[1] * b[0];
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
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

function snapPointToMetadataGrid(point: Point, bpMetadata?: FOLDFormat["bp_metadata"]): Point {
  const gridSize = bpMetadata?.gridSize;
  if (!gridSize || gridSize <= 0) return roundPoint(point);
  const denominator = gridSize * 2;
  return [
    round(Math.round(point[0] * denominator) / denominator),
    round(Math.round(point[1] * denominator) / denominator),
  ];
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
