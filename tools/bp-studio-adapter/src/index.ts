#!/usr/bin/env bun

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { Bridge } from "client/plugins/optimizer/bridge";
import { TreeController } from "core/controller/treeController";
import { Tree } from "core/design/context/tree";
import { heightTask } from "core/design/tasks/height";
import { Processor } from "core/service/processor";
import { State, fullReset } from "core/service/state";
import optimizer from "lib/optimizer/debug/optimizer.js";
import { GridType } from "shared/json";
import { CreaseType } from "shared/types/cp";

import type { OptimizerRequest } from "client/plugins/optimizer/types";
import type { CPLine } from "shared/types/cp";
import type { JEdge, JFlap, NodeId } from "shared/json";
import type { ILine, Path, Polygon } from "shared/types/geometry";
import type {
  AdapterMetadata,
  Assignment,
  BPRole,
  BPStudioEdgeSource,
  BPStudioExportMode,
  BpStudioAdapterSpec,
  EdgeSpec,
  FlapSpec,
  FoldDocument,
  GenerationResult,
  NodeLayoutSpec,
  SheetSpec,
  StretchMetadata
} from "./types";

interface TaggedCPLine extends CPLine {
  role: BPRole;
  source: BPStudioEdgeSource;
}

const ADAPTER_NAME = "@cp-detector/bp-studio-adapter";
const ADAPTER_VERSION = "0.1.0";
const BP_STUDIO_VERSION = "0.7.14";
const BP_STUDIO_SOURCE = "../../third_party/bp-studio";

const ASSIGNMENT_BY_TYPE: Record<CreaseType, Assignment> = {
  [CreaseType.Border]: "B",
  [CreaseType.Mountain]: "M",
  [CreaseType.Valley]: "V",
  [CreaseType.Auxiliary]: "F",
  [CreaseType.None]: "U"
};

const FOLD_ANGLE_BY_ASSIGNMENT: Record<Assignment, number> = {
  B: 0,
  M: -180,
  V: 180,
  F: 0,
  U: 0
};
const GEOMETRY_EPSILON = 1e-8;

export async function generate(specInput: unknown): Promise<GenerationResult> {
  const spec = normalizeSpec(specInput);
  const { edges } = getTreeParts(spec);
  let { flaps } = getTreeParts(spec);
  let sheet = spec.sheet;
  const inputLayout = cloneLayout(sheet, edges, flaps);
  const useAuxiliary = spec.useAuxiliary ?? false;
  const completeRepositories = spec.completeRepositories ?? true;
  const exportMode = spec.exportMode ?? "outer";
  const optimizeLayout = spec.optimizeLayout ?? false;
  const optimizerLayout = spec.optimizerLayout ?? "view";
  const optimizerSeed = spec.optimizerSeed ?? null;

  validateLeafFlaps(edges, flaps);
  createBpStudioTree(edges, flaps, { runTasks: !optimizeLayout || optimizerLayout === "random" });
  if(optimizeLayout) {
    const optimized = await optimizeTreeLayout(spec, flaps);
    sheet = optimized.sheet;
    flaps = optimized.flaps;
    spec.sheet = sheet;
    validateLeafFlaps(edges, flaps);
    createBpStudioTree(edges, flaps, { runTasks: true });
  }
  if(completeRepositories) completeAllStretches();

  const border = sheetBorder(sheet);
  const cpLines = getCPByMode(border, useAuxiliary, exportMode);
  const fold = toFold(cpLines, spec);
  const metadata = collectMetadata(spec, fold, cpLines, edges, flaps, inputLayout, useAuxiliary, completeRepositories, exportMode, optimizeLayout, optimizerLayout, optimizerSeed);
  return { fold, metadata };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if(args.help) {
    printHelp();
    return;
  }
  if(!args.spec || !args.out) {
    printHelp();
    throw new Error("Missing required --spec or --out argument.");
  }

  const specPath = resolve(args.spec);
  const spec = JSON.parse(await readFile(specPath, "utf8")) as unknown;
  const { fold, metadata } = await generate(spec);

  await writeJson(args.out, fold);
  if(args.metadata) await writeJson(args.metadata, metadata);
}

function createBpStudioTree(edges: EdgeSpec[], flaps: FlapSpec[], options: { runTasks?: boolean } = {}): void {
  fullReset();
  const tree = new Tree(
    edges.map(e => ({
      n1: e.n1 as NodeId,
      n2: e.n2 as NodeId,
      length: e.length
    })) as JEdge[],
    flaps.map(f => ({
      id: f.id as NodeId,
      x: f.x,
      y: f.y,
      width: f.width ?? 0,
      height: f.height ?? 0
    })) as JFlap[]
  );
  State.m.$tree = tree;
  if(options.runTasks ?? true) Processor.$run(heightTask);
}

function completeAllStretches(): void {
  for(const stretch of State.$stretches.values()) {
    stretch.$repo.$complete();
  }
}

async function optimizeTreeLayout(
  spec: BpStudioAdapterSpec,
  flaps: FlapSpec[],
): Promise<{ sheet: SheetSpec; flaps: FlapSpec[] }> {
  const flapMap = new Map(flaps.map(flap => [flap.id, flap]));
  const hierarchies = TreeController.getHierarchy(spec.optimizerLayout === "random", spec.optimizerUseDimension ?? true);
  const orderedLeafIds = hierarchies[hierarchies.length - 1].leaves;
  const orderedFlaps = orderedLeafIds.map(id => {
    const flap = flapMap.get(id);
    return flap ?? { id, x: spec.sheet.width / 2, y: spec.sheet.height / 2, width: 0, height: 0 };
  });
  const request: OptimizerRequest = {
    command: "start",
    useBH: false,
    layout: spec.optimizerLayout ?? "view",
    random: 5,
    problem: {
      type: GridType.rectangular,
      flaps: orderedFlaps.map(flap => ({
        id: flap.id as NodeId,
        width: flap.width ?? 0,
        height: flap.height ?? 0,
      })),
      hierarchies,
    },
    vec: (spec.optimizerLayout ?? "view") === "view"
      ? orderedFlaps.map(flap => ({ x: flap.x / spec.sheet.width, y: flap.y / spec.sheet.height }))
      : null,
  };
  const instance = await optimizer({
    print: () => undefined,
    checkInterrupt: () => false,
  });
  const bridge = new Bridge(instance);
  const result = await bridge.solve(request, spec.optimizerSeed);
  const sizeById = new Map(flaps.map(flap => [flap.id, { width: flap.width ?? 0, height: flap.height ?? 0 }]));
  const optimizedFlaps = result.flaps.map(flap => ({
    id: flap.id,
    x: flap.x,
    y: flap.y,
    width: sizeById.get(flap.id)?.width ?? 0,
    height: sizeById.get(flap.id)?.height ?? 0,
  }));
  return {
    sheet: sheetForOptimizedFlaps(result.width, result.height, optimizedFlaps),
    flaps: optimizedFlaps,
  };
}

function sheetForOptimizedFlaps(width: number, height: number, flaps: FlapSpec[]): SheetSpec {
  let maxX = width;
  let maxY = height;
  for(const flap of flaps) {
    maxX = Math.max(maxX, flap.x + (flap.width ?? 0));
    maxY = Math.max(maxY, flap.y + (flap.height ?? 0));
  }
  const squareSize = Math.max(Math.ceil(maxX), Math.ceil(maxY));
  return { width: squareSize, height: squareSize };
}

function getCPByMode(borders: Path, useAuxiliary: boolean, exportMode: BPStudioExportMode): TaggedCPLine[] {
  if(exportMode === "expanded") return getExpandedCP(borders, useAuxiliary);
  if(exportMode === "final") return getFinalRenderedCP(borders, useAuxiliary);
  return getOuterCP(borders, useAuxiliary);
}

function getOuterCP(borders: Path, useAuxiliary: boolean): TaggedCPLine[] {
  const hingeType = useAuxiliary ? CreaseType.Auxiliary : CreaseType.Valley;
  const lines: TaggedCPLine[] = [];
  addPolygon(lines, [borders], CreaseType.Border, "border", { kind: "sheet-border", mandatory: true });

  for(const node of State.m.$tree.$nodes) {
    if(!node || !node.$parent) continue;
    addPolygon(lines, node.$graphics.$contours.map(c => c.outer), hingeType, "hinge", {
      kind: "node-contour",
      ownerId: node.id,
      mandatory: true,
    });
    addLines(lines, node.$graphics.$ridges, CreaseType.Mountain, "ridge", {
      kind: "node-ridge",
      ownerId: node.id,
      mandatory: true,
    });
  }

  for(const [stretchId, stretch] of State.$stretches.entries()) {
    const pattern = stretch.$repo.$pattern;
    if(!pattern) continue;
    for(const [deviceIndex, device] of pattern.$devices.entries()) {
      addLines(lines, device.$drawRidges, CreaseType.Mountain, "ridge", {
        kind: "device-draw-ridge",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
      addLines(lines, device.$axisParallels, CreaseType.Valley, "axis", {
        kind: "device-axis-parallel",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
    }
  }

  return clipTagged(lines);
}

function getFinalRenderedCP(borders: Path, useAuxiliary: boolean): TaggedCPLine[] {
  const hingeType = useAuxiliary ? CreaseType.Auxiliary : CreaseType.Valley;
  const lines: TaggedCPLine[] = [];
  addPolygon(lines, [borders], CreaseType.Border, "border", { kind: "sheet-border", mandatory: true });

  for(const node of State.m.$tree.$nodes) {
    if(!node || !node.$parent) continue;
    for(const contour of node.$graphics.$contours) {
      addPolygon(lines, [contour.outer], hingeType, "hinge", {
        kind: "node-final-contour-outer",
        ownerId: node.id,
        mandatory: true,
      });
      if(contour.inner) {
        addPolygon(lines, contour.inner, hingeType, "hinge", {
          kind: "node-final-contour-inner",
          ownerId: node.id,
          mandatory: false,
        });
      }
    }
    addLines(lines, node.$graphics.$ridges, CreaseType.Mountain, "ridge", {
      kind: "node-ridge",
      ownerId: node.id,
      mandatory: true,
    });
  }

  for(const [stretchId, stretch] of State.$stretches.entries()) {
    const pattern = stretch.$repo.$pattern;
    if(!pattern) continue;
    for(const [deviceIndex, device] of pattern.$devices.entries()) {
      addLines(lines, device.$drawRidges, CreaseType.Mountain, "ridge", {
        kind: "device-draw-ridge",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
      addLines(lines, device.$axisParallels, CreaseType.Valley, "axis", {
        kind: "device-axis-parallel",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
    }
  }

  return clipTagged(lines);
}

function getExpandedCP(borders: Path, useAuxiliary: boolean): TaggedCPLine[] {
  const hingeType = useAuxiliary ? CreaseType.Auxiliary : CreaseType.Valley;
  const lines: TaggedCPLine[] = [];
  addPolygon(lines, [borders], CreaseType.Border, "border", { kind: "sheet-border", mandatory: true });

  for(const node of State.m.$tree.$nodes) {
    if(!node || !node.$parent) continue;
    for(const contour of node.$graphics.$contours) {
      addPolygon(lines, [contour.outer], hingeType, "hinge", {
        kind: "node-contour-outer",
        ownerId: node.id,
        mandatory: true,
      });
      if(contour.inner) {
        addPolygon(lines, contour.inner, hingeType, "hinge", {
          kind: "node-contour-inner",
          ownerId: node.id,
          mandatory: false,
        });
      }
    }
    addPolygon(lines, node.$graphics.$patternContours, hingeType, "hinge", {
      kind: "node-pattern-contour",
      ownerId: node.id,
      mandatory: false,
    });
    addPolygon(lines, node.$graphics.$traceContours, hingeType, "hinge", {
      kind: "node-trace-contour",
      ownerId: node.id,
      mandatory: false,
    });
    addPolygon(lines, node.$graphics.$roughContours, hingeType, "hinge", {
      kind: "node-rough-contour",
      ownerId: node.id,
      mandatory: false,
    });
    addLines(lines, node.$graphics.$ridges, CreaseType.Mountain, "ridge", {
      kind: "node-ridge",
      ownerId: node.id,
      mandatory: true,
    });
  }

  for(const [stretchId, stretch] of State.$stretches.entries()) {
    const pattern = stretch.$repo.$pattern;
    if(!pattern) continue;
    for(const [deviceIndex, device] of pattern.$devices.entries()) {
      addLines(lines, device.$drawRidges, CreaseType.Mountain, "ridge", {
        kind: "device-draw-ridge",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
      addLines(lines, device.$traceRidges.map(ridge => ridge.$toILine()), CreaseType.Mountain, "ridge", {
        kind: "device-trace-ridge",
        stretchId,
        deviceIndex,
        mandatory: false,
      });
      addLines(lines, device.$axisParallels, CreaseType.Valley, "axis", {
        kind: "device-axis-parallel",
        stretchId,
        deviceIndex,
        mandatory: true,
      });
      for(const contour of device.$contour) {
        addPolygon(lines, [contour.outer], hingeType, "hinge", {
          kind: "device-contour-outer",
          stretchId,
          deviceIndex,
          mandatory: false,
        });
        if(contour.inner) {
          addPolygon(lines, contour.inner, hingeType, "hinge", {
            kind: "device-contour-inner",
            stretchId,
            deviceIndex,
            mandatory: false,
          });
        }
      }
    }
  }

  return clipTagged(lines);
}

function clipTagged(lines: TaggedCPLine[]): TaggedCPLine[] {
  const sanitized = sanitizeTaggedLines(lines);
  const bounds = clipBoundsFromBorder(sanitized);
  const clipped = sanitized.flatMap((line, clippedSegmentIndex) => {
    const segment = line.type === CreaseType.Border
      ? [line.p1, line.p2] as [IPoint, IPoint]
      : clipSegmentToBounds(line.p1, line.p2, bounds);
    if(!segment) return [];
    const [p1, p2] = segment;
    return [{
      ...line,
      p1,
      p2,
      source: {
        ...line.source,
        creaseType: line.type,
        clippedSegmentIndex,
      },
    }];
  });
  return sanitizeTaggedLines(clipped);
}

function sanitizeTaggedLines(lines: TaggedCPLine[]): TaggedCPLine[] {
  const byGeometry = new Map<string, TaggedCPLine>();
  for(const line of lines) {
    if(!isFinitePoint(line.p1) || !isFinitePoint(line.p2)) continue;
    if(distance(line.p1, line.p2) <= GEOMETRY_EPSILON) continue;

    const key = lineGeometryKey(line);
    const existing = byGeometry.get(key);
    if(!existing || sourcePriority(line) > sourcePriority(existing)) {
      byGeometry.set(key, {
        ...line,
        p1: roundPoint(line.p1),
        p2: roundPoint(line.p2),
      });
    }
  }
  return [...byGeometry.values()];
}

function sourcePriority(line: TaggedCPLine): number {
  const rolePriority: Record<BPRole, number> = { border: 50, ridge: 40, axis: 30, stretch: 20, hinge: 10 };
  return rolePriority[line.role] + (line.source.mandatory ? 1 : 0);
}

interface ClipBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function clipBoundsFromBorder(lines: TaggedCPLine[]): ClipBounds {
  const borderPoints = lines
    .filter(line => line.type === CreaseType.Border)
    .flatMap(line => [line.p1, line.p2]);
  if(borderPoints.length === 0) {
    const points = lines.flatMap(line => [line.p1, line.p2]);
    return boundsForPoints(points);
  }
  return boundsForPoints(borderPoints);
}

function boundsForPoints(points: IPoint[]): ClipBounds {
  return {
    minX: Math.min(...points.map(point => point.x)),
    minY: Math.min(...points.map(point => point.y)),
    maxX: Math.max(...points.map(point => point.x)),
    maxY: Math.max(...points.map(point => point.y)),
  };
}

function clipSegmentToBounds(p1: IPoint, p2: IPoint, bounds: ClipBounds): [IPoint, IPoint] | null {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  let t0 = 0;
  let t1 = 1;
  for(const [p, q] of [
    [-dx, p1.x - bounds.minX],
    [dx, bounds.maxX - p1.x],
    [-dy, p1.y - bounds.minY],
    [dy, bounds.maxY - p1.y],
  ] as const) {
    if(Math.abs(p) <= GEOMETRY_EPSILON) {
      if(q < -GEOMETRY_EPSILON) return null;
      continue;
    }
    const r = q / p;
    if(p < 0) {
      if(r > t1) return null;
      if(r > t0) t0 = r;
    } else {
      if(r < t0) return null;
      if(r < t1) t1 = r;
    }
  }
  const a = { x: p1.x + t0 * dx, y: p1.y + t0 * dy };
  const b = { x: p1.x + t1 * dx, y: p1.y + t1 * dy };
  return distance(a, b) <= GEOMETRY_EPSILON ? null : [roundPoint(a), roundPoint(b)];
}

function toFold(lines: TaggedCPLine[], spec: BpStudioAdapterSpec): FoldDocument {
  const vertices = new VertexSet();
  const edgeData = lines.map((line, lineIndex) => {
    const assignment = ASSIGNMENT_BY_TYPE[line.type];
    return {
      assignment,
      role: line.role,
      source: { ...line.source, creaseType: line.type, lineIndex },
      p1: vertices.add(line.p1),
      p2: vertices.add(line.p2)
    };
  });

  return {
    file_spec: 1.1,
    file_creator: `${ADAPTER_NAME} using Box Pleating Studio ${BP_STUDIO_VERSION}`,
    file_title: spec.title ?? "BP Studio adapter output",
    file_description: spec.description ?? "",
    vertices_coords: vertices.list(),
    edges_vertices: edgeData.map(edge => [edge.p1, edge.p2]),
    edges_assignment: edgeData.map(edge => edge.assignment),
    edges_foldAngle: edgeData.map(edge => FOLD_ANGLE_BY_ASSIGNMENT[edge.assignment]),
    edges_bpRole: edgeData.map(edge => edge.role),
    edges_bpStudioSource: edgeData.map(edge => edge.source)
  };
}

function collectMetadata(
  spec: BpStudioAdapterSpec,
  fold: FoldDocument,
  lines: CPLine[],
  edges: EdgeSpec[],
  flaps: FlapSpec[],
  inputLayout: { sheet: SheetSpec; edges: EdgeSpec[]; flaps: FlapSpec[] },
  useAuxiliary: boolean,
  completeRepositories: boolean,
  exportMode: BPStudioExportMode,
  optimizeLayout: boolean,
  optimizerLayout: "view" | "random",
  optimizerSeed: number | null
): AdapterMetadata {
  const nodes = collectNodeLayout();
  return {
    adapter: {
      name: ADAPTER_NAME,
      version: ADAPTER_VERSION
    },
    bpStudio: {
      version: BP_STUDIO_VERSION,
      source: BP_STUDIO_SOURCE
    },
    spec: {
      title: spec.title ?? "",
      sheet: spec.sheet,
      useAuxiliary,
      completeRepositories,
      exportMode,
      optimizeLayout,
      optimizerLayout,
      optimizerSeed,
      edgeCount: edges.length,
      flapCount: flaps.length
    },
    layout: {
      optimized: optimizeLayout,
      optimizerLayout,
      sheet: spec.sheet,
      edges,
      flaps,
      nodes
    },
    inputLayout,
    optimizedLayout: {
      optimized: optimizeLayout,
      optimizerLayout,
      sheet: spec.sheet,
      edges,
      flaps,
      nodes
    },
    cp: {
      lineCount: lines.length,
      vertexCount: fold.vertices_coords.length,
      edgeCount: fold.edges_vertices.length,
      assignmentCounts: countAssignments(fold.edges_assignment),
      roleCounts: countRoles(fold.edges_bpRole)
    },
    stretches: collectStretchMetadata()
  };
}

function collectNodeLayout(): NodeLayoutSpec[] {
  const tree = State.m.$tree;
  const result: NodeLayoutSpec[] = [];
  for(const node of tree.$nodes) {
    if(!node) continue;
    const [top, right, bottom, left] = node.$AABB.$toValues();
    result.push({
      id: node.id,
      parentId: node.$parent?.id,
      length: node.$length,
      dist: node.$dist,
      isLeaf: node.$isLeaf,
      bounds: { top, right, bottom, left }
    });
  }
  return result.sort((a, b) => a.id - b.id);
}

function cloneLayout(sheet: SheetSpec, edges: EdgeSpec[], flaps: FlapSpec[]): { sheet: SheetSpec; edges: EdgeSpec[]; flaps: FlapSpec[] } {
  return {
    sheet: { ...sheet },
    edges: edges.map(edge => ({ ...edge })),
    flaps: flaps.map(flap => ({ ...flap })),
  };
}

function collectStretchMetadata(): StretchMetadata[] {
  const result: StretchMetadata[] = [];
  for(const [id, stretch] of State.$stretches.entries()) {
    const repo = stretch.$repo;
    const serialized = stretch.toJSON();
    const repoJson = serialized.repo;
    const configuration = repo.$configuration;
    const pattern = repo.$pattern;

    result.push({
      id,
      active: stretch.$isActive,
      repository: {
        signature: repo.$signature,
        isValid: repo.$isValid,
        complete: Boolean(repoJson),
        configurationCount: repo.$configurations.length,
        selectedConfigurationIndex: repoJson?.index ?? null,
        selectedPatternIndex: configuration?.$index ?? null,
        selectedPatternCount: configuration?.$patterns.length ?? null,
        selectedDeviceCount: pattern?.$devices.length ?? null,
        selectedGadgetCount: pattern?.$gadgets.length ?? null,
        selectedAddOnCount: pattern?.$devices.reduce((sum, device) => sum + device.$addOns.length, 0) ?? null,
        quadrantCount: repo.$quadrants.size,
        junctionCount: repo.$junctions.length,
        serialized: repoJson ? serialized : undefined
      }
    });
  }
  return result;
}

function countAssignments(assignments: Assignment[]): Record<Assignment, number> {
  const counts: Record<Assignment, number> = { B: 0, M: 0, V: 0, F: 0, U: 0 };
  for(const assignment of assignments) counts[assignment]++;
  return counts;
}

function sheetBorder(sheet: SheetSpec): IPoint[] {
  return [
    { x: 0, y: 0 },
    { x: sheet.width, y: 0 },
    { x: sheet.width, y: sheet.height },
    { x: 0, y: sheet.height }
  ];
}

function addPolygon(
  set: TaggedCPLine[],
  polygon: Polygon,
  type: CreaseType,
  role: BPRole,
  source: Omit<BPStudioEdgeSource, "creaseType" | "lineIndex">,
): void {
  for(const path of polygon) {
    const l = path.length;
    for(let i = 0; i < l; i++) {
      const p1 = path[i], p2 = path[i + 1] || path[0];
      set.push({ type, p1, p2, role, source: { ...source, creaseType: type } });
    }
  }
}

function addLines(
  set: TaggedCPLine[],
  lines: readonly ILine[],
  type: CreaseType,
  role: BPRole,
  source: Omit<BPStudioEdgeSource, "creaseType" | "lineIndex">,
): void {
  for(const [lineIndex, line] of lines.entries()) {
    const [p1, p2] = line;
    set.push({ type, p1, p2, role, source: { ...source, creaseType: type, lineIndex } });
  }
}

function countRoles(roles: BPRole[]): Record<BPRole, number> {
  const counts: Record<BPRole, number> = { border: 0, hinge: 0, ridge: 0, axis: 0, stretch: 0 };
  for(const role of roles) counts[role]++;
  return counts;
}

function validateLeafFlaps(edges: EdgeSpec[], flaps: FlapSpec[]): void {
  const degree = new Map<number, number>();
  for(const edge of edges) {
    degree.set(edge.n1, (degree.get(edge.n1) ?? 0) + 1);
    degree.set(edge.n2, (degree.get(edge.n2) ?? 0) + 1);
  }
  const flapIds = new Set(flaps.map(flap => flap.id));
  const missing = [...degree.entries()]
    .filter(([, count]) => count <= 1)
    .map(([id]) => id)
    .filter(id => !flapIds.has(id))
    .sort((a, b) => a - b);
  if(missing.length) {
    throw new Error(`Every BP Studio leaf must have flap geometry; missing flap ids: ${missing.join(", ")}`);
  }
}

function isFinitePoint(point: IPoint): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.y);
}

function distance(a: IPoint, b: IPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function lineGeometryKey(line: TaggedCPLine): string {
  const p1 = pointKey(roundPoint(line.p1));
  const p2 = pointKey(roundPoint(line.p2));
  const [a, b] = p1 <= p2 ? [p1, p2] : [p2, p1];
  return `${line.type}:${a}:${b}`;
}

function pointKey(point: IPoint): string {
  return `${point.x.toFixed(9)},${point.y.toFixed(9)}`;
}

function roundPoint(point: IPoint): IPoint {
  return {
    x: round(point.x),
    y: round(point.y),
  };
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function getTreeParts(spec: BpStudioAdapterSpec): { edges: EdgeSpec[]; flaps: FlapSpec[] } {
  return {
    edges: spec.tree?.edges ?? spec.edges ?? [],
    flaps: spec.tree?.flaps ?? spec.flaps ?? []
  };
}

function normalizeSpec(input: unknown): BpStudioAdapterSpec {
  if(!isObject(input)) throw new Error("Spec must be a JSON object.");
  const spec = input as BpStudioAdapterSpec;
  if(!isObject(spec.sheet)) throw new Error("Spec requires sheet.width and sheet.height.");
  assertPositiveFiniteNumber(spec.sheet.width, "sheet.width");
  assertPositiveFiniteNumber(spec.sheet.height, "sheet.height");

  const { edges, flaps } = getTreeParts(spec);
  if(edges.length === 0) throw new Error("Spec requires at least one tree edge.");
  for(const [index, edge] of edges.entries()) {
    assertInteger(edge.n1, `edges[${index}].n1`);
    assertInteger(edge.n2, `edges[${index}].n2`);
    assertPositiveFiniteNumber(edge.length, `edges[${index}].length`);
  }
  for(const [index, flap] of flaps.entries()) {
    assertInteger(flap.id, `flaps[${index}].id`);
    assertFiniteNumber(flap.x, `flaps[${index}].x`);
    assertFiniteNumber(flap.y, `flaps[${index}].y`);
    assertNonnegativeFiniteNumber(flap.width ?? 0, `flaps[${index}].width`);
    assertNonnegativeFiniteNumber(flap.height ?? 0, `flaps[${index}].height`);
  }
  return spec;
}

function parseArgs(argv: string[]): Record<string, string | boolean> {
  const result: Record<string, string | boolean> = {};
  for(let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if(arg === "--help" || arg === "-h") {
      result.help = true;
      continue;
    }
    if(!arg.startsWith("--")) throw new Error(`Unexpected positional argument: ${arg}`);
    const key = arg.slice(2);
    const value = argv[i + 1];
    if(!value || value.startsWith("--")) throw new Error(`Missing value for --${key}.`);
    result[key] = value;
    i++;
  }
  return result;
}

async function writeJson(path: string, value: unknown): Promise<void> {
  const absolute = resolve(path);
  await mkdir(dirname(absolute), { recursive: true });
  await writeFile(absolute, `${JSON.stringify(value, null, 2)}\n`);
}

function printHelp(): void {
  console.log(`Usage: bun run generate -- --spec fixtures/two-flap.json --out /tmp/bps.fold --metadata /tmp/bps.meta.json`);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertInteger(value: unknown, name: string): void {
  if(!Number.isInteger(value)) throw new Error(`${name} must be an integer.`);
}

function assertPositiveFiniteNumber(value: unknown, name: string): void {
  assertFiniteNumber(value, name);
  if((value as number) <= 0) throw new Error(`${name} must be positive.`);
}

function assertNonnegativeFiniteNumber(value: unknown, name: string): void {
  assertFiniteNumber(value, name);
  if((value as number) < 0) throw new Error(`${name} must be nonnegative.`);
}

function assertFiniteNumber(value: unknown, name: string): void {
  if(typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number.`);
  }
}

class VertexSet {
  private readonly keys = new Map<string, number>();
  private readonly vertices: IPoint[] = [];

  public add(point: IPoint): number {
    const key = `${point.x},${point.y}`;
    const existing = this.keys.get(key);
    if(existing !== undefined) return existing;
    const index = this.vertices.length;
    this.vertices.push(point);
    this.keys.set(key, index);
    return index;
  }

  public list(): [number, number][] {
    return this.vertices.map(point => [point.x, point.y]);
  }
}

if(import.meta.main) {
  main().catch(error => {
    if(process.env.BP_STUDIO_ADAPTER_STACK === "1" && error instanceof Error) {
      console.error(error.stack ?? error.message);
    } else {
      console.error(error instanceof Error ? error.message : error);
    }
    process.exit(1);
  });
}
