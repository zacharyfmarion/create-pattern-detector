#!/usr/bin/env bun

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { LayoutController } from "core/controller/layoutController";
import { Tree } from "core/design/context/tree";
import { heightTask } from "core/design/tasks/height";
import { Clip } from "core/math/sweepLine/clip/clip";
import { Processor } from "core/service/processor";
import { State, fullReset } from "core/service/state";
import { CreaseType } from "shared/types/cp";

import type { CPLine } from "shared/types/cp";
import type { JEdge, JFlap, NodeId } from "shared/json";
import type { ILine, Path, Polygon } from "shared/types/geometry";
import type {
  AdapterMetadata,
  Assignment,
  BpStudioAdapterSpec,
  EdgeSpec,
  FlapSpec,
  FoldDocument,
  GenerationResult,
  SheetSpec,
  StretchMetadata
} from "./types";

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

export function generate(specInput: unknown): GenerationResult {
  const spec = normalizeSpec(specInput);
  const { edges, flaps } = getTreeParts(spec);
  const useAuxiliary = spec.useAuxiliary ?? false;
  const completeRepositories = spec.completeRepositories ?? true;
  const exportMode = spec.exportMode ?? "outer";

  createBpStudioTree(edges, flaps);
  if(completeRepositories) completeAllStretches();

  const border = sheetBorder(spec.sheet);
  const cpLines = exportMode === "expanded"
    ? getExpandedCP(border, useAuxiliary)
    : LayoutController.getCP(border, useAuxiliary);
  const fold = toFold(cpLines, spec);
  const metadata = collectMetadata(spec, fold, cpLines, edges, flaps, useAuxiliary, completeRepositories, exportMode);
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
  const { fold, metadata } = generate(spec);

  await writeJson(args.out, fold);
  if(args.metadata) await writeJson(args.metadata, metadata);
}

function createBpStudioTree(edges: EdgeSpec[], flaps: FlapSpec[]): void {
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
  Processor.$run(heightTask);
}

function completeAllStretches(): void {
  for(const stretch of State.$stretches.values()) {
    stretch.$repo.$complete();
  }
}

function getExpandedCP(borders: Path, useAuxiliary: boolean): CPLine[] {
  const hingeType = useAuxiliary ? CreaseType.Auxiliary : CreaseType.Valley;
  const lines: CPLine[] = [];
  addPolygon(lines, [borders], CreaseType.Border);

  for(const node of State.m.$tree.$nodes) {
    if(!node || !node.$parent) continue;
    for(const contour of node.$graphics.$contours) {
      addPolygon(lines, [contour.outer], hingeType);
      if(contour.inner) addPolygon(lines, contour.inner, hingeType);
    }
    addPolygon(lines, node.$graphics.$patternContours, hingeType);
    addPolygon(lines, node.$graphics.$traceContours, hingeType);
    addPolygon(lines, node.$graphics.$roughContours, hingeType);
    addLines(lines, node.$graphics.$ridges, CreaseType.Mountain);
  }

  for(const stretch of State.$stretches.values()) {
    const pattern = stretch.$repo.$pattern;
    if(!pattern) continue;
    for(const device of pattern.$devices) {
      addLines(lines, device.$drawRidges, CreaseType.Mountain);
      addLines(lines, device.$traceRidges.map(ridge => ridge.$toILine()), CreaseType.Mountain);
      addLines(lines, device.$axisParallels, CreaseType.Valley);
      for(const contour of device.$contour) {
        addPolygon(lines, [contour.outer], hingeType);
        if(contour.inner) addPolygon(lines, contour.inner, hingeType);
      }
    }
  }

  return new Clip().$get(lines);
}

function toFold(lines: CPLine[], spec: BpStudioAdapterSpec): FoldDocument {
  const vertices = new VertexSet();
  const edgeData = lines.map(line => {
    const assignment = ASSIGNMENT_BY_TYPE[line.type];
    return {
      assignment,
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
    edges_foldAngle: edgeData.map(edge => FOLD_ANGLE_BY_ASSIGNMENT[edge.assignment])
  };
}

function collectMetadata(
  spec: BpStudioAdapterSpec,
  fold: FoldDocument,
  lines: CPLine[],
  edges: EdgeSpec[],
  flaps: FlapSpec[],
  useAuxiliary: boolean,
  completeRepositories: boolean,
  exportMode: "outer" | "expanded"
): AdapterMetadata {
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
      edgeCount: edges.length,
      flapCount: flaps.length
    },
    cp: {
      lineCount: lines.length,
      vertexCount: fold.vertices_coords.length,
      edgeCount: fold.edges_vertices.length,
      assignmentCounts: countAssignments(fold.edges_assignment)
    },
    stretches: collectStretchMetadata()
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

function addPolygon(set: CPLine[], polygon: Polygon, type: CreaseType): void {
  for(const path of polygon) {
    const l = path.length;
    for(let i = 0; i < l; i++) {
      const p1 = path[i], p2 = path[i + 1] || path[0];
      set.push({ type, p1, p2 });
    }
  }
}

function addLines(set: CPLine[], lines: readonly ILine[], type: CreaseType): void {
  for(const line of lines) {
    const [p1, p2] = line;
    set.push({ type, p1, p2 });
  }
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
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
