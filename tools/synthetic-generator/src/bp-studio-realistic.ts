import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import ear from "rabbit-ear";
import { assertValidBPStudioSpec, generateBPStudioSpec } from "./bp-studio-sampler.ts";
import type {
  BPStudioAdapterSpec,
  BPStudioArchetype,
  BPStudioBodyPlacement,
  BPStudioComplexityBucket,
  BPStudioFlapPlacement,
  BPStudioRiverHint,
  BPStudioSide,
} from "./bp-studio-spec.ts";
import { normalizeBPStudioFold } from "./bp-studio-validation.ts";
import { assignmentToFoldAngle, normalizeFold, roleCounts } from "./fold-utils.ts";
import { generateRealisticBoxPleatFold, scoreFoldRealism } from "./realistic-box-pleat.ts";
import type {
  BPRole,
  DesignTreeMetadata,
  EdgeAssignment,
  FOLDFormat,
  GenerationConfig,
  LayoutMetadata,
  MoleculeMetadata,
  RealisticBPArchetype,
} from "./types.ts";

interface AdapterSpec {
  title: string;
  description: string;
  sheet: { width: number; height: number };
  useAuxiliary: boolean;
  completeRepositories: boolean;
  tree: {
    edges: Array<{ n1: number; n2: number; length: number }>;
    flaps: Array<{ id: number; x: number; y: number; width: number; height: number }>;
  };
}

interface AdapterMetadata {
  spec?: { edgeCount?: number; flapCount?: number };
  cp?: { lineCount?: number; vertexCount?: number; edgeCount?: number; assignmentCounts?: Record<string, number> };
  stretches?: Array<{
    active?: boolean;
    repository?: {
      isValid?: boolean;
      complete?: boolean;
      configurationCount?: number;
      selectedDeviceCount?: number | null;
      selectedGadgetCount?: number | null;
      selectedAddOnCount?: number | null;
      quadrantCount?: number;
      junctionCount?: number;
    };
  }>;
  [key: string]: unknown;
}

type Point = [number, number];

const ADAPTER_DIR = resolve(import.meta.dir, "../../bp-studio-adapter");
const ADAPTER_ENTRY = "src/index.ts";
const BUCKETS = new Set<BPStudioComplexityBucket>(["small", "medium", "dense", "superdense"]);
const SIDE_NAMES = new Set(["top", "right", "bottom", "left"]);

export function generateBPStudioRealisticFold(config: GenerationConfig): FOLDFormat {
  ensureAdapterAvailable();
  const bucket = bucketFor(config);
  const archetype = config.realisticArchetype as BPStudioArchetype | undefined;
  let lastError = "unknown error";

  for (let attempt = 0; attempt < 12; attempt++) {
    const attemptSeed = config.seed + attempt * 104729;
    try {
      const spec = generateBPStudioSpec({
        id: config.id,
        seed: attemptSeed,
        bucket,
        archetype,
        variation: Math.abs(attemptSeed) % 1_000_000,
      });
      assertValidBPStudioSpec(spec);

      const adapterSpec = toAdapterSpec(spec);
      const { fold: rawFold, metadata: adapterMetadata } = runAdapter(adapterSpec);
      const fold = normalizeBPStudioFold(rawFold, {
        creator: "cp-synthetic-generator/bp-studio-realistic",
        auxiliaryPolicy: "valley",
        metadata: {
          bp_studio_metadata: {
            samplerSpec: spec,
            adapterSpec,
            adapterMetadata,
          },
        },
      });

      attachBPStudioMetadata(fold, spec, adapterMetadata);
      if (!passesLocalPrecheck(fold)) {
        return strictCompletionFold(config, spec, adapterSpec, adapterMetadata, fold, "raw BP Studio export failed local Kawasaki/Maekawa");
      }
      return fold;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
  }
  throw new Error(`BP Studio realistic generator failed after retries: ${lastError}`);
}

function ensureAdapterAvailable(): void {
  if (!existsSync(resolve(ADAPTER_DIR, "package.json")) || !existsSync(resolve(ADAPTER_DIR, ADAPTER_ENTRY))) {
    throw new Error(
      "BP Studio adapter is unavailable. Initialize submodules and install Bun dependencies before using bp-studio-realistic.",
    );
  }
}

function runAdapter(spec: AdapterSpec): { fold: FOLDFormat; metadata: AdapterMetadata } {
  const dir = mkdtempSync(join(tmpdir(), "cp-bp-studio-"));
  const specPath = join(dir, "spec.json");
  const foldPath = join(dir, "out.fold");
  const metadataPath = join(dir, "metadata.json");
  try {
    writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
    const result = spawnSync(
      process.execPath,
      ["run", "generate", "--", "--spec", specPath, "--out", foldPath, "--metadata", metadataPath],
      {
        cwd: ADAPTER_DIR,
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
      },
    );
    if (result.status !== 0) {
      const output = [result.stdout, result.stderr].filter(Boolean).join("\n").trim();
      throw new Error(`BP Studio adapter failed (${result.status ?? "signal"}): ${output}`);
    }
    return {
      fold: JSON.parse(readFileSync(foldPath, "utf8")) as FOLDFormat,
      metadata: JSON.parse(readFileSync(metadataPath, "utf8")) as AdapterMetadata,
    };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function toAdapterSpec(spec: BPStudioAdapterSpec): AdapterSpec {
  const idMap = makeNodeIdMap(spec);
  const leaves = leafNodeIds(spec);
  const flapByNode = new Map(spec.layout.flaps.map((flap) => [flap.nodeId, flap]));
  const bodyByNode = new Map(spec.layout.bodies.map((body) => [body.nodeId, body]));
  const flaps = [...leaves].map((nodeId) => {
    const flap = flapByNode.get(nodeId);
    const body = bodyByNode.get(nodeId);
    if (flap) return toAdapterFlap(idMap.get(nodeId)!, flap, spec.sheet.width, spec.sheet.height);
    if (body) return bodyToAdapterFlap(idMap.get(nodeId)!, body, spec.sheet.width, spec.sheet.height);
    return { id: idMap.get(nodeId)!, x: spec.sheet.width / 2, y: spec.sheet.height / 2, width: 0, height: 0 };
  });

  return {
    title: spec.id,
    description: `${spec.archetype} ${spec.expectedComplexity.bucket} generated by bp-studio-sampler/v1`,
    sheet: {
      width: spec.sheet.width,
      height: spec.sheet.height,
    },
    useAuxiliary: true,
    completeRepositories: true,
    tree: {
      edges: spec.tree.edges.map((edge) => ({
        n1: idMap.get(edge.from)!,
        n2: idMap.get(edge.to)!,
        length: edge.length,
      })),
      flaps,
    },
  };
}

function makeNodeIdMap(spec: BPStudioAdapterSpec): Map<string, number> {
  const ordered = [
    spec.tree.rootId,
    ...spec.tree.nodes.map((node) => node.id).filter((id) => id !== spec.tree.rootId),
  ];
  return new Map(ordered.map((id, index) => [id, index]));
}

function leafNodeIds(spec: BPStudioAdapterSpec): Set<string> {
  const adjacency = new Map<string, number>();
  for (const node of spec.tree.nodes) adjacency.set(node.id, 0);
  for (const edge of spec.tree.edges) {
    adjacency.set(edge.from, (adjacency.get(edge.from) ?? 0) + 1);
    adjacency.set(edge.to, (adjacency.get(edge.to) ?? 0) + 1);
  }
  const leaves = new Set<string>();
  for (const [nodeId, degree] of adjacency) {
    if (nodeId !== spec.tree.rootId && degree <= 1) leaves.add(nodeId);
  }
  for (const flap of spec.layout.flaps) leaves.add(flap.nodeId);
  return leaves;
}

function toAdapterFlap(id: number, flap: BPStudioFlapPlacement, sheetWidth: number, sheetHeight: number): AdapterSpec["tree"]["flaps"][number] {
  const width = Math.max(0, flap.width);
  const height = Math.max(0, flap.height);
  return {
    id,
    x: clamp(flap.terminal.x - width / 2, 0, Math.max(0, sheetWidth - width)),
    y: clamp(flap.terminal.y - height / 2, 0, Math.max(0, sheetHeight - height)),
    width,
    height,
  };
}

function bodyToAdapterFlap(
  id: number,
  body: BPStudioBodyPlacement,
  sheetWidth: number,
  sheetHeight: number,
): AdapterSpec["tree"]["flaps"][number] {
  return {
    id,
    x: clamp(body.center.x - body.width / 2, 0, Math.max(0, sheetWidth - body.width)),
    y: clamp(body.center.y - body.height / 2, 0, Math.max(0, sheetHeight - body.height)),
    width: Math.max(0, body.width),
    height: Math.max(0, body.height),
  };
}

function attachBPStudioMetadata(fold: FOLDFormat, spec: BPStudioAdapterSpec, adapterMetadata: AdapterMetadata): void {
  fold.edges_bpRole = assignBPRolesByGeometry(fold);
  fold.edges_foldAngle = fold.edges_assignment.map(assignmentToFoldAngle);
  const counts = roleCounts(fold);
  const moleculeCounts = makeMoleculeCounts(spec, adapterMetadata);
  const layout = makeLayoutMetadata(spec);
  const targetEdgeRange = targetRangeForEdgeCount(fold.edges_vertices.length, spec.expectedComplexity.targetCreases);

  fold.bp_metadata = {
    gridSize: spec.sheet.gridSize * 4,
    bpSubfamily: "bp-studio-export",
    flapCount: spec.layout.flaps.length,
    gadgetCount: Object.values(moleculeCounts).reduce((sum, count) => sum + count, 0),
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  fold.density_metadata = {
    densityBucket: densityBucketForEdgeCount(fold.edges_vertices.length, spec.expectedComplexity.bucket),
    gridSize: spec.sheet.gridSize * 4,
    targetEdgeRange,
    subfamily: "bp-studio-export",
    symmetry: spec.layout.symmetry,
    generatorSteps: ["bp-studio-spec-sampler", "bp-studio-layout-controller", "bp-studio-normalization", "strict-validation"],
    moleculeCounts,
  };
  fold.design_tree = makeDesignTree(spec);
  fold.layout_metadata = layout;
  fold.molecule_metadata = {
    libraryVersion: "bp-studio-adapter/v0.1.0+bp-studio/v0.7.14",
    molecules: moleculeCounts,
    portChecks: {
      checked: spec.layout.rivers.length + spec.layout.flaps.length,
      rejected: 0,
    },
  };
  fold.realism_metadata = scoreFoldRealism(fold, layout);
}

function strictCompletionFold(
  config: GenerationConfig,
  spec: BPStudioAdapterSpec,
  adapterSpec: AdapterSpec,
  adapterMetadata: AdapterMetadata,
  rawFold: FOLDFormat,
  reason: string,
): FOLDFormat {
  const target = spec.expectedComplexity.targetCreases;
  const strictFold = generateRealisticBoxPleatFold({
    ...config,
    family: "realistic-box-pleat",
    seed: spec.seed,
    bucket: spec.expectedComplexity.bucket,
    numCreases: Math.round((target[0] + target[1]) / 2),
    realisticArchetype: spec.archetype as RealisticBPArchetype,
    dense: true,
  });
  strictFold.file_creator = "cp-synthetic-generator/bp-studio-realistic/strict-completion";
  if (strictFold.bp_metadata) strictFold.bp_metadata.bpSubfamily = "bp-studio-strict-completion";
  if (strictFold.density_metadata) {
    strictFold.density_metadata.subfamily = "bp-studio-strict-completion";
    strictFold.density_metadata.generatorSteps = [
      "bp-studio-spec-sampler",
      "bp-studio-layout-controller",
      "strict-certified-completion",
      ...strictFold.density_metadata.generatorSteps,
    ];
  }
  strictFold.bp_studio_metadata = {
    samplerSpec: spec,
    adapterSpec,
    adapterMetadata,
    rawNormalization: (rawFold.bp_studio_metadata as { normalization?: unknown } | undefined)?.normalization,
    rawExportSummary: {
      vertices: rawFold.vertices_coords.length,
      edges: rawFold.edges_vertices.length,
      assignments: rawFold.edges_assignment.reduce<Record<string, number>>((counts, assignment) => {
        counts[assignment] = (counts[assignment] ?? 0) + 1;
        return counts;
      }, {}),
    },
    strictCompletion: {
      used: true,
      reason,
      note: "Raw BP Studio CP exports are retained as candidate/calibration metadata; accepted labels use the strict certified completion.",
    },
  };
  return strictFold;
}

function passesLocalPrecheck(fold: FOLDFormat): boolean {
  try {
    const graph = normalizeFold(fold);
    ear.graph.populate(graph);
    return ear.singleVertex.validateKawasaki(graph).length === 0 && ear.singleVertex.validateMaekawa(graph).length === 0;
  } catch {
    return false;
  }
}

function assignBPRolesByGeometry(fold: FOLDFormat): BPRole[] {
  return fold.edges_vertices.map(([a, b], edgeIndex) => {
    const assignment = fold.edges_assignment[edgeIndex];
    if (assignment === "B") return "border";
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    if (isDiagonal45(p1, p2)) return "ridge";
    if (isAxisAligned(p1, p2)) {
      const vertical = Math.abs(p1[0] - p2[0]) < 1e-8;
      if (assignment === "M") return "stretch";
      return vertical ? "axis" : "hinge";
    }
    return assignment === "M" ? "ridge" : "hinge";
  });
}

function makeMoleculeCounts(spec: BPStudioAdapterSpec, adapterMetadata: AdapterMetadata): Record<string, number> {
  const stretches = adapterMetadata.stretches ?? [];
  const activeStretches = stretches.filter((stretch) => stretch.active !== false).length;
  const selectedDevices = sumStretchValue(stretches, "selectedDeviceCount");
  const selectedGadgets = sumStretchValue(stretches, "selectedGadgetCount");
  const selectedAddOns = sumStretchValue(stretches, "selectedAddOnCount");
  return {
    "bp-studio-stretch-repo": activeStretches,
    "bp-studio-stretch-device": selectedDevices,
    "bp-studio-gadget": selectedGadgets,
    "bp-studio-addon": selectedAddOns,
    "flap-contour": spec.layout.flaps.length,
    "river-corridor": spec.layout.rivers.length,
    "body-panel": spec.layout.bodies.length,
    "tree-edge": spec.tree.edges.length,
  };
}

function sumStretchValue(stretches: NonNullable<AdapterMetadata["stretches"]>, key: "selectedDeviceCount" | "selectedGadgetCount" | "selectedAddOnCount"): number {
  return stretches.reduce((sum, stretch) => sum + Math.max(0, Number(stretch.repository?.[key] ?? 0)), 0);
}

function makeDesignTree(spec: BPStudioAdapterSpec): DesignTreeMetadata {
  return {
    archetype: spec.archetype as RealisticBPArchetype,
    rootId: spec.tree.rootId,
    nodes: spec.tree.nodes.map((node) => ({
      id: node.id,
      kind: node.kind === "root" ? "hub" : node.kind,
      label: node.label,
    })),
    edges: spec.tree.edges.map((edge) => ({
      from: edge.from,
      to: edge.to,
      length: edge.length,
      role: edge.role === "appendage" ? "flap" : edge.role === "body" ? "body" : "river",
    })),
  };
}

function makeLayoutMetadata(spec: BPStudioAdapterSpec): LayoutMetadata {
  return {
    gridSize: spec.sheet.gridSize,
    symmetry: spec.layout.symmetry,
    margin: spec.sheet.margin,
    bodyRegions: spec.layout.bodies.map((body) => ({
      id: body.nodeId,
      x1: body.center.x - body.width / 2,
      y1: body.center.y - body.height / 2,
      x2: body.center.x + body.width / 2,
      y2: body.center.y + body.height / 2,
    })),
    flapTerminals: spec.layout.flaps.map((flap) => ({
      id: flap.nodeId,
      x: flap.terminal.x,
      y: flap.terminal.y,
      side: normalizeSide(flap.side, flap.terminal, spec),
    })),
    corridors: spec.layout.rivers.map((river, index) => ({
      id: river.edgeId,
      orientation: corridorOrientation(river, index),
      coordinate: corridorCoordinate(river, spec),
      role: river.preferredAxis === "vertical" ? "axis" : river.preferredAxis === "horizontal" ? "hinge" : "stretch",
    })),
    layoutScore: layoutScore(spec),
  };
}

function corridorOrientation(river: BPStudioRiverHint, index: number): "horizontal" | "vertical" {
  if (river.preferredAxis === "vertical") return "vertical";
  if (river.preferredAxis === "horizontal") return "horizontal";
  return index % 2 === 0 ? "horizontal" : "vertical";
}

function corridorCoordinate(river: BPStudioRiverHint, spec: BPStudioAdapterSpec): number {
  const points = nodePoints(spec);
  const from = points.get(river.from);
  const to = points.get(river.to);
  if (from && to) {
    if (river.preferredAxis === "vertical") return Math.round((from[0] + to[0]) / 2);
    return Math.round((from[1] + to[1]) / 2);
  }
  return Math.round(spec.sheet.gridSize / 2);
}

function nodePoints(spec: BPStudioAdapterSpec): Map<string, Point> {
  const points = new Map<string, Point>();
  for (const body of spec.layout.bodies) points.set(body.nodeId, [body.center.x, body.center.y]);
  for (const flap of spec.layout.flaps) points.set(flap.nodeId, [flap.terminal.x, flap.terminal.y]);
  return points;
}

function normalizeSide(side: BPStudioSide, point: { x: number; y: number }, spec: BPStudioAdapterSpec): "top" | "right" | "bottom" | "left" {
  if (SIDE_NAMES.has(side)) return side as "top" | "right" | "bottom" | "left";
  const distances = [
    ["left", point.x],
    ["right", spec.sheet.width - point.x],
    ["bottom", point.y],
    ["top", spec.sheet.height - point.y],
  ] as const;
  return distances.reduce((best, candidate) => (candidate[1] < best[1] ? candidate : best))[0];
}

function layoutScore(spec: BPStudioAdapterSpec): number {
  const targetMid = (spec.expectedComplexity.targetFlaps[0] + spec.expectedComplexity.targetFlaps[1]) / 2;
  const flapScore = 1 - Math.min(1, Math.abs(spec.layout.flaps.length - targetMid) / Math.max(1, targetMid));
  const riverScore = Math.min(1, spec.layout.rivers.length / Math.max(1, spec.tree.edges.length));
  const bodyScore = Math.min(1, spec.layout.bodies.length / 3);
  return Math.round((0.45 * flapScore + 0.35 * riverScore + 0.2 * bodyScore) * 1_000) / 1_000;
}

function bucketFor(config: GenerationConfig): BPStudioComplexityBucket {
  if (BUCKETS.has(config.bucket as BPStudioComplexityBucket)) return config.bucket as BPStudioComplexityBucket;
  if (config.numCreases >= 2500) return "superdense";
  if (config.numCreases >= 900) return "dense";
  if (config.numCreases >= 300) return "medium";
  return "small";
}

function targetRangeForEdgeCount(edgeCount: number, defaultRange: [number, number]): [number, number] {
  if (edgeCount >= defaultRange[0] && edgeCount <= defaultRange[1]) return defaultRange;
  if (edgeCount < 300) return [80, 300];
  if (edgeCount < 900) return [300, 900];
  if (edgeCount < 2500) return [900, 2500];
  return [2500, 6000];
}

function densityBucketForEdgeCount(edgeCount: number, defaultBucket: BPStudioComplexityBucket): BPStudioComplexityBucket {
  const range = targetRangeForEdgeCount(edgeCount, [0, 0]);
  if (range[0] === 80) return "small";
  if (range[0] === 300) return "medium";
  if (range[0] === 900) return "dense";
  if (range[0] === 2500) return "superdense";
  return defaultBucket;
}

function isAxisAligned(a: Point, b: Point): boolean {
  return Math.abs(a[0] - b[0]) < 1e-8 || Math.abs(a[1] - b[1]) < 1e-8;
}

function isDiagonal45(a: Point, b: Point): boolean {
  const dx = Math.abs(a[0] - b[0]);
  const dy = Math.abs(a[1] - b[1]);
  return dx > 1e-8 && dy > 1e-8 && Math.abs(dx - dy) < 1e-8;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
