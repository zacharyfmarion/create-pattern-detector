import { runBpStudioLayout, type BpStudioLayoutContour, type BpStudioLayoutGraphics, type BpStudioLayoutLine } from "./bp-studio-layout.ts";
import { solveWithBpStudioOptimizer } from "./bp-studio-optimizer.ts";
import { SeededRandom } from "./random.ts";
import { fillBoxPleatedGaps, type GapRect, type OccupiedPolygon } from "./box-pleated-gap-fill.ts";
import type { OriSegment } from "./ori-parser.ts";
import type { BpOptimizerHierarchy, BpOptimizerRequest, BpOptimizerResult } from "./bp-studio-optimizer.ts";

export type BoxPleatedPackingSymmetry = "vertical" | "horizontal" | "none";

export interface BoxPleatedPackingConfig {
  id: string;
  seed: number;
  numCreases: number;
  bucket: string;
  symmetry?: BoxPleatedPackingSymmetry;
  bpStudioRoot?: string;
  maxAttempts?: number;
  targetLeafCount?: number;
  optimizerDistanceScale?: number;
  noStretches?: boolean;
  /**
   * Tight mode: run BP Studio's optimizer the way the app's "Optimize Layout"
   * does - basin-hopping across several random restarts - and keep the tightest
   * layout. Pythagorean stretch devices are allowed (their creases are well
   * defined), trading the clean-grid guarantee for paper efficiency.
   */
  tight?: boolean;
  tightRestarts?: number;
}

export interface BoxPleatedBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BoxPleatedTreeNode {
  id: number;
  parentId: number | null;
  kind: "root" | "hub" | "terminal";
  label: string;
  lengthToParent: number;
}

export interface BoxPleatedTreeEdge {
  id: string;
  from: number;
  to: number;
  length: number;
}

export interface BoxPleatedFlap {
  id: number;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  radius: number;
}

export interface BoxPleatedLayoutObject {
  id: string;
  kind: "flap" | "river" | "stretch-device" | "root";
  nodeId: number | null;
  contours: BpStudioLayoutContour[];
  ridges: BpStudioLayoutLine[];
  axisParallel: BpStudioLayoutLine[];
}

export interface BoxPleatedValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface BoxPleatedPacking {
  schemaVersion: "box-pleated-packing/v3";
  id: string;
  seed: number;
  bucket: string;
  symmetry: BoxPleatedPackingSymmetry;
  constraints: {
    noStretches: boolean;
  };
  tree: {
    nodes: BoxPleatedTreeNode[];
    edges: BoxPleatedTreeEdge[];
    leafIds: number[];
    optimizerDistanceScale: number;
  };
  optimizer: {
    source: "bp-studio";
    request: BpOptimizerRequest;
    result: BpOptimizerResult;
  };
  sheet: BoxPleatedBounds;
  flaps: BoxPleatedFlap[];
  layout: {
    source: "bp-studio-core";
    patternNotFound: boolean;
    objects: BoxPleatedLayoutObject[];
  };
  validation: BoxPleatedValidation;
  stats: {
    leaves: number;
    treeEdges: number;
    layoutObjects: number;
    hingeContours: number;
    ridgeCreases: number;
    axisParallelCreases: number;
    stretchDevices: number;
    offGridRidgeCreases: number;
    sheetCells: number;
  };
}

interface MutableTreeNode {
  id: number;
  parentId: number | null;
  kind: "root" | "hub" | "terminal";
  label: string;
  lengthToParent: number;
  children: number[];
  width?: number;
  height?: number;
}

interface SampledTree {
  nodes: MutableTreeNode[];
  leafIds: number[];
  symmetry: BoxPleatedPackingSymmetry;
  optimizerDistanceScale: number;
  noStretches: boolean;
}

interface BuildResult {
  packing: BoxPleatedPacking;
  errors: string[];
}

const DEFAULT_MAX_ATTEMPTS = 24;
const DEFAULT_NO_STRETCH_MAX_ATTEMPTS = 96;
const DEFAULT_OPTIMIZER_DISTANCE_SCALE = 1;
const DEFAULT_NO_STRETCH_OPTIMIZER_DISTANCE_SCALE = 1.25;
const DEFAULT_TIGHT_RESTARTS = 16;

export async function generateBoxPleatedPacking(config: BoxPleatedPackingConfig): Promise<BoxPleatedPacking> {
  if (config.tight) return generateTightBoxPleatedPacking(config);
  const maxAttempts = config.maxAttempts ?? (config.noStretches ? DEFAULT_NO_STRETCH_MAX_ATTEMPTS : DEFAULT_MAX_ATTEMPTS);
  const failures: string[] = [];

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const rng = new SeededRandom(config.seed + attempt * 1009);
    const sampled = sampleTree(config, rng);
    const request = optimizerRequestForTree(sampled, rng);
    let optimizerResult: BpOptimizerResult;
    try {
      optimizerResult = await solveWithBpStudioOptimizer(request, {
        bpStudioRoot: config.bpStudioRoot,
        seed: config.seed + attempt,
      });
    } catch (error) {
      failures.push(`attempt ${attempt}: BP Studio optimizer failed: ${errorMessage(error)}`);
      continue;
    }

    let result: BuildResult;
    try {
      result = await buildPacking({
        id: attempt === 0 ? config.id : `${config.id}-attempt-${attempt}`,
        seed: config.seed,
        bucket: config.bucket,
        bpStudioRoot: config.bpStudioRoot,
        noStretches: config.noStretches ?? false,
        sampled,
        optimizerRequest: request,
        optimizerResult,
      });
    } catch (error) {
      failures.push(`attempt ${attempt}: BP Studio core failed: ${errorMessage(error)}`);
      continue;
    }
    if (result.errors.length === 0) return result.packing;
    failures.push(...result.errors.map((error) => `attempt ${attempt}: ${error}`));
  }

  const uniqueFailures = [...new Set(failures)].slice(0, 12);
  throw new Error(`Unable to generate a valid BP Studio packing for ${config.id}: ${uniqueFailures.join("; ")}`);
}

async function generateTightBoxPleatedPacking(config: BoxPleatedPackingConfig): Promise<BoxPleatedPacking> {
  const restarts = config.tightRestarts ?? DEFAULT_TIGHT_RESTARTS;
  const rng = new SeededRandom(config.seed);
  // Tight mode optimizes a single fixed tree; stretches are permitted, so the
  // 45/90-biased distance scale is not needed.
  const sampled = sampleTree({ ...config, noStretches: false }, rng);
  const flaps = sampled.leafIds.map((id) => {
    const node = sampled.nodes[id];
    return { id, width: node.width ?? 0, height: node.height ?? 0 };
  });
  const hierarchy: BpOptimizerHierarchy = { leaves: sampled.leafIds, distMap: distMap(sampled), parents: [] };

  // Restart the optimizer from independent random seeds (basin-hopping on) and
  // keep every distinct layout; the tightest one wins.
  const candidates: Array<{ request: BpOptimizerRequest; result: BpOptimizerResult }> = [];
  const failures: string[] = [];
  for (let restart = 0; restart < restarts; restart++) {
    const request: BpOptimizerRequest = {
      type: "rect",
      flaps,
      hierarchies: [hierarchy],
      layout: "view",
      useBH: true,
      random: 0,
      vec: sampled.leafIds.map(() => ({ x: rng.float(0.08, 0.92), y: rng.float(0.08, 0.92) })),
    };
    try {
      const result = await solveWithBpStudioOptimizer(request, {
        bpStudioRoot: config.bpStudioRoot,
        seed: config.seed + restart,
      });
      candidates.push({ request, result });
    } catch (error) {
      failures.push(errorMessage(error));
    }
  }
  candidates.sort((a, b) => a.result.width - b.result.width);

  // Build the tightest layout that BP Studio's core can actually realize
  // (skip any that report patternNotFound or violate packing constraints).
  for (const candidate of candidates) {
    let result: BuildResult;
    try {
      result = await buildPacking({
        id: config.id,
        seed: config.seed,
        bucket: config.bucket,
        bpStudioRoot: config.bpStudioRoot,
        noStretches: false,
        sampled,
        optimizerRequest: candidate.request,
        optimizerResult: candidate.result,
      });
    } catch (error) {
      failures.push(`BP Studio core failed: ${errorMessage(error)}`);
      continue;
    }
    if (result.errors.length === 0) return result.packing;
    failures.push(...result.errors);
  }

  const uniqueFailures = [...new Set(failures)].slice(0, 12);
  throw new Error(`Unable to optimize a tight BP Studio packing for ${config.id}: ${uniqueFailures.join("; ")}`);
}

export function validateBoxPleatedPacking(packing: BoxPleatedPacking): string[] {
  return validatePacking(packing);
}

export interface PackingGapFill {
  /** New filler flaps (sheet coordinates) that consume the empty paper. */
  flaps: GapRect[];
  /** Straight-skeleton ridge creases for the filler flaps. */
  ridges: OriSegment[];
  /** True when all empty paper was filled (the packing is complete, rule #4). */
  complete: boolean;
  /** Empty regions that could not be filled (the packing should be rejected). */
  unresolved: GapRect[];
}

/**
 * Fill the empty paper of a BP Studio packing with filler flaps so it consumes
 * all paper (ODS polygon-packing rule #4). Flap, river, and stretch-device
 * contours are treated as occupied; the remaining empty rectangles are tiled
 * with valid flaps. A packing with an unfillable void (e.g. a 1-wide fully
 * interior strip) is reported incomplete and should be rejected.
 */
export function fillPackingGaps(packing: BoxPleatedPacking): PackingGapFill {
  const occupied: OccupiedPolygon[] = [];
  for (const object of packing.layout.objects) {
    if (object.kind === "root") continue;
    for (const contour of object.contours) {
      occupied.push({ outer: contour.outer, inner: contour.inner });
    }
  }
  const result = fillBoxPleatedGaps(packing.sheet, occupied);
  return {
    flaps: result.flaps,
    ridges: result.ridges,
    complete: result.resolved,
    unresolved: result.unresolved,
  };
}

export function renderBoxPleatedPackingSvg(
  packing: BoxPleatedPacking,
  options: { cellSize?: number; includeLegend?: boolean; includeLayoutContours?: boolean } = {},
): string {
  const cellSize = options.cellSize ?? 16;
  const includeLegend = options.includeLegend ?? true;
  const includeLayoutContours = options.includeLayoutContours ?? false;
  const margin = 44;
  const legendWidth = includeLegend ? 250 : 0;
  const sheetWidth = packing.sheet.width * cellSize;
  const sheetHeight = packing.sheet.height * cellSize;
  const width = sheetWidth + margin * 2 + legendWidth;
  const height = sheetHeight + margin * 2;
  const transform = `translate(${margin} ${margin + sheetHeight}) scale(${cellSize} ${-cellSize})`;
  const clipId = `${sanitizeId(packing.id)}-clip`;
  const grid = renderGrid(packing.sheet);
  const hingeContours = includeLayoutContours ? packing.layout.objects.map(renderHingeContours).join("\n") : "";
  const flapOutlines = packing.flaps.map(renderFlapCircle).join("\n");
  const ridges = packing.layout.objects.map(renderRidges).join("\n");
  const axisParallels = packing.layout.objects.map(renderAxisParallels).join("\n");
  const dots = packing.flaps.map(renderFlapDots).join("\n");

  const contourLegend = includeLayoutContours ? `    <rect x="0" y="26" width="42" height="12" class="flap-box"/>
    <text class="legend" x="54" y="36">amber flap boxes (aux lines)</text>
    <line x1="0" y1="62" x2="42" y2="62" class="river-box"/>
    <text class="legend" x="54" y="66">cyan river contours (aux lines)</text>
    <line x1="0" y1="92" x2="42" y2="92" class="circle"/>
    <text class="legend" x="54" y="96">blue flap radius outlines</text>
    <line x1="0" y1="122" x2="42" y2="122" class="ridge"/>
    <text class="legend" x="54" y="126">red ridge creases</text>
    <line x1="0" y1="152" x2="42" y2="152" class="axis-parallel"/>
    <text class="legend" x="54" y="156">green stretch lines</text>
    <circle cx="8" cy="182" r="4" class="dot"/>
    <text class="legend" x="54" y="186">flap anchor dots</text>` : `    <line x1="0" y1="32" x2="42" y2="32" class="circle"/>
    <text class="legend" x="54" y="36">blue flap radius outlines</text>
    <line x1="0" y1="62" x2="42" y2="62" class="ridge"/>
    <text class="legend" x="54" y="66">red ridge creases</text>
    <line x1="0" y1="92" x2="42" y2="92" class="axis-parallel"/>
    <text class="legend" x="54" y="96">green stretch lines</text>
    <circle cx="8" cy="122" r="4" class="dot"/>
    <text class="legend" x="54" y="126">flap anchor dots</text>`;
  const legend = includeLegend ? `  <g transform="translate(${margin + sheetWidth + 28} ${margin})">
    <text class="title" x="0" y="0">Layers</text>
${contourLegend}
  </g>` : "";

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <clipPath id="${clipId}">
      <rect x="0" y="0" width="${packing.sheet.width}" height="${packing.sheet.height}"/>
    </clipPath>
  </defs>
  <style>
    .background { fill: #1f1f1f; }
    .sheet { fill: none; stroke: #d4d4d4; stroke-width: 3; vector-effect: non-scaling-stroke; }
    .grid { stroke: #6d6d6d; stroke-width: 0.55; opacity: 0.65; vector-effect: non-scaling-stroke; }
    .hinge { fill: none; stroke: #5f93ff; stroke-width: 2.5; stroke-linejoin: round; vector-effect: non-scaling-stroke; }
    .flap-box { fill: none; stroke: #f2b134; stroke-width: 2; stroke-linejoin: round; vector-effect: non-scaling-stroke; }
    .river-box { fill: none; stroke: #38bdf8; stroke-width: 1.8; stroke-linejoin: round; vector-effect: non-scaling-stroke; }
    .circle { fill: none; stroke: #5f93ff; stroke-width: 1.25; vector-effect: non-scaling-stroke; }
    .ridge { stroke: #ff3657; stroke-width: 1.7; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .axis-parallel { stroke: #53df63; stroke-width: 1.25; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .dot { fill: #5f93ff; stroke: #d4d4d4; stroke-width: 0.8; vector-effect: non-scaling-stroke; }
    .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: 700; fill: #f5f5f5; }
    .meta { font-family: Arial, sans-serif; font-size: 12px; fill: #c4c4c4; }
    .legend { font-family: Arial, sans-serif; font-size: 12px; fill: #e5e7eb; }
  </style>
  <rect width="100%" height="100%" class="background"/>
  <text x="${margin}" y="24" class="title">${escapeXml(packing.id)}</text>
  <text x="${margin}" y="40" class="meta">${packing.sheet.width}x${packing.sheet.height} sheet, ${packing.flaps.length} flaps, ${packing.stats.ridgeCreases} ridge creases, noStretches=${packing.constraints.noStretches}</text>
  <g transform="${transform}">
    <rect x="0" y="0" width="${packing.sheet.width}" height="${packing.sheet.height}" class="sheet"/>
    ${grid}
    <g clip-path="url(#${clipId})">
      ${hingeContours}
      ${flapOutlines}
      ${axisParallels}
      ${ridges}
      ${dots}
    </g>
  </g>
${legend}
</svg>
`;
}

function sampleTree(config: BoxPleatedPackingConfig, rng: SeededRandom): SampledTree {
  const symmetry = config.symmetry ?? "none";
  const noStretches = config.noStretches ?? false;
  const optimizerDistanceScale = config.optimizerDistanceScale ??
    (noStretches ? DEFAULT_NO_STRETCH_OPTIMIZER_DISTANCE_SCALE : DEFAULT_OPTIMIZER_DISTANCE_SCALE);
  const requestedLeaves = config.targetLeafCount ?? leafCountForTarget(config.numCreases);
  const hubCount = Math.max(2, Math.ceil(requestedLeaves / 2));
  const nodes: MutableTreeNode[] = [{
    id: 0,
    parentId: null,
    kind: "root",
    label: "trunk",
    lengthToParent: 0,
    children: [],
  }];
  const trunkId = 0;

  const leafIds: number[] = [];
  let nextId = 1;
  for (let hubIndex = 0; hubIndex < hubCount; hubIndex++) {
    const hubId = nextId++;
    nodes.push({
      id: hubId,
      parentId: trunkId,
      kind: "hub",
      label: `hub-${hubIndex}`,
      lengthToParent: scaledLength(rng.int(3, 6), optimizerDistanceScale),
      children: [],
    });
    nodes[trunkId].children.push(hubId);

    const leavesOnHub = hubIndex === hubCount - 1 && requestedLeaves % 2 === 1 ? 1 : 2;
    for (let leafIndex = 0; leafIndex < leavesOnHub; leafIndex++) {
      const leafId = nextId++;
      nodes.push({
        id: leafId,
        parentId: hubId,
        kind: "terminal",
        label: `leaf-${hubIndex}-${leafIndex}`,
        lengthToParent: scaledLength(rng.int(3, 7), optimizerDistanceScale),
        children: [],
        width: rng.choice([0, 0, 1, 2, 3]),
        height: rng.choice([0, 0, 1, 2, 3]),
      });
      nodes[hubId].children.push(leafId);
      leafIds.push(leafId);
    }
  }

  return { nodes, leafIds, symmetry, optimizerDistanceScale, noStretches };
}

function optimizerRequestForTree(sampled: SampledTree, rng: SeededRandom): BpOptimizerRequest {
  const flaps = sampled.leafIds.map((id) => {
    const node = sampled.nodes[id];
    return {
      id,
      width: node.width ?? 0,
      height: node.height ?? 0,
    };
  });
  const hierarchy: BpOptimizerHierarchy = {
    leaves: sampled.leafIds,
    distMap: distMap(sampled),
    parents: [],
  };
  return {
    type: "rect",
    flaps,
    hierarchies: [hierarchy],
    layout: "view",
    useBH: false,
    random: 0,
    vec: initialVectors(sampled, rng),
  };
}

async function buildPacking(args: {
  id: string;
  seed: number;
  bucket: string;
  bpStudioRoot?: string;
  noStretches: boolean;
  sampled: SampledTree;
  optimizerRequest: BpOptimizerRequest;
  optimizerResult: BpOptimizerResult;
}): Promise<BuildResult> {
  const sheet = {
    x: 0,
    y: 0,
    width: args.optimizerResult.width,
    height: args.optimizerResult.height,
  };
  const flaps = args.optimizerResult.flaps.map((result) => {
    const node = args.sampled.nodes[result.id];
    return {
      id: result.id,
      label: node.label,
      x: result.x,
      y: result.y,
      width: node.width ?? 0,
      height: node.height ?? 0,
      radius: node.lengthToParent,
    };
  }).sort((a, b) => a.id - b.id);
  const edges = treeEdges(args.sampled);
  const layoutResult = await runBpStudioLayout({
    bpStudioRoot: args.bpStudioRoot,
    edges: edges.map((edge) => ({ n1: edge.from, n2: edge.to, length: edge.length })),
    flaps,
    sheet,
  });
  const objects = layoutObjects(layoutResult.graphics);
  const packing: BoxPleatedPacking = {
    schemaVersion: "box-pleated-packing/v3",
    id: args.id,
    seed: args.seed,
    bucket: args.bucket,
    symmetry: args.sampled.symmetry,
    constraints: {
      noStretches: args.noStretches,
    },
    tree: {
      nodes: args.sampled.nodes.map((node) => ({
        id: node.id,
        parentId: node.parentId,
        kind: node.kind,
        label: node.label,
        lengthToParent: node.lengthToParent,
      })),
      edges,
      leafIds: args.sampled.leafIds,
      optimizerDistanceScale: args.sampled.optimizerDistanceScale,
    },
    optimizer: {
      source: "bp-studio",
      request: args.optimizerRequest,
      result: args.optimizerResult,
    },
    sheet,
    flaps,
    layout: {
      source: "bp-studio-core",
      patternNotFound: layoutResult.patternNotFound,
      objects,
    },
    validation: {
      valid: false,
      errors: [],
      warnings: [],
    },
    stats: statsFor(args.sampled, objects, sheet),
  };
  const errors = validatePacking(packing);
  packing.validation = {
    valid: errors.length === 0,
    errors,
    warnings: packing.layout.patternNotFound ? ["BP Studio did not find a stretch pattern for at least one junction"] : [],
  };
  return { packing, errors };
}

function treeEdges(sampled: SampledTree): BoxPleatedTreeEdge[] {
  return sampled.nodes
    .filter((node) => node.parentId !== null)
    .map((node) => ({
      id: `${node.parentId}-${node.id}`,
      from: node.parentId!,
      to: node.id,
      length: node.lengthToParent,
    }));
}

function layoutObjects(graphics: Record<string, BpStudioLayoutGraphics>): BoxPleatedLayoutObject[] {
  return Object.entries(graphics).map(([id, data]) => ({
    id,
    kind: objectKind(id),
    nodeId: nodeIdFromTag(id),
    contours: data.contours ?? [],
    ridges: withoutZeroLength(data.ridges ?? []),
    axisParallel: withoutZeroLength(data.axisParallel ?? []),
  })).sort((a, b) => objectRank(a.kind) - objectRank(b.kind) || a.id.localeCompare(b.id));
}

function objectKind(id: string): BoxPleatedLayoutObject["kind"] {
  if (id.startsWith("f")) return "flap";
  if (id.startsWith("re")) return "river";
  if (id.startsWith("s")) return "stretch-device";
  return "root";
}

function nodeIdFromTag(id: string): number | null {
  if (id.startsWith("f")) return Number(id.slice(1));
  const river = /^re(\d+),(\d+)$/.exec(id);
  if (river) return Number(river[2]);
  return null;
}

function objectRank(kind: BoxPleatedLayoutObject["kind"]): number {
  if (kind === "river") return 0;
  if (kind === "flap") return 1;
  if (kind === "stretch-device") return 2;
  return 3;
}

function statsFor(sampled: SampledTree, objects: BoxPleatedLayoutObject[], sheet: BoxPleatedBounds): BoxPleatedPacking["stats"] {
  const ridges = objects.flatMap((object) => object.ridges);
  return {
    leaves: sampled.leafIds.length,
    treeEdges: sampled.nodes.filter((node) => node.parentId !== null).length,
    layoutObjects: objects.length,
    hingeContours: objects.reduce((sum, object) => sum + object.contours.length, 0),
    ridgeCreases: ridges.length,
    axisParallelCreases: objects.reduce((sum, object) => sum + object.axisParallel.length, 0),
    stretchDevices: objects.filter((object) => object.kind === "stretch-device").length,
    offGridRidgeCreases: ridges.filter((ridge) => !isPure4590Line(ridge)).length,
    sheetCells: sheet.width * sheet.height,
  };
}

function validatePacking(packing: BoxPleatedPacking): string[] {
  const errors: string[] = [];
  if (packing.layout.patternNotFound) errors.push("BP Studio core did not find a stretch pattern");
  if (packing.constraints.noStretches) {
    if (packing.stats.stretchDevices > 0) errors.push(`no-stretch packing has ${packing.stats.stretchDevices} stretch devices`);
    if (packing.stats.axisParallelCreases > 0) {
      errors.push(`no-stretch packing has ${packing.stats.axisParallelCreases} stretch axis-parallel creases`);
    }
    if (packing.stats.offGridRidgeCreases > 0) {
      errors.push(`no-stretch packing has ${packing.stats.offGridRidgeCreases} non-45/90 ridge creases`);
    }
  }
  if (packing.flaps.length !== packing.tree.leafIds.length) errors.push("optimizer did not return one flap per leaf");
  for (const flap of packing.flaps) {
    if (flap.x < 0 || flap.y < 0 || flap.x > packing.sheet.width || flap.y > packing.sheet.height) {
      errors.push(`flap ${flap.id} anchor is outside the sheet`);
    }
    if (flap.radius < 0) errors.push(`flap ${flap.id} has negative radius`);
  }
  for (let i = 0; i < packing.flaps.length; i++) {
    for (let j = i + 1; j < packing.flaps.length; j++) {
      const a = packing.flaps[i];
      const b = packing.flaps[j];
      const clearance = roundedRectDistance(a, b) - a.radius - b.radius;
      if (clearance < -1e-9) {
        errors.push(`flap radius outlines ${a.id} and ${b.id} overlap by ${(-clearance).toFixed(3)}`);
      }
    }
  }
  for (const [aId, bId, distance] of distMapFromPacking(packing)) {
    const a = packing.flaps.find((flap) => flap.id === aId);
    const b = packing.flaps.find((flap) => flap.id === bId);
    if (!a || !b) continue;
    const violation = roundedConstraint(a, b, distance);
    if (violation > 1e-9) {
      errors.push(`flaps ${aId} and ${bId} violate BP Studio tree-distance constraint by ${violation.toFixed(3)}`);
    }
  }
  const flapObjects = new Set(packing.layout.objects.filter((object) => object.kind === "flap").map((object) => object.nodeId));
  const leafIds = new Set(packing.tree.leafIds);
  for (const leafId of packing.tree.leafIds) {
    if (!flapObjects.has(leafId)) errors.push(`missing BP Studio flap graphics for leaf ${leafId}`);
  }
  for (const flapObjectId of flapObjects) {
    if (flapObjectId === null || !leafIds.has(flapObjectId)) {
      errors.push(`unexpected BP Studio flap graphics for non-leaf node ${flapObjectId}`);
    }
  }
  return [...new Set(errors)];
}

function initialVectors(sampled: SampledTree, rng: SeededRandom): Array<{ x: number; y: number }> {
  const count = sampled.leafIds.length;
  if (sampled.noStretches && sampled.symmetry === "none") {
    // Grid-native placement: the tree sampler emits leaves as sibling pairs
    // sharing a hub, so the arrangement BP Studio box-pleats cleanly is the
    // canonical two-band layout - each sibling pair straddles a central axis
    // while hubs spread along it. Every resulting neighbour separation is axial
    // or 45-degree, so BP Studio needs no stretch devices. We pick the band
    // axis once per sample (not per flap) to keep the whole layout coherent;
    // the random per-leaf edge lengths still break exact mirror symmetry and
    // supply diversity. This avoids the rejection-sampling cliff that scattered
    // radial layouts fell off of.
    const horizontal = rng.next() < 0.5;
    return sampled.leafIds.map((_, index) => {
      const band = (Math.floor(index / 2) + 1) / (Math.ceil(count / 2) + 1);
      const side = index % 2 === 0 ? 0.22 : 0.78;
      const jitter = rng.float(-0.02, 0.02);
      return horizontal ? { x: band + jitter, y: side } : { x: side, y: band + jitter };
    });
  }
  if (sampled.symmetry === "vertical" || sampled.symmetry === "horizontal") {
    const horizontal = sampled.symmetry === "horizontal";
    return sampled.leafIds.map((_, index) => {
      const band = (Math.floor(index / 2) + 1) / (Math.ceil(count / 2) + 1);
      const side = index % 2 === 0 ? 0.22 : 0.78;
      return {
        x: horizontal ? band : side,
        y: horizontal ? side : band,
      };
    });
  }
  const radius = 0.34;
  return sampled.leafIds.map((_, index) => {
    const theta = (Math.PI * 2 * index) / count + rng.float(-0.12, 0.12);
    return {
      x: 0.5 + Math.cos(theta) * radius,
      y: 0.5 + Math.sin(theta) * radius,
    };
  });
}

function distMap(sampled: SampledTree): Array<[number, number, number]> {
  const result: Array<[number, number, number]> = [];
  for (let i = 0; i < sampled.leafIds.length; i++) {
    for (let j = i + 1; j < sampled.leafIds.length; j++) {
      result.push([
        sampled.leafIds[i],
        sampled.leafIds[j],
        distanceBetween(sampled, sampled.leafIds[i], sampled.leafIds[j]),
      ]);
    }
  }
  return result;
}

function distMapFromPacking(packing: BoxPleatedPacking): Array<[number, number, number]> {
  const childrenByParent = new Map<number, number[]>();
  for (const node of packing.tree.nodes) {
    if (node.parentId !== null) {
      const children = childrenByParent.get(node.parentId) ?? [];
      children.push(node.id);
      childrenByParent.set(node.parentId, children);
    }
  }
  const sampled: SampledTree = {
    nodes: packing.tree.nodes.map((node) => ({
      ...node,
      children: childrenByParent.get(node.id) ?? [],
    })),
    leafIds: packing.tree.leafIds,
    symmetry: packing.symmetry,
    optimizerDistanceScale: packing.tree.optimizerDistanceScale,
    noStretches: packing.constraints.noStretches,
  };
  return distMap(sampled);
}

function distanceBetween(sampled: SampledTree, a: number, b: number): number {
  const lca = lcaOf(sampled, a, b);
  return distanceToAncestor(sampled, a, lca) + distanceToAncestor(sampled, b, lca);
}

function distanceToAncestor(sampled: SampledTree, id: number, ancestor: number): number {
  let cursor = sampled.nodes[id];
  let distance = 0;
  while (cursor.id !== ancestor) {
    distance += cursor.lengthToParent;
    if (cursor.parentId === null) throw new Error(`Node ${ancestor} is not an ancestor of ${id}`);
    cursor = sampled.nodes[cursor.parentId];
  }
  return distance;
}

function lcaOf(sampled: SampledTree, a: number, b: number): number {
  const ancestors = new Set<number>();
  let cursor: MutableTreeNode | undefined = sampled.nodes[a];
  while (cursor) {
    ancestors.add(cursor.id);
    cursor = cursor.parentId === null ? undefined : sampled.nodes[cursor.parentId];
  }
  cursor = sampled.nodes[b];
  while (cursor) {
    if (ancestors.has(cursor.id)) return cursor.id;
    cursor = cursor.parentId === null ? undefined : sampled.nodes[cursor.parentId];
  }
  return 0;
}

function roundedConstraint(a: BoxPleatedFlap, b: BoxPleatedFlap, distance: number): number {
  const dx = intervalDistance(a.x, a.width, b.x, b.width);
  const dy = intervalDistance(a.y, a.height, b.y, b.height);
  return distance * distance - dx * dx - dy * dy;
}

function roundedRectDistance(a: BoxPleatedFlap, b: BoxPleatedFlap): number {
  const dx = intervalDistance(a.x, a.width, b.x, b.width);
  const dy = intervalDistance(a.y, a.height, b.y, b.height);
  return Math.hypot(dx, dy);
}

function intervalDistance(l1: number, w1: number, l2: number, w2: number): number {
  return Math.max(l1 - l2 - w2, 0) + Math.min(l1 + w1 - l2, 0);
}

function withoutZeroLength(lines: BpStudioLayoutLine[]): BpStudioLayoutLine[] {
  return lines.filter(([a, b]) => a.x !== b.x || a.y !== b.y);
}

function isPure4590Line(line: BpStudioLayoutLine): boolean {
  const [a, b] = line;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return dx === 0 || dy === 0 || Math.abs(dx) === Math.abs(dy);
}

function renderGrid(sheet: BoxPleatedBounds): string {
  const lines: string[] = [];
  for (let x = 0; x <= sheet.width; x++) {
    lines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${sheet.height}" class="grid"/>`);
  }
  for (let y = 0; y <= sheet.height; y++) {
    lines.push(`<line x1="0" y1="${y}" x2="${sheet.width}" y2="${y}" class="grid"/>`);
  }
  return lines.join("\n");
}

function renderHingeContours(object: BoxPleatedLayoutObject): string {
  // BP Studio's flap and river fold-region contours - these are the "boxes"
  // around flap circles and the river polygons that BP Studio draws and exports
  // as FOLD auxiliary lines. Style flaps and rivers distinctly so they read
  // separately from the blue flap radius circles.
  const className = object.kind === "flap" ? "flap-box" : object.kind === "river" ? "river-box" : "hinge";
  return object.contours.map((contour) => {
    const paths = [pathData(contour.outer), ...(contour.inner ?? []).map(pathData)];
    return paths.map((path) => `<path d="${path}" class="${className}"/>`).join("\n");
  }).join("\n");
}

function renderFlapCircle(flap: BoxPleatedFlap): string {
  const x = flap.x - flap.radius;
  const y = flap.y - flap.radius;
  const width = flap.width + flap.radius * 2;
  const height = flap.height + flap.radius * 2;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${flap.radius}" ry="${flap.radius}" class="circle"/>`;
}

function renderRidges(object: BoxPleatedLayoutObject): string {
  return object.ridges.map((line) => renderLine(line, "ridge")).join("\n");
}

function renderAxisParallels(object: BoxPleatedLayoutObject): string {
  return object.axisParallel.map((line) => renderLine(line, "axis-parallel")).join("\n");
}

function renderFlapDots(flap: BoxPleatedFlap): string {
  const points = uniquePoints([
    { x: flap.x, y: flap.y },
    { x: flap.x + flap.width, y: flap.y },
    { x: flap.x + flap.width, y: flap.y + flap.height },
    { x: flap.x, y: flap.y + flap.height },
  ]);
  return points.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="0.13" class="dot"/>`).join("\n");
}

function renderLine(line: BpStudioLayoutLine, className: string): string {
  return `<line x1="${line[0].x}" y1="${line[0].y}" x2="${line[1].x}" y2="${line[1].y}" class="${className}"/>`;
}

function pathData(path: Array<{ x: number; y: number }>): string {
  if (!path.length) return "";
  return `M${path.map((point) => `${point.x},${point.y}`).join("L")}Z`;
}

function uniquePoints(points: Array<{ x: number; y: number }>): Array<{ x: number; y: number }> {
  const seen = new Set<string>();
  return points.filter((point) => {
    const key = `${point.x},${point.y}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function sanitizeId(value: string): string {
  return value.replaceAll(/[^a-zA-Z0-9_-]/g, "-");
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

function leafCountForTarget(numCreases: number): number {
  if (numCreases < 220) return 4;
  if (numCreases < 420) return 6;
  return 8;
}

function scaledLength(length: number, scale: number): number {
  return Math.max(length, Math.ceil(length * scale));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
