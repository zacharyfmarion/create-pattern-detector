import { runBpStudioLayout, type BpStudioLayoutContour, type BpStudioLayoutGraphics, type BpStudioLayoutLine } from "./bp-studio-layout.ts";
import { solveWithBpStudioOptimizer } from "./bp-studio-optimizer.ts";
import { SeededRandom } from "./random.ts";
import {
  fillBoxPleatedGapsFromGrid,
  insidePolygon,
  type GapRect,
  type OccupiedPolygon,
} from "./box-pleated-gap-fill.ts";
import type { GridPoint, OriSegment } from "./ori-parser.ts";

const EPS = 1e-9;
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
  /**
   * Filler ridges grouped per flap (parallel to `flaps`). Axial seeds come from
   * each filler flap's own straight skeleton, so a point where two adjacent
   * fillers merely touch on their shared boundary is never seeded.
   */
  ridgesByFlap: OriSegment[][];
  /**
   * Axial-seed centers for the filler flaps (flattened): each filler's straight-
   * skeleton convergence points, computed from its reflected full rectangle. These
   * are the authoritative centers - an edge/corner filler's center sits on (or just
   * beyond) the paper boundary, which the clipped `ridgesByFlap` no longer carries.
   */
  centers: GridPoint[];
  /** True when all empty paper was filled (the packing is complete, rule #4). */
  complete: boolean;
  /** Empty regions that could not be filled (the packing should be rejected). */
  unresolved: GapRect[];
}

/**
 * Fill the empty paper of a BP Studio packing with filler flaps so it consumes
 * all paper (ODS polygon-packing rule #4). The empty regions ("holes") come from
 * findPackingHoles - the gaps between facing flaps that exceed their river width.
 * Each hole is tiled with valid flaps; a hole that cannot be tiled (e.g. a 1-wide
 * fully interior strip) is reported incomplete and the packing is rejected.
 */
export function fillPackingGaps(packing: BoxPleatedPacking): PackingGapFill {
  const result = fillBoxPleatedGapsFromGrid(packing.sheet, packingEmptyGrid(packing));
  return {
    flaps: result.flaps,
    ridges: result.ridges,
    ridgesByFlap: result.ridgesByFlap,
    centers: result.centersByFlap.flat(),
    complete: result.resolved,
    unresolved: result.unresolved,
  };
}

/**
 * The cells (grid centres) that a river band covers. Every internal tree node is
 * a river: its band is the children's subtree grown outward by the node's edge
 * length (BP's RoughContour expansion). But that expansion fills interior gaps
 * between sibling flaps too, so we keep only the OUTWARD ring: a cell within the
 * width of the subtree is river unless the subtree sandwiches it (subtree cells on
 * opposite axis-sides), which marks it as an interior hole instead.
 *
 * Returns a map from each river cell ("x,y") to the tree node that owns it - the
 * river whose subtree is nearest - so callers (and the debug renderer) can tell
 * adjacent rivers apart.
 */
export function packingRiverCells(packing: BoxPleatedPacking): Map<string, number> {
  const W = Math.round(packing.sheet.width);
  const H = Math.round(packing.sheet.height);
  const childrenOf = new Map<number, number[]>();
  for (const n of packing.tree.nodes) childrenOf.set(n.id, []);
  for (const n of packing.tree.nodes) if (n.parentId != null) childrenOf.get(n.parentId)?.push(n.id);
  const flapById = new Map(packing.flaps.map((f) => [f.id, f]));
  // Real flap polygons keyed by their tree node, so the river subtree matches the
  // coverage test (a cell inside a flap's bounding square but outside its real
  // polygon - common next to a stretch - is not treated as flap).
  const flapPolysByNode = new Map<number, OccupiedPolygon[]>();
  for (const object of packing.layout.objects) {
    if (object.kind !== "flap" || object.nodeId == null) continue;
    flapPolysByNode.set(
      object.nodeId,
      object.contours.map((c) => ({ outer: c.outer, inner: c.inner })),
    );
  }
  const leavesUnder = (id: number): number[] => {
    const kids = childrenOf.get(id) ?? [];
    if (kids.length === 0) return flapById.has(id) ? [id] : [];
    return kids.flatMap(leavesUnder);
  };
  const key = (x: number, y: number): string => `${x},${y}`;

  const owner = new Map<string, { node: number; dist: number }>();
  for (const v of packing.tree.nodes) {
    const kids = childrenOf.get(v.id) ?? [];
    if (kids.length === 0 || v.parentId == null) continue; // skip flaps (leaves) and the root
    const width = Math.round(v.lengthToParent);
    if (width <= 0) continue;

    const polys = leavesUnder(v.id).flatMap((leaf) => flapPolysByNode.get(leaf) ?? []);
    if (polys.length === 0) continue;
    const subtree = new Set<string>();
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = { x: x + 0.5, y: y + 0.5 };
        if (polys.some((p) => insidePolygon(c, p))) subtree.add(key(x, y));
      }
    }
    if (subtree.size === 0) continue;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (subtree.has(key(x, y))) continue;
        let dist = Infinity;
        for (let dy = -width; dy <= width; dy++) {
          for (let dx = -width; dx <= width; dx++) {
            if (subtree.has(key(x + dx, y + dy))) dist = Math.min(dist, Math.max(Math.abs(dx), Math.abs(dy)));
          }
        }
        if (!Number.isFinite(dist)) continue;
        // Sandwiched = the subtree lies on opposite sides (anywhere along the row
        // or column), so this cell is an interior gap (a hole), not the outward
        // river ring. Scan the full row/column, not just within the width, so a
        // gap taller than the river width is still recognised as enclosed.
        let up = false;
        let down = false;
        let left = false;
        let right = false;
        for (let k = y + 1; k < H && !up; k++) if (subtree.has(key(x, k))) up = true;
        for (let k = y - 1; k >= 0 && !down; k--) if (subtree.has(key(x, k))) down = true;
        for (let k = x + 1; k < W && !right; k++) if (subtree.has(key(k, y))) right = true;
        for (let k = x - 1; k >= 0 && !left; k--) if (subtree.has(key(k, y))) left = true;
        if ((up && down) || (left && right)) continue;
        const prev = owner.get(key(x, y));
        if (!prev || dist < prev.dist) owner.set(key(x, y), { node: v.id, dist });
      }
    }
  }
  return new Map([...owner].map(([k, v]) => [k, v.node]));
}

/**
 * The empty (hole) cells of a packing: paper not covered by a flap polygon, a
 * stretch device, or a river band. Real flap/stretch polygons are used for
 * coverage so non-rectangular flaps near Pythagorean stretches are not mis-counted
 * as empty.
 */
export function packingEmptyGrid(packing: BoxPleatedPacking): boolean[][] {
  const W = Math.round(packing.sheet.width);
  const H = Math.round(packing.sheet.height);
  const covers: OccupiedPolygon[] = [];
  for (const object of packing.layout.objects) {
    if (object.kind !== "flap" && object.kind !== "stretch-device") continue;
    for (const contour of object.contours) covers.push({ outer: contour.outer, inner: contour.inner });
  }
  const rivers = packingRiverCells(packing);
  const empty: boolean[][] = Array.from({ length: H }, () => new Array<boolean>(W).fill(false));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (rivers.has(`${x},${y}`)) continue;
      const c = { x: x + 0.5, y: y + 0.5 };
      if (!covers.some((s) => insidePolygon(c, s))) empty[y][x] = true;
    }
  }
  return empty;
}

/** The packing's holes as the bounding rectangles of its connected empty regions. */
export function findPackingHoles(packing: BoxPleatedPacking): GapRect[] {
  const empty = packingEmptyGrid(packing);
  const H = empty.length;
  const W = H > 0 ? empty[0].length : 0;
  const seen = Array.from({ length: H }, () => new Array<boolean>(W).fill(false));
  const holes: GapRect[] = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!empty[y][x] || seen[y][x]) continue;
      let x0 = x;
      let y0 = y;
      let x1 = x + 1;
      let y1 = y + 1;
      const stack: Array<[number, number]> = [[x, y]];
      seen[y][x] = true;
      while (stack.length) {
        const [cx, cy] = stack.pop()!;
        x0 = Math.min(x0, cx);
        y0 = Math.min(y0, cy);
        x1 = Math.max(x1, cx + 1);
        y1 = Math.max(y1, cy + 1);
        for (const [nx, ny] of [
          [cx + 1, cy],
          [cx - 1, cy],
          [cx, cy + 1],
          [cx, cy - 1],
        ] as Array<[number, number]>) {
          if (nx >= 0 && ny >= 0 && nx < W && ny < H && empty[ny][nx] && !seen[ny][nx]) {
            seen[ny][nx] = true;
            stack.push([nx, ny]);
          }
        }
      }
      holes.push({ x0, y0, x1, y1 });
    }
  }
  return holes;
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
      const { width, height } = evenShorterFlap(rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 0, 1, 2, 3]));
      nodes.push({
        id: leafId,
        parentId: hubId,
        kind: "terminal",
        label: `leaf-${hubIndex}-${leafIndex}`,
        lengthToParent: scaledLength(rng.int(3, 7), optimizerDistanceScale),
        children: [],
        width,
        height,
      });
      nodes[hubId].children.push(leafId);
      leafIds.push(leafId);
    }
  }

  return { nodes, leafIds, symmetry, optimizerDistanceScale, noStretches };
}

/**
 * Constrain a flap's rectangular cross-section so its box-pleat molecule lands
 * on the grid.
 *
 * A leaf flap occupies a `width x height` rectangle of paper that must be creased
 * with its own straight skeleton (a spine plus 45-degree corner miters) to
 * collapse flat. The spine runs down the middle of the SHORTER side, so it sits
 * on an integer grid line only when the shorter side is even. A flap with an odd
 * shorter side >= 1 (e.g. 1x2) would put its spine on a half-grid line - a
 * sub-grid crease, which box pleating forbids - and there is no valid on-grid
 * molecule for it. We therefore never sample such a flap: reduce an odd shorter
 * side to the next even value (1 -> 0, 3 -> 2). A zero dimension is a degenerate
 * (point/edge) flap with no interior to crease and is always fine; valid
 * mixed-parity rectangles like 2x3 (shorter side 2) are left untouched.
 *
 * (When we later feed in real treemaker trees, which we do not control, the same
 * rule needs to live as a validation/rejection check on the generated packing.)
 */
export function evenShorterFlap(width: number, height: number): { width: number; height: number } {
  if (Math.min(width, height) % 2 === 1) {
    if (width <= height) width -= 1;
    else height -= 1;
  }
  return { width, height };
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
