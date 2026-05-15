import type { AdapterMetadata } from "./bp-studio-realistic.ts";
import type { BPStudioAdapterSpec, BPStudioFlapPlacement } from "./bp-studio-spec.ts";

export interface BPStudioPackingCircle {
  adapterId: number;
  nodeId: string;
  label: string;
  center: { x: number; y: number };
  radius: number;
  radiusSource: "tree-edge-length" | "terminal-radius";
}

export interface BPStudioPackingMetrics {
  circleCount: number;
  overlapCount: number;
  outsideCount: number;
  minGap: number | null;
  sheetWidth: number;
  sheetHeight: number;
}

export interface BPStudioPackingValidation {
  ok: boolean;
  errors: string[];
  warnings: string[];
  circles: BPStudioPackingCircle[];
  metrics: BPStudioPackingMetrics;
}

type AdapterLayout = NonNullable<AdapterMetadata["layout"]> |
  NonNullable<AdapterMetadata["optimizedLayout"]> |
  NonNullable<AdapterMetadata["inputLayout"]>;

const EPSILON = 1e-8;

export function validateBPStudioPacking(
  spec: BPStudioAdapterSpec,
  metadata: AdapterMetadata,
): BPStudioPackingValidation {
  const layout = chooseBPStudioPackingLayout(metadata);
  if (!layout) {
    return emptyPackingValidation(spec, ["missing-bp-studio-layout"]);
  }
  return validateBPStudioPackingLayout(spec, layout);
}

export function validateBPStudioPackingLayout(
  spec: BPStudioAdapterSpec,
  layout: AdapterLayout,
): BPStudioPackingValidation {
  const sheetWidth = Math.max(1, Number(layout.sheet?.width ?? spec.sheet.width));
  const sheetHeight = Math.max(1, Number(layout.sheet?.height ?? spec.sheet.height));
  const circles = bpStudioPackingCircles(spec, layout);
  const errors: string[] = [];
  const warnings: string[] = [];
  let minGap: number | null = null;
  let overlapCount = 0;
  let outsideCount = 0;

  for (let aIndex = 0; aIndex < circles.length; aIndex += 1) {
    for (let bIndex = aIndex + 1; bIndex < circles.length; bIndex += 1) {
      const a = circles[aIndex];
      const b = circles[bIndex];
      const gap = distance(a.center, b.center) - (a.radius + b.radius);
      minGap = minGap === null ? gap : Math.min(minGap, gap);
      if (gap < -EPSILON) {
        overlapCount += 1;
        errors.push(
          `flap-circle-overlap:${a.nodeId}:${b.nodeId}:gap=${round3(gap)}:r=${round3(a.radius)}+${round3(b.radius)}`,
        );
      }
    }
  }

  for (const circle of circles) {
    const outside = Math.max(
      0,
      circle.radius - circle.center.x,
      circle.center.x + circle.radius - sheetWidth,
      circle.radius - circle.center.y,
      circle.center.y + circle.radius - sheetHeight,
    );
    if (outside > EPSILON) {
      outsideCount += 1;
      warnings.push(`flap-circle-outside-sheet:${circle.nodeId}:overflow=${round3(outside)}`);
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    circles,
    metrics: {
      circleCount: circles.length,
      overlapCount,
      outsideCount,
      minGap: minGap === null ? null : round(minGap),
      sheetWidth,
      sheetHeight,
    },
  };
}

export function bpStudioPackingCircles(
  spec: BPStudioAdapterSpec,
  layout: AdapterLayout,
): BPStudioPackingCircle[] {
  const adapterIds = adapterNodeIds(spec);
  const nodeByAdapterId = new Map([...adapterIds.entries()].map(([nodeId, adapterId]) => [adapterId, nodeId]));
  const terminalByNodeId = new Map(spec.layout.flaps.map((flap) => [flap.nodeId, flap]));
  const treeLengthByNodeId = flapLengthByNodeId(spec);
  const circles: BPStudioPackingCircle[] = [];
  for (const flap of layout.flaps ?? []) {
    const nodeId = nodeByAdapterId.get(flap.id);
    const terminal = nodeId ? terminalByNodeId.get(nodeId) : undefined;
    if (!nodeId || !terminal) continue;
    const treeLength = treeLengthByNodeId.get(nodeId);
    const radius = Math.max(0, Number(treeLength ?? terminal.terminalRadius ?? 0));
    if (radius <= EPSILON) continue;
    circles.push({
      adapterId: flap.id,
      nodeId,
      label: terminal.label,
      center: centerOfAdapterFlap(flap),
      radius,
      radiusSource: treeLength === undefined ? "terminal-radius" : "tree-edge-length",
    });
  }
  return circles;
}

export function chooseBPStudioPackingLayout(metadata: AdapterMetadata): AdapterLayout | undefined {
  return metadata.optimizedLayout ?? metadata.layout ?? metadata.inputLayout;
}

function emptyPackingValidation(spec: BPStudioAdapterSpec, errors: string[]): BPStudioPackingValidation {
  return {
    ok: false,
    errors,
    warnings: [],
    circles: [],
    metrics: {
      circleCount: 0,
      overlapCount: 0,
      outsideCount: 0,
      minGap: null,
      sheetWidth: spec.sheet.width,
      sheetHeight: spec.sheet.height,
    },
  };
}

function adapterNodeIds(spec: BPStudioAdapterSpec): Map<string, number> {
  const referenced = new Set<string>();
  for (const edge of adapterTreeEdges(spec)) {
    referenced.add(edge.from);
    referenced.add(edge.to);
  }
  for (const flap of spec.layout.flaps) referenced.add(flap.nodeId);
  return new Map(
    spec.tree.nodes
      .map((node) => node.id)
      .filter((id) => id !== spec.tree.rootId && referenced.has(id))
      .map((id, index) => [id, index]),
  );
}

function adapterTreeEdges(spec: BPStudioAdapterSpec): BPStudioAdapterSpec["tree"]["edges"] {
  return spec.tree.edges.filter((edge) => edge.from !== spec.tree.rootId && edge.to !== spec.tree.rootId);
}

function flapLengthByNodeId(spec: BPStudioAdapterSpec): Map<string, number> {
  const result = new Map<string, number>();
  for (const terminal of spec.layout.flaps) {
    const edge = terminalTreeEdge(spec, terminal);
    if (edge) result.set(terminal.nodeId, edge.length);
  }
  return result;
}

function terminalTreeEdge(
  spec: BPStudioAdapterSpec,
  terminal: BPStudioFlapPlacement,
): BPStudioAdapterSpec["tree"]["edges"][number] | undefined {
  return spec.tree.edges.find((edge) => edge.to === terminal.nodeId && edge.from !== spec.tree.rootId) ??
    spec.tree.edges.find((edge) => edge.from === terminal.nodeId && edge.to !== spec.tree.rootId);
}

function centerOfAdapterFlap(flap: { x: number; y: number; width?: number; height?: number }): { x: number; y: number } {
  return {
    x: flap.x + (flap.width ?? 0) / 2,
    y: flap.y + (flap.height ?? 0) / 2,
  };
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function round3(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, "");
}
