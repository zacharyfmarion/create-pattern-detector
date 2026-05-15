import type { AdapterMetadata, AdapterSpec } from "./bp-studio-realistic.ts";
import type { BPStudioAdapterSpec } from "./bp-studio-spec.ts";

export interface BPStudioLayoutGraphNode {
  adapterId: number;
  nodeId: string;
  label: string;
  point: { x: number; y: number };
  normalized: { x: number; y: number };
  kind: "terminal" | "hub";
  source: "bp-studio-optimized-flap" | "bp-studio-inferred-internal";
}

export interface BPStudioLayoutGraphEdge {
  id: string;
  fromAdapterId: number;
  toAdapterId: number;
  from: string;
  to: string;
  length: number;
}

export interface BPStudioLayoutGraph {
  sheet: { width: number; height: number };
  nodes: BPStudioLayoutGraphNode[];
  edges: BPStudioLayoutGraphEdge[];
}

type AdapterLayout = NonNullable<AdapterMetadata["layout"]> |
  NonNullable<AdapterMetadata["optimizedLayout"]> |
  NonNullable<AdapterMetadata["inputLayout"]>;

export function buildBPStudioLayoutGraph(
  spec: BPStudioAdapterSpec,
  options: { adapterSpec?: AdapterSpec; adapterMetadata?: AdapterMetadata } = {},
): BPStudioLayoutGraph | undefined {
  const layout = chooseBPStudioLayout(options.adapterMetadata);
  if (!layout) return undefined;
  const sheet = {
    width: Math.max(1, Number(layout.sheet?.width ?? spec.sheet.width)),
    height: Math.max(1, Number(layout.sheet?.height ?? spec.sheet.height)),
  };
  const adapterToNodeId = adapterNodeIdLookup(spec, options.adapterSpec);
  const nodeById = new Map(spec.tree.nodes.map((node) => [node.id, node]));
  const terminalNodeIds = new Set(spec.layout.flaps.map((flap) => flap.nodeId));
  const known = new Map<number, { x: number; y: number }>();
  const knownSource = new Map<number, BPStudioLayoutGraphNode["source"]>();
  for (const flap of layout.flaps ?? []) {
    known.set(flap.id, {
      x: flap.x + (flap.width ?? 0) / 2,
      y: flap.y + (flap.height ?? 0) / 2,
    });
    knownSource.set(flap.id, "bp-studio-optimized-flap");
  }

  const edgeSpecs = layout.edges ?? [];
  const adapterIds = new Set<number>();
  for (const edge of edgeSpecs) {
    adapterIds.add(edge.n1);
    adapterIds.add(edge.n2);
  }
  for (const flap of layout.flaps ?? []) adapterIds.add(flap.id);

  const inferred = inferInternalPoints(adapterIds, edgeSpecs, known, sheet);
  const allPoints = new Map([...known, ...inferred]);
  const nodes = [...adapterIds].sort((a, b) => a - b).flatMap((adapterId): BPStudioLayoutGraphNode[] => {
    const nodeId = adapterToNodeId.get(adapterId) ?? String(adapterId);
    const point = allPoints.get(adapterId);
    if (!point) return [];
    const terminal = terminalNodeIds.has(nodeId);
    return [{
      adapterId,
      nodeId,
      label: nodeById.get(nodeId)?.label ?? nodeId,
      point,
      normalized: {
        x: point.x / sheet.width,
        y: point.y / sheet.height,
      },
      kind: terminal ? "terminal" : "hub",
      source: knownSource.get(adapterId) ?? "bp-studio-inferred-internal",
    }];
  });

  const specEdgeByNodes = new Map(
    spec.tree.edges.map((edge) => [unorderedNodeKey(edge.from, edge.to), edge.id]),
  );
  const edges = edgeSpecs.flatMap((edge, index): BPStudioLayoutGraphEdge[] => {
    const from = adapterToNodeId.get(edge.n1);
    const to = adapterToNodeId.get(edge.n2);
    if (!from || !to) return [];
    return [{
      id: specEdgeByNodes.get(unorderedNodeKey(from, to)) ?? `bp-studio-edge-${index}-${edge.n1}-${edge.n2}`,
      fromAdapterId: edge.n1,
      toAdapterId: edge.n2,
      from,
      to,
      length: edge.length,
    }];
  });
  return { sheet, nodes, edges };
}

export function chooseBPStudioLayout(metadata: AdapterMetadata | undefined): AdapterLayout | undefined {
  return metadata?.optimizedLayout ?? metadata?.layout ?? metadata?.inputLayout;
}

function adapterNodeIdLookup(spec: BPStudioAdapterSpec, adapterSpec?: AdapterSpec): Map<number, string> {
  if (adapterSpec?.nodeIdByAdapterId) {
    return new Map(Object.entries(adapterSpec.nodeIdByAdapterId).map(([adapterId, nodeId]) => [Number(adapterId), nodeId]));
  }
  const referenced = new Set<string>();
  for (const edge of spec.tree.edges) {
    if (edge.from === spec.tree.rootId || edge.to === spec.tree.rootId) continue;
    referenced.add(edge.from);
    referenced.add(edge.to);
  }
  for (const flap of spec.layout.flaps) referenced.add(flap.nodeId);
  return new Map(
    spec.tree.nodes
      .map((node) => node.id)
      .filter((id) => id !== spec.tree.rootId && referenced.has(id))
      .map((id, index) => [index, id]),
  );
}

function inferInternalPoints(
  adapterIds: Set<number>,
  edges: Array<{ n1: number; n2: number; length: number }>,
  known: Map<number, { x: number; y: number }>,
  sheet: { width: number; height: number },
): Map<number, { x: number; y: number }> {
  const unknownIds = [...adapterIds].filter((id) => !known.has(id));
  const points = new Map<number, { x: number; y: number }>();
  for (const id of unknownIds) {
    const neighbors = edges
      .flatMap((edge) => edge.n1 === id ? [edge.n2] : edge.n2 === id ? [edge.n1] : [])
      .map((neighborId) => known.get(neighborId))
      .filter((point): point is { x: number; y: number } => Boolean(point));
    points.set(id, neighbors.length ? averagePoint(neighbors) : { x: sheet.width / 2, y: sheet.height / 2 });
  }

  for (let iteration = 0; iteration < 300; iteration += 1) {
    const next = new Map(points);
    for (const id of unknownIds) {
      const current = points.get(id);
      if (!current) continue;
      let fx = 0;
      let fy = 0;
      let degree = 0;
      for (const edge of edges) {
        const neighborId = edge.n1 === id ? edge.n2 : edge.n2 === id ? edge.n1 : undefined;
        if (neighborId === undefined) continue;
        const neighbor = known.get(neighborId) ?? points.get(neighborId);
        if (!neighbor) continue;
        const dx = current.x - neighbor.x;
        const dy = current.y - neighbor.y;
        const distance = Math.hypot(dx, dy) || 1e-6;
        const error = distance - edge.length;
        fx -= (error * dx) / distance;
        fy -= (error * dy) / distance;
        degree += 1;
      }
      if (degree > 0) {
        next.set(id, {
          x: clamp(current.x + (fx / degree) * 0.18, 0, sheet.width),
          y: clamp(current.y + (fy / degree) * 0.18, 0, sheet.height),
        });
      }
    }
    points.clear();
    for (const [id, point] of next) points.set(id, point);
  }
  return points;
}

function averagePoint(points: Array<{ x: number; y: number }>): { x: number; y: number } {
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function unorderedNodeKey(a: string, b: string): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
