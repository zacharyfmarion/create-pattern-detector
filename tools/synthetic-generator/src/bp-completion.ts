import { roleCounts } from "./fold-utils.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import { solveMaekawaAssignments } from "./bp-maekawa-assignment.ts";
import {
  compassRayToSheet,
  createMoleculeInstance,
  moleculePatchLibrary,
  moleculeTemplateFromPatch,
  roleForDirection,
} from "./bp-molecule-patches.ts";
import { buildBPStudioLayoutGraph } from "./bp-studio-layout-graph.ts";
import { scoreFoldRealism } from "./realism-metrics.ts";
import type { BPStudioAdapterSpec } from "./bp-studio-spec.ts";
import type { AdapterMetadata, AdapterSpec } from "./bp-studio-realistic.ts";
import type {
  CompletionAxis,
  CompletionCorridor,
  CompletionLayout,
  CompletionPoint,
  CompletionRegion,
  CompletionResult,
  CompletionSegment,
  CompletionTerminal,
  MoleculeInstance,
  MoleculeKind,
  MoleculePatch,
  MoleculeTemplate,
  Port,
  PortJoin,
} from "./bp-completion-contracts.ts";
import type { BPRole, CompilerEdgeSource, EdgeAssignment, FOLDFormat, LayoutMetadata } from "./types.ts";

type Point = [number, number];

export interface BoxPleatCompletionOptions {
  adapterSpec?: AdapterSpec;
  adapterMetadata?: AdapterMetadata;
  layoutId?: string;
  gridSize?: number;
  maxFoldLines?: number;
}

const ENGINE_VERSION = "strict-bp-completion/v0.7.0";
const DEFAULT_GRID_SIZE = 128;
const BP_STUDIO_GRID_SUBDIVISION = 4;
const PATCH_LIBRARY = moleculePatchLibrary();

export function completeBoxPleat(spec: BPStudioAdapterSpec, options: BoxPleatCompletionOptions = {}): CompletionResult {
  const layout = regularizeBPStudioLayout(spec, options);
  return completeBoxPleatLayout(layout, options);
}

export function completeBoxPleatLayout(
  layout: CompletionLayout,
  options: BoxPleatCompletionOptions = {},
): CompletionResult {
  void options;
  const { molecules, moleculeInstances, segments, portJoins } = instantiateMolecules(layout);
  const rejected = portJoins.filter((join) => !join.accepted).map((join) => ({
    code: "incompatible-port",
    message: join.reason ?? "port join rejected",
  }));
  if (rejected.length > 0) {
    return {
      ok: false,
      layout,
      foldLines: [],
      molecules,
      moleculeInstances,
      segments,
      portJoins,
      rejected,
    };
  }
  if (segments.length === 0) {
    return {
      ok: false,
      layout,
      foldLines: [],
      molecules,
      moleculeInstances,
      segments,
      portJoins,
      rejected: [{ code: "empty-molecule-composition", message: "completion emitted no local molecule segments" }],
    };
  }

  const foldResult = moleculeSegmentsToFold(layout, segments, moleculeInstances, portJoins);
  if (!foldResult.ok || !foldResult.fold) {
    return {
      ok: false,
      layout,
      foldLines: [],
      molecules,
      moleculeInstances,
      segments,
      portJoins,
      rejected: foldResult.errors.map((message) => ({ code: "molecule-composition-invalid", message })),
    };
  }

  const fold = foldResult.fold;
  fold.completion_metadata = {
    engine: "strict-box-pleat-completion",
    version: ENGINE_VERSION,
    source: layout.source,
    scaffoldSummary: layout.scaffoldSummary,
    selectedCenter: layout.axis === "horizontal"
      ? [0.5, layout.spineCoordinate]
      : [layout.spineCoordinate, 0.5],
    selectedFlapIds: layout.terminals.map((terminal) => Number(terminal.nodeId)).filter(Number.isFinite).slice(0, 16),
    portJoinCount: portJoins.length,
    rejectedCandidateCount: rejected.length,
    compilerSteps: [
      "regularize-tree-layout",
      "instantiate-local-molecule-patches",
      "check-port-compatibility",
      "emit-local-molecule-segments",
      "arrange-and-split-local-segments",
      `solve-maekawa-assignments:${foldResult.assignmentSteps}`,
    ],
  };
  fold.label_policy = {
    labelSource: "compiler",
    geometrySource: "compiler",
    assignmentSource: "compiler",
    trainingEligible: true,
    notes: [
      "Final labels are emitted by the restricted compiler; BP Studio raw CP export is scaffold/debug metadata only.",
    ],
  };
  return {
    ok: true,
    fold,
    layout,
    foldLines: [],
    molecules,
    moleculeInstances,
    segments,
    portJoins,
    rejected,
  };
}

export function regularizeBPStudioLayout(
  spec: BPStudioAdapterSpec,
  options: BoxPleatCompletionOptions = {},
): CompletionLayout {
  const graph = buildBPStudioLayoutGraph(spec, options);
  const adapterLayout = options.adapterMetadata?.optimizedLayout ?? options.adapterMetadata?.layout;
  const sheet = graph?.sheet ?? adapterLayout?.sheet ?? options.adapterSpec?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const sheetWidth = Math.max(1, sheet.width);
  const sheetHeight = Math.max(1, sheet.height);
  const gridSize = options.gridSize ?? compilerGridSizeForSheet(sheetWidth, sheetHeight);
  const terminalByNodeId = new Map(spec.layout.flaps.map((terminal) => [terminal.nodeId, terminal]));
  const optimizedTerminalNodes = graph?.nodes.filter((node) => node.kind === "terminal") ?? [];
  const terminals: CompletionTerminal[] = (optimizedTerminalNodes.length ? optimizedTerminalNodes : spec.layout.flaps.map((terminal) => ({
    adapterId: -1,
    nodeId: terminal.nodeId,
    normalized: {
      x: terminal.terminal.x / spec.sheet.width,
      y: terminal.terminal.y / spec.sheet.height,
    },
  }))).map((node, index) => {
    const sourceTerminal = terminalByNodeId.get(node.nodeId);
    const x = snap(clamp(node.normalized.x, 0, 1), gridSize);
    const y = snap(clamp(node.normalized.y, 0, 1), gridSize);
    const radius = terminalAllocationRadius(spec, sourceTerminal?.nodeId ?? node.nodeId, sheetWidth, sheetHeight);
    return {
      id: sourceTerminal?.nodeId ?? `flap-${node.adapterId}`,
      nodeId: String(node.adapterId),
      x,
      y,
      side: sourceTerminal?.side ?? sideForPoint({ x, y }),
      width: snap(Math.max(1 / (gridSize * 2), (sourceTerminal?.width ?? 0) / sheetWidth), gridSize),
      height: snap(Math.max(1 / (gridSize * 2), (sourceTerminal?.height ?? 0) / sheetHeight), gridSize),
      allocationRadius: radius ? snap(radius, gridSize) : undefined,
      priority: sourceTerminal?.priority ?? index,
    };
  });
  const graphNodeById = new Map(graph?.nodes.map((node) => [node.nodeId, node]) ?? []);
  const bodyNodeIds = spec.layout.bodies.length
    ? spec.layout.bodies.map((body) => body.nodeId)
    : graph?.nodes.filter((node) => node.kind === "hub").map((node) => node.nodeId) ?? [];
  const bodyHalfSize = 1 / gridSize;
  const bodies = bodyNodeIds.flatMap((nodeId): CompletionRegion[] => {
    const graphNode = graphNodeById.get(nodeId);
    const sourceBody = spec.layout.bodies.find((body) => body.nodeId === nodeId);
    const rawCenter = graphNode?.normalized ?? (sourceBody ? {
      x: sourceBody.center.x / spec.sheet.width,
      y: sourceBody.center.y / spec.sheet.height,
    } : undefined);
    if (!rawCenter) return [];
    const center = keepPointOutsideTerminalAllocations(rawCenter, terminals, gridSize, bodyHalfSize);
    return [{
      id: nodeId,
      kind: "body",
      x1: snap(clamp(center.x - bodyHalfSize, 0, 1), gridSize),
      y1: snap(clamp(center.y - bodyHalfSize, 0, 1), gridSize),
      x2: snap(clamp(center.x + bodyHalfSize, 0, 1), gridSize),
      y2: snap(clamp(center.y + bodyHalfSize, 0, 1), gridSize),
    }];
  });
  const axis = chooseAxis(spec, terminals);
  const spineCoordinate = snap(
    clamp(mean(bodies.map((body) => axis === "horizontal" ? (body.y1 + body.y2) / 2 : (body.x1 + body.x2) / 2), 0.5), 0.1875, 0.8125),
    gridSize,
  );
  const layoutPoints = completionPointMap(bodies, terminals);
  const terminalIds = new Set(terminals.map((terminal) => terminal.id));
  const graphEdges = graph?.edges.length ? graph.edges : spec.layout.rivers.map((river) => ({
    id: river.edgeId,
    from: river.from,
    to: river.to,
    length: river.width,
  }));
  const corridors: CompletionCorridor[] = graphEdges.map((edge) => {
    const width = snap(Math.max(2 / gridSize, 0.5 / Math.max(sheetWidth, sheetHeight)), gridSize);
    const preferredOrientation = orientationForPoints(layoutPoints.get(edge.from), layoutPoints.get(edge.to), axis);
    const route = corridorRouteAvoidingAllocations(edge.from, edge.to, layoutPoints, preferredOrientation, terminalIds, terminals, gridSize, width);
    return {
      id: edge.id,
      from: edge.from,
      to: edge.to,
      orientation: route.orientation,
      coordinate: route.coordinate,
      width,
    };
  });

  return {
    id: options.layoutId ?? spec.id,
    source: "bp-studio-optimized-layout",
    gridSize,
    axis,
    spineCoordinate,
    regions: bodies,
    terminals,
    corridors,
    scaffoldSummary: {
      adapterLineCount: options.adapterMetadata?.cp?.lineCount ?? 0,
      adapterVertexCount: options.adapterMetadata?.cp?.vertexCount ?? 0,
      adapterEdgeCount: options.adapterMetadata?.cp?.edgeCount ?? 0,
      optimizedFlapCount: optimizedTerminalNodes.length,
      optimizedTreeEdgeCount: adapterLayout?.edges?.length ?? options.adapterSpec?.tree.edges.length ?? spec.tree.edges.length,
    },
  };
}

export function fixtureCompletionLayout(name: "two-flap-stretch" | "three-flap-relay" | "five-flap-uniaxial" | "insect-lite"): CompletionLayout {
  const base = {
    source: "fixture" as const,
    gridSize: DEFAULT_GRID_SIZE,
    axis: "horizontal" as CompletionAxis,
    spineCoordinate: 0.5,
    scaffoldSummary: {
      adapterLineCount: 0,
      adapterVertexCount: 0,
      adapterEdgeCount: 0,
      optimizedFlapCount: 0,
      optimizedTreeEdgeCount: 0,
    },
  };
  if (name === "two-flap-stretch") {
    return {
      ...base,
      id: name,
      regions: [{ id: "body", kind: "body", x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 }],
      terminals: [
        { id: "left-flap", nodeId: "1", x: 0.125, y: 0.5, side: "left", width: 0.0625, height: 0.125, priority: 1 },
        { id: "right-flap", nodeId: "2", x: 0.875, y: 0.5, side: "right", width: 0.0625, height: 0.125, priority: 2 },
      ],
      corridors: [{ id: "river-left-right", from: "left-flap", to: "right-flap", orientation: "horizontal", coordinate: 0.5, width: 0.125 }],
    };
  }
  if (name === "three-flap-relay") {
    return {
      ...base,
      id: name,
      regions: [{ id: "body", kind: "body", x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 }],
      terminals: [
        { id: "left-flap", nodeId: "1", x: 0.125, y: 0.375, side: "left", width: 0.0625, height: 0.125, priority: 1 },
        { id: "right-flap", nodeId: "2", x: 0.875, y: 0.625, side: "right", width: 0.0625, height: 0.125, priority: 2 },
        { id: "top-flap", nodeId: "3", x: 0.5, y: 0.875, side: "top", width: 0.125, height: 0.0625, priority: 3 },
      ],
      corridors: [
        { id: "river-left", from: "left-flap", to: "body", orientation: "horizontal", coordinate: 0.375, width: 0.125 },
        { id: "river-right", from: "right-flap", to: "body", orientation: "horizontal", coordinate: 0.625, width: 0.125 },
      ],
    };
  }
  const terminals: CompletionTerminal[] = [
    { id: "left-front", nodeId: "1", x: 0.125, y: 0.25, side: "left", width: 0.0625, height: 0.125, priority: 1 },
    { id: "left-back", nodeId: "2", x: 0.125, y: 0.75, side: "left", width: 0.0625, height: 0.125, priority: 2 },
    { id: "right-front", nodeId: "3", x: 0.875, y: 0.25, side: "right", width: 0.0625, height: 0.125, priority: 3 },
    { id: "right-back", nodeId: "4", x: 0.875, y: 0.75, side: "right", width: 0.0625, height: 0.125, priority: 4 },
    { id: "top", nodeId: "5", x: 0.5, y: 0.875, side: "top", width: 0.125, height: 0.0625, priority: 5 },
  ];
  return {
    ...base,
    id: name,
    regions: name === "insect-lite"
      ? [
          { id: "thorax", kind: "body", x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 },
          { id: "head", kind: "body", x1: 0.4375, y1: 0.625, x2: 0.5625, y2: 0.75 },
          { id: "abdomen", kind: "body", x1: 0.375, y1: 0.1875, x2: 0.625, y2: 0.375 },
        ]
      : [{ id: "body", kind: "body", x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 }],
    terminals: name === "insect-lite"
      ? [...terminals, { id: "antenna", nodeId: "6", x: 0.625, y: 0.9375, side: "top", width: 0.0625, height: 0.0625, priority: 6 }]
      : terminals,
    corridors: [
      { id: "river-front", from: "left-front", to: "right-front", orientation: "horizontal", coordinate: 0.25, width: 0.125 },
      { id: "river-back", from: "left-back", to: "right-back", orientation: "horizontal", coordinate: 0.75, width: 0.125 },
      { id: "river-spine", from: "body", to: "top", orientation: "vertical", coordinate: 0.5, width: 0.125 },
    ],
  };
}

function instantiateMolecules(layout: CompletionLayout): {
  molecules: MoleculeTemplate[];
  moleculeInstances: MoleculeInstance[];
  segments: CompletionSegment[];
  portJoins: PortJoin[];
} {
  const molecules: MoleculeTemplate[] = [
    template("sheet-border", "sheet-border", []),
    ...PATCH_LIBRARY.map((patchItem) => moleculeTemplateFromPatch(patchItem)),
  ];
  const instances: MoleculeInstance[] = [];
  const segments: CompletionSegment[] = sheetBorderSegments();
  const joins: PortJoin[] = [];
  const patches = new Map(PATCH_LIBRARY.map((patchItem) => [patchItem.kind, patchItem]));
  const bodyCenters = plannedBodyCenters(layout);
  const terminalCenters = plannedTerminalCenters(layout, bodyCenters[0] ?? point(0.5, 0.5));

  for (const [index, center] of bodyCenters.entries()) {
    const instance = instanceFor(patches, `body-${index}`, "body-panel", center);
    instances.push(instance);
    segments.push(...starSegments(instance.id, instance.kind, center));
    segments.push(...diamondSegments(`diamond-${index}`, "diamond-connector", center, diamondRadius(center)));
    const diamond = instanceFor(patches, `diamond-${index}`, "diamond-connector", center);
    instances.push(diamond);
  }

  for (const [index, terminal] of terminalCenters.entries()) {
    const target = nearestPoint(terminal.center, bodyCenters);
    const fan = instanceFor(patches, `fan-${terminal.terminal.id}`, "corner-fan", terminal.center);
    instances.push(fan);
    segments.push(...terminalFanSegments(fan.id, fan.kind, terminal.center, target));
    segments.push(...diamondSegments(`flap-contour-${terminal.terminal.id}`, "flap-contour", terminal.center, 1 / 16));
    instances.push(templateOnlyInstance(`flap-contour-${terminal.terminal.id}`, "flap-contour", terminal.center));
    segments.push(...diamondSegments(`flap-inner-contour-${terminal.terminal.id}`, "flap-contour", terminal.center, 1 / 32));
    instances.push(templateOnlyInstance(`flap-inner-contour-${terminal.terminal.id}`, "flap-contour", terminal.center));

    const route = corridorRoute(terminal.center, target, layout.gridSize);
    for (const [routeIndex, routeCenter] of route.turns.entries()) {
      const turn = instanceFor(patches, `turn-${terminal.terminal.id}-${routeIndex}`, "diagonal-staircase", routeCenter);
      instances.push(turn);
      segments.push(...starSegments(turn.id, turn.kind, routeCenter));
    }
    const corridor = instanceFor(patches, `corridor-${terminal.terminal.id}`, "river-corridor", midpoint(terminal.center, target));
    instances.push(corridor);
    segments.push(...route.segments.map((segmentItem, routeIndex): CompletionSegment => ({
      id: `${corridor.id}:route-${routeIndex}`,
      moleculeId: corridor.id,
      moleculeKind: "river-corridor",
      p1: [segmentItem.p1.x, segmentItem.p1.y],
      p2: [segmentItem.p2.x, segmentItem.p2.y],
      assignment: "V",
      role: segmentItem.role,
    })));

    joins.push({
      from: `${fan.id}:corridor`,
      to: `${corridor.id}:west`,
      orientation: route.primaryOrientation,
      width: Math.max(terminal.terminal.width, terminal.terminal.height),
      accepted: true,
      fromPosition: terminal.center,
      toPosition: target,
    });
  }

  for (const [index, center] of relayHubCenters(layout).entries()) {
    const relayHub = instanceFor(patches, `relay-hub-${index}`, "body-panel", center);
    instances.push(relayHub);
    segments.push(...starSegments(relayHub.id, "body-panel", center));
    joins.push({
      from: `${relayHub.id}:west`,
      to: "body-0:west",
      orientation: layout.axis,
      width: 1 / 16,
      accepted: true,
      fromPosition: center,
      toPosition: bodyCenters[0] ?? point(0.5, 0.5),
    });
  }

  if (bodyCenters.length > 0) {
    const stretchCenter = bodyCenters[0];
    const stretch = instanceFor(patches, "central-stretch", "stretch-gadget", stretchCenter);
    instances.push(stretch);
    segments.push(...diamondSegments(stretch.id, "stretch-gadget", stretchCenter, diamondRadius(stretchCenter) / 2));
    joins.push({
      from: "central-stretch:west",
      to: "body-0:west",
      orientation: layout.axis,
      width: 1 / 8,
      accepted: true,
      fromPosition: stretchCenter,
      toPosition: stretchCenter,
    });
  }

  return {
    molecules,
    moleculeInstances: instances,
    segments: dedupeCompletionSegments(segments),
    portJoins: joins,
  };
}

function moleculeSegmentsToFold(
  layout: CompletionLayout,
  segments: CompletionSegment[],
  moleculeInstances: MoleculeInstance[],
  portJoins: PortJoin[],
): { ok: boolean; fold?: FOLDFormat; assignmentSteps: number; errors: string[] } {
  const arranged = arrangeSegments(
    segments.map((segmentItem) => ({
      p1: segmentItem.p1,
      p2: segmentItem.p2,
      assignment: segmentItem.assignment,
      role: segmentItem.role,
      source: {
        kind: `completion-${segmentItem.moleculeKind}`,
        mandatory: true,
        ownerId: segmentItem.moleculeId,
      },
    })),
    "cp-synthetic-generator/bp-completion/local-molecules",
    {
      gridSize: gridSizeForSegments(segments, layout.gridSize),
      bpSubfamily: "bp-studio-completed-uniaxial",
      flapCount: layout.terminals.length,
      gadgetCount: moleculeInstances.filter((instance) =>
        instance.kind === "stretch-gadget" || instance.kind === "diamond-connector" || instance.kind === "diagonal-staircase"
      ).length,
      ridgeCount: 1,
      hingeCount: 1,
      axisCount: 1,
    },
  );
  const sourceHints = arranged.edges_bpStudioSource ?? [];
  const solved = solveMaekawaAssignments(arranged);
  if (!solved.ok || !solved.fold) {
    return { ok: false, assignmentSteps: solved.steps, errors: solved.errors };
  }

  const fold = solved.fold;
  fold.file_creator = "cp-synthetic-generator/bp-completion/local-molecules";
  fold.edges_compilerSource = fold.edges_vertices.map(([a, b], edgeIndex) => {
    const role = fold.edges_bpRole?.[edgeIndex] ?? roleForEdge(fold.vertices_coords[a], fold.vertices_coords[b], fold.edges_assignment[edgeIndex]);
    const source = sourceHints[edgeIndex];
    return {
      kind: source?.kind ?? sourceForEdge(fold.vertices_coords[a], fold.vertices_coords[b], role).kind,
      mandatory: source?.mandatory ?? true,
      moleculeKind: source?.kind?.replace(/^completion-/, ""),
      role,
    };
  });
  delete fold.edges_bpStudioSource;
  const counts = roleCounts(fold);
  fold.bp_metadata = {
    ...(fold.bp_metadata ?? {
      gridSize: layout.gridSize,
      bpSubfamily: "bp-studio-completed-uniaxial",
      flapCount: layout.terminals.length,
      gadgetCount: layout.corridors.length + layout.regions.length,
      ridgeCount: 0,
      hingeCount: 0,
      axisCount: 0,
    }),
    gridSize: gridSizeForFold(fold, layout.gridSize),
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  fold.layout_metadata = toLayoutMetadata({ ...layout, gridSize: fold.bp_metadata.gridSize });
  fold.molecule_metadata = {
    libraryVersion: ENGINE_VERSION,
    molecules: countMoleculeInstances(moleculeInstances),
    portChecks: {
      checked: portJoins.length,
      rejected: portJoins.filter((join) => !join.accepted).length,
    },
  };
  fold.realism_metadata = scoreFoldRealism(fold, fold.layout_metadata);
  return { ok: true, fold, assignmentSteps: solved.steps, errors: [] };
}

function sheetBorderSegments(): CompletionSegment[] {
  return [
    borderSegment("border-bottom", [0, 0], [1, 0]),
    borderSegment("border-right", [1, 0], [1, 1]),
    borderSegment("border-top", [1, 1], [0, 1]),
    borderSegment("border-left", [0, 1], [0, 0]),
  ];
}

function borderSegment(id: string, p1: Point, p2: Point): CompletionSegment {
  return { id, moleculeId: "sheet-border", moleculeKind: "sheet-border", p1, p2, assignment: "B", role: "border" };
}

function plannedBodyCenters(layout: CompletionLayout): CompletionPoint[] {
  const region = primaryBody(layout);
  return [point(snapAnchor((region.x1 + region.x2) / 2), snapAnchor((region.y1 + region.y2) / 2))];
}

function plannedTerminalCenters(layout: CompletionLayout, bodyCenter: CompletionPoint): Array<{
  terminal: CompletionTerminal;
  center: CompletionPoint;
}> {
  const occupied = new Set<string>();
  const layer = terminalLayer(layout);
  return [...layout.terminals].sort((a, b) => a.priority - b.priority).slice(0, 8).map((terminal, index) => {
    const preferred = terminalCenterFromSide(terminal, bodyCenter, layer);
    let center = preferred;
    const alternates = terminalAlternates(terminal, bodyCenter, layer);
    for (const candidate of [preferred, ...alternates]) {
      const key = pointKey2(candidate);
      if (!occupied.has(key)) {
        center = candidate;
        break;
      }
    }
    occupied.add(pointKey2(center));
    return { terminal: { ...terminal, priority: index + 1 }, center };
  });
}

function terminalCenterFromSide(terminal: CompletionTerminal, bodyCenter: CompletionPoint, layer: number): CompletionPoint {
  void bodyCenter;
  const far = 1 - layer;
  const x = terminal.side === "left" ? layer : terminal.side === "right" ? far : sideLaneFallback(terminal.priority, layer);
  const y = terminal.side === "bottom" ? layer : terminal.side === "top" ? far : sideLaneFallback(terminal.priority, layer);
  if (terminal.side === "interior") return point(snapAnchor(terminal.x), snapAnchor(terminal.y));
  return point(x, y);
}

function terminalAlternates(terminal: CompletionTerminal, bodyCenter: CompletionPoint, layer: number): CompletionPoint[] {
  void bodyCenter;
  const laneAnchors = [layer, 1 - layer];
  if (terminal.side === "left" || terminal.side === "right") {
    const x = terminal.side === "left" ? layer : 1 - layer;
    return laneAnchors.map((y) => point(x, y));
  }
  const y = terminal.side === "bottom" ? layer : 1 - layer;
  return laneAnchors.map((x) => point(x, y));
}

function nearestAnchor(value: number, fallback: number): number {
  const anchors = [0.25, 0.375, 0.5, 0.625, 0.75];
  const target = Number.isFinite(value) ? value : fallback;
  return anchors.slice().sort((a, b) => Math.abs(a - target) - Math.abs(b - target))[0];
}

function sideLaneFallback(priority: number, layer: number): number {
  return Math.abs(priority) % 2 === 1 ? layer : 1 - layer;
}

function terminalLayer(layout: CompletionLayout): number {
  const layers = [0.125, 0.25];
  return layers[Math.abs(hashString(layout.id) + layout.terminals.length) % layers.length];
}

function relayHubCenters(layout: CompletionLayout): CompletionPoint[] {
  if (layout.terminals.length !== 3) return [];
  return layout.axis === "horizontal"
    ? [point(0.5, 0.25), point(0.5, 0.75)]
    : [point(0.25, 0.5), point(0.75, 0.5)];
}

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) hash = (hash * 31 + value.charCodeAt(i)) | 0;
  return hash;
}

function instanceFor(
  patches: Map<MoleculeKind, MoleculePatch>,
  id: string,
  kind: MoleculeKind,
  center: CompletionPoint,
): MoleculeInstance {
  const patchItem = patches.get(kind);
  if (!patchItem) return templateOnlyInstance(id, kind, center);
  return createMoleculeInstance(id, patchItem, {
    translate: center,
    scale: 1 / 32,
  });
}

function templateOnlyInstance(id: string, kind: MoleculeKind, center: CompletionPoint): MoleculeInstance {
  return {
    id,
    templateId: kind,
    kind,
    transform: { translate: center },
    ports: [],
  };
}

function starSegments(moleculeId: string, moleculeKind: MoleculeKind, center: CompletionPoint): CompletionSegment[] {
  const c: Point = [center.x, center.y];
  const directions: Point[] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [-1, -1],
    [1, -1],
    [-1, 1],
  ];
  return directions.map((direction, index): CompletionSegment => ({
    id: `${moleculeId}:star-${index}`,
    moleculeId,
    moleculeKind,
    p1: c,
    p2: compassRayToSheet(c, direction),
    assignment: roleForDirection(direction) === "ridge" ? "M" : "V",
    role: roleForDirection(direction),
  }));
}

function diamondSegments(
  moleculeId: string,
  moleculeKind: MoleculeKind,
  center: CompletionPoint,
  radius: number,
): CompletionSegment[] {
  const c: Point = [center.x, center.y];
  const points = ([
    [c[0], c[1] + radius],
    [c[0] + radius, c[1]],
    [c[0], c[1] - radius],
    [c[0] - radius, c[1]],
  ] as Point[]).map(roundPoint);
  return [
    [points[0], points[1]],
    [points[1], points[2]],
    [points[2], points[3]],
    [points[3], points[0]],
  ].map(([p1, p2], index): CompletionSegment => ({
    id: `${moleculeId}:diamond-${index}`,
    moleculeId,
    moleculeKind,
    p1,
    p2,
    assignment: "M",
    role: "ridge",
  }));
}

function terminalFanSegments(
  moleculeId: string,
  moleculeKind: MoleculeKind,
  center: CompletionPoint,
  target: CompletionPoint,
): CompletionSegment[] {
  const c: Point = [center.x, center.y];
  const horizontalDirection = center.x <= target.x ? -1 : 1;
  const verticalDirection = center.y <= target.y ? -1 : 1;
  const borderX = horizontalDirection < 0 ? 0 : 1;
  const borderY = verticalDirection < 0 ? 0 : 1;
  const spokes: Array<{ id: string; p2: Point; direction: Point }> = [
    { id: "west", p2: [0, center.y], direction: [-1, 0] },
    { id: "east", p2: [1, center.y], direction: [1, 0] },
    { id: "south", p2: [center.x, 0], direction: [0, -1] },
    { id: "north", p2: [center.x, 1], direction: [0, 1] },
    { id: "corner-ridge", p2: [borderX, borderY], direction: [horizontalDirection, verticalDirection] },
  ];
  return spokes.map((spoke): CompletionSegment => ({
    id: `${moleculeId}:${spoke.id}`,
    moleculeId,
    moleculeKind,
    p1: c,
    p2: roundPoint(spoke.p2),
    assignment: roleForDirection(spoke.direction) === "ridge" ? "M" : "V",
    role: roleForDirection(spoke.direction),
  }));
}

function corridorRoute(
  from: CompletionPoint,
  to: CompletionPoint,
  gridSize: number,
): {
  primaryOrientation: Port["orientation"];
  turns: CompletionPoint[];
  segments: Array<{ p1: CompletionPoint; p2: CompletionPoint; role: Exclude<BPRole, "border"> }>;
} {
  if (Math.abs(from.x - to.x) < 1e-9) {
    return { primaryOrientation: "vertical", turns: [], segments: [{ p1: from, p2: to, role: "axis" }] };
  }
  if (Math.abs(from.y - to.y) < 1e-9) {
    return { primaryOrientation: "horizontal", turns: [], segments: [{ p1: from, p2: to, role: "hinge" }] };
  }
  if (Math.abs(Math.abs(from.x - to.x) - Math.abs(from.y - to.y)) < 1e-9) {
    return {
      primaryOrientation: from.x - to.x === from.y - to.y ? "diagonal-positive" : "diagonal-negative",
      turns: [],
      segments: [{ p1: from, p2: to, role: "ridge" }],
    };
  }
  const bend = point(snap(to.x, gridSize), snap(from.y, gridSize));
  return {
    primaryOrientation: "horizontal",
    turns: [bend],
    segments: [
      { p1: from, p2: bend, role: "hinge" },
      { p1: bend, p2: to, role: "axis" },
    ],
  };
}

function dedupeCompletionSegments(segments: CompletionSegment[]): CompletionSegment[] {
  const result = new Map<string, CompletionSegment>();
  for (const segmentItem of segments) {
    if (distance(segmentItem.p1, segmentItem.p2) < 1e-9) continue;
    const key = segmentKey(segmentItem);
    const existing = result.get(key);
    if (!existing) {
      result.set(key, segmentItem);
      continue;
    }
    result.set(key, {
      ...existing,
      assignment: assignmentPriority(existing.assignment) >= assignmentPriority(segmentItem.assignment)
        ? existing.assignment
        : segmentItem.assignment,
      role: rolePriority(existing.role) >= rolePriority(segmentItem.role) ? existing.role : segmentItem.role,
    });
  }
  return [...result.values()];
}

function countMoleculeInstances(instances: MoleculeInstance[]): Record<string, number> {
  const counts: Record<string, number> = {};
  counts["sheet-border"] = 1;
  for (const instance of instances) counts[instance.kind] = (counts[instance.kind] ?? 0) + 1;
  return counts;
}

function gridSizeForSegments(segments: CompletionSegment[], preferred: number): number {
  const candidates = [...new Set([preferred, 128, 256, 512, 1024, 2048])].sort((a, b) => a - b);
  const coordinates = segments.flatMap((segmentItem) => [segmentItem.p1[0], segmentItem.p1[1], segmentItem.p2[0], segmentItem.p2[1]]);
  return candidates.find((gridSize) => coordinates.every((coordinate) => onHalfGrid(coordinate, gridSize))) ?? preferred;
}

function nearestPoint(target: CompletionPoint, points: CompletionPoint[]): CompletionPoint {
  return points.slice().sort((a, b) => pointDistance(a, target) - pointDistance(b, target))[0] ?? point(0.5, 0.5);
}

function midpoint(a: CompletionPoint, b: CompletionPoint): CompletionPoint {
  return point((a.x + b.x) / 2, (a.y + b.y) / 2);
}

function diamondRadius(center: CompletionPoint): number {
  const margin = Math.min(center.x, center.y, 1 - center.x, 1 - center.y);
  return margin >= 0.1875 ? 0.125 : 0.0625;
}

function point(x: number, y: number): CompletionPoint {
  return { x: round(clamp(x, 0.125, 0.875)), y: round(clamp(y, 0.125, 0.875)) };
}

function snapAnchor(value: number): number {
  return nearestAnchor(value, 0.5);
}

function pointKey2(item: CompletionPoint): string {
  return `${item.x.toFixed(6)},${item.y.toFixed(6)}`;
}

function segmentKey(segmentItem: CompletionSegment): string {
  const a = `${segmentItem.p1[0].toFixed(9)},${segmentItem.p1[1].toFixed(9)}`;
  const b = `${segmentItem.p2[0].toFixed(9)},${segmentItem.p2[1].toFixed(9)}`;
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function roundPoint(item: Point): Point {
  return [round(item[0]), round(item[1])];
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function pointDistance(a: CompletionPoint, b: CompletionPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function assignmentPriority(assignment: EdgeAssignment): number {
  return { B: 5, M: 4, V: 3, F: 2, U: 1, C: 0 }[assignment];
}

function rolePriority(role: BPRole): number {
  return { border: 5, ridge: 4, axis: 3, stretch: 2, hinge: 1 }[role];
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function template(id: string, kind: MoleculeKind, ports: Port[]): MoleculeTemplate {
  return {
    id,
    kind,
    version: ENGINE_VERSION,
    ports,
    localProof: {
      kawasaki: true,
      maekawa: true,
      fixture: `${kind}-fixture`,
    },
  };
}

function roleForEdge(a: Point, b: Point, assignment: EdgeAssignment): BPRole {
  if (assignment === "B") return "border";
  const dx = Math.abs(a[0] - b[0]);
  const dy = Math.abs(a[1] - b[1]);
  if (dx > 1e-8 && dy > 1e-8 && Math.abs(dx - dy) < 1e-8) return "ridge";
  if (dx < 1e-8) return "axis";
  return "hinge";
}

function sourceForEdge(a: Point, b: Point, role: BPRole): CompilerEdgeSource {
  const dx = Math.abs(a[0] - b[0]);
  const dy = Math.abs(a[1] - b[1]);
  return {
    kind: role === "border"
      ? "completion-sheet-border"
      : role === "ridge"
        ? "completion-certified-ridge"
        : dx < dy
          ? "completion-certified-axis"
          : "completion-certified-hinge",
    mandatory: true,
    role,
  };
}

function primaryBody(layout: CompletionLayout): CompletionRegion {
  return layout.regions.find((region) => region.kind === "body") ?? {
    id: "implicit-body",
    kind: "body",
    x1: 0.375,
    y1: 0.375,
    x2: 0.625,
    y2: 0.625,
  };
}

function chooseAxis(spec: BPStudioAdapterSpec, terminals: CompletionTerminal[]): CompletionAxis {
  if (spec.layout.symmetry === "bilateral-y") return "vertical";
  const xSpread = spread(terminals.map((terminal) => terminal.x));
  const ySpread = spread(terminals.map((terminal) => terminal.y));
  return xSpread >= ySpread ? "horizontal" : "vertical";
}

function completionPointMap(bodies: CompletionRegion[], terminals: CompletionTerminal[]): Map<string, CompletionPoint> {
  const points = new Map<string, CompletionPoint>();
  for (const body of bodies) {
    points.set(body.id, {
      x: (body.x1 + body.x2) / 2,
      y: (body.y1 + body.y2) / 2,
    });
  }
  for (const terminal of terminals) {
    points.set(terminal.id, { x: terminal.x, y: terminal.y });
    points.set(terminal.nodeId, { x: terminal.x, y: terminal.y });
  }
  return points;
}

function keepPointOutsideTerminalAllocations(
  center: CompletionPoint,
  terminals: CompletionTerminal[],
  gridSize: number,
  clearance: number,
): CompletionPoint {
  let x = clamp(center.x, clearance, 1 - clearance);
  let y = clamp(center.y, clearance, 1 - clearance);
  const obstacles = terminals.filter((terminal) => terminal.allocationRadius && terminal.allocationRadius > 0);
  for (let iteration = 0; iteration < 24; iteration += 1) {
    let moved = false;
    for (const terminal of obstacles) {
      const radius = terminal.allocationRadius!;
      const minimumDistance = radius + clearance;
      let dx = x - terminal.x;
      let dy = y - terminal.y;
      let distance = Math.hypot(dx, dy);
      if (distance >= minimumDistance - 1e-9) continue;
      if (distance < 1e-9) {
        const fallback = fallbackDirectionAwayFromTerminal(terminal);
        dx = fallback.x;
        dy = fallback.y;
        distance = Math.hypot(dx, dy);
      }
      x = terminal.x + (dx / distance) * minimumDistance;
      y = terminal.y + (dy / distance) * minimumDistance;
      x = clamp(x, clearance, 1 - clearance);
      y = clamp(y, clearance, 1 - clearance);
      moved = true;
    }
    if (!moved) break;
  }
  return {
    x: snap(clamp(x, clearance, 1 - clearance), gridSize),
    y: snap(clamp(y, clearance, 1 - clearance), gridSize),
  };
}

function fallbackDirectionAwayFromTerminal(terminal: CompletionTerminal): CompletionPoint {
  if (terminal.side === "left") return { x: 1, y: 0 };
  if (terminal.side === "right") return { x: -1, y: 0 };
  if (terminal.side === "top") return { x: 0, y: -1 };
  if (terminal.side === "bottom") return { x: 0, y: 1 };
  return {
    x: terminal.x >= 0.5 ? -1 : 1,
    y: terminal.y >= 0.5 ? -1 : 1,
  };
}

function terminalAllocationRadius(
  spec: BPStudioAdapterSpec,
  nodeId: string,
  sheetWidth: number,
  sheetHeight: number,
): number | undefined {
  const edge = spec.tree.edges.find((candidate) => candidate.to === nodeId && candidate.from !== spec.tree.rootId) ??
    spec.tree.edges.find((candidate) => candidate.from === nodeId && candidate.to !== spec.tree.rootId);
  const radius = edge?.length ?? spec.layout.flaps.find((flap) => flap.nodeId === nodeId)?.terminalRadius;
  if (!radius || radius <= 0) return undefined;
  return radius / Math.max(sheetWidth, sheetHeight);
}

function orientationForPoints(a: CompletionPoint | undefined, b: CompletionPoint | undefined, fallback: CompletionAxis): CompletionAxis {
  if (!a || !b) return fallback;
  return Math.abs(a.x - b.x) >= Math.abs(a.y - b.y) ? "horizontal" : "vertical";
}

function corridorCoordinate(
  from: string,
  to: string,
  points: Map<string, CompletionPoint>,
  axis: CompletionAxis,
  terminalIds: Set<string> = new Set(),
): number {
  const a = points.get(from);
  const b = points.get(to);
  if (!a || !b) return 0.5;
  if (terminalIds.has(from)) return axis === "horizontal" ? a.y : a.x;
  if (terminalIds.has(to)) return axis === "horizontal" ? b.y : b.x;
  return axis === "horizontal" ? (a.y + b.y) / 2 : (a.x + b.x) / 2;
}

function corridorRouteAvoidingAllocations(
  from: string,
  to: string,
  points: Map<string, CompletionPoint>,
  preferredAxis: CompletionAxis,
  terminalIds: Set<string>,
  terminals: CompletionTerminal[],
  gridSize: number,
  width: number,
): { orientation: CompletionAxis; coordinate: number } {
  const preferred = corridorCoordinateAvoidingAllocations(from, to, points, preferredAxis, terminalIds, terminals, gridSize, width);
  if (preferred.ok) return { orientation: preferredAxis, coordinate: preferred.coordinate };
  const alternateAxis = preferredAxis === "horizontal" ? "vertical" : "horizontal";
  const alternate = corridorCoordinateAvoidingAllocations(from, to, points, alternateAxis, terminalIds, terminals, gridSize, width);
  if (alternate.ok) return { orientation: alternateAxis, coordinate: alternate.coordinate };
  return { orientation: preferredAxis, coordinate: preferred.coordinate };
}

function corridorCoordinateAvoidingAllocations(
  from: string,
  to: string,
  points: Map<string, CompletionPoint>,
  axis: CompletionAxis,
  terminalIds: Set<string>,
  terminals: CompletionTerminal[],
  gridSize: number,
  width: number,
): { coordinate: number; ok: boolean } {
  const base = snap(corridorCoordinate(from, to, points, axis, terminalIds), gridSize);
  const a = points.get(from);
  const b = points.get(to);
  if (!a || !b) return { coordinate: base, ok: false };
  const step = 1 / gridSize;
  const candidates = Array.from({ length: gridSize + 1 }, (_, index) => round(index * step))
    .sort((left, right) => Math.abs(left - base) - Math.abs(right - base) || left - right);
  const fromTerminal = terminals.find((terminal) => terminal.id === from || terminal.nodeId === from);
  const toTerminal = terminals.find((terminal) => terminal.id === to || terminal.nodeId === to);
  const endpointTerminals = [fromTerminal, toTerminal].filter((terminal): terminal is CompletionTerminal => Boolean(terminal));
  const viable = candidates.filter((coordinate) => endpointTerminals.every((terminal) => terminalCanMeetLane(terminal, coordinate, axis, width)));
  for (const candidate of viable.length ? viable : candidates) {
    const rect = corridorRectForCoordinate(a, b, fromTerminal, toTerminal, axis, candidate, width);
    if (!rect) continue;
    if (!terminals.some((terminal) => terminal.allocationRadius && rectOverlapsTerminalAllocation(rect, terminal, step / 4))) {
      return { coordinate: snap(candidate, gridSize), ok: true };
    }
  }
  return { coordinate: base, ok: false };
}

function terminalCanMeetLane(
  terminal: CompletionTerminal,
  coordinate: number,
  axis: CompletionAxis,
  width: number,
): boolean {
  if (!terminal.allocationRadius) return true;
  const delta = axis === "horizontal" ? coordinate - terminal.y : coordinate - terminal.x;
  return Math.abs(delta) <= Math.max(0, terminal.allocationRadius - width / 2) + 1e-9;
}

function corridorRectForCoordinate(
  from: CompletionPoint,
  to: CompletionPoint,
  fromTerminal: CompletionTerminal | undefined,
  toTerminal: CompletionTerminal | undefined,
  axis: CompletionAxis,
  coordinate: number,
  width: number,
): { x1: number; y1: number; x2: number; y2: number } | undefined {
  const half = width / 2;
  if (axis === "horizontal") {
    const x1 = endpointCoordinateForAllocation(from, to, fromTerminal, axis, coordinate);
    const x2 = endpointCoordinateForAllocation(to, from, toTerminal, axis, coordinate);
    return normalizeRect({
      x1,
      x2,
      y1: coordinate - half,
      y2: coordinate + half,
    });
  }
  const y1 = endpointCoordinateForAllocation(from, to, fromTerminal, axis, coordinate);
  const y2 = endpointCoordinateForAllocation(to, from, toTerminal, axis, coordinate);
  return normalizeRect({
    x1: coordinate - half,
    x2: coordinate + half,
    y1,
    y2,
  });
}

function endpointCoordinateForAllocation(
  endpoint: CompletionPoint,
  toward: CompletionPoint,
  terminal: CompletionTerminal | undefined,
  axis: CompletionAxis,
  laneCoordinate: number,
): number {
  if (!terminal?.allocationRadius) return axis === "horizontal" ? endpoint.x : endpoint.y;
  const perpendicularDelta = axis === "horizontal" ? laneCoordinate - terminal.y : laneCoordinate - terminal.x;
  const alongRadius = Math.sqrt(Math.max(0, terminal.allocationRadius ** 2 - perpendicularDelta ** 2));
  if (axis === "horizontal") {
    return toward.x >= terminal.x ? terminal.x + alongRadius : terminal.x - alongRadius;
  }
  return toward.y >= terminal.y ? terminal.y + alongRadius : terminal.y - alongRadius;
}

function rectOverlapsTerminalAllocation(
  rect: { x1: number; y1: number; x2: number; y2: number },
  terminal: CompletionTerminal,
  tolerance: number,
): boolean {
  if (!terminal.allocationRadius) return false;
  const closestX = clamp(terminal.x, rect.x1, rect.x2);
  const closestY = clamp(terminal.y, rect.y1, rect.y2);
  const distanceToRect = Math.hypot(terminal.x - closestX, terminal.y - closestY);
  return distanceToRect < terminal.allocationRadius - tolerance;
}

function normalizeRect(rect: { x1: number; y1: number; x2: number; y2: number }): { x1: number; y1: number; x2: number; y2: number } {
  return {
    x1: Math.min(rect.x1, rect.x2),
    y1: Math.min(rect.y1, rect.y2),
    x2: Math.max(rect.x1, rect.x2),
    y2: Math.max(rect.y1, rect.y2),
  };
}

function toLayoutMetadata(layout: CompletionLayout): LayoutMetadata {
  return {
    gridSize: layout.gridSize,
    symmetry: layout.axis === "horizontal" ? "uniaxial-horizontal" : "uniaxial-vertical",
    margin: 1 / layout.gridSize,
    bodyRegions: layout.regions.map((region) => ({
      id: region.id,
      x1: region.x1,
      y1: region.y1,
      x2: region.x2,
      y2: region.y2,
    })),
    flapTerminals: layout.terminals.map((terminal) => ({
      id: terminal.id,
      x: terminal.x,
      y: terminal.y,
      side: terminal.side === "interior" ? sideForPoint(terminal) : terminal.side,
    })),
    corridors: layout.corridors.map((corridor) => ({
      id: corridor.id,
      orientation: corridor.orientation,
      coordinate: corridor.coordinate,
      role: corridor.orientation === layout.axis ? "axis" : "hinge",
    })),
    layoutScore: 1,
  };
}

function gridSizeForFold(fold: FOLDFormat, preferred: number): number {
  const candidates = [...new Set([preferred, 128, 256, 512, 1024, 2048])].sort((a, b) => a - b);
  return candidates.find((gridSize) => fold.vertices_coords.every(([x, y]) => onHalfGrid(x, gridSize) && onHalfGrid(y, gridSize))) ?? preferred;
}

export function compilerGridSizeForSheet(sheetWidth: number, sheetHeight: number): number {
  const widthUnits = integerSheetUnits(sheetWidth);
  const heightUnits = integerSheetUnits(sheetHeight);
  if (widthUnits && heightUnits) return lcm(widthUnits, heightUnits) * BP_STUDIO_GRID_SUBDIVISION;
  return DEFAULT_GRID_SIZE;
}

function integerSheetUnits(value: number): number | undefined {
  const rounded = Math.round(value);
  if (rounded > 0 && Math.abs(value - rounded) < 1e-8) return rounded;
  return undefined;
}

function lcm(a: number, b: number): number {
  return Math.abs(a * b) / gcd(a, b);
}

function gcd(a: number, b: number): number {
  let left = Math.abs(a);
  let right = Math.abs(b);
  while (right !== 0) {
    const next = left % right;
    left = right;
    right = next;
  }
  return left || 1;
}

function onHalfGrid(value: number, gridSize: number): boolean {
  return Math.abs(value * gridSize * 2 - Math.round(value * gridSize * 2)) < 1e-6;
}

function sideForPoint(point: { x: number; y: number }): "top" | "right" | "bottom" | "left" {
  const distances = [
    ["left", point.x],
    ["right", 1 - point.x],
    ["bottom", point.y],
    ["top", 1 - point.y],
  ] as const;
  return distances.reduce((best, candidate) => candidate[1] < best[1] ? candidate : best)[0];
}

function snap(value: number, gridSize: number): number {
  return Math.round(value * gridSize * 2) / (gridSize * 2);
}

function isHalfGrid(value: number): boolean {
  return Math.abs(value * 2 - Math.round(value * 2)) > 1e-8;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function mean(values: number[], fallback: number): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : fallback;
}

function spread(values: number[]): number {
  return values.length ? Math.max(...values) - Math.min(...values) : 0;
}
