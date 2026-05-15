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

const ENGINE_VERSION = "strict-bp-completion/v0.2.0";
const DEFAULT_GRID_SIZE = 128;
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
  const gridSize = options.gridSize ?? DEFAULT_GRID_SIZE;
  const adapterLayout = options.adapterMetadata?.layout;
  const sheet = adapterLayout?.sheet ?? options.adapterSpec?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const sheetWidth = Math.max(1, sheet.width);
  const sheetHeight = Math.max(1, sheet.height);
  const optimizedFlaps = adapterLayout?.flaps ?? [];
  const terminalByNodeId = new Map(spec.layout.flaps.map((terminal) => [terminal.nodeId, terminal]));
  const terminalByIndex = new Map(spec.layout.flaps.map((terminal, index) => [index, terminal]));
  const terminals: CompletionTerminal[] = (optimizedFlaps.length ? optimizedFlaps : spec.layout.flaps.map((terminal, index) => ({
    id: index,
    x: terminal.terminal.x,
    y: terminal.terminal.y,
    width: terminal.width,
    height: terminal.height,
  }))).map((flap, index) => {
    const adapterNodeId = options.adapterSpec?.nodeIdByAdapterId?.[String(flap.id)];
    const sourceTerminal = adapterNodeId ? terminalByNodeId.get(adapterNodeId) : terminalByIndex.get(index);
    const x = snap(clamp((flap.x + (flap.width ?? 0) / 2) / sheetWidth, 0.05, 0.95), gridSize);
    const y = snap(clamp((flap.y + (flap.height ?? 0) / 2) / sheetHeight, 0.05, 0.95), gridSize);
    return {
      id: sourceTerminal?.nodeId ?? `flap-${flap.id}`,
      nodeId: String(flap.id),
      x,
      y,
      side: sourceTerminal?.side ?? sideForPoint({ x, y }),
      width: snap(Math.max(1 / (gridSize * 2), (flap.width ?? 0) / sheetWidth), gridSize),
      height: snap(Math.max(1 / (gridSize * 2), (flap.height ?? 0) / sheetHeight), gridSize),
      priority: sourceTerminal?.priority ?? index,
    };
  });
  const bodies = spec.layout.bodies.map((body): CompletionRegion => ({
    id: body.nodeId,
    kind: "body",
    x1: snap(clamp((body.center.x - body.width / 2) / spec.sheet.width, 0, 1), gridSize),
    y1: snap(clamp((body.center.y - body.height / 2) / spec.sheet.height, 0, 1), gridSize),
    x2: snap(clamp((body.center.x + body.width / 2) / spec.sheet.width, 0, 1), gridSize),
    y2: snap(clamp((body.center.y + body.height / 2) / spec.sheet.height, 0, 1), gridSize),
  }));
  const axis = chooseAxis(spec, terminals);
  const spineCoordinate = snap(
    clamp(mean(bodies.map((body) => axis === "horizontal" ? (body.y1 + body.y2) / 2 : (body.x1 + body.x2) / 2), 0.5), 0.1875, 0.8125),
    gridSize,
  );
  const corridors: CompletionCorridor[] = spec.layout.rivers.map((river, index) => ({
    id: river.edgeId,
    from: river.from,
    to: river.to,
    orientation: river.preferredAxis === "vertical" ? "vertical" : river.preferredAxis === "horizontal" ? "horizontal" : axis,
    coordinate: snap(corridorCoordinate(river.from, river.to, spec, axis), gridSize),
    width: snap(Math.max(1 / gridSize, river.width / spec.sheet.gridSize), gridSize),
  }));

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
      optimizedFlapCount: optimizedFlaps.length,
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
    const fan = instanceFor(patches, `fan-${terminal.terminal.id}`, "corner-fan", terminal.center);
    instances.push(fan);
    segments.push(...starSegments(fan.id, fan.kind, terminal.center));
    segments.push(...diamondSegments(`flap-contour-${terminal.terminal.id}`, "flap-contour", terminal.center, 1 / 16));
    instances.push(templateOnlyInstance(`flap-contour-${terminal.terminal.id}`, "flap-contour", terminal.center));

    const target = nearestPoint(terminal.center, bodyCenters);
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

function uniquePointsByKey(points: CompletionPoint[]): CompletionPoint[] {
  const seen = new Set<string>();
  const result: CompletionPoint[] = [];
  for (const item of points) {
    const key = pointKey2(item);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }
  return result;
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

function corridorCoordinate(from: string, to: string, spec: BPStudioAdapterSpec, axis: CompletionAxis): number {
  const points = new Map<string, CompletionPoint>();
  for (const body of spec.layout.bodies) points.set(body.nodeId, { x: body.center.x / spec.sheet.width, y: body.center.y / spec.sheet.height });
  for (const flap of spec.layout.flaps) points.set(flap.nodeId, { x: flap.terminal.x / spec.sheet.width, y: flap.terminal.y / spec.sheet.height });
  const a = points.get(from);
  const b = points.get(to);
  if (!a || !b) return 0.5;
  return axis === "horizontal" ? (a.y + b.y) / 2 : (a.x + b.x) / 2;
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
