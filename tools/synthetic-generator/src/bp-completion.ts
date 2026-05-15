import ear from "rabbit-ear";
import { assignmentToFoldAngle, normalizeFold, roleCounts } from "./fold-utils.ts";
import { scoreFoldRealism } from "./realism-metrics.ts";
import type { BPStudioAdapterSpec } from "./bp-studio-spec.ts";
import type { AdapterMetadata, AdapterSpec } from "./bp-studio-realistic.ts";
import type {
  CompletionAxis,
  CompletionCorridor,
  CompletionFoldLine,
  CompletionLayout,
  CompletionPoint,
  CompletionRegion,
  CompletionResult,
  CompletionTerminal,
  MoleculeKind,
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

const ENGINE_VERSION = "strict-bp-completion/v0.1.0";
const DEFAULT_GRID_SIZE = 128;

export function completeBoxPleat(spec: BPStudioAdapterSpec, options: BoxPleatCompletionOptions = {}): CompletionResult {
  const layout = regularizeBPStudioLayout(spec, options);
  return completeBoxPleatLayout(layout, options);
}

export function completeBoxPleatLayout(
  layout: CompletionLayout,
  options: BoxPleatCompletionOptions = {},
): CompletionResult {
  const molecules = instantiateMolecules(layout);
  const portJoins = joinPorts(molecules);
  const rejected = portJoins
    .filter((join) => !join.accepted)
    .map((join) => ({ code: "incompatible-port", message: join.reason ?? "port join rejected" }));
  if (rejected.length > 0) {
    return {
      ok: false,
      layout,
      foldLines: [],
      molecules,
      portJoins,
      rejected,
    };
  }
  const foldLines = buildFoldProgram(layout, molecules, options.maxFoldLines ?? 96);
  if (foldLines.length === 0) {
    return {
      ok: false,
      layout,
      foldLines,
      molecules,
      portJoins,
      rejected: [{ code: "empty-fold-program", message: "completion emitted no fold lines" }],
    };
  }

  const fold = foldProgramToFold(layout, foldLines, portJoins);
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
      "instantiate-certified-molecules",
      "check-port-compatibility",
      "emit-restricted-fold-program",
      "construct-fold-with-rabbit-ear-flatfold",
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
  return { ok: true, fold, layout, foldLines, molecules, portJoins, rejected };
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

function buildFoldProgram(
  layout: CompletionLayout,
  molecules: MoleculeTemplate[],
  maxFoldLines: number,
): CompletionFoldLine[] {
  const lines = new Map<string, CompletionFoldLine>();
  const add = (line: CompletionFoldLine): void => {
    if (lines.size >= maxFoldLines) return;
    const key = `${line.vector.join(",")}:${line.origin.map((value) => value.toFixed(6)).join(",")}`;
    if (!lines.has(key)) lines.set(key, line);
  };

  const body = primaryBody(layout);
  const xCoordinates = rankedUnique([
    body.x1,
    body.x2,
    ...layout.terminals.map((terminal) => terminal.x),
    ...layout.corridors.filter((corridor) => corridor.orientation === "vertical").map((corridor) => corridor.coordinate),
  ], layout.gridSize);
  const yCoordinates = rankedUnique([
    layout.spineCoordinate,
    body.y1,
    body.y2,
    ...layout.terminals.map((terminal) => terminal.y),
    ...layout.corridors.filter((corridor) => corridor.orientation === "horizontal").map((corridor) => corridor.coordinate),
  ], layout.gridSize);

  for (const x of xCoordinates) add(axisLine("axis", x, "V", "river-corridor"));
  for (const y of yCoordinates) add(axisLine("hinge", y, y === layout.spineCoordinate ? "V" : "M", "hinge-corridor"));

  const terminalOrder = [...layout.terminals].sort((a, b) => a.priority - b.priority);
  for (const terminal of terminalOrder) {
    add(diagonalLine("positive", terminal.y - terminal.x, terminal.priority % 2 === 0 ? "V" : "M", "corner-fan"));
    add(diagonalLine("negative", terminal.y + terminal.x, "M", "diagonal-staircase"));
  }

  const bodyCenter = { x: (body.x1 + body.x2) / 2, y: (body.y1 + body.y2) / 2 };
  for (const offset of [-0.125, 0, 0.125]) {
    add(diagonalLine("positive", snap(bodyCenter.y - bodyCenter.x + offset, layout.gridSize), "M", "diamond-connector"));
    add(diagonalLine("negative", snap(bodyCenter.y + bodyCenter.x + offset, layout.gridSize), offset === 0 ? "V" : "M", "stretch-gadget"));
  }

  for (const molecule of molecules) {
    if (molecule.kind !== "body-panel") continue;
    add(axisLine("axis", bodyCenter.x, "V", "body-panel"));
    add(axisLine("hinge", bodyCenter.y, "V", "body-panel"));
  }

  return [...lines.values()]
    .filter((line) => lineIsInsideSheet(line))
    .sort((a, b) => foldLineOrder(a) - foldLineOrder(b));
}

function foldProgramToFold(layout: CompletionLayout, foldLines: CompletionFoldLine[], portJoins: PortJoin[]): FOLDFormat {
  const graph = ear.graph.square();
  for (const line of foldLines) {
    ear.graph.flatFold(graph, line.vector, line.origin, line.assignment);
  }
  const fold = normalizeFold({
    file_spec: 1.1,
    file_creator: "cp-synthetic-generator/bp-completion",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: graph.vertices_coords,
    edges_vertices: graph.edges_vertices,
    edges_assignment: graph.edges_assignment,
  }, "cp-synthetic-generator/bp-completion");
  fold.edges_bpRole = fold.edges_vertices.map(([a, b], edgeIndex) => roleForEdge(fold.vertices_coords[a], fold.vertices_coords[b], fold.edges_assignment[edgeIndex]));
  fold.edges_compilerSource = fold.edges_vertices.map(([a, b], edgeIndex) => sourceForEdge(fold.vertices_coords[a], fold.vertices_coords[b], fold.edges_bpRole?.[edgeIndex] ?? "hinge"));
  fold.edges_foldAngle = fold.edges_assignment.map(assignmentToFoldAngle);
  const counts = roleCounts(fold);
  const outputGridSize = gridSizeForFold(fold, layout.gridSize);
  fold.bp_metadata = {
    gridSize: outputGridSize,
    bpSubfamily: "bp-studio-completed-uniaxial",
    flapCount: layout.terminals.length,
    gadgetCount: layout.corridors.length + layout.regions.length,
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  const layoutMetadata = toLayoutMetadata({ ...layout, gridSize: outputGridSize });
  fold.layout_metadata = layoutMetadata;
  fold.molecule_metadata = {
    libraryVersion: ENGINE_VERSION,
    molecules: countMolecules(foldLines),
    portChecks: {
      checked: portJoins.length,
      rejected: portJoins.filter((join) => !join.accepted).length,
    },
  };
  fold.realism_metadata = scoreFoldRealism(fold, layoutMetadata);
  return fold;
}

function instantiateMolecules(layout: CompletionLayout): MoleculeTemplate[] {
  const molecules: MoleculeTemplate[] = [
    template("sheet-border", "sheet-border", []),
    template("body-panel", "body-panel", [
      port("body-north", "body-panel", "horizontal", "top", layout.spineCoordinate, 1 / layout.gridSize, "hinge"),
      port("body-south", "body-panel", "horizontal", "bottom", layout.spineCoordinate, 1 / layout.gridSize, "hinge"),
    ]),
  ];
  for (const terminal of layout.terminals) {
    molecules.push(template(`flap-${terminal.id}`, "flap-contour", [
      port(`flap-${terminal.id}-corridor`, `flap-${terminal.id}`, layout.axis, terminal.side, terminal.x, Math.max(terminal.width, terminal.height), "hinge"),
    ]));
    molecules.push(template(`fan-${terminal.id}`, "corner-fan", [
      port(`fan-${terminal.id}-terminal`, `fan-${terminal.id}`, terminal.side === "left" || terminal.side === "right" ? "horizontal" : "vertical", terminal.side, terminal.x, Math.max(terminal.width, terminal.height), "ridge"),
    ]));
  }
  for (const corridor of layout.corridors) {
    molecules.push(template(`corridor-${corridor.id}`, corridor.orientation === layout.axis ? "river-corridor" : "hinge-corridor", [
      port(`corridor-${corridor.id}-a`, `corridor-${corridor.id}`, corridor.orientation, "interior", corridor.coordinate, corridor.width, "axis"),
      port(`corridor-${corridor.id}-b`, `corridor-${corridor.id}`, corridor.orientation, "interior", corridor.coordinate, corridor.width, "axis"),
    ]));
  }
  molecules.push(template("central-diamond", "diamond-connector", [
    port("central-diamond-left", "central-diamond", "diagonal-positive", "interior", layout.spineCoordinate, 1 / layout.gridSize, "ridge"),
    port("central-diamond-right", "central-diamond", "diagonal-negative", "interior", layout.spineCoordinate, 1 / layout.gridSize, "ridge"),
  ]));
  molecules.push(template("central-stretch", "stretch-gadget", [
    port("central-stretch-left", "central-stretch", "horizontal", "interior", layout.spineCoordinate, 1 / layout.gridSize, "axis"),
    port("central-stretch-right", "central-stretch", "vertical", "interior", 0.5, 1 / layout.gridSize, "axis"),
  ]));
  return molecules;
}

function joinPorts(molecules: MoleculeTemplate[]): PortJoin[] {
  const joins: PortJoin[] = [];
  const ports = molecules.flatMap((molecule) => molecule.ports);
  const interiorPorts = ports.filter((port) => port.side === "interior");
  for (const port of ports.filter((candidate) => candidate.side !== "interior")) {
    const mate = nearestCompatiblePort(port, interiorPorts);
    if (!mate) {
      joins.push({
        from: port.id,
        to: "unmatched",
        orientation: port.orientation,
        width: port.width,
        accepted: false,
        reason: `no compatible interior port for ${port.orientation}`,
      });
      continue;
    }
    joins.push({
      from: port.id,
      to: mate.id,
      orientation: port.orientation,
      width: Math.min(port.width, mate.width),
      accepted: true,
    });
  }
  return joins;
}

function nearestCompatiblePort(port: Port, candidates: Port[]): Port | undefined {
  return candidates
    .filter((candidate) => portsCompatible(port, candidate))
    .sort((a, b) => Math.abs(a.coordinate - port.coordinate) - Math.abs(b.coordinate - port.coordinate))[0];
}

function portsCompatible(a: Port, b: Port): boolean {
  if (a.orientation === b.orientation) return true;
  if (a.orientation.startsWith("diagonal") && b.orientation.startsWith("diagonal")) return true;
  return false;
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

function port(
  id: string,
  moleculeId: string,
  orientation: Port["orientation"],
  side: Port["side"],
  coordinate: number,
  width: number,
  role: BPRole,
): Port {
  return {
    id,
    moleculeId,
    orientation,
    side,
    coordinate,
    width,
    parity: isHalfGrid(coordinate) ? "half" : "integer",
    role,
  };
}

function axisLine(role: "axis" | "hinge", coordinate: number, assignment: "M" | "V", moleculeKind: MoleculeKind): CompletionFoldLine {
  return role === "axis"
    ? {
        id: `axis-x-${coordinate.toFixed(6)}`,
        moleculeId: moleculeKind,
        moleculeKind,
        vector: [0, 1],
        origin: [coordinate, 0],
        assignment,
        role,
      }
    : {
        id: `hinge-y-${coordinate.toFixed(6)}`,
        moleculeId: moleculeKind,
        moleculeKind,
        vector: [1, 0],
        origin: [0, coordinate],
        assignment,
        role,
      };
}

function diagonalLine(kind: "positive" | "negative", rawConstant: number, assignment: "M" | "V", moleculeKind: MoleculeKind): CompletionFoldLine {
  const constant = kind === "positive" ? clamp(rawConstant, -0.9375, 0.9375) : clamp(rawConstant, 0.0625, 1.9375);
  const origin = kind === "positive"
    ? (constant >= 0 ? [0, constant] : [-constant, 0])
    : (constant <= 1 ? [0, constant] : [constant - 1, 1]);
  return {
    id: `${kind}-diagonal-${constant.toFixed(6)}`,
    moleculeId: moleculeKind,
    moleculeKind,
    vector: kind === "positive" ? [1, 1] : [1, -1],
    origin: origin as Point,
    assignment,
    role: "ridge",
  };
}

function lineIsInsideSheet(line: CompletionFoldLine): boolean {
  return line.origin.every((value) => Number.isFinite(value) && value >= -1e-8 && value <= 1 + 1e-8);
}

function foldLineOrder(line: CompletionFoldLine): number {
  if (line.role === "ridge") return 0;
  if (line.moleculeKind === "body-panel") return 1;
  if (line.role === "hinge") return 2;
  return 3;
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

function rankedUnique(values: number[], gridSize: number): number[] {
  const seen = new Set<string>();
  const result: number[] = [];
  for (const value of values.map((item) => snap(clamp(item, 0.0625, 0.9375), gridSize)).sort((a, b) => Math.abs(a - 0.5) - Math.abs(b - 0.5))) {
    const key = value.toFixed(6);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(value);
  }
  return result;
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

function countMolecules(foldLines: CompletionFoldLine[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const line of foldLines) counts[line.moleculeKind] = (counts[line.moleculeKind] ?? 0) + 1;
  return counts;
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
