import ear from "rabbit-ear";
import { fixtureCompletionLayout } from "./bp-completion.ts";
import {
  alternatingSequence,
  port as solverPort,
  solvePortAssignmentProblem,
  type PortAssignment,
  type PortSolverProblem,
  type PortSolverResult,
  type RegionState,
} from "./bp-port-assignment-solver.ts";
import type {
  BodyPanelRegion,
  BoundaryPort,
  CompletionCorridor,
  CompletionLayout,
  CompletionPoint,
  CompletionTerminal,
  FlapRegion,
  PleatStripOrientation,
  PleatStripRegion,
  RegionCandidateSegment,
  RegionCompletionCandidate,
  RegionLocalFlatFoldProbe,
  RegionLayout,
  RegionRect,
  StairBoundary,
} from "./bp-completion-contracts.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import type { EdgeAssignment } from "./types.ts";

type Point = [number, number];

const DEFAULT_CORRIDOR_WIDTH = 0.25;

interface CorridorEndpoint {
  id: string;
  center: CompletionPoint;
  kind: "flap" | "body";
  allocationRadius?: number;
  rect?: RegionRect;
}

type BodyPortSide = "left" | "right" | "top" | "bottom";
type BodyPortMap = Map<string, CompletionPoint>;

export type RegionFixtureName = "two-flap-stretch" | "three-flap-relay" | "five-flap-uniaxial" | "insect-lite";

export interface CompileRegionCandidateOptions {
  solvePortPhases?: boolean;
}

export interface RegionPhaseSolveResult {
  ok: boolean;
  layout: RegionLayout;
  solver?: PortSolverResult;
  errors: string[];
}

export function fixtureRegionLayout(name: RegionFixtureName): RegionLayout {
  return regionLayoutFromCompletionLayout(fixtureCompletionLayout(name));
}

export function regionLayoutFromCompletionLayout(layout: CompletionLayout): RegionLayout {
  const pitch = pleatPitchForGrid(layout.gridSize);
  const baseBodies = layout.regions.filter((region) => region.kind === "body").map((region): BodyPanelRegion => {
    const rect = snapRectToStep({ x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2 }, pitch);
    return { id: region.id, rect, center: rectCenter(rect) };
  });
  const primaryBody = baseBodies[0] ?? {
    id: "implicit-body",
    rect: { x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 },
    center: { x: 0.5, y: 0.5 },
  };
  const flaps = layout.terminals.map((terminal) => flapRegion(terminal, layout.gridSize));
  const bodies = expandBodyPanelsForCorridorPorts(layout, flaps, baseBodies.length ? baseBodies : [primaryBody]);
  const pleatStrips = layout.corridors.length
    ? corridorPleatStripRegions(layout, flaps, bodies.length ? bodies : [primaryBody])
    : flaps.map((flap, index) => pleatStripRegion(flap, primaryBody, layout, index));
  return {
    id: `${layout.id}-region-layout`,
    sourceLayoutId: layout.id,
    gridSize: layout.gridSize,
    axis: layout.axis,
    bodies: bodies.length ? bodies : [primaryBody],
    flaps,
    pleatStrips,
    boundaryPorts: pleatStrips.flatMap((strip) => boundaryPortsForStrip(strip)),
    portConstraints: [],
  };
}

export function compileRegionCandidate(layout: RegionLayout, options: CompileRegionCandidateOptions = {}): RegionCompletionCandidate {
  const phaseSolve = options.solvePortPhases === false ? undefined : solveRegionPleatStripPhases(layout);
  const workingLayout = phaseSolve?.ok ? phaseSolve.layout : layout;
  const segments: RegionCandidateSegment[] = [
    ...sheetBorderSegments(),
    ...workingLayout.bodies.flatMap((body) => rectBoundarySegments(`body-${body.id}`, body.id, "body-boundary", body.rect, "V")),
    ...workingLayout.flaps.flatMap((flap) => rectBoundarySegments(`flap-${flap.id}`, flap.id, "flap-boundary", flap.rect, "V")),
  ];
  const stairBoundaries: StairBoundary[] = [];

  for (const strip of workingLayout.pleatStrips) {
    segments.push(...rectBoundarySegments(`strip-${strip.id}`, strip.id, "body-boundary", strip.rect, "V"));
    segments.push(...pleatSegments(strip));
  }

  const arrangedSegments = dedupeSegments(segments);
  const rejectionReasons = [
    ...(phaseSolve && !phaseSolve.ok ? phaseSolve.errors : []),
    ...offGridRejections(arrangedSegments, layout.gridSize),
    ...overlapRejections(workingLayout),
  ];
  const localProbe = probeRegionCandidateLocalFlatFoldability(workingLayout, arrangedSegments);

  return {
    id: `${workingLayout.id}-candidate`,
    layout: workingLayout,
    validity: rejectionReasons.length ? "rejected" : localProbe.locallyFlatFoldable ? "locally-valid" : "layout-valid",
    segments: arrangedSegments,
    stairBoundaries,
    localProbe,
    rejectionReasons,
  };
}

export function probeRegionCandidateLocalFlatFoldability(
  layout: RegionLayout,
  segments: RegionCandidateSegment[],
): RegionLocalFlatFoldProbe {
  const activeSegmentKinds: RegionCandidateSegment["kind"][] = ["border", "strip-pleat", "stair-boundary"];
  const active = segments.filter((segmentItem) => activeSegmentKinds.includes(segmentItem.kind));
  const arranged = arrangeSegments(
    active.map((segmentItem) => ({
      p1: segmentItem.p1,
      p2: segmentItem.p2,
      assignment: segmentItem.assignment,
      role: segmentItem.role,
      source: {
        kind: `region-${segmentItem.kind}`,
        mandatory: true,
        ownerId: segmentItem.regionId,
      },
    })),
    "cp-synthetic-generator/bp-region/local-probe",
    {
      gridSize: layout.gridSize,
      bpSubfamily: "bp-studio-completed-uniaxial",
      flapCount: layout.flaps.length,
      gadgetCount: segments.filter((segmentItem) => segmentItem.kind === "stair-boundary").length,
      ridgeCount: 1,
      hingeCount: 1,
      axisCount: 1,
    },
  );
  const kawasakiProbe = {
    ...arranged,
    edges_assignment: arranged.edges_assignment.map((assignment) => assignment === "B" ? "B" : "M"),
  };
  ear.graph.populate(kawasakiProbe);
  ear.graph.populate(arranged);
  const kawasakiBadVertices = sortedUnique(ear.singleVertex.validateKawasaki(kawasakiProbe) as number[]);
  const maekawaBadVertices = sortedUnique(ear.singleVertex.validateMaekawa(arranged) as number[]);
  const badVertexSet = new Set([...kawasakiBadVertices, ...maekawaBadVertices]);
  const failureReasons: Record<string, number> = {};
  const badVertices = [...badVertexSet].sort((a, b) => a - b);
  for (const vertex of badVertices) {
    const reason = classifyLocalFailure(arranged, vertex, kawasakiBadVertices.includes(vertex));
    failureReasons[reason] = (failureReasons[reason] ?? 0) + 1;
  }
  const failurePoints = badVertices.slice(0, 240).flatMap((vertex) => {
    const pointItem = arranged.vertices_coords[vertex];
    if (!pointItem) return [];
    return [{
      x: round(pointItem[0]),
      y: round(pointItem[1]),
      kawasaki: kawasakiBadVertices.includes(vertex),
      maekawa: maekawaBadVertices.includes(vertex),
    }];
  });
  return {
    activeSegmentKinds,
    arrangedVertices: arranged.vertices_coords.length,
    arrangedEdges: arranged.edges_vertices.length,
    kawasakiBad: kawasakiBadVertices.length,
    maekawaBad: maekawaBadVertices.length,
    badVertices: badVertexSet.size,
    locallyFlatFoldable: badVertexSet.size === 0,
    failureReasons,
    failurePoints,
  };
}

function classifyLocalFailure(
  graph: ReturnType<typeof arrangeSegments> & { vertices_edges?: number[][] },
  vertex: number,
  kawasakiFailed: boolean,
): string {
  if (kawasakiFailed) return "kawasaki-geometry";
  const incidentEdges = graph.vertices_edges?.[vertex] ?? [];
  const foldedDegree = incidentEdges.filter((edge) => graph.edges_assignment[edge] !== "B").length;
  if (foldedDegree <= 1) return "dangling-active-endpoint";
  if (foldedDegree % 2 === 1) return "odd-active-degree";
  return "maekawa-assignment";
}

export function solveRegionPleatStripPhases(layout: RegionLayout): RegionPhaseSolveResult {
  if (!layout.portConstraints?.length) return { ok: true, layout, errors: [] };
  const problem = regionPhaseProblem(layout);
  const solver = solvePortAssignmentProblem(problem);
  if (!solver.ok) {
    return {
      ok: false,
      layout,
      solver,
      errors: solver.errors.map((error) => `port-phase:${error}`),
    };
  }
  return {
    ok: true,
    layout: applyRegionPhaseSolution(layout, solver),
    solver,
    errors: [],
  };
}

export function regionPhaseProblem(layout: RegionLayout): PortSolverProblem {
  return {
    id: `${layout.id}-port-phase-problem`,
    regions: layout.pleatStrips.map((strip, index) => ({
      id: strip.id,
      rank: index,
      states: pleatStripStates(strip),
    })),
    constraints: (layout.portConstraints ?? []).map((constraint) => ({
      id: constraint.id,
      aRegion: constraint.aStripId,
      aPort: constraint.aSide,
      bRegion: constraint.bStripId,
      bPort: constraint.bSide,
      sequenceOrder: constraint.sequenceOrder,
    })),
  };
}

export interface RegionSvgOptions {
  showGrid?: boolean;
  showLegend?: boolean;
  showFlapTargets?: boolean;
  showFlapBoundaries?: boolean;
}

export function regionCandidateToSvg(candidate: RegionCompletionCandidate, size = 900, options: RegionSvgOptions = {}): string {
  const showGrid = options.showGrid ?? true;
  const showLegend = options.showLegend ?? true;
  const showFlapTargets = options.showFlapTargets ?? true;
  const showFlapBoundaries = options.showFlapBoundaries ?? true;
  const strokeScale = size / 900;
  const toPx = ([x, y]: Point): Point => [round(x * size), round((1 - y) * size)];
  const colors = {
    mountain: "#ff1f1f",
    valley: "#0057ff",
    border: "#111827",
    pleatStrip: "#facc15",
    body: "#93c5fd",
    flap: "#4ade80",
    legendText: "#111827",
    legendMuted: "#475569",
  };
  const line = (segment: RegionCandidateSegment): string => {
    const [x1, y1] = toPx(segment.p1);
    const [x2, y2] = toPx(segment.p2);
    const color = segment.assignment === "M" ? colors.mountain : segment.assignment === "V" ? colors.valley : colors.border;
    const width = segment.kind === "border" ? 3.4 : segment.kind === "strip-pleat" ? 3.6 : segment.kind === "stair-boundary" ? 3.5 : 1.8;
    const dash = segment.kind === "body-boundary" || segment.kind === "flap-boundary" ? " stroke-dasharray=\"5 4\"" : "";
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${round(width * strokeScale)}" stroke-linecap="round" stroke-opacity="0.98"${dash}/>`;
  };
  const rect = (item: { rect: RegionRect }, color: string, opacity: number): string => {
    const [x1, y1] = toPx([item.rect.x1, item.rect.y2]);
    const [x2, y2] = toPx([item.rect.x2, item.rect.y1]);
    return `<rect x="${x1}" y="${y1}" width="${round(x2 - x1)}" height="${round(y2 - y1)}" fill="${color}" opacity="${opacity}"/>`;
  };
  const latticeSize = Math.min(Math.max(8, Math.round(1 / pleatPitchForGrid(candidate.layout.gridSize))), 32);
  const majorEvery = Math.max(1, Math.round(latticeSize / 16));
  const mediumEvery = Math.max(1, Math.round(latticeSize / 64));
  const gridLines = Array.from({ length: latticeSize + 1 }, (_, index) => index / latticeSize).flatMap((v, index) => {
    const [x, y] = toPx([v, v]);
    const major = index % majorEvery === 0;
    const medium = index % mediumEvery === 0;
    const color = major ? "#94a3b8" : medium ? "#cbd5e1" : "#e2e8f0";
    const width = major ? 0.7 : medium ? 0.42 : 0.24;
    const opacity = major ? 0.22 : medium ? 0.10 : 0.045;
    return [
      `<line x1="${x}" y1="0" x2="${x}" y2="${size}" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}"/>`,
      `<line x1="0" y1="${y}" x2="${size}" y2="${y}" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}"/>`,
    ];
  });
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" shape-rendering="geometricPrecision">`,
    `<rect width="${size}" height="${size}" fill="white"/>`,
    ...(showGrid ? gridLines : []),
    ...candidate.layout.pleatStrips.map((strip) => rect(strip, colors.pleatStrip, 0.34)),
    ...candidate.layout.bodies.map((body) => rect(body, colors.body, 0.32)),
    ...(showFlapTargets ? candidate.layout.flaps.map((flap) => rect(flap, colors.flap, 0.34)) : []),
    ...candidate.segments.filter((segment) => showFlapBoundaries || segment.kind !== "flap-boundary").map(line),
    showLegend ? legendSvg(size, colors) : "",
    `</svg>`,
  ].join("\n");
}

function legendSvg(
  size: number,
  colors: {
    mountain: string;
    valley: string;
    border: string;
    pleatStrip: string;
    body: string;
    flap: string;
    legendText: string;
    legendMuted: string;
  },
): string {
  const x = round(size * 0.022);
  const y = round(size * 0.022);
  const width = round(size * 0.27);
  const row = round(size * 0.024);
  const swatch = round(size * 0.014);
  const lineX1 = x + swatch * 0.2;
  const lineX2 = x + swatch * 1.45;
  const labelX = x + swatch * 2.0;
  const titleY = y + row * 0.95;
  const firstY = y + row * 2.0;
  const height = round(row * 8.0);
  const fontSize = Math.max(10, round(size * 0.014));
  const titleSize = Math.max(11, round(size * 0.016));
  const item = (index: number, label: string, mark: string): string => {
    const cy = firstY + row * index;
    return `${mark}<text x="${labelX}" y="${cy + fontSize * 0.35}" font-family="Inter, Arial, sans-serif" font-size="${fontSize}" fill="${colors.legendText}">${label}</text>`;
  };
  return [
    `<g data-debug-legend="bp-region">`,
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="7" fill="white" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1"/>`,
    `<text x="${x + swatch}" y="${titleY}" font-family="Inter, Arial, sans-serif" font-size="${titleSize}" font-weight="700" fill="${colors.legendText}">Debug legend</text>`,
    item(0, "Pleat corridor", `<rect x="${lineX1}" y="${firstY - swatch * 0.55}" width="${swatch * 1.25}" height="${swatch * 0.95}" fill="${colors.pleatStrip}" fill-opacity="0.70" stroke="#ca8a04" stroke-width="0.8"/>`),
    item(1, "Body panel", `<rect x="${lineX1}" y="${firstY + row - swatch * 0.55}" width="${swatch * 1.25}" height="${swatch * 0.95}" fill="${colors.body}" fill-opacity="0.76" stroke="#2563eb" stroke-width="0.8"/>`),
    item(2, "Flap target", `<rect x="${lineX1}" y="${firstY + row * 2 - swatch * 0.55}" width="${swatch * 1.25}" height="${swatch * 0.95}" fill="${colors.flap}" fill-opacity="0.76" stroke="#16a34a" stroke-width="0.8"/>`),
    item(3, "Mountain crease", `<line x1="${lineX1}" y1="${firstY + row * 3}" x2="${lineX2}" y2="${firstY + row * 3}" stroke="${colors.mountain}" stroke-width="3" stroke-linecap="round"/>`),
    item(4, "Valley crease", `<line x1="${lineX1}" y1="${firstY + row * 4}" x2="${lineX2}" y2="${firstY + row * 4}" stroke="${colors.valley}" stroke-width="3" stroke-linecap="round"/>`),
    item(5, "Debug boundary", `<line x1="${lineX1}" y1="${firstY + row * 5}" x2="${lineX2}" y2="${firstY + row * 5}" stroke="${colors.valley}" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round"/>`),
    `<text x="${x + swatch}" y="${firstY + row * 6.35}" font-family="Inter, Arial, sans-serif" font-size="${Math.max(9, round(size * 0.0115))}" fill="${colors.legendMuted}">Fills are scaffold/debug only</text>`,
    `</g>`,
  ].join("\n");
}

function applyRegionPhaseSolution(layout: RegionLayout, solver: PortSolverResult): RegionLayout {
  const stateByRegion = new Map(solver.states.map((state) => [state.regionId, state]));
  return {
    ...layout,
    pleatStrips: layout.pleatStrips.map((strip) => {
      const state = stateByRegion.get(strip.id);
      if (!state) return strip;
      const sequence = state.ports[0]?.sequence ?? pleatSequence(strip, strip.startAssignment);
      return {
        ...strip,
        phase: state.phase ?? strip.phase,
        startAssignment: sequence[0] ?? strip.startAssignment,
      };
    }),
  };
}

function pleatStripStates(strip: PleatStripRegion): RegionState[] {
  const laneCount = pleatLaneCount(strip);
  const phase0 = strip.startAssignment;
  const phase1 = flip(strip.startAssignment);
  return [
    pleatStripState(strip, phase0, 0, laneCount),
    pleatStripState(strip, phase1, 1, laneCount),
  ];
}

function pleatStripState(
  strip: PleatStripRegion,
  startAssignment: PortAssignment,
  phase: number,
  laneCount: number,
): RegionState {
  const sequence = alternatingSequence(startAssignment, laneCount);
  return {
    id: `${strip.id}:phase-${phase}`,
    regionId: strip.id,
    phase,
    cost: phase,
    ports: [
      solverPort("start", sequence, {
        orientation: strip.orientation === "vertical" ? "vertical" : "horizontal",
        side: strip.orientation === "vertical" ? "left" : "bottom",
        width: laneCount,
        parity: strip.phase % 2 === 0 ? "integer" : "half",
      }),
      solverPort("end", sequence, {
        orientation: strip.orientation === "vertical" ? "vertical" : "horizontal",
        side: strip.orientation === "vertical" ? "right" : "top",
        width: laneCount,
        parity: strip.phase % 2 === 0 ? "integer" : "half",
      }),
    ],
  };
}

function pleatSequence(strip: PleatStripRegion, startAssignment: PortAssignment): PortAssignment[] {
  return alternatingSequence(startAssignment, pleatLaneCount(strip));
}

function pleatLaneCount(strip: PleatStripRegion): number {
  const min = strip.orientation === "vertical" ? strip.rect.x1 : strip.rect.y1;
  const max = strip.orientation === "vertical" ? strip.rect.x2 : strip.rect.y2;
  let count = 0;
  for (let coordinate = nextGridInside(min, strip.pitch); coordinate < max - 1e-9; coordinate += strip.pitch) {
    count += 1;
  }
  return Math.max(1, count);
}

function expandBodyPanelsForCorridorPorts(
  layout: CompletionLayout,
  flaps: FlapRegion[],
  bodies: BodyPanelRegion[],
): BodyPanelRegion[] {
  if (!layout.corridors.length) return bodies;
  const pitch = pleatPitchForGrid(layout.gridSize);
  const corridorWidth = Math.max(pitch * 2, 2 / layout.gridSize);
  const endpoints = endpointMap(flaps, bodies);
  const counts = new Map<string, Record<BodyPortSide, number>>();
  for (const corridor of layout.corridors) {
    const from = endpoints.get(corridor.from);
    const to = endpoints.get(corridor.to);
    if (!from || !to) continue;
    for (const [body, other] of [[from, to], [to, from]] as const) {
      if (body.kind !== "body" || !body.rect) continue;
      const side = bodyPortSideForCorridor(corridor, body, other, pitch);
      const record = counts.get(body.id) ?? { left: 0, right: 0, top: 0, bottom: 0 };
      record[side] += 1;
      counts.set(body.id, record);
    }
  }
  return bodies.map((body) => {
    const record = counts.get(body.id);
    if (!record) return body;
    const verticalPorts = Math.max(record.left, record.right, 1);
    const horizontalPorts = Math.max(record.top, record.bottom, 1);
    const width = Math.max(body.rect.x2 - body.rect.x1, horizontalPorts * corridorWidth);
    const height = Math.max(body.rect.y2 - body.rect.y1, verticalPorts * corridorWidth);
    const rect = snapRectToStep(rectAround(body.center, width, height), pitch);
    return { ...body, rect, center: rectCenter(rect) };
  });
}

function endpointMap(flaps: FlapRegion[], bodies: BodyPanelRegion[]): Map<string, CorridorEndpoint> {
  const endpoints = new Map<string, CorridorEndpoint>();
  for (const flap of flaps) {
    endpoints.set(flap.terminalId, {
      id: flap.terminalId,
      center: flap.center,
      kind: "flap",
      allocationRadius: flap.allocationRadius,
      rect: flap.rect,
    });
  }
  for (const body of bodies) endpoints.set(body.id, { id: body.id, center: body.center, kind: "body", rect: body.rect });
  if (!endpoints.has("body") && bodies[0]) endpoints.set("body", { id: "body", center: bodies[0].center, kind: "body", rect: bodies[0].rect });
  return endpoints;
}

function corridorPleatStripRegions(
  layout: CompletionLayout,
  flaps: FlapRegion[],
  bodies: BodyPanelRegion[],
): PleatStripRegion[] {
  const endpoints = endpointMap(flaps, bodies);
  const bodyPorts = bodyPortMapForCorridors(layout, endpoints, bodies);

  return layout.corridors.flatMap((corridor, index) => {
    const from = endpoints.get(corridor.from);
    const to = endpoints.get(corridor.to);
    if (!from || !to) return [];
    return corridorPleatStripRegionsForCorridor(corridor, from, to, layout.gridSize, index, bodyPorts);
  });
}

function bodyPortMapForCorridors(layout: CompletionLayout, endpoints: Map<string, CorridorEndpoint>, bodies: BodyPanelRegion[]): BodyPortMap {
  const pitch = pleatPitchForGrid(layout.gridSize);
  const grouped = new Map<string, Array<{
    key: string;
    body: CorridorEndpoint;
    other: CorridorEndpoint;
    corridor: CompletionCorridor;
    side: BodyPortSide;
  }>>();
  for (const corridor of layout.corridors) {
    const from = endpoints.get(corridor.from);
    const to = endpoints.get(corridor.to);
    if (!from || !to) continue;
    for (const [body, other] of [[from, to], [to, from]] as const) {
      if (body.kind !== "body" || !body.rect) continue;
      const side = bodyPortSideForCorridor(corridor, body, other, pitch);
      const groupKey = `${body.id}:${side}`;
      const list = grouped.get(groupKey) ?? [];
      list.push({ key: bodyPortKey(corridor.id, body.id), body, other, corridor, side });
      grouped.set(groupKey, list);
    }
  }

  const result: BodyPortMap = new Map();
  for (const list of grouped.values()) {
    list.sort((a, b) => bodySideSortCoordinate(a.other, a.side) - bodySideSortCoordinate(b.other, b.side) ||
      a.key.localeCompare(b.key));
    for (const [index, item] of list.entries()) {
      const basePort = bodyPortPoint(item.body, item.side, index, list.length, pitch);
      const flapClamped = clampBodyPortAwayFromFlapAllocation(basePort, item.corridor, item.other, pitch);
      result.set(item.key, clampBodyPortAwayFromBodyRegions(flapClamped, item.corridor, item.body, item.other, bodies, pitch));
    }
  }
  return result;
}

function bodyPortKey(corridorId: string, bodyId: string): string {
  return `${corridorId}:${bodyId}`;
}

function bodyPortSideForCorridor(
  corridor: CompletionCorridor,
  body: CorridorEndpoint,
  other: CorridorEndpoint,
  pitch: number,
): BodyPortSide {
  const rect = body.rect!;
  const lane = snapToStep(corridor.coordinate, pitch);
  if (corridor.orientation === "vertical") {
    if (lane < rect.x1 - 1e-9) return "left";
    if (lane > rect.x2 + 1e-9) return "right";
    return other.center.y >= body.center.y ? "top" : "bottom";
  }
  if (lane < rect.y1 - 1e-9) return "bottom";
  if (lane > rect.y2 + 1e-9) return "top";
  return other.center.x >= body.center.x ? "right" : "left";
}

function bodySideSortCoordinate(endpoint: CorridorEndpoint, side: BodyPortSide): number {
  return side === "left" || side === "right" ? endpoint.center.y : endpoint.center.x;
}

function bodyPortPoint(
  body: CorridorEndpoint,
  side: BodyPortSide,
  index: number,
  count: number,
  pitch: number,
): CompletionPoint {
  const rect = body.rect!;
  const fraction = (index + 1) / (count + 1);
  if (side === "left" || side === "right") {
    const y = snapToStep(rect.y1 + (rect.y2 - rect.y1) * fraction, pitch);
    return point(side === "left" ? rect.x1 : rect.x2, clamp(y, rect.y1, rect.y2));
  }
  const x = snapToStep(rect.x1 + (rect.x2 - rect.x1) * fraction, pitch);
  return point(clamp(x, rect.x1, rect.x2), side === "bottom" ? rect.y1 : rect.y2);
}

function clampBodyPortAwayFromFlapAllocation(
  port: CompletionPoint,
  corridor: CompletionCorridor,
  other: CorridorEndpoint,
  pitch: number,
): CompletionPoint {
  if (other.kind !== "flap" || !other.allocationRadius) return port;
  const halfWidth = pitch;
  if (corridor.orientation === "vertical") {
    const perpendicular = Math.abs(snapToStep(corridor.coordinate, pitch) - other.center.x);
    const along = Math.sqrt(Math.max(0, other.allocationRadius ** 2 - perpendicular ** 2));
    if (other.center.y >= port.y) return point(port.x, Math.min(port.y, other.center.y - along - halfWidth));
    return point(port.x, Math.max(port.y, other.center.y + along + halfWidth));
  }
  const perpendicular = Math.abs(snapToStep(corridor.coordinate, pitch) - other.center.y);
  const along = Math.sqrt(Math.max(0, other.allocationRadius ** 2 - perpendicular ** 2));
  if (other.center.x >= port.x) return point(Math.min(port.x, other.center.x - along - halfWidth), port.y);
  return point(Math.max(port.x, other.center.x + along + halfWidth), port.y);
}

function clampBodyPortAwayFromBodyRegions(
  port: CompletionPoint,
  corridor: CompletionCorridor,
  body: CorridorEndpoint,
  other: CorridorEndpoint,
  bodies: BodyPanelRegion[],
  pitch: number,
): CompletionPoint {
  let result = port;
  for (const obstacle of bodies) {
    if (obstacle.id === body.id || obstacle.id === other.id) continue;
    const approach = bodyApproachRectForPort(result, corridor, pitch);
    const overlap = rectIntersection(approach, obstacle.rect);
    if (!overlap || rectArea(overlap) < 1e-9) continue;
    if (corridor.orientation === "vertical") {
      const y = other.center.y <= result.y ? obstacle.rect.y1 - pitch : obstacle.rect.y2 + pitch;
      result = point(result.x, y);
    } else {
      const x = other.center.x <= result.x ? obstacle.rect.x1 - pitch : obstacle.rect.x2 + pitch;
      result = point(x, result.y);
    }
  }
  return result;
}

function bodyApproachRectForPort(
  port: CompletionPoint,
  corridor: CompletionCorridor,
  pitch: number,
): RegionRect {
  const lane = snapToStep(corridor.coordinate, pitch);
  if (corridor.orientation === "vertical") {
    return normalizeRect({
      x1: port.x,
      x2: lane,
      y1: port.y - pitch,
      y2: port.y + pitch,
    });
  }
  return normalizeRect({
    x1: port.x - pitch,
    x2: port.x + pitch,
    y1: port.y,
    y2: lane,
  });
}

function corridorPleatStripRegionsForCorridor(
  corridor: CompletionCorridor,
  from: CorridorEndpoint,
  to: CorridorEndpoint,
  gridSize: number,
  index: number,
  bodyPorts: BodyPortMap,
): PleatStripRegion[] {
  const unit = 1 / gridSize;
  const pitch = pleatPitchForGrid(gridSize);
  const width = Math.max(snapEvenDistanceToStep(corridor.width, pitch), pitch * 2, unit * 2);
  const fromPort = bodyPorts.get(bodyPortKey(corridor.id, from.id));
  const toPort = bodyPorts.get(bodyPortKey(corridor.id, to.id));
  const core = coreCorridorPleatStripRegion(corridor, from, to, gridSize, index, width, pitch, fromPort, toPort);
  const strips: PleatStripRegion[] = [];
  const fromApproach = bodyApproachStripRegion(corridor, from, to, gridSize, index, width, pitch, "from", fromPort);
  const toApproach = bodyApproachStripRegion(corridor, to, from, gridSize, index, width, pitch, "to", toPort);
  if (fromApproach) strips.push(fromApproach);
  if (rectLongEnough(core.rect, corridor.orientation, pitch)) strips.push(core);
  if (toApproach) strips.push(toApproach);
  return strips;
}

function coreCorridorPleatStripRegion(
  corridor: CompletionCorridor,
  from: CorridorEndpoint,
  to: CorridorEndpoint,
  gridSize: number,
  index: number,
  width: number,
  pitch: number,
  fromPort?: CompletionPoint,
  toPort?: CompletionPoint,
): PleatStripRegion {
  const half = width / 2;
  let rect: RegionRect;
  if (corridor.orientation === "horizontal") {
    const y = snapToStep(corridor.coordinate, pitch);
    const x1 = from.kind === "body" && fromPort ? fromPort.x : endpointBoundaryCoordinate(from, to.center, "horizontal", y);
    const x2 = to.kind === "body" && toPort ? toPort.x : endpointBoundaryCoordinate(to, from.center, "horizontal", y);
    rect = normalizeRect({
      x1: snapToStep(x1, pitch),
      y1: snapToStep(y - half, pitch),
      x2: snapToStep(x2, pitch),
      y2: snapToStep(y + half, pitch),
    });
  } else {
    const x = snapToStep(corridor.coordinate, pitch);
    const y1 = from.kind === "body" && fromPort ? fromPort.y : endpointBoundaryCoordinate(from, to.center, "vertical", x);
    const y2 = to.kind === "body" && toPort ? toPort.y : endpointBoundaryCoordinate(to, from.center, "vertical", x);
    rect = normalizeRect({
      x1: snapToStep(x - half, pitch),
      y1: snapToStep(y1, pitch),
      x2: snapToStep(x + half, pitch),
      y2: snapToStep(y2, pitch),
    });
  }
  return pleatStripFromRect(
    `strip-${index}-${corridor.id}`,
    corridor.from,
    corridor.to,
    rect,
    corridor.orientation,
    pitch,
    index,
    corridor.id,
  );
}

function bodyApproachStripRegion(
  corridor: CompletionCorridor,
  body: CorridorEndpoint,
  other: CorridorEndpoint,
  gridSize: number,
  index: number,
  width: number,
  pitch: number,
  side: "from" | "to",
  port?: CompletionPoint,
): PleatStripRegion | undefined {
  if (body.kind !== "body" || !body.rect) return undefined;
  if (!port) return undefined;
  const lane = snapToStep(corridor.coordinate, pitch);
  const half = width / 2;
  let rect: RegionRect | undefined;
  let approachOrientation: CompletionCorridor["orientation"];

  if (corridor.orientation === "vertical") {
    if (lane >= body.rect.x1 - 1e-9 && lane <= body.rect.x2 + 1e-9) return undefined;
    rect = normalizeRect({
      x1: snapToStep(port.x, pitch),
      x2: snapToStep(lane, pitch),
      y1: snapToStep(port.y - half, pitch),
      y2: snapToStep(port.y + half, pitch),
    });
    approachOrientation = "horizontal";
  } else {
    if (lane >= body.rect.y1 - 1e-9 && lane <= body.rect.y2 + 1e-9) return undefined;
    rect = normalizeRect({
      x1: snapToStep(port.x - half, pitch),
      x2: snapToStep(port.x + half, pitch),
      y1: snapToStep(port.y, pitch),
      y2: snapToStep(lane, pitch),
    });
    approachOrientation = "vertical";
  }

  const clamped = clampRect(rect);
  if (!rectLongEnough(clamped, approachOrientation, pitch)) return undefined;
  return pleatStripFromRect(
    `strip-${index}-${corridor.id}-${side}-body-approach`,
    corridor.from,
    corridor.to,
    clamped,
    approachOrientation,
    pitch,
    index + (side === "from" ? 1000 : 2000),
    corridor.id,
  );
}

function pleatStripFromRect(
  id: string,
  from: string,
  to: string,
  rect: RegionRect,
  corridorOrientation: CompletionCorridor["orientation"],
  pitch: number,
  index: number,
  treeEdgeId?: string,
): PleatStripRegion {
  void corridorOrientation;
  const clamped = clampRect(rect);
  const longAxis: PleatStripOrientation = (clamped.x2 - clamped.x1) >= (clamped.y2 - clamped.y1) ? "horizontal" : "vertical";
  return {
    id,
    from,
    to,
    rect: clamped,
    orientation: longAxis,
    pitch,
    phase: 0,
    startAssignment: index % 2 === 0 ? "M" : "V",
    treeEdgeId,
  };
}

function rectLongEnough(rect: RegionRect, corridorOrientation: CompletionCorridor["orientation"], pitch: number): boolean {
  const length = corridorOrientation === "horizontal" ? Math.abs(rect.x2 - rect.x1) : Math.abs(rect.y2 - rect.y1);
  const width = corridorOrientation === "horizontal" ? Math.abs(rect.y2 - rect.y1) : Math.abs(rect.x2 - rect.x1);
  return length >= pitch - 1e-9 && width >= pitch - 1e-9;
}

function endpointBoundaryCoordinate(
  endpoint: CorridorEndpoint,
  toward: CompletionPoint,
  corridorOrientation: CompletionCorridor["orientation"],
  laneCoordinate: number,
): number {
  if (endpoint.kind === "flap") {
    if (!endpoint.allocationRadius) return corridorOrientation === "horizontal" ? endpoint.center.x : endpoint.center.y;
    const perpendicularDelta = corridorOrientation === "horizontal"
      ? laneCoordinate - endpoint.center.y
      : laneCoordinate - endpoint.center.x;
    const alongRadius = Math.sqrt(Math.max(0, endpoint.allocationRadius ** 2 - perpendicularDelta ** 2));
    if (corridorOrientation === "horizontal") {
      return toward.x >= endpoint.center.x
        ? endpoint.center.x + alongRadius
        : endpoint.center.x - alongRadius;
    }
    return toward.y >= endpoint.center.y
      ? endpoint.center.y + alongRadius
      : endpoint.center.y - alongRadius;
  }
  if (!endpoint.rect) return corridorOrientation === "horizontal" ? endpoint.center.x : endpoint.center.y;
  if (corridorOrientation === "horizontal") {
    return toward.x >= endpoint.center.x ? endpoint.rect.x2 : endpoint.rect.x1;
  }
  return toward.y >= endpoint.center.y ? endpoint.rect.y2 : endpoint.rect.y1;
}

function flapRegion(terminal: CompletionTerminal, gridSize: number): FlapRegion {
  const pitch = pleatPitchForGrid(gridSize);
  const center = point(snapToStep(terminal.x, pitch), snapToStep(terminal.y, pitch));
  const width = snapEvenDistanceToStep(clamp(Math.max(terminal.width, 1 / 16), 1 / 16, 3 / 16), pitch);
  const height = snapEvenDistanceToStep(clamp(Math.max(terminal.height, 1 / 16), 1 / 16, 3 / 16), pitch);
  return {
    id: `flap-${terminal.id}`,
    terminalId: terminal.id,
    nodeId: terminal.nodeId,
    side: terminal.side,
    center,
    allocationRadius: terminal.allocationRadius,
    rect: snapRectToStep(rectAround(center, width, height), pitch),
  };
}

function pleatStripRegion(
  flap: FlapRegion,
  body: BodyPanelRegion,
  layout: CompletionLayout,
  index: number,
): PleatStripRegion {
  const horizontal = flap.side === "left" || flap.side === "right" ||
    (flap.side === "interior" && Math.abs(flap.center.x - body.center.x) >= Math.abs(flap.center.y - body.center.y));
  const unit = 1 / layout.gridSize;
  const pitch = pleatPitchForGrid(layout.gridSize);
  const half = Math.max(snapEvenDistanceToStep(DEFAULT_CORRIDOR_WIDTH, pitch), unit * 2) / 2;
  let rect: RegionRect;
  if (horizontal) {
    const y = snapToStep(flap.center.y, pitch);
    const bodyEdge = flap.center.x < body.center.x ? body.rect.x1 : body.rect.x2;
    rect = normalizeRect({
      x1: snapToStep(flap.center.x, pitch),
      y1: snapToStep(y - half, pitch),
      x2: snapToStep(bodyEdge, pitch),
      y2: snapToStep(y + half, pitch),
    });
  } else {
    const x = snapToStep(flap.center.x, pitch);
    const bodyEdge = flap.center.y < body.center.y ? body.rect.y1 : body.rect.y2;
    rect = normalizeRect({
      x1: snapToStep(x - half, pitch),
      y1: snapToStep(flap.center.y, pitch),
      x2: snapToStep(x + half, pitch),
      y2: snapToStep(bodyEdge, pitch),
    });
  }
  return {
    id: `strip-${index}-${flap.terminalId}`,
    from: flap.id,
    to: body.id,
    rect: clampRect(rect),
    orientation: horizontal ? "vertical" : "horizontal",
    pitch,
    phase: 0,
    startAssignment: index % 2 === 0 ? "M" : "V",
  };
}

function pleatSegments(strip: PleatStripRegion): RegionCandidateSegment[] {
  const result: RegionCandidateSegment[] = [];
  const min = strip.orientation === "vertical" ? strip.rect.x1 : strip.rect.y1;
  const max = strip.orientation === "vertical" ? strip.rect.x2 : strip.rect.y2;
  let lineIndex = 0;
  for (let coordinate = nextGridInside(min, strip.pitch); coordinate < max - 1e-9; coordinate += strip.pitch) {
    const assignment = alternate(strip.startAssignment, lineIndex);
    const p1: Point = strip.orientation === "vertical" ? [coordinate, strip.rect.y1] : [strip.rect.x1, coordinate];
    const p2: Point = strip.orientation === "vertical" ? [coordinate, strip.rect.y2] : [strip.rect.x2, coordinate];
    result.push({
      id: `${strip.id}-pleat-${lineIndex}`,
      regionId: strip.id,
      kind: "strip-pleat",
      p1: roundPoint(p1),
      p2: roundPoint(p2),
      assignment,
      role: strip.orientation === "vertical" ? "axis" : "hinge",
    });
    lineIndex += 1;
  }
  return result;
}

function stairBoundariesForStrip(strip: PleatStripRegion): StairBoundary[] {
  const start = stairBoundaryForStripSide(strip, "start");
  const end = stairBoundaryForStripSide(strip, "end");
  return [start, end];
}

function stairBoundaryForStripSide(strip: PleatStripRegion, side: "start" | "end"): StairBoundary {
  const rect = strip.rect;
  const firstAssignment = side === "start" ? "M" : "V";
  const lines: StairBoundary["lines"] = [];
  if (strip.orientation === "vertical") {
    const x = side === "start" ? rect.x1 : rect.x2;
    const dx = side === "start" ? strip.pitch : -strip.pitch;
    const stepCount = Math.max(1, Math.round((rect.y2 - rect.y1) / strip.pitch));
    for (let index = 0; index < stepCount; index += 1) {
      const y0 = round(rect.y1 + index * strip.pitch);
      const y1 = round(Math.min(rect.y2, y0 + strip.pitch));
      const rising = index % 2 === 0;
      lines.push({
        p1: roundPoint([x, rising ? y0 : y1]),
        p2: roundPoint([x + dx, rising ? y1 : y0]),
        assignment: alternate(firstAssignment, index),
        role: "ridge",
      });
    }
  } else {
    const y = side === "start" ? rect.y1 : rect.y2;
    const dy = side === "start" ? strip.pitch : -strip.pitch;
    const stepCount = Math.max(1, Math.round((rect.x2 - rect.x1) / strip.pitch));
    for (let index = 0; index < stepCount; index += 1) {
      const x0 = round(rect.x1 + index * strip.pitch);
      const x1 = round(Math.min(rect.x2, x0 + strip.pitch));
      const rising = index % 2 === 0;
      lines.push({
        p1: roundPoint([rising ? x0 : x1, y]),
        p2: roundPoint([rising ? x1 : x0, y + dy]),
        assignment: alternate(firstAssignment, index),
        role: "ridge",
      });
    }
  }
  return { id: `${strip.id}-${side}-stair`, stripId: strip.id, side, lines };
}

function boundaryPortsForStrip(strip: PleatStripRegion): BoundaryPort[] {
  const rect = strip.rect;
  if (strip.orientation === "vertical") {
    return [
      { id: `${strip.id}-left`, regionId: strip.id, side: "left", orientation: "vertical", position: point(rect.x1, (rect.y1 + rect.y2) / 2), width: rect.y2 - rect.y1 },
      { id: `${strip.id}-right`, regionId: strip.id, side: "right", orientation: "vertical", position: point(rect.x2, (rect.y1 + rect.y2) / 2), width: rect.y2 - rect.y1 },
    ];
  }
  return [
    { id: `${strip.id}-bottom`, regionId: strip.id, side: "bottom", orientation: "horizontal", position: point((rect.x1 + rect.x2) / 2, rect.y1), width: rect.x2 - rect.x1 },
    { id: `${strip.id}-top`, regionId: strip.id, side: "top", orientation: "horizontal", position: point((rect.x1 + rect.x2) / 2, rect.y2), width: rect.x2 - rect.x1 },
  ];
}

function sheetBorderSegments(): RegionCandidateSegment[] {
  return [
    segment("border-bottom", "sheet", "border", [0, 0], [1, 0], "B", "border"),
    segment("border-right", "sheet", "border", [1, 0], [1, 1], "B", "border"),
    segment("border-top", "sheet", "border", [1, 1], [0, 1], "B", "border"),
    segment("border-left", "sheet", "border", [0, 1], [0, 0], "B", "border"),
  ];
}

function rectBoundarySegments(
  id: string,
  regionId: string,
  kind: RegionCandidateSegment["kind"],
  rect: RegionRect,
  assignment: Extract<EdgeAssignment, "M" | "V" | "B">,
): RegionCandidateSegment[] {
  return [
    segment(`${id}-bottom`, regionId, kind, [rect.x1, rect.y1], [rect.x2, rect.y1], assignment, assignment === "B" ? "border" : "hinge"),
    segment(`${id}-right`, regionId, kind, [rect.x2, rect.y1], [rect.x2, rect.y2], assignment, assignment === "B" ? "border" : "axis"),
    segment(`${id}-top`, regionId, kind, [rect.x2, rect.y2], [rect.x1, rect.y2], assignment, assignment === "B" ? "border" : "hinge"),
    segment(`${id}-left`, regionId, kind, [rect.x1, rect.y2], [rect.x1, rect.y1], assignment, assignment === "B" ? "border" : "axis"),
  ];
}

function segment(
  id: string,
  regionId: string,
  kind: RegionCandidateSegment["kind"],
  p1: Point,
  p2: Point,
  assignment: RegionCandidateSegment["assignment"],
  role: RegionCandidateSegment["role"],
): RegionCandidateSegment {
  return { id, regionId, kind, p1: roundPoint(p1), p2: roundPoint(p2), assignment, role };
}

function dedupeSegments(segments: RegionCandidateSegment[]): RegionCandidateSegment[] {
  const seen = new Map<string, RegionCandidateSegment>();
  for (const item of segments) {
    if (distance(item.p1, item.p2) < 1e-9) continue;
    const key = segmentKey(item.p1, item.p2);
    if (!seen.has(key)) seen.set(key, item);
  }
  return [...seen.values()];
}

function offGridRejections(segments: RegionCandidateSegment[], gridSize: number): string[] {
  const result: string[] = [];
  for (const item of segments) {
    const axisAligned = item.p1[0] === item.p2[0] || item.p1[1] === item.p2[1];
    const diagonal45 = Math.abs(Math.abs(item.p1[0] - item.p2[0]) - Math.abs(item.p1[1] - item.p2[1])) < 1e-6;
    if (!axisAligned && !diagonal45) {
      result.push(`non-grid-angle:${item.id}`);
      continue;
    }
    if (![...item.p1, ...item.p2].every((value) => isOnGrid(value, gridSize))) {
      result.push(`off-grid-coordinate:${item.id}`);
    }
  }
  return result;
}

function overlapRejections(layout: RegionLayout): string[] {
  const result: string[] = [];
  for (const strip of layout.pleatStrips) {
    for (const body of layout.bodies) {
      if (stripTouchesBody(strip, body.id)) continue;
      const overlap = rectIntersection(strip.rect, body.rect);
      if (!overlap || rectArea(overlap) < 1e-9) continue;
      result.push(`pleat-strip-body-overlap:${strip.id}:${body.id}`);
    }
  }
  for (let aIndex = 0; aIndex < layout.pleatStrips.length; aIndex += 1) {
    for (let bIndex = aIndex + 1; bIndex < layout.pleatStrips.length; bIndex += 1) {
      const a = layout.pleatStrips[aIndex];
      const b = layout.pleatStrips[bIndex];
      const overlap = rectIntersection(a.rect, b.rect);
      if (!overlap || rectArea(overlap) < 1e-9) continue;
      const bodyOwner = layout.bodies.find((body) => rectContainsPoint(body.rect, rectCenter(overlap)));
      if (bodyOwner) {
        if (stripTouchesBody(a, bodyOwner.id) && stripTouchesBody(b, bodyOwner.id)) continue;
        result.push(`pleat-strip-body-overlap:${a.id}:${b.id}:${bodyOwner.id}`);
        continue;
      }
      if (a.treeEdgeId && a.treeEdgeId === b.treeEdgeId) {
        const turnArea = Math.max(a.pitch, b.pitch) ** 2 * 4 + 1e-9;
        if (rectArea(overlap) <= turnArea) continue;
      }
      if ((a.treeEdgeId || b.treeEdgeId) && stripsShareEndpoint(a, b)) {
        const branchArea = Math.max(a.pitch, b.pitch) ** 2 * 4 + 1e-9;
        if (rectArea(overlap) <= branchArea) continue;
      }
      result.push(`pleat-strip-overlap:${a.id}:${b.id}`);
    }
  }
  result.push(...flapAllocationOverlapRejections(layout));
  return result;
}

function stripTouchesBody(strip: PleatStripRegion, bodyId: string): boolean {
  return strip.from === bodyId || strip.to === bodyId;
}

function flapAllocationOverlapRejections(layout: RegionLayout): string[] {
  const result: string[] = [];
  const pitch = Math.min(...layout.pleatStrips.map((strip) => strip.pitch), 1 / Math.max(1, layout.gridSize));
  const tolerance = pitch / 4;
  const flaps = layout.flaps.filter((flap) => flap.allocationRadius && flap.allocationRadius > tolerance);
  for (const flap of flaps) {
    const radius = flap.allocationRadius!;
    for (const body of layout.bodies) {
      if (rectCircleInteriorOverlap(body.rect, flap.center, radius, tolerance)) {
        result.push(`flap-allocation-overlap:body-${body.id}:${flap.terminalId}`);
      }
    }
    for (const strip of layout.pleatStrips) {
      if (rectCircleInteriorOverlap(strip.rect, flap.center, radius, tolerance)) {
        result.push(`flap-allocation-overlap:${strip.id}:${flap.terminalId}`);
      }
    }
  }
  return result;
}

function stripsShareEndpoint(a: PleatStripRegion, b: PleatStripRegion): boolean {
  return a.from === b.from || a.from === b.to || a.to === b.from || a.to === b.to;
}

function rectCircleInteriorOverlap(
  rect: RegionRect,
  center: CompletionPoint,
  radius: number,
  tolerance: number,
): boolean {
  if (rectArea(rect) < 1e-12) return false;
  const closestX = clamp(center.x, rect.x1, rect.x2);
  const closestY = clamp(center.y, rect.y1, rect.y2);
  const distanceToRect = Math.hypot(center.x - closestX, center.y - closestY);
  return distanceToRect < radius - tolerance;
}

function rectIntersection(a: RegionRect, b: RegionRect): RegionRect | undefined {
  const rect = {
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
    x2: Math.min(a.x2, b.x2),
    y2: Math.min(a.y2, b.y2),
  };
  return rect.x2 > rect.x1 && rect.y2 > rect.y1 ? rect : undefined;
}

function rectContainsPoint(container: RegionRect, inner: CompletionPoint): boolean {
  return inner.x >= container.x1 - 1e-9 &&
    inner.y >= container.y1 - 1e-9 &&
    inner.x <= container.x2 + 1e-9 &&
    inner.y <= container.y2 + 1e-9;
}

function rectArea(rect: RegionRect): number {
  return Math.max(0, rect.x2 - rect.x1) * Math.max(0, rect.y2 - rect.y1);
}

function sortedUnique(values: readonly number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function alternate(start: Extract<EdgeAssignment, "M" | "V">, index: number): Extract<EdgeAssignment, "M" | "V"> {
  return index % 2 === 0 ? start : flip(start);
}

function flip(assignment: Extract<EdgeAssignment, "M" | "V">): Extract<EdgeAssignment, "M" | "V"> {
  return assignment === "M" ? "V" : "M";
}

function rectAround(center: CompletionPoint, width: number, height: number): RegionRect {
  return clampRect({
    x1: center.x - width / 2,
    y1: center.y - height / 2,
    x2: center.x + width / 2,
    y2: center.y + height / 2,
  });
}

function snapRectToStep(rect: RegionRect, step: number): RegionRect {
  const normalized = normalizeRect(rect);
  const center = rectCenter(normalized);
  const width = Math.max(step * 2, snapEvenDistanceToStep(normalized.x2 - normalized.x1, step));
  const height = Math.max(step * 2, snapEvenDistanceToStep(normalized.y2 - normalized.y1, step));
  return rectAround(
    point(snapToStep(center.x, step), snapToStep(center.y, step)),
    width,
    height,
  );
}

function normalizeRect(rect: RegionRect): RegionRect {
  return {
    x1: Math.min(rect.x1, rect.x2),
    y1: Math.min(rect.y1, rect.y2),
    x2: Math.max(rect.x1, rect.x2),
    y2: Math.max(rect.y1, rect.y2),
  };
}

function clampRect(rect: RegionRect): RegionRect {
  const normalized = normalizeRect(rect);
  return {
    x1: round(clamp(normalized.x1, 0, 1)),
    y1: round(clamp(normalized.y1, 0, 1)),
    x2: round(clamp(normalized.x2, 0, 1)),
    y2: round(clamp(normalized.y2, 0, 1)),
  };
}

function rectCenter(rect: RegionRect): CompletionPoint {
  return point((rect.x1 + rect.x2) / 2, (rect.y1 + rect.y2) / 2);
}

function point(x: number, y: number): CompletionPoint {
  return { x: round(clamp(x, 0, 1)), y: round(clamp(y, 0, 1)) };
}

function snapToGrid(value: number, gridSize: number): number {
  return round(Math.round(value * gridSize) / gridSize);
}

function snapDistance(value: number, gridSize: number): number {
  return round(Math.max(1, Math.round(value * gridSize)) / gridSize);
}

function snapDistanceToStep(value: number, step: number): number {
  return round(Math.max(1, Math.round(value / step)) * step);
}

function snapEvenDistanceToStep(value: number, step: number): number {
  const steps = Math.max(2, Math.round(value / step));
  return round((steps % 2 === 0 ? steps : steps + 1) * step);
}

function snapToStep(value: number, step: number): number {
  return round(Math.round(value / step) * step);
}

function snapEvenDistance(value: number, gridSize: number): number {
  const lattice = 1 / (gridSize * 2);
  const steps = Math.max(2, Math.round(value / lattice));
  return round((steps % 2 === 0 ? steps : steps + 1) * lattice);
}

function pleatPitchForGrid(gridSize: number): number {
  return 1 / Math.min(gridSize, 32);
}

function nextGridInside(value: number, pitch: number): number {
  return round(Math.ceil((value + 1e-9) / pitch) * pitch);
}

function isOnGrid(value: number, gridSize: number): boolean {
  const denominator = Math.min(gridSize, 32);
  return Math.abs(value * denominator - Math.round(value * denominator)) < 1e-6;
}

function roundPoint(point: Point): Point {
  return [round(point[0]), round(point[1])];
}

function segmentKey(a: Point, b: Point): string {
  const left = `${a[0].toFixed(9)},${a[1].toFixed(9)}`;
  const right = `${b[0].toFixed(9)},${b[1].toFixed(9)}`;
  return left < right ? `${left}:${right}` : `${right}:${left}`;
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
