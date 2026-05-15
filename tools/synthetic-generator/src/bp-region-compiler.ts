import { fixtureCompletionLayout } from "./bp-completion.ts";
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
  RegionLayout,
  RegionRect,
  StairBoundary,
} from "./bp-completion-contracts.ts";
import type { EdgeAssignment } from "./types.ts";

type Point = [number, number];

const DEFAULT_CORRIDOR_WIDTH = 0.25;

export type RegionFixtureName = "two-flap-stretch" | "three-flap-relay" | "five-flap-uniaxial" | "insect-lite";

export function fixtureRegionLayout(name: RegionFixtureName): RegionLayout {
  return regionLayoutFromCompletionLayout(fixtureCompletionLayout(name));
}

export function regionLayoutFromCompletionLayout(layout: CompletionLayout): RegionLayout {
  const bodies = layout.regions.filter((region) => region.kind === "body").map((region): BodyPanelRegion => {
    const rect = normalizeRect({ x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2 });
    return { id: region.id, rect, center: rectCenter(rect) };
  });
  const primaryBody = bodies[0] ?? {
    id: "implicit-body",
    rect: { x1: 0.375, y1: 0.375, x2: 0.625, y2: 0.625 },
    center: { x: 0.5, y: 0.5 },
  };
  const flaps = layout.terminals.slice(0, 8).map((terminal) => flapRegion(terminal, layout.gridSize));
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
  };
}

export function compileRegionCandidate(layout: RegionLayout): RegionCompletionCandidate {
  const segments: RegionCandidateSegment[] = [
    ...sheetBorderSegments(),
    ...layout.bodies.flatMap((body) => rectBoundarySegments(`body-${body.id}`, body.id, "body-boundary", body.rect, "V")),
    ...layout.flaps.flatMap((flap) => rectBoundarySegments(`flap-${flap.id}`, flap.id, "flap-boundary", flap.rect, "V")),
  ];
  const stairBoundaries: StairBoundary[] = [];

  for (const strip of layout.pleatStrips) {
    segments.push(...rectBoundarySegments(`strip-${strip.id}`, strip.id, "body-boundary", strip.rect, "V"));
    segments.push(...pleatSegments(strip));
    const boundaries = stairBoundariesForStrip(strip);
    stairBoundaries.push(...boundaries);
    for (const boundary of boundaries) {
      segments.push(...boundary.lines.map((line, index): RegionCandidateSegment => ({
        id: `${boundary.id}-${index}`,
        regionId: strip.id,
        kind: "stair-boundary",
        p1: line.p1,
        p2: line.p2,
        assignment: line.assignment,
        role: line.role,
      })));
    }
  }

  const arrangedSegments = dedupeSegments(segments);
  const rejectionReasons = [
    ...offGridRejections(arrangedSegments, layout.gridSize),
    ...overlapRejections(layout),
  ];

  return {
    id: `${layout.id}-candidate`,
    layout,
    validity: rejectionReasons.length ? "rejected" : "candidate-complete",
    segments: arrangedSegments,
    stairBoundaries,
    rejectionReasons,
  };
}

export function regionCandidateToSvg(candidate: RegionCompletionCandidate, size = 900): string {
  const strokeScale = size / 900;
  const toPx = ([x, y]: Point): Point => [round(x * size), round((1 - y) * size)];
  const line = (segment: RegionCandidateSegment): string => {
    const [x1, y1] = toPx(segment.p1);
    const [x2, y2] = toPx(segment.p2);
    const color = segment.assignment === "M" ? "#ef4444" : segment.assignment === "V" ? "#2563eb" : "#111827";
    const width = segment.kind === "border" ? 3.2 : segment.kind === "strip-pleat" ? 3.0 : segment.kind === "stair-boundary" ? 3.0 : 1.5;
    const dash = segment.kind === "body-boundary" || segment.kind === "flap-boundary" ? " stroke-dasharray=\"5 4\"" : "";
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${round(width * strokeScale)}" stroke-linecap="round" stroke-opacity="0.96"${dash}/>`;
  };
  const rect = (item: { rect: RegionRect }, color: string, opacity: number): string => {
    const [x1, y1] = toPx([item.rect.x1, item.rect.y2]);
    const [x2, y2] = toPx([item.rect.x2, item.rect.y1]);
    return `<rect x="${x1}" y="${y1}" width="${round(x2 - x1)}" height="${round(y2 - y1)}" fill="${color}" opacity="${opacity}"/>`;
  };
  const gridSize = Math.min(Math.max(4, candidate.layout.gridSize), 32);
  const gridLines = Array.from({ length: gridSize + 1 }, (_, index) => index / gridSize).flatMap((v, index) => {
    const [x, y] = toPx([v, v]);
    const major = index % 4 === 0;
    const color = major ? "#e5e7eb" : "#f3f4f6";
    const width = major ? 0.75 : 0.45;
    return [
      `<line x1="${x}" y1="0" x2="${x}" y2="${size}" stroke="${color}" stroke-width="${width}"/>`,
      `<line x1="0" y1="${y}" x2="${size}" y2="${y}" stroke="${color}" stroke-width="${width}"/>`,
    ];
  });
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">`,
    `<rect width="${size}" height="${size}" fill="white"/>`,
    ...gridLines,
    ...candidate.layout.pleatStrips.map((strip) => rect(strip, "#fde68a", 0.22)),
    ...candidate.layout.bodies.map((body) => rect(body, "#c7d2fe", 0.28)),
    ...candidate.layout.flaps.map((flap) => rect(flap, "#bbf7d0", 0.25)),
    ...candidate.segments.map(line),
    `</svg>`,
  ].join("\n");
}

function corridorPleatStripRegions(
  layout: CompletionLayout,
  flaps: FlapRegion[],
  bodies: BodyPanelRegion[],
): PleatStripRegion[] {
  const centers = new Map<string, CompletionPoint>();
  for (const flap of flaps) centers.set(flap.terminalId, flap.center);
  for (const body of bodies) centers.set(body.id, body.center);
  if (!centers.has("body") && bodies[0]) centers.set("body", bodies[0].center);

  return layout.corridors.flatMap((corridor, index) => {
    const from = centers.get(corridor.from);
    const to = centers.get(corridor.to);
    if (!from || !to) return [];
    return [corridorPleatStripRegion(corridor, from, to, layout.gridSize, index)];
  });
}

function corridorPleatStripRegion(
  corridor: CompletionCorridor,
  from: CompletionPoint,
  to: CompletionPoint,
  gridSize: number,
  index: number,
): PleatStripRegion {
  const unit = 1 / gridSize;
  const pitch = pleatPitchForGrid(gridSize);
  const width = Math.max(snapDistance(corridor.width, gridSize), unit * 2);
  const half = width / 2;
  let rect: RegionRect;
  if (corridor.orientation === "horizontal") {
    const y = snapToGrid(corridor.coordinate, gridSize);
    rect = normalizeRect({
      x1: snapToGrid(from.x, gridSize),
      y1: snapToGrid(y - half, gridSize),
      x2: snapToGrid(to.x, gridSize),
      y2: snapToGrid(y + half, gridSize),
    });
  } else {
    const x = snapToGrid(corridor.coordinate, gridSize);
    rect = normalizeRect({
      x1: snapToGrid(x - half, gridSize),
      y1: snapToGrid(from.y, gridSize),
      x2: snapToGrid(x + half, gridSize),
      y2: snapToGrid(to.y, gridSize),
    });
  }
  return {
    id: `strip-${index}-${corridor.id}`,
    from: corridor.from,
    to: corridor.to,
    rect: clampRect(rect),
    orientation: corridor.orientation === "horizontal" ? "vertical" : "horizontal",
    pitch,
    phase: 0,
    startAssignment: index % 2 === 0 ? "M" : "V",
    treeEdgeId: corridor.id,
  };
}

function flapRegion(terminal: CompletionTerminal, gridSize: number): FlapRegion {
  const center = point(terminal.x, terminal.y);
  const width = snapEvenDistance(clamp(Math.max(terminal.width, 1 / 16), 1 / 16, 3 / 16), gridSize);
  const height = snapEvenDistance(clamp(Math.max(terminal.height, 1 / 16), 1 / 16, 3 / 16), gridSize);
  return {
    id: `flap-${terminal.id}`,
    terminalId: terminal.id,
    nodeId: terminal.nodeId,
    side: terminal.side,
    center,
    rect: rectAround(center, width, height),
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
  const half = Math.max(snapDistance(DEFAULT_CORRIDOR_WIDTH, layout.gridSize), unit * 2) / 2;
  let rect: RegionRect;
  if (horizontal) {
    const y = snapToGrid(flap.center.y, layout.gridSize);
    const bodyEdge = flap.center.x < body.center.x ? body.rect.x1 : body.rect.x2;
    rect = normalizeRect({
      x1: flap.center.x,
      y1: y - half,
      x2: bodyEdge,
      y2: y + half,
    });
  } else {
    const x = snapToGrid(flap.center.x, layout.gridSize);
    const bodyEdge = flap.center.y < body.center.y ? body.rect.y1 : body.rect.y2;
    rect = normalizeRect({
      x1: x - half,
      y1: flap.center.y,
      x2: x + half,
      y2: bodyEdge,
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
  const assignment = side === "start" ? "M" : "V";
  let lines: StairBoundary["lines"];
  if (strip.orientation === "vertical") {
    const x = side === "start" ? rect.x1 : rect.x2;
    const dx = side === "start" ? (rect.y2 - rect.y1) / 2 : -(rect.y2 - rect.y1) / 2;
    const midY = (rect.y1 + rect.y2) / 2;
    lines = [
      { p1: roundPoint([x, rect.y1]), p2: roundPoint([x + dx, midY]), assignment, role: "ridge" },
      { p1: roundPoint([x, rect.y2]), p2: roundPoint([x + dx, midY]), assignment: flip(assignment), role: "ridge" },
    ];
  } else {
    const y = side === "start" ? rect.y1 : rect.y2;
    const dy = side === "start" ? (rect.x2 - rect.x1) / 2 : -(rect.x2 - rect.x1) / 2;
    const midX = (rect.x1 + rect.x2) / 2;
    lines = [
      { p1: roundPoint([rect.x1, y]), p2: roundPoint([midX, y + dy]), assignment, role: "ridge" },
      { p1: roundPoint([rect.x2, y]), p2: roundPoint([midX, y + dy]), assignment: flip(assignment), role: "ridge" },
    ];
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
    const diagonal45 = Math.abs(Math.abs(item.p1[0] - item.p2[0]) - Math.abs(item.p1[1] - item.p2[1])) < 1e-9;
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
  for (let aIndex = 0; aIndex < layout.pleatStrips.length; aIndex += 1) {
    for (let bIndex = aIndex + 1; bIndex < layout.pleatStrips.length; bIndex += 1) {
      const a = layout.pleatStrips[aIndex];
      const b = layout.pleatStrips[bIndex];
      const overlap = rectIntersection(a.rect, b.rect);
      if (!overlap || rectArea(overlap) < 1e-9) continue;
      if (layout.bodies.some((body) => rectContainsPoint(body.rect, rectCenter(overlap)))) continue;
      result.push(`pleat-strip-overlap:${a.id}:${b.id}`);
    }
  }
  return result;
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

function snapEvenDistance(value: number, gridSize: number): number {
  const lattice = 1 / (gridSize * 2);
  const steps = Math.max(2, Math.round(value / lattice));
  return round((steps % 2 === 0 ? steps : steps + 1) * lattice);
}

function pleatPitchForGrid(gridSize: number): number {
  return round(1 / Math.min(gridSize, 32));
}

function nextGridInside(value: number, pitch: number): number {
  return round(Math.ceil((value + 1e-9) / pitch) * pitch);
}

function isOnGrid(value: number, gridSize: number): boolean {
  return Math.abs(value * gridSize * 2 - Math.round(value * gridSize * 2)) < 1e-9;
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
