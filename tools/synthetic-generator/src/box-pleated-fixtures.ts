// Hand-authored box-pleated polygon-packing fixtures.
//
// Unlike the BP Studio packings (which only satisfy the circle/tree non-overlap
// condition and leave large unused regions), these layouts are authored to tile
// the sheet exactly: every point of paper is inside exactly one flap or river
// polygon (Origami Design Secrets polygon-packing rule #4). They are the
// ground-truth inputs for building and unit-testing the crease construction
// (axials -> pleats -> M/V) before pointing it at real packings.
//
// This module is intentionally crease-free. It defines the polygon layout, the
// tiling metrics, and a renderer. Reference points / axes / creases come later.

export interface GridPoint {
  x: number;
  y: number;
}

export type BoxPleatedPolygonKind = "flap" | "river";

export interface BoxPleatedPolygon {
  id: string;
  kind: BoxPleatedPolygonKind;
  /** Closed polygon as ordered grid vertices; edges must be axis-aligned or 45deg. */
  vertices: GridPoint[];
}

export interface BoxPleatedPolygonLayout {
  id: string;
  description: string;
  sheet: { width: number; height: number };
  polygons: BoxPleatedPolygon[];
}

export interface PolygonLayoutMetrics {
  sheetArea: number;
  /** Sum of |shoelace area| over all polygons; equals sheetArea for an exact tiling. */
  polygonAreaSum: number;
  /** Sampled fraction of sheet inside exactly one polygon. */
  coveragePct: number;
  /** Sampled fraction inside two or more polygons (overlap). */
  overlapPct: number;
  /** Sampled fraction inside no polygon (unused space). */
  gapPct: number;
  flapCount: number;
  riverCount: number;
  offGridVertices: number;
  nonBoxPleatEdges: number;
  tiles: boolean;
  errors: string[];
}

const SAMPLE_STEP = 0.2;
const EPS = 1e-9;

export function validatePolygonLayout(layout: BoxPleatedPolygonLayout): PolygonLayoutMetrics {
  const { width, height } = layout.sheet;
  const sheetArea = width * height;
  const errors: string[] = [];

  let offGridVertices = 0;
  let nonBoxPleatEdges = 0;
  let polygonAreaSum = 0;
  for (const polygon of layout.polygons) {
    for (const vertex of polygon.vertices) {
      if (!Number.isInteger(vertex.x) || !Number.isInteger(vertex.y)) offGridVertices++;
    }
    for (let i = 0; i < polygon.vertices.length; i++) {
      const a = polygon.vertices[i];
      const b = polygon.vertices[(i + 1) % polygon.vertices.length];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      if (!(dx === 0 || dy === 0 || Math.abs(dx) === Math.abs(dy))) nonBoxPleatEdges++;
    }
    polygonAreaSum += Math.abs(shoelaceArea(polygon.vertices));
  }

  // Sampled coverage classification.
  let total = 0;
  let covered = 0;
  let overlap = 0;
  let gap = 0;
  for (let x = SAMPLE_STEP / 2; x < width; x += SAMPLE_STEP) {
    for (let y = SAMPLE_STEP / 2; y < height; y += SAMPLE_STEP) {
      total++;
      let count = 0;
      for (const polygon of layout.polygons) {
        if (pointInPolygon({ x, y }, polygon.vertices)) count++;
      }
      if (count === 0) gap++;
      else if (count === 1) covered++;
      else overlap++;
    }
  }

  const coveragePct = (100 * covered) / total;
  const overlapPct = (100 * overlap) / total;
  const gapPct = (100 * gap) / total;

  if (offGridVertices > 0) errors.push(`${offGridVertices} off-grid (non-integer) vertices`);
  if (nonBoxPleatEdges > 0) errors.push(`${nonBoxPleatEdges} edges not axis-aligned or 45deg`);
  if (Math.abs(polygonAreaSum - sheetArea) > EPS) {
    errors.push(`polygon area sum ${polygonAreaSum} != sheet area ${sheetArea} (gaps or overlaps)`);
  }
  if (overlap > 0) errors.push(`${overlapPct.toFixed(2)}% of samples lie in overlapping polygons`);
  if (gap > 0) errors.push(`${gapPct.toFixed(2)}% of samples lie in unused space`);

  const tiles =
    errors.length === 0 &&
    Math.abs(polygonAreaSum - sheetArea) <= EPS &&
    overlap === 0 &&
    gap === 0;

  return {
    sheetArea,
    polygonAreaSum,
    coveragePct,
    overlapPct,
    gapPct,
    flapCount: layout.polygons.filter((p) => p.kind === "flap").length,
    riverCount: layout.polygons.filter((p) => p.kind === "river").length,
    offGridVertices,
    nonBoxPleatEdges,
    tiles,
    errors,
  };
}

function shoelaceArea(vertices: GridPoint[]): number {
  let sum = 0;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return sum / 2;
}

function pointInPolygon(point: GridPoint, vertices: GridPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
    const xi = vertices[i].x;
    const yi = vertices[i].y;
    const xj = vertices[j].x;
    const yj = vertices[j].y;
    if ((yi > point.y) !== (yj > point.y) && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// ---------------------------------------------------------------------------
// Fixtures. Each tiles its sheet exactly.
// ---------------------------------------------------------------------------

function rect(id: string, kind: BoxPleatedPolygonKind, x0: number, y0: number, x1: number, y1: number): BoxPleatedPolygon {
  return {
    id,
    kind,
    vertices: [
      { x: x0, y: y0 },
      { x: x1, y: y0 },
      { x: x1, y: y1 },
      { x: x0, y: y1 },
    ],
  };
}

/** F1: the whole sheet is a single square flap. */
export function fixtureSingleSquare(): BoxPleatedPolygonLayout {
  return {
    id: "F1-single-square",
    description: "Single square flap filling a 4x4 sheet.",
    sheet: { width: 4, height: 4 },
    polygons: [rect("f0", "flap", 0, 0, 4, 4)],
  };
}

/** F2: two square flaps sharing a vertical hinge. */
export function fixtureTwoFlaps(): BoxPleatedPolygonLayout {
  return {
    id: "F2-two-flaps",
    description: "Two 4x4 flaps tiling an 8x4 sheet, sharing one hinge.",
    sheet: { width: 8, height: 4 },
    polygons: [rect("f0", "flap", 0, 0, 4, 4), rect("f1", "flap", 4, 0, 8, 4)],
  };
}

/** F3: a 2x2 grid of square flaps. */
export function fixtureFourFlaps(): BoxPleatedPolygonLayout {
  return {
    id: "F3-four-flaps",
    description: "Four 4x4 flaps tiling an 8x8 sheet in a 2x2 grid.",
    sheet: { width: 8, height: 8 },
    polygons: [
      rect("f0", "flap", 0, 0, 4, 4),
      rect("f1", "flap", 4, 0, 8, 4),
      rect("f2", "flap", 0, 4, 4, 8),
      rect("f3", "flap", 4, 4, 8, 8),
    ],
  };
}

/** F4: two flaps separated by a river band. */
export function fixtureFlapRiverFlap(): BoxPleatedPolygonLayout {
  return {
    id: "F4-flap-river-flap",
    description: "Two 8x4 flaps separated by a width-4 river, tiling an 8x12 sheet.",
    sheet: { width: 8, height: 12 },
    polygons: [
      rect("f0", "flap", 0, 0, 8, 4),
      rect("re0", "river", 0, 4, 8, 8),
      rect("f1", "flap", 0, 8, 8, 12),
    ],
  };
}

/** F5: an L-shaped flap absorbing a corner gap that a small flap leaves behind. */
export function fixtureLShape(): BoxPleatedPolygonLayout {
  return {
    id: "F5-l-shape",
    description: "An L-shaped flap absorbs the space a 4x4 corner flap does not use (rule #4).",
    sheet: { width: 8, height: 8 },
    polygons: [
      {
        id: "f0",
        kind: "flap",
        vertices: [
          { x: 0, y: 0 },
          { x: 8, y: 0 },
          { x: 8, y: 4 },
          { x: 4, y: 4 },
          { x: 4, y: 8 },
          { x: 0, y: 8 },
        ],
      },
      rect("f1", "flap", 4, 4, 8, 8),
    ],
  };
}

export function allFixtures(): BoxPleatedPolygonLayout[] {
  return [
    fixtureSingleSquare(),
    fixtureTwoFlaps(),
    fixtureFourFlaps(),
    fixtureFlapRiverFlap(),
    fixtureLShape(),
  ];
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const FLAP_FILLS = ["#f2b134", "#5f93ff", "#53df63", "#e879f9", "#fb7185", "#22d3ee"];

export function renderPolygonLayoutSvg(
  layout: BoxPleatedPolygonLayout,
  options: { cellSize?: number } = {},
): string {
  const cellSize = options.cellSize ?? 28;
  const margin = 44;
  const sheetWidth = layout.sheet.width * cellSize;
  const sheetHeight = layout.sheet.height * cellSize;
  const width = sheetWidth + margin * 2;
  const height = sheetHeight + margin * 2;
  const transform = `translate(${margin} ${margin + sheetHeight}) scale(${cellSize} ${-cellSize})`;
  const metrics = validatePolygonLayout(layout);

  let flapIndex = 0;
  const shapes = layout.polygons
    .map((polygon) => {
      const fill = polygon.kind === "river" ? "#38bdf8" : FLAP_FILLS[flapIndex++ % FLAP_FILLS.length];
      const opacity = polygon.kind === "river" ? 0.22 : 0.32;
      const d = `M${polygon.vertices.map((v) => `${v.x},${v.y}`).join("L")}Z`;
      const cx = polygon.vertices.reduce((s, v) => s + v.x, 0) / polygon.vertices.length;
      const cy = polygon.vertices.reduce((s, v) => s + v.y, 0) / polygon.vertices.length;
      const dots = polygon.vertices
        .map((v) => `<circle cx="${v.x}" cy="${v.y}" r="0.12" fill="#f8fafc" stroke="#0f172a" stroke-width="0.03"/>`)
        .join("");
      return `<path d="${d}" fill="${fill}" fill-opacity="${opacity}" stroke="${fill}" stroke-width="2.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
    <text x="${cx}" y="${cy}" font-size="${cellSize * 0.018}" fill="#f8fafc" text-anchor="middle" transform="scale(1 -1)" transform-origin="${cx} ${cy}">${escapeXml(polygon.id)}</text>
    ${dots}`;
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .background { fill: #1f1f1f; }
    .sheet { fill: none; stroke: #e2e8f0; stroke-width: 3; vector-effect: non-scaling-stroke; }
    .grid { stroke: #4b5563; stroke-width: 0.5; opacity: 0.6; vector-effect: non-scaling-stroke; }
    .title { font-family: Arial, sans-serif; font-size: 17px; font-weight: 700; fill: #f5f5f5; }
    .meta { font-family: Arial, sans-serif; font-size: 12px; fill: #c4c4c4; }
  </style>
  <rect width="100%" height="100%" class="background"/>
  <text x="${margin}" y="22" class="title">${escapeXml(layout.id)}</text>
  <text x="${margin}" y="${margin + sheetHeight + 26}" class="meta">${layout.sheet.width}x${layout.sheet.height}, ${metrics.flapCount} flaps, ${metrics.riverCount} rivers — coverage ${metrics.coveragePct.toFixed(1)}%, gap ${metrics.gapPct.toFixed(1)}%, overlap ${metrics.overlapPct.toFixed(1)}%, tiles=${metrics.tiles}</text>
  <g transform="${transform}">
    <rect x="0" y="0" width="${layout.sheet.width}" height="${layout.sheet.height}" class="sheet"/>
    ${renderGrid(layout.sheet.width, layout.sheet.height)}
    ${shapes}
  </g>
</svg>
`;
}

function renderGrid(width: number, height: number): string {
  const lines: string[] = [];
  for (let x = 0; x <= width; x++) lines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${height}" class="grid"/>`);
  for (let y = 0; y <= height; y++) lines.push(`<line x1="0" y1="${y}" x2="${width}" y2="${y}" class="grid"/>`);
  return lines.join("\n");
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
