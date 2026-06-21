import type { BoxPleatedFlap, BoxPleatedPacking } from "./box-pleated-packing.ts";
import type { BpStudioLayoutLine, BpStudioLayoutPoint } from "./bp-studio-layout.ts";

export type BoxPleatedScaffoldLineKind =
  | "boundary"
  | "bp-ridge"
  | "bp-contour"
  | "computed-axial"
  | "gap-ridge"
  | "stretch";

export interface BoxPleatedScaffoldLine {
  kind: BoxPleatedScaffoldLineKind;
  source: string;
  line: BpStudioLayoutLine;
}

export interface BoxPleatedScaffoldConfig {
  fillGaps?: boolean;
}

export interface BoxPleatedCreaseScaffold {
  schemaVersion: "box-pleated-crease-scaffold/v1";
  packingId: string;
  lines: BoxPleatedScaffoldLine[];
  stats: {
    boundary: number;
    bpRidges: number;
    bpContours: number;
    computedAxials: number;
    gapRidges: number;
    stretches: number;
    skippedFlapCells: number;
    alreadyCreasedCells: number;
    filledGapCells: number;
    unfilledGapCells: number;
    rejectedContourSegments: number;
  };
  warnings: string[];
}

interface MutableScaffoldStats {
  skippedFlapCells: number;
  alreadyCreasedCells: number;
  filledGapCells: number;
  unfilledGapCells: number;
  rejectedContourSegments: number;
}

const KIND_PRIORITY: Record<BoxPleatedScaffoldLineKind, number> = {
  boundary: 0,
  "bp-ridge": 1,
  stretch: 2,
  "bp-contour": 3,
  "computed-axial": 4,
  "gap-ridge": 5,
};

export function completeBoxPleatedCreaseScaffold(
  packing: BoxPleatedPacking,
  config: BoxPleatedScaffoldConfig = {},
): BoxPleatedCreaseScaffold {
  const fillGaps = config.fillGaps ?? true;
  const lines = new Map<string, BoxPleatedScaffoldLine>();
  const stats: MutableScaffoldStats = {
    skippedFlapCells: 0,
    alreadyCreasedCells: 0,
    filledGapCells: 0,
    unfilledGapCells: 0,
    rejectedContourSegments: 0,
  };

  addLine(lines, "boundary", "sheet", [{ x: 0, y: 0 }, { x: packing.sheet.width, y: 0 }]);
  addLine(lines, "boundary", "sheet", [{ x: packing.sheet.width, y: 0 }, { x: packing.sheet.width, y: packing.sheet.height }]);
  addLine(lines, "boundary", "sheet", [{ x: packing.sheet.width, y: packing.sheet.height }, { x: 0, y: packing.sheet.height }]);
  addLine(lines, "boundary", "sheet", [{ x: 0, y: packing.sheet.height }, { x: 0, y: 0 }]);

  for (const object of packing.layout.objects) {
    for (const ridge of object.ridges) addLine(lines, "bp-ridge", object.id, ridge);
    for (const axisParallel of object.axisParallel) addLine(lines, "stretch", object.id, axisParallel);
    for (const [contourIndex, contour] of object.contours.entries()) {
      for (const segment of pathSegments(contour.outer)) {
        if (isPure4590Line(segment)) addLine(lines, "bp-contour", `${object.id}:outer:${contourIndex}`, segment);
        else stats.rejectedContourSegments++;
      }
      for (const [innerIndex, inner] of (contour.inner ?? []).entries()) {
        for (const segment of pathSegments(inner)) {
          if (isPure4590Line(segment)) addLine(lines, "bp-contour", `${object.id}:inner:${contourIndex}:${innerIndex}`, segment);
          else stats.rejectedContourSegments++;
        }
      }
    }
  }

  for (const line of [...lines.values()].filter((candidate) => candidate.kind === "bp-ridge")) {
    if (!isDiagonal45(line.line)) continue;
    for (const axial of boundingBoxPerimeter(line.line)) {
      addLine(lines, "computed-axial", `bbox:${line.source}`, axial);
    }
  }

  if (fillGaps) fillEmptyUnitCells(lines, packing.flaps, packing.sheet.width, packing.sheet.height, stats);

  const resultLines = [...lines.values()].sort((a, b) =>
    KIND_PRIORITY[a.kind] - KIND_PRIORITY[b.kind] ||
    lineKey(a.line).localeCompare(lineKey(b.line))
  );
  const counts = countKinds(resultLines);
  const warnings = [
    "Scaffold is unassigned crease geometry only; it is not a flat-foldability proof.",
  ];
  if (stats.rejectedContourSegments > 0) {
    warnings.push(`${stats.rejectedContourSegments} non-45/90 BP contour segments were not imported into the scaffold`);
  }
  return {
    schemaVersion: "box-pleated-crease-scaffold/v1",
    packingId: packing.id,
    lines: resultLines,
    stats: {
      boundary: counts.boundary ?? 0,
      bpRidges: counts["bp-ridge"] ?? 0,
      bpContours: counts["bp-contour"] ?? 0,
      computedAxials: counts["computed-axial"] ?? 0,
      gapRidges: counts["gap-ridge"] ?? 0,
      stretches: counts.stretch ?? 0,
      ...stats,
    },
    warnings,
  };
}

export function renderBoxPleatedCreaseScaffoldSvg(
  packing: BoxPleatedPacking,
  scaffold: BoxPleatedCreaseScaffold,
  options: { cellSize?: number; includeLegend?: boolean } = {},
): string {
  const cellSize = options.cellSize ?? 14;
  const includeLegend = options.includeLegend ?? true;
  const margin = 44;
  const legendWidth = includeLegend ? 280 : 0;
  const sheetWidth = packing.sheet.width * cellSize;
  const sheetHeight = packing.sheet.height * cellSize;
  const width = sheetWidth + margin * 2 + legendWidth;
  const height = sheetHeight + margin * 2;
  const transform = `translate(${margin} ${margin + sheetHeight}) scale(${cellSize} ${-cellSize})`;
  const clipId = `${sanitizeId(packing.id)}-scaffold-clip`;

  const base = scaffold.lines.filter((line) => line.kind !== "boundary");
  const gapRidges = base.filter((line) => line.kind === "gap-ridge").map(renderScaffoldLine).join("\n");
  const bpContours = base.filter((line) => line.kind === "bp-contour").map(renderScaffoldLine).join("\n");
  const computedAxials = base.filter((line) => line.kind === "computed-axial").map(renderScaffoldLine).join("\n");
  const stretches = base.filter((line) => line.kind === "stretch").map(renderScaffoldLine).join("\n");
  const bpRidges = base.filter((line) => line.kind === "bp-ridge").map(renderScaffoldLine).join("\n");
  const flapOutlines = packing.flaps.map(renderFlapOutline).join("\n");
  const dots = packing.flaps.map(renderFlapDots).join("\n");
  const legend = includeLegend ? `  <g transform="translate(${margin + sheetWidth + 28} ${margin})">
    <text class="title" x="0" y="0">Scaffold</text>
    <line x1="0" y1="32" x2="42" y2="32" class="bp-ridge"/>
    <text class="legend" x="54" y="36">BP ridges</text>
    <line x1="0" y1="62" x2="42" y2="62" class="gap-ridge"/>
    <text class="legend" x="54" y="66">gap-fill ridge candidates</text>
    <line x1="0" y1="92" x2="42" y2="92" class="computed-axial"/>
    <text class="legend" x="54" y="96">computed axial candidates</text>
    <line x1="0" y1="122" x2="42" y2="122" class="bp-contour"/>
    <text class="legend" x="54" y="126">BP contour candidates</text>
    <line x1="0" y1="152" x2="42" y2="152" class="stretch"/>
    <text class="legend" x="54" y="156">stretch lines</text>
    <line x1="0" y1="182" x2="42" y2="182" class="flap-outline"/>
    <text class="legend" x="54" y="186">flap outlines</text>
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
    .grid { stroke: #6d6d6d; stroke-width: 0.45; opacity: 0.38; vector-effect: non-scaling-stroke; }
    .flap-outline { fill: none; stroke: #75a7ff; stroke-width: 1.1; vector-effect: non-scaling-stroke; }
    .bp-contour { stroke: #66e0ff; stroke-width: 1.0; opacity: 0.9; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .computed-axial { stroke: #2dd4bf; stroke-width: 0.85; opacity: 0.95; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .gap-ridge { stroke: #f7b84b; stroke-width: 0.55; opacity: 0.7; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .bp-ridge { stroke: #ff3657; stroke-width: 1.55; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .stretch { stroke: #53df63; stroke-width: 1.2; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .dot { fill: #75a7ff; stroke: #d4d4d4; stroke-width: 0.8; vector-effect: non-scaling-stroke; }
    .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: 700; fill: #f5f5f5; }
    .meta { font-family: Arial, sans-serif; font-size: 12px; fill: #c4c4c4; }
    .legend { font-family: Arial, sans-serif; font-size: 12px; fill: #e5e7eb; }
  </style>
  <rect width="100%" height="100%" class="background"/>
  <text x="${margin}" y="24" class="title">${escapeXml(packing.id)} scaffold</text>
  <text x="${margin}" y="40" class="meta">${packing.sheet.width}x${packing.sheet.height}, ${scaffold.stats.bpRidges} BP ridges, ${scaffold.stats.computedAxials} axial, ${scaffold.stats.gapRidges} gap-fill, unassigned</text>
  <g transform="${transform}">
    <rect x="0" y="0" width="${packing.sheet.width}" height="${packing.sheet.height}" class="sheet"/>
    ${renderGrid(packing.sheet.width, packing.sheet.height)}
    <g clip-path="url(#${clipId})">
      ${flapOutlines}
      ${gapRidges}
      ${bpContours}
      ${computedAxials}
      ${stretches}
      ${bpRidges}
      ${dots}
    </g>
  </g>
${legend}
</svg>
`;
}

function fillEmptyUnitCells(
  lines: Map<string, BoxPleatedScaffoldLine>,
  flaps: BoxPleatedFlap[],
  width: number,
  height: number,
  stats: MutableScaffoldStats,
): void {
  const existing = [...lines.values()].filter((line) => line.kind !== "boundary");
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const center = { x: x + 0.5, y: y + 0.5 };
      if (flaps.some((flap) => pointInsideRoundedFlap(center, flap))) {
        stats.skippedFlapCells++;
        continue;
      }
      if (existing.some((line) => pointOnSegment(center, line.line))) {
        stats.alreadyCreasedCells++;
        continue;
      }
      const line: BpStudioLayoutLine = (x + y) % 2 === 0
        ? [{ x, y }, { x: x + 1, y: y + 1 }]
        : [{ x, y: y + 1 }, { x: x + 1, y }];
      addLine(lines, "gap-ridge", "unit-gap-fill", line);
      stats.filledGapCells++;
    }
  }
}

function countKinds(lines: BoxPleatedScaffoldLine[]): Partial<Record<BoxPleatedScaffoldLineKind, number>> {
  const counts: Partial<Record<BoxPleatedScaffoldLineKind, number>> = {};
  for (const line of lines) counts[line.kind] = (counts[line.kind] ?? 0) + 1;
  return counts;
}

function addLine(
  lines: Map<string, BoxPleatedScaffoldLine>,
  kind: BoxPleatedScaffoldLineKind,
  source: string,
  line: BpStudioLayoutLine,
): void {
  if (samePoint(line[0], line[1])) return;
  const normalized = normalizeLine(line);
  const key = lineKey(normalized);
  const existing = lines.get(key);
  if (!existing || KIND_PRIORITY[kind] < KIND_PRIORITY[existing.kind]) {
    lines.set(key, { kind, source, line: normalized });
  }
}

function pathSegments(path: BpStudioLayoutPoint[]): BpStudioLayoutLine[] {
  const result: BpStudioLayoutLine[] = [];
  if (path.length < 2) return result;
  for (let i = 0; i < path.length; i++) {
    result.push([path[i], path[(i + 1) % path.length]]);
  }
  return result;
}

function boundingBoxPerimeter(line: BpStudioLayoutLine): BpStudioLayoutLine[] {
  const minX = Math.min(line[0].x, line[1].x);
  const maxX = Math.max(line[0].x, line[1].x);
  const minY = Math.min(line[0].y, line[1].y);
  const maxY = Math.max(line[0].y, line[1].y);
  return [
    [{ x: minX, y: minY }, { x: maxX, y: minY }],
    [{ x: maxX, y: minY }, { x: maxX, y: maxY }],
    [{ x: maxX, y: maxY }, { x: minX, y: maxY }],
    [{ x: minX, y: maxY }, { x: minX, y: minY }],
  ];
}

function pointInsideRoundedFlap(point: BpStudioLayoutPoint, flap: BoxPleatedFlap): boolean {
  const dx = Math.max(flap.x - point.x, 0, point.x - (flap.x + flap.width));
  const dy = Math.max(flap.y - point.y, 0, point.y - (flap.y + flap.height));
  return dx * dx + dy * dy < flap.radius * flap.radius - 1e-9;
}

function pointOnSegment(point: BpStudioLayoutPoint, line: BpStudioLayoutLine): boolean {
  const [a, b] = line;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const cross = (point.x - a.x) * dy - (point.y - a.y) * dx;
  if (Math.abs(cross) > 1e-9) return false;
  const dot = (point.x - a.x) * dx + (point.y - a.y) * dy;
  if (dot < 1e-9) return false;
  const lengthSquared = dx * dx + dy * dy;
  return dot < lengthSquared - 1e-9;
}

function normalizeLine(line: BpStudioLayoutLine): BpStudioLayoutLine {
  const a = normalizePoint(line[0]);
  const b = normalizePoint(line[1]);
  if (pointKey(a) <= pointKey(b)) return [a, b];
  return [b, a];
}

function normalizePoint(point: BpStudioLayoutPoint): BpStudioLayoutPoint {
  return { x: cleanNumber(point.x), y: cleanNumber(point.y) };
}

function cleanNumber(value: number): number {
  const rounded = Math.round(value);
  return Math.abs(value - rounded) < 1e-9 ? rounded : Number(value.toFixed(9));
}

function lineKey(line: BpStudioLayoutLine): string {
  const normalized = normalizeLine(line);
  return `${pointKey(normalized[0])}|${pointKey(normalized[1])}`;
}

function pointKey(point: BpStudioLayoutPoint): string {
  return `${cleanNumber(point.x)},${cleanNumber(point.y)}`;
}

function samePoint(a: BpStudioLayoutPoint, b: BpStudioLayoutPoint): boolean {
  return a.x === b.x && a.y === b.y;
}

function isDiagonal45(line: BpStudioLayoutLine): boolean {
  const dx = line[1].x - line[0].x;
  const dy = line[1].y - line[0].y;
  return dx !== 0 && dy !== 0 && Math.abs(dx) === Math.abs(dy);
}

function isPure4590Line(line: BpStudioLayoutLine): boolean {
  const dx = line[1].x - line[0].x;
  const dy = line[1].y - line[0].y;
  return dx === 0 || dy === 0 || Math.abs(dx) === Math.abs(dy);
}

function renderScaffoldLine(line: BoxPleatedScaffoldLine): string {
  return `<line x1="${line.line[0].x}" y1="${line.line[0].y}" x2="${line.line[1].x}" y2="${line.line[1].y}" class="${line.kind}"/>`;
}

function renderFlapOutline(flap: BoxPleatedFlap): string {
  const x = flap.x - flap.radius;
  const y = flap.y - flap.radius;
  const width = flap.width + flap.radius * 2;
  const height = flap.height + flap.radius * 2;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${flap.radius}" ry="${flap.radius}" class="flap-outline"/>`;
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

function uniquePoints(points: BpStudioLayoutPoint[]): BpStudioLayoutPoint[] {
  const seen = new Set<string>();
  return points.filter((point) => {
    const key = pointKey(point);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function renderGrid(width: number, height: number): string {
  const lines: string[] = [];
  for (let x = 0; x <= width; x++) lines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${height}" class="grid"/>`);
  for (let y = 0; y <= height; y++) lines.push(`<line x1="0" y1="${y}" x2="${width}" y2="${y}" class="grid"/>`);
  return lines.join("\n");
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
