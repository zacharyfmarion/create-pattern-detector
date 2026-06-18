// Parser for Oriedita `.ori` files containing hand-authored box-pleating
// fixtures. A single file holds many fixtures laid out across the canvas; we
// split them into connected components and normalize each into grid units.
//
// Color semantics (confirmed against the authored fixtures):
//   RED   -> ridge creases (the straight-skeleton ridges; all 45deg)
//   BLUE  -> packing annotation only (flap/river polygon outlines) - DISCARDED
//            for crease assignment; not ground truth.
//   BLACK -> the paper boundary (sheet edge), split where creases meet it.
//
// The crease-construction pipeline consumes { sheet, boundary, ridges } and
// must regenerate axials / pleats / hinges and an M/V assignment from scratch.

export interface GridPoint {
  x: number;
  y: number;
}

export interface OriSegment {
  a: GridPoint;
  b: GridPoint;
}

export interface OriFixture {
  index: number;
  /** Sheet size in grid units. */
  sheet: { width: number; height: number };
  /** BLACK segments: the paper boundary, in local grid units. */
  boundary: OriSegment[];
  /** RED segments: ridge creases, in local grid units. */
  ridges: OriSegment[];
  /** BLUE segments: packing annotation, in local grid units (not used for assignment). */
  packing: OriSegment[];
}

interface RawSegment {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  color: string;
}

export interface OriDocument {
  lineSegments?: Array<{ a: string; b: string; color: string }>;
}

const DEFAULT_UNIT = 50;

export function parseOriFixtures(doc: OriDocument, unit: number = DEFAULT_UNIT): OriFixture[] {
  const raw = (doc.lineSegments ?? []).map((s) => {
    const [ax, ay] = s.a.split(",").map(Number);
    const [bx, by] = s.b.split(",").map(Number);
    // Snap to the half-unit lattice to absorb floating-point fuzz.
    return {
      ax: snap(ax, unit),
      ay: snap(ay, unit),
      bx: snap(bx, unit),
      by: snap(by, unit),
      color: s.color,
    } satisfies RawSegment;
  });

  const components = connectedComponents(raw);
  const fixtures = components.map((segs) => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const s of segs) {
      minX = Math.min(minX, s.ax, s.bx);
      minY = Math.min(minY, s.ay, s.by);
      maxX = Math.max(maxX, s.ax, s.bx);
      maxY = Math.max(maxY, s.ay, s.by);
    }
    const toLocal = (x: number, y: number): GridPoint => ({
      x: (x - minX) / unit,
      y: (y - minY) / unit,
    });
    const byColor = (predicate: (color: string) => boolean): OriSegment[] =>
      segs
        .filter((s) => predicate(s.color))
        .map((s) => ({ a: toLocal(s.ax, s.ay), b: toLocal(s.bx, s.by) }));
    return {
      minX,
      minY,
      sheet: { width: (maxX - minX) / unit, height: (maxY - minY) / unit },
      boundary: byColor((c) => c.startsWith("BLACK")),
      ridges: byColor((c) => c.startsWith("RED")),
      packing: byColor((c) => c.startsWith("BLUE")),
    };
  });

  // Stable order: top-to-bottom, then left-to-right, matching the canvas layout.
  fixtures.sort((a, b) => a.minY - b.minY || a.minX - b.minX);
  return fixtures.map((f, index) => ({
    index,
    sheet: f.sheet,
    boundary: f.boundary,
    ridges: f.ridges,
    packing: f.packing,
  }));
}

function snap(value: number, unit: number): number {
  const half = unit / 2;
  return Math.round(value / half) * half;
}

function connectedComponents(segs: RawSegment[]): RawSegment[][] {
  const parent = new Map<string, string>();
  const key = (x: number, y: number): string => `${x},${y}`;
  const find = (k: string): string => {
    if (!parent.has(k)) parent.set(k, k);
    let root = k;
    while (parent.get(root) !== root) root = parent.get(root)!;
    while (parent.get(k) !== root) {
      const next = parent.get(k)!;
      parent.set(k, root);
      k = next;
    }
    return root;
  };
  const union = (a: string, b: string): void => {
    parent.set(find(a), find(b));
  };
  for (const s of segs) union(key(s.ax, s.ay), key(s.bx, s.by));

  const groups = new Map<string, RawSegment[]>();
  for (const s of segs) {
    const root = find(key(s.ax, s.ay));
    const group = groups.get(root);
    if (group) group.push(s);
    else groups.set(root, [s]);
  }
  return [...groups.values()];
}

export function renderOriFixtureSvg(fixture: OriFixture, options: { cellSize?: number; includePacking?: boolean } = {}): string {
  const cell = options.cellSize ?? 44;
  const includePacking = options.includePacking ?? false;
  const margin = 24;
  const { width, height } = fixture.sheet;
  const sheetW = width * cell;
  const sheetH = height * cell;
  const grid: string[] = [];
  for (let x = 0; x <= width; x++) grid.push(`<line x1="${x * cell}" y1="0" x2="${x * cell}" y2="${sheetH}" stroke="#d1d5db" stroke-width="1"/>`);
  for (let y = 0; y <= height; y++) grid.push(`<line x1="0" y1="${y * cell}" x2="${sheetW}" y2="${y * cell}" stroke="#d1d5db" stroke-width="1"/>`);
  const line = (s: OriSegment, stroke: string, w: number): string =>
    `<line x1="${s.a.x * cell}" y1="${s.a.y * cell}" x2="${s.b.x * cell}" y2="${s.b.y * cell}" stroke="${stroke}" stroke-width="${w}" stroke-linecap="round"/>`;
  const packing = includePacking ? fixture.packing.map((s) => line(s, "#3b6fe0", 2)).join("") : "";
  const ridges = fixture.ridges.map((s) => line(s, "#e8453c", 2)).join("");
  const boundary = fixture.boundary.map((s) => line(s, "#1f2937", 3)).join("");
  return `<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${sheetW + margin * 2}" height="${sheetH + margin * 2 + 22}" viewBox="0 0 ${sheetW + margin * 2} ${sheetH + margin * 2 + 22}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="${margin}" y="16" font-family="Arial" font-size="13" font-weight="700" fill="#111">#${fixture.index} — ${width}x${height}u (ridges:${fixture.ridges.length})</text>
<g transform="translate(${margin} ${margin + 10})">${grid.join("")}${packing}${ridges}${boundary}</g>
</svg>
`;
}
