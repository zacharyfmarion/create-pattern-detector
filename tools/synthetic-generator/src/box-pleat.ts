import { roleCounts } from "./fold-utils.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import { SeededRandom } from "./random.ts";
import type { BPSubfamily, BPRole, EdgeAssignment, FOLDFormat, GenerationConfig } from "./types.ts";

type Point = [number, number];
type AxisOrientation = "vertical" | "horizontal";
type DiagonalFamily = "main" | "anti";

interface BPSegment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
}

export interface GOPSPiece {
  ox: number;
  oy: number;
  u: number;
  v: number;
}

const GRID_SIZES = [8, 10, 12, 14, 16, 18, 20, 24] as const;
const SUBFAMILIES: BPSubfamily[] = ["two-flap-stretch", "uniaxial-chain", "symmetric-insect-lite"];
const DIAGONAL_DIRECTIONS: Point[] = [
  [1, 1],
  [-1, -1],
  [1, -1],
  [-1, 1],
];

export function generateBoxPleatFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  let lastError = "unknown error";
  for (let attempt = 0; attempt < 80; attempt++) {
    try {
      const gridSize = rng.choice(GRID_SIZES);
      const subfamily = chooseSubfamily(rng, config.numCreases);
      if (subfamily === "two-flap-stretch") return twoFlapStretch(rng, gridSize);
      if (subfamily === "uniaxial-chain") return uniaxialChain(rng, gridSize);
      return symmetricInsectLite(rng, gridSize);
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
  }
  throw new Error(`Box-pleat generator failed after retries: ${lastError}`);
}

export function findGopsPiece(ox: number, oy: number, maxSpan = Number.POSITIVE_INFINITY): GOPSPiece | null {
  if (ox <= 0 || oy <= 0) return null;
  if (ox % 2 === 1 && oy % 2 === 1) return null;
  const halfArea = (ox * oy) / 2;
  if (!Number.isInteger(halfArea)) return null;

  let best: GOPSPiece | null = null;
  for (let u = Math.floor(Math.sqrt(halfArea)); u > 0; u--) {
    if (halfArea % u !== 0) continue;
    const candidates: GOPSPiece[] = [
      { ox, oy, u, v: halfArea / u },
      { ox, oy, u: halfArea / u, v: u },
    ];
    for (const candidate of candidates) {
      if (candidate.u + candidate.v + candidate.oy > maxSpan) continue;
      if (!best || gopsRank(candidate) < gopsRank(best)) best = candidate;
    }
    if (best) return best;
  }
  return null;
}

function twoFlapStretch(rng: SeededRandom, gridSize: number): FOLDFormat {
  const overlap = chooseEvenOverlap(rng, gridSize);
  const piece = findGopsPiece(overlap, overlap);
  if (!piece) throw new Error("could not find GOPS piece for two-flap stretch");

  const center = gridPoint(
    clampGrid(Math.round(gridSize / 2) + ((piece.u + piece.v) % 3) - 1, 2, gridSize - 2),
    clampGrid(Math.round(gridSize / 2) + (piece.u % 3) - 1, 2, gridSize - 2),
    gridSize,
  );
  const diagonalFamily = rng.choice<DiagonalFamily>(["main", "anti"]);
  const builder = new BPBuilder(gridSize, "two-flap-stretch", 2, 1);
  builder.addPlusDiagonalGadget(center, diagonalFamily);
  return builder.finish();
}

function uniaxialChain(rng: SeededRandom, gridSize: number): FOLDFormat {
  const family = rng.choice<DiagonalFamily>(["main", "anti"]);
  const margin = rng.int(2, Math.max(3, Math.floor(gridSize / 4)));
  const centers = diagonalPair(gridSize, margin, family);
  const orientation = rng.choice<AxisOrientation>(["vertical", "horizontal"]);
  const builder = new BPBuilder(gridSize, "uniaxial-chain", rng.int(3, 7), 2);
  centers.forEach((center, index) => {
    builder.addXGadget(center, orientation, index % 2 === 0 ? ["axis", "hinge"] : ["hinge", "axis"], index % 2 === 0 ? 1 : -1);
  });
  return builder.finish();
}

function symmetricInsectLite(rng: SeededRandom, gridSize: number): FOLDFormat {
  const family = rng.choice<DiagonalFamily>(["main", "anti"]);
  const margin = rng.choice([2, 3, 4].filter((value) => value < gridSize / 2));
  const centers = diagonalPair(gridSize, margin, family);
  const orientation: AxisOrientation = family === "main"
    ? rng.choice<AxisOrientation>(["vertical", "horizontal"])
    : rng.choice<AxisOrientation>(["horizontal", "vertical"]);
  const builder = new BPBuilder(gridSize, "symmetric-insect-lite", 6, 2);
  centers.forEach((center, index) => {
    builder.addXGadget(center, orientation, index === 0 ? ["axis", "hinge", "stretch"] : ["hinge", "axis"], index === 0 ? 1 : -1);
  });
  return builder.finish();
}

function chooseSubfamily(rng: SeededRandom, numCreases: number): BPSubfamily {
  if (numCreases < 24) return rng.choice(["two-flap-stretch", "symmetric-insect-lite"]);
  if (numCreases > 90) return rng.choice(["uniaxial-chain", "symmetric-insect-lite"]);
  return rng.choice(SUBFAMILIES);
}

function chooseEvenOverlap(rng: SeededRandom, gridSize: number): number {
  const max = Math.max(4, Math.floor(gridSize / 2));
  const values = [];
  for (let value = 2; value <= max; value += 2) values.push(value);
  return rng.choice(values);
}

function diagonalPair(gridSize: number, margin: number, family: DiagonalFamily): [Point, Point] {
  const a = margin;
  const b = gridSize - margin;
  if (a <= 0 || b >= gridSize || a >= b) throw new Error("invalid BP diagonal pair");
  if (family === "main") return [gridPoint(a, a, gridSize), gridPoint(b, b, gridSize)];
  return [gridPoint(a, b, gridSize), gridPoint(b, a, gridSize)];
}

function gopsRank(piece: GOPSPiece): number {
  return Math.max(reducedNumerator(piece.oy + piece.v, piece.oy), reducedNumerator(piece.ox + piece.u, piece.ox));
}

function reducedNumerator(a: number, b: number): number {
  return a / gcd(a, b);
}

function gcd(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y) [x, y] = [y, x % y];
  return x || 1;
}

class BPBuilder {
  private readonly segments: BPSegment[] = [
    { p1: [0, 0], p2: [1, 0], assignment: "B", role: "border" },
    { p1: [1, 0], p2: [1, 1], assignment: "B", role: "border" },
    { p1: [1, 1], p2: [0, 1], assignment: "B", role: "border" },
    { p1: [0, 1], p2: [0, 0], assignment: "B", role: "border" },
  ];

  constructor(
    private readonly gridSize: number,
    private readonly subfamily: BPSubfamily,
    private readonly flapCount: number,
    private readonly gadgetCount: number,
  ) {}

  addXGadget(center: Point, orientation: AxisOrientation, axisRoles: BPRole[], stretchDirection: 1 | -1): void {
    this.addAxisSegments(center, orientation, axisRoles, stretchDirection);
    for (const direction of DIAGONAL_DIRECTIONS) {
      this.segments.push({
        p1: center,
        p2: rayToSquareBoundary(center, direction),
        assignment: "M",
        role: "ridge",
      });
    }
  }

  addPlusDiagonalGadget(center: Point, diagonalFamily: DiagonalFamily): void {
    this.addAxisSegments(center, "vertical", ["axis", "hinge"], 1);
    this.addAxisSegments(center, "horizontal", ["hinge", "stretch"], 1);
    const directions: Point[] = diagonalFamily === "main"
      ? [[1, 1], [-1, -1]]
      : [[1, -1], [-1, 1]];
    for (const direction of directions) {
      this.segments.push({
        p1: center,
        p2: rayToSquareBoundary(center, direction),
        assignment: "M",
        role: "ridge",
      });
    }
  }

  finish(): FOLDFormat {
    const arranged = arrangeSegments(this.segments, `cp-synthetic-generator/box-pleat/${this.subfamily}`, {
      gridSize: this.gridSize,
      bpSubfamily: this.subfamily,
      flapCount: this.flapCount,
      gadgetCount: this.gadgetCount,
      ridgeCount: 0,
      hingeCount: 0,
      axisCount: 0,
    });
    const counts = roleCounts(arranged);
    arranged.bp_metadata = {
      gridSize: this.gridSize,
      bpSubfamily: this.subfamily,
      flapCount: this.flapCount,
      gadgetCount: this.gadgetCount,
      ridgeCount: counts.ridge ?? 0,
      hingeCount: counts.hinge ?? 0,
      axisCount: counts.axis ?? 0,
    };
    if ((counts.ridge ?? 0) < 1 || (counts.hinge ?? 0) < 1 || (counts.axis ?? 0) + (counts.stretch ?? 0) < 1) {
      throw new Error("BP gadget collapsed below role requirements");
    }
    return arranged;
  }

  private addAxisSegments(center: Point, orientation: AxisOrientation, roles: BPRole[], stretchDirection: 1 | -1): void {
    const endpoints: [Point, Point] = orientation === "vertical"
      ? [[center[0], 0], [center[0], 1]]
      : [[0, center[1]], [1, center[1]]];
    const sortedPoints = uniqueAxisPoints([
      endpoints[0],
      center,
      ...extraAxisSplitPoints(center, orientation, this.gridSize, stretchDirection, roles.length),
      endpoints[1],
    ], orientation);

    for (let i = 0; i < sortedPoints.length - 1; i++) {
      const p1 = sortedPoints[i];
      const p2 = sortedPoints[i + 1];
      if (distance(p1, p2) < 1e-9) continue;
      this.segments.push({
        p1,
        p2,
        assignment: "V",
        role: roles[Math.min(i, roles.length - 1)] ?? "hinge",
      });
    }
  }
}

function extraAxisSplitPoints(
  center: Point,
  orientation: AxisOrientation,
  gridSize: number,
  direction: 1 | -1,
  desiredRoles: number,
): Point[] {
  if (desiredRoles <= 2) return [];
  const step = direction / gridSize;
  const point: Point = orientation === "vertical"
    ? [center[0], clamp(center[1] + step, 0, 1)]
    : [clamp(center[0] + step, 0, 1), center[1]];
  return distance(point, center) < 1e-9 ? [] : [point];
}

function uniqueAxisPoints(points: Point[], orientation: AxisOrientation): Point[] {
  const seen = new Set<string>();
  const unique: Point[] = [];
  for (const point of points) {
    const key = point.map((value) => value.toFixed(9)).join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(point);
  }
  return unique.sort((a, b) => (orientation === "vertical" ? a[1] - b[1] : a[0] - b[0]));
}

function rayToSquareBoundary(center: Point, direction: Point): Point {
  const [cx, cy] = center;
  const [dx, dy] = direction;
  const candidates: number[] = [];
  if (dx > 0) candidates.push((1 - cx) / dx);
  if (dx < 0) candidates.push((0 - cx) / dx);
  if (dy > 0) candidates.push((1 - cy) / dy);
  if (dy < 0) candidates.push((0 - cy) / dy);
  const t = Math.min(...candidates.filter((candidate) => candidate > 1e-9));
  return [clamp(cx + dx * t, 0, 1), clamp(cy + dy * t, 0, 1)];
}

function gridPoint(x: number, y: number, gridSize: number): Point {
  return [x / gridSize, y / gridSize];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampGrid(value: number, min: number, max: number): number {
  return Math.round(Math.min(max, Math.max(min, value)));
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
