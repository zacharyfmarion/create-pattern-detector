import ear from "rabbit-ear";
import { generateBoxPleatFold } from "./box-pleat.ts";
import { normalizeFold } from "./fold-utils.ts";
import { SeededRandom } from "./random.ts";
import { GENERATOR_FAMILIES } from "./types.ts";
import type { EdgeAssignment, FOLDFormat, GenerationConfig, GeneratorFamily } from "./types.ts";

type Line = { vector: [number, number]; origin: [number, number] };

const CLASSIC_BASES = ["kite"] as const;

export function generateFold(config: GenerationConfig): FOLDFormat {
  if (config.family === "axiom") return generateAxiomFold(config);
  if (config.family === "classic") return generateClassicFold(config);
  if (config.family === "single-vertex") return generateSingleVertexFold(config);
  if (config.family === "box-pleat") return generateBoxPleatFold(config);
  if (config.family === "grid-baseline") return generateGridBaselineFold(config);
  throw new Error(`Unknown generator family: ${String(config.family)}`);
}

export function availableFamilies(): GeneratorFamily[] {
  return [...GENERATOR_FAMILIES];
}

export function generateAxiomFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const graph = ear.graph.square() as FOLDFormat;
  const targetFolds = Math.max(2, Math.min(18, Math.ceil(config.numCreases / 8)));
  let applied = 0;

  for (let i = 0; i < targetFolds * 12 && applied < targetFolds; i++) {
    const before = JSON.parse(JSON.stringify(graph)) as FOLDFormat;
    const line = randomAxiomLine(rng);
    const assignment: EdgeAssignment = rng.next() < 0.5 ? "M" : "V";
    try {
      ear.graph.flatFold(graph, line.vector, line.origin, assignment);
      applied += 1;
      if ((graph.edges_vertices?.length ?? 0) > config.numCreases * 4 + 32) break;
    } catch {
      for (const key of Object.keys(graph)) delete graph[key];
      Object.assign(graph, before);
    }
  }

  if (applied === 0) {
    throw new Error("Axiom generator could not apply any folds");
  }

  return normalizeFold(graph, "cp-synthetic-generator/axiom");
}

export function generateClassicFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const baseName = rng.choice(CLASSIC_BASES);
  const factory = ear.graph[baseName] as () => FOLDFormat;
  if (typeof factory !== "function") {
    throw new Error(`Rabbit Ear classic base is unavailable: ${baseName}`);
  }
  const graph = factory();
  return normalizeFold(graph, `cp-synthetic-generator/classic/${baseName}`);
}

export function generateSingleVertexFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  let lastError = "unknown error";
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      return generateSingleVertexAttempt(rng);
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
  }
  throw new Error(`Single-vertex generator failed after retries: ${lastError}`);
}

function generateSingleVertexAttempt(rng: SeededRandom): FOLDFormat {
  const degree = rng.choice([4, 6, 8, 10]);
  const center: [number, number] = [rng.float(0.35, 0.65), rng.float(0.35, 0.65)];
  const angles = kawasakiAngles(rng, degree);
  const start = rng.float(0, Math.PI * 2);
  const directions: number[] = [];
  let cursor = start;
  for (const sector of angles) {
    directions.push(cursor);
    cursor += sector;
  }

  const rayPoints = directions.map((angle) => rayToSquare(center, angle));
  const border = buildBorderWithPoints(rayPoints);
  if (new Set(border.rayPointIndices).size !== degree) {
    throw new Error("ray collision at square boundary");
  }
  const borderPointCount = border.points.length;
  const vertices: [number, number][] = border.points.map((point) => [...point] as [number, number]);
  const centerIndex = vertices.length;
  vertices.push(center);

  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];

  for (let i = 0; i < borderPointCount; i++) {
    edges.push([i, (i + 1) % borderPointCount]);
    assignments.push("B");
  }

  const creaseAssignments = maekawaAssignments(rng, degree);
  for (let i = 0; i < border.rayPointIndices.length; i++) {
    edges.push([centerIndex, border.rayPointIndices[i]]);
    assignments.push(creaseAssignments[i]);
  }

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: "cp-synthetic-generator/single-vertex",
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
    },
    "cp-synthetic-generator/single-vertex",
  );
}

export function generateGridBaselineFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const divisions = Math.max(2, Math.min(8, Math.round(Math.sqrt(config.numCreases / 2))));
  const vertices: [number, number][] = [];
  const index = (x: number, y: number) => y * (divisions + 1) + x;

  for (let y = 0; y <= divisions; y++) {
    for (let x = 0; x <= divisions; x++) {
      vertices.push([x / divisions, y / divisions]);
    }
  }

  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const verticalAssignment: EdgeAssignment = rng.next() < 0.5 ? "M" : "V";
  const opposite: EdgeAssignment = verticalAssignment === "M" ? "V" : "M";

  for (let y = 0; y <= divisions; y++) {
    for (let x = 0; x < divisions; x++) {
      edges.push([index(x, y), index(x + 1, y)]);
      if (y === 0 || y === divisions) assignments.push("B");
      else assignments.push(x % 2 === 0 ? verticalAssignment : opposite);
    }
  }

  for (let x = 0; x <= divisions; x++) {
    for (let y = 0; y < divisions; y++) {
      edges.push([index(x, y), index(x, y + 1)]);
      if (x === 0 || x === divisions) assignments.push("B");
      else assignments.push(verticalAssignment);
    }
  }

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: "cp-synthetic-generator/grid",
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
    },
    "cp-synthetic-generator/grid-baseline",
  );
}

function randomAxiomLine(rng: SeededRandom): Line {
  const point = (): [number, number] => [rng.float(0.08, 0.92), rng.float(0.08, 0.92)];
  const line = (): Line => ({
    origin: point(),
    vector: normalize([rng.float(-1, 1), rng.float(-1, 1)]),
  });

  const choice = rng.next();
  let result: unknown;
  if (choice < 0.45) {
    result = ear.axiom.axiom1(point(), point());
  } else if (choice < 0.75) {
    result = ear.axiom.axiom2(point(), point());
  } else {
    result = ear.axiom.axiom3(line(), line());
  }
  return firstLine(result);
}

function firstLine(result: unknown): Line {
  const line = Array.isArray(result) ? result[0] : result;
  if (!line || typeof line !== "object") {
    throw new Error("Rabbit Ear axiom did not return a fold line");
  }
  const maybeLine = line as Partial<Line>;
  if (!maybeLine.vector || !maybeLine.origin) {
    throw new Error("Rabbit Ear axiom returned an invalid fold line");
  }
  return {
    vector: normalize(maybeLine.vector),
    origin: [maybeLine.origin[0], maybeLine.origin[1]],
  };
}

function normalize(vector: readonly number[]): [number, number] {
  const length = Math.hypot(vector[0], vector[1]);
  if (length < 1e-9) return [1, 0];
  return [vector[0] / length, vector[1] / length];
}

function kawasakiAngles(rng: SeededRandom, degree: number): number[] {
  const half = degree / 2;
  const odd = normalizedPositiveParts(rng, half, Math.PI);
  const even = normalizedPositiveParts(rng, half, Math.PI);
  const result: number[] = [];
  for (let i = 0; i < half; i++) {
    result.push(odd[i], even[i]);
  }
  return result;
}

function normalizedPositiveParts(rng: SeededRandom, count: number, total: number): number[] {
  const raw = Array.from({ length: count }, () => rng.float(0.25, 1.0));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map((value) => (value / sum) * total);
}

function maekawaAssignments(rng: SeededRandom, degree: number): EdgeAssignment[] {
  const mountainCount = degree / 2 + 1;
  const valleyCount = degree - mountainCount;
  const assignments: EdgeAssignment[] = [
    ...Array.from({ length: mountainCount }, () => "M" as EdgeAssignment),
    ...Array.from({ length: valleyCount }, () => "V" as EdgeAssignment),
  ];
  for (let i = assignments.length - 1; i > 0; i--) {
    const j = rng.int(0, i);
    [assignments[i], assignments[j]] = [assignments[j], assignments[i]];
  }
  return assignments;
}

function rayToSquare(origin: [number, number], angle: number): [number, number] {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const candidates: [number, number][] = [];
  if (Math.abs(dx) > 1e-9) {
    candidates.push([(0 - origin[0]) / dx, 0]);
    candidates.push([(1 - origin[0]) / dx, 1]);
  }
  if (Math.abs(dy) > 1e-9) {
    candidates.push([(0 - origin[1]) / dy, 2]);
    candidates.push([(1 - origin[1]) / dy, 3]);
  }
  const hits = candidates
    .filter(([t]) => t > 1e-8)
    .map(([t]) => [origin[0] + t * dx, origin[1] + t * dy] as [number, number])
    .filter(([x, y]) => x >= -1e-8 && x <= 1 + 1e-8 && y >= -1e-8 && y <= 1 + 1e-8);
  if (hits.length === 0) throw new Error("Ray missed unit square");
  const [x, y] = hits[0];
  return [clamp01(x), clamp01(y)];
}

function buildBorderWithPoints(rayPoints: [number, number][]): {
  points: [number, number][];
  rayPointIndices: number[];
} {
  const tagged = [
    { point: [0, 0] as [number, number], ray: false },
    { point: [1, 0] as [number, number], ray: false },
    { point: [1, 1] as [number, number], ray: false },
    { point: [0, 1] as [number, number], ray: false },
    ...rayPoints.map((point) => ({ point, ray: true })),
  ];
  tagged.sort((a, b) => perimeterPosition(a.point) - perimeterPosition(b.point));

  const points: [number, number][] = [];
  const rayPointIndices: number[] = [];
  for (const item of tagged) {
    const existing = points.findIndex((point) => distance(point, item.point) < 1e-6);
    const index = existing >= 0 ? existing : points.push(item.point) - 1;
    if (item.ray) rayPointIndices.push(index);
  }
  return { points, rayPointIndices };
}

function perimeterPosition([x, y]: [number, number]): number {
  if (Math.abs(y) < 1e-8) return x;
  if (Math.abs(x - 1) < 1e-8) return 1 + y;
  if (Math.abs(y - 1) < 1e-8) return 2 + (1 - x);
  return 3 + (1 - y);
}

function distance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
