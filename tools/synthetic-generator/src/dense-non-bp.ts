import { denseBucketSpec, perimeterPosition } from "./dense-utils.ts";
import { normalizeFold } from "./fold-utils.ts";
import { SeededRandom } from "./random.ts";
import type { DenseNonBPSubfamily, EdgeAssignment, FOLDFormat, GenerationConfig } from "./types.ts";

type Point = [number, number];

const DENSE_NON_BP_SUBFAMILIES: DenseNonBPSubfamily[] = [
  "recursive-axiom",
  "expanded-classic",
  "radial-multi-vertex",
  "tessellation-like",
];

export function generateDenseNonBPFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const subfamily = chooseDenseNonBPSubfamily(config.denseSubfamily, rng);
  const spec = denseBucketSpec(config.bucket, config.seed, "non-bp");
  if (subfamily === "tessellation-like") {
    const size = miuraSizeForBucket(spec.bucket, config.seed);
    return makeMiuraTessellation(size, size, miuraOffset(config.seed), spec.bucket, spec.targetEdgeRange, subfamily);
  }
  const spokes = subfamily === "expanded-classic" ? 4 : 8;
  const rings = Math.max(3, ringsForTargetEdges(spokes, spec.gridSize, spec.targetEdgeRange[0]));
  return makeRadialRingPattern(spokes, rings, spec.bucket, spec.targetEdgeRange, subfamily);
}

function miuraSizeForBucket(bucket: string, seed: number): number {
  const choices = bucket === "dense" ? [20, 24] : bucket === "medium" ? [12, 16] : [8, 10];
  return choices[Math.abs(seed) % choices.length];
}

function miuraOffset(seed: number): number {
  return [0.22, 0.3, 0.38, 0.46][Math.abs(seed) % 4];
}

function ringsForTargetEdges(spokes: number, baseRings: number, minEdges: number): number {
  return Math.max(baseRings, Math.ceil((minEdges - spokes) / (2 * spokes)));
}

export function chooseDenseNonBPSubfamily(value: string | undefined, rng: SeededRandom): DenseNonBPSubfamily {
  if (value && DENSE_NON_BP_SUBFAMILIES.includes(value as DenseNonBPSubfamily)) {
    return value as DenseNonBPSubfamily;
  }
  return rng.choice(DENSE_NON_BP_SUBFAMILIES);
}

export function makeRadialRingPattern(
  spokes: 4 | 8,
  rings: number,
  densityBucket: string,
  targetEdgeRange: [number, number],
  subfamily: DenseNonBPSubfamily,
): FOLDFormat {
  const center: Point = [0.5, 0.5];
  const angles = Array.from({ length: spokes }, (_, index) => -Math.PI / 2 + (index * Math.PI * 2) / spokes);
  const endpoints = angles.map((angle) => rayToSquare(center, angle));
  const vertices: Point[] = [center];
  const index = (spoke: number, level: number): number => 1 + (level - 1) * spokes + spoke;
  for (let level = 1; level <= rings + 1; level++) {
    const t = level / (rings + 1);
    for (let spoke = 0; spoke < spokes; spoke++) {
      const endpoint = endpoints[spoke];
      vertices.push([round(center[0] + (endpoint[0] - center[0]) * t), round(center[1] + (endpoint[1] - center[1]) * t)]);
    }
  }

  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const rayAssignments = Array.from({ length: spokes }, (_, index) => index < spokes / 2 + 1 ? "M" : "V") as EdgeAssignment[];
  const add = (a: number, b: number, assignment: EdgeAssignment): void => {
    edges.push([a, b]);
    assignments.push(assignment);
  };

  for (let spoke = 0; spoke < spokes; spoke++) {
    add(0, index(spoke, 1), rayAssignments[spoke]);
    for (let level = 1; level <= rings; level++) {
      add(index(spoke, level), index(spoke, level + 1), rayAssignments[spoke]);
    }
  }
  for (let level = 1; level <= rings; level++) {
    for (let spoke = 0; spoke < spokes; spoke++) {
      add(index(spoke, level), index((spoke + 1) % spokes, level), spoke % 2 === 0 ? "V" : "M");
    }
  }

  const boundary = endpoints
    .map((point, spoke) => ({ point, spoke, position: perimeterPosition(point) }))
    .sort((a, b) => a.position - b.position);
  for (let i = 0; i < boundary.length; i++) {
    add(index(boundary[i].spoke, rings + 1), index(boundary[(i + 1) % boundary.length].spoke, rings + 1), "B");
  }

  const fold = normalizeFold(
    {
      file_spec: 1.1,
      file_creator: `cp-synthetic-generator/dense-non-bp/${subfamily}`,
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
      density_metadata: {
        densityBucket,
        gridSize: rings,
        targetEdgeRange,
        subfamily,
        symmetry: spokes === 8 ? "octagonal-radial" : "kite-radial",
        generatorSteps: ["radial-spokes", "concentric-rings", "alternating-ring-assignments"],
        moleculeCounts: {
          spoke: spokes,
          ring: rings,
        },
      },
    },
    `cp-synthetic-generator/dense-non-bp/${subfamily}`,
  );
  return fold;
}

export function makeMiuraTessellation(
  columns: number,
  rows: number,
  offset: number,
  densityBucket: string,
  targetEdgeRange: [number, number],
  subfamily: DenseNonBPSubfamily,
): FOLDFormat {
  const vertices: Point[] = [];
  const index = (x: number, y: number): number => y * (columns + 1) + x;
  for (let y = 0; y <= rows; y++) {
    const rowOffset = y % 2 === 0 ? 0 : offset;
    for (let x = 0; x <= columns; x++) {
      const px = x === 0 ? 0 : x === columns ? 1 : round((x + rowOffset) / columns);
      vertices.push([px, round(y / rows)]);
    }
  }

  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const add = (a: number, b: number, assignment: EdgeAssignment): void => {
    edges.push([a, b]);
    assignments.push(assignment);
  };

  for (let y = 0; y <= rows; y++) {
    for (let x = 0; x < columns; x++) {
      add(
        index(x, y),
        index(x + 1, y),
        y === 0 || y === rows ? "B" : x % 2 === 0 ? "V" : "M",
      );
    }
  }
  for (let x = 0; x <= columns; x++) {
    for (let y = 0; y < rows; y++) {
      add(index(x, y), index(x, y + 1), x === 0 || x === columns ? "B" : "M");
    }
  }

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: `cp-synthetic-generator/dense-non-bp/${subfamily}`,
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
      density_metadata: {
        densityBucket,
        gridSize: Math.max(columns, rows),
        targetEdgeRange,
        subfamily,
        symmetry: "miura-like-translational",
        generatorSteps: ["slanted-row-lattice", "alternating-horizontal-assignments", "parallel-mountain-ribs"],
        moleculeCounts: {
          parallelogram: columns * rows,
          row: rows,
          column: columns,
        },
      },
    },
    `cp-synthetic-generator/dense-non-bp/${subfamily}`,
  );
}

export function makeDenseNonBPXGrid(
  gridSize: number,
  densityBucket: string,
  targetEdgeRange: [number, number],
  subfamily: DenseNonBPSubfamily,
): FOLDFormat {
  const vertices: Point[] = [];
  const vertexKeys = new Map<string, number>();
  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const vertexIndex = (x: number, y: number): number => {
    const point: Point = [round(x / gridSize), round(y / gridSize)];
    const key = `${point[0].toFixed(9)},${point[1].toFixed(9)}`;
    const existing = vertexKeys.get(key);
    if (existing !== undefined) return existing;
    const index = vertices.length;
    vertices.push(point);
    vertexKeys.set(key, index);
    return index;
  };
  const add = (a: number, b: number, assignment: EdgeAssignment): void => {
    edges.push([a, b]);
    assignments.push(assignment);
  };

  for (let y = 0; y <= gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      add(vertexIndex(x, y), vertexIndex(x + 1, y), y === 0 || y === gridSize ? "B" : "V");
    }
  }
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      add(vertexIndex(x, y), vertexIndex(x, y + 1), x === 0 || x === gridSize ? "B" : "M");
    }
  }
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const center = vertexIndex(x + 0.5, y + 0.5);
      const corners = [
        [x, y],
        [x + 1, y],
        [x + 1, y + 1],
        [x, y + 1],
      ] as const;
      for (const [cornerX, cornerY] of corners) {
        add(center, vertexIndex(cornerX, cornerY), cornerX === x && cornerY === y ? "V" : "M");
      }
    }
  }

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: `cp-synthetic-generator/dense-non-bp/${subfamily}`,
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
      density_metadata: {
        densityBucket,
        gridSize,
        targetEdgeRange,
        subfamily,
        symmetry: "waterbomb-grid",
        generatorSteps: ["orthogonal-grid", "cell-center-diagonals", "lower-left-valley-choice"],
        moleculeCounts: {
          "cell-x": gridSize * gridSize,
          "pleat-line": gridSize * 2 - 2,
        },
      },
    },
    `cp-synthetic-generator/dense-non-bp/${subfamily}`,
  );
}

function rayToSquare(origin: Point, angle: number): Point {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const candidates: number[] = [];
  if (dx > 0) candidates.push((1 - origin[0]) / dx);
  if (dx < 0) candidates.push((0 - origin[0]) / dx);
  if (dy > 0) candidates.push((1 - origin[1]) / dy);
  if (dy < 0) candidates.push((0 - origin[1]) / dy);
  const t = Math.min(...candidates.filter((candidate) => candidate > 1e-9));
  return [round(clamp(origin[0] + dx * t)), round(clamp(origin[1] + dy * t))];
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
