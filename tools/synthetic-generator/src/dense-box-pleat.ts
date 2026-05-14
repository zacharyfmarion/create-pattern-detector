import { roleCounts } from "./fold-utils.ts";
import { denseBucketSpec, portsCompatible } from "./dense-utils.ts";
import { SeededRandom } from "./random.ts";
import type { TilePortSignature } from "./dense-utils.ts";
import type { BPRole, EdgeAssignment, FOLDFormat, GenerationConfig } from "./types.ts";

type Point = [number, number];
type ValleyCorner = "ll" | "lr" | "ur" | "ul";
type DenseBPPattern =
  | "row-chevrons"
  | "column-fans"
  | "anti-diagonal-diamond-chain"
  | "checker-diamonds"
  | "uniform-twist";

interface DenseEdge {
  a: number;
  b: number;
  assignment: EdgeAssignment;
  role: BPRole;
}

interface DenseBPOptions {
  seed?: number;
  pattern?: DenseBPPattern;
}

export { portsCompatible };
export type { TilePortSignature };

export function generateDenseBoxPleatFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const spec = denseBucketSpec(config.bucket, config.seed, "bp");
  return makeDenseBoxPleatTessellation(spec.gridSize, config.bucket, spec.targetEdgeRange, {
    seed: config.seed,
    pattern: chooseDenseBPPattern(rng),
  });
}

export function makeDenseBoxPleatTessellation(
  gridSize: number,
  densityBucket = "medium",
  targetEdgeRange: [number, number] = [180, 750],
  options: DenseBPOptions = {},
): FOLDFormat {
  const seed = options.seed ?? 0;
  const pattern = options.pattern ?? "row-chevrons";
  const horizontalStretchLane = 1 + Math.abs(seed) % Math.max(2, gridSize - 1);
  const verticalStretchLane = 1 + Math.abs(Math.floor(seed / 7)) % Math.max(2, gridSize - 1);
  const sparseChecker = pattern === "checker-diamonds" && sparseCheckerEdgeCount(gridSize) >= targetEdgeRange[0];
  const vertices: Point[] = [];
  const vertexKeys = new Map<string, number>();
  const edges: DenseEdge[] = [];
  let activeMoleculeCount = 0;
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
  const addEdge = (a: number, b: number, assignment: EdgeAssignment, role: BPRole): void => {
    if (a === b) return;
    edges.push({ a, b, assignment, role });
  };

  for (let y = 0; y <= gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      addEdge(
        vertexIndex(x, y),
        vertexIndex(x + 1, y),
        y === 0 || y === gridSize ? "B" : "V",
        y === 0 || y === gridSize ? "border" : y === horizontalStretchLane ? "stretch" : "hinge",
      );
    }
  }
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      addEdge(
        vertexIndex(x, y),
        vertexIndex(x, y + 1),
        x === 0 || x === gridSize ? "B" : "M",
        x === 0 || x === gridSize ? "border" : x === verticalStretchLane ? "stretch" : "axis",
      );
    }
  }

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      if (sparseChecker && (x + y + Math.abs(seed)) % 2 !== 0) continue;
      activeMoleculeCount += 1;
      const center = vertexIndex(x + 0.5, y + 0.5);
      const corners = [
        [x, y],
        [x + 1, y],
        [x + 1, y + 1],
        [x, y + 1],
      ] as const;
      const valleyCorner = valleyCornerForCell(x, y, gridSize, seed, pattern);
      for (const [cornerX, cornerY] of corners) {
        addEdge(
          center,
          vertexIndex(cornerX, cornerY),
          cornerName(cornerX - x, cornerY - y) === valleyCorner ? "V" : "M",
          "ridge",
        );
      }
    }
  }

  const fold: FOLDFormat = {
    file_spec: 1.1,
    file_creator: "cp-synthetic-generator/box-pleat/dense-molecule-tessellation",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: vertices,
    edges_vertices: edges.map((edge) => [edge.a, edge.b]),
    edges_assignment: edges.map((edge) => edge.assignment),
    edges_bpRole: edges.map((edge) => edge.role),
  };
  const counts = roleCounts(fold);
  fold.bp_metadata = {
    gridSize,
    bpSubfamily: "dense-molecule-tessellation",
    flapCount: 0,
    gadgetCount: activeMoleculeCount,
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  fold.density_metadata = {
    densityBucket,
    gridSize,
    targetEdgeRange,
    subfamily: "dense-molecule-tessellation",
    symmetry: sparseChecker ? "checker-diamond-chain-sparse" : pattern,
    generatorSteps: ["pleat-strip", "cell-x-molecule", pattern, "stretch-corridors"],
    moleculeCounts: {
      "pleat-strip": gridSize * 2 - 2,
      "cell-x": activeMoleculeCount,
      "diamond-chain": sparseChecker ? Math.ceil(activeMoleculeCount / gridSize) : Math.max(0, gridSize - 1),
      "corner-fan": 4,
      "river-corridor": Math.max(1, Math.floor(gridSize / 2)),
    },
  };
  return fold;
}

function chooseDenseBPPattern(rng: SeededRandom): DenseBPPattern {
  return rng.choice([
    "row-chevrons",
    "column-fans",
    "anti-diagonal-diamond-chain",
    "checker-diamonds",
    "uniform-twist",
  ]);
}

function valleyCornerForCell(
  x: number,
  y: number,
  gridSize: number,
  seed: number,
  pattern: DenseBPPattern,
): ValleyCorner {
  if (pattern === "row-chevrons") {
    return hashedBand(y, seed, 1 + Math.abs(seed) % 3) ? "ll" : "lr";
  }
  if (pattern === "column-fans") {
    return hashedBand(x, seed * 17 + 5, 1 + Math.abs(seed) % 3) ? "ll" : "ul";
  }
  if (pattern === "anti-diagonal-diamond-chain") {
    const width = 1 + Math.abs(seed) % 3;
    const lane = Math.floor((x - y + gridSize + Math.abs(seed % gridSize)) / width);
    return hashedBit(lane, seed * 31 + 11) ? "ll" : "ur";
  }
  if (pattern === "checker-diamonds") {
    return ["ll", "lr", "ul", "ur"][x % 4] as ValleyCorner;
  }
  return ["ll", "lr", "ur", "ul"][Math.abs(seed) % 4] as ValleyCorner;
}

function sparseCheckerEdgeCount(gridSize: number): number {
  const axisEdges = 2 * gridSize * (gridSize + 1);
  const activeCells = Math.ceil((gridSize * gridSize) / 2);
  return axisEdges + activeCells * 4;
}

function hashedBand(index: number, seed: number, width: number): boolean {
  return hashedBit(Math.floor(index / width), seed);
}

function hashedBit(index: number, seed: number): boolean {
  let value = (index + 1) * 0x9e3779b1;
  value ^= seed * 0x85ebca6b;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  return (value & 1) === 0;
}

function cornerName(dx: number, dy: number): ValleyCorner {
  if (dx === 0 && dy === 0) return "ll";
  if (dx === 1 && dy === 0) return "lr";
  if (dx === 1 && dy === 1) return "ur";
  return "ul";
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
