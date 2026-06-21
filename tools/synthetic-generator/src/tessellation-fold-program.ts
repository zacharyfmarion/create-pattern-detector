import { normalizeFold } from "./fold-utils.ts";
import { SeededRandom } from "./random.ts";
import type {
  EdgeAssignment,
  FOLDFormat,
  GenerationConfig,
  LabelPolicy,
  TessellationMetadata,
  TessellationSamplerConfig,
  TessellationSubfamily,
} from "./types.ts";

interface GridCandidate {
  gridSizeX: number;
  gridSizeY: number;
  horizontalPleatInterval: number;
  verticalPleatInterval: number;
  cols: number;
  rows: number;
  activeCreases: number;
  totalEdges: number;
  intervalWeight: number;
  horizontalCreaseLengthFraction: number;
  verticalCreaseLengthFraction: number;
}

interface MiuraCandidate {
  cols: number;
  rows: number;
  skewFactor: number;
  activeCreases: number;
  totalEdges: number;
}

export function generateTessellationFoldProgram(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const subfamily = chooseSubfamily(rng, config.tessellationSampler);
  if (subfamily === "orthogonal-bp-grid") return generateOrthogonalBpGrid(config, rng);
  if (subfamily === "miura-ori") return generateMiuraOri(config, rng);
  throw new Error(`Unsupported tessellation subfamily: ${String(subfamily)}`);
}

function generateOrthogonalBpGrid(config: GenerationConfig, rng: SeededRandom): FOLDFormat {
  const minActive = config.numCreases;
  const maxActive = Math.max(minActive, config.maxCreases ?? minActive);
  const sampler = config.tessellationSampler ?? {};
  const verticalBias = rng.next() < clamp01(sampler.verticalBiasProbability ?? 0.72);
  const candidate = chooseGridCandidate(rng, minActive, maxActive, sampler, verticalBias);
  const assignmentMode: TessellationMetadata["assignmentMode"] =
    verticalBias || rng.next() < 0.5 ? "vertical-line-alternating" : "horizontal-line-alternating";
  const primary: EdgeAssignment = rng.next() < 0.5 ? "M" : "V";
  const secondary: EdgeAssignment = primary === "M" ? "V" : "M";

  const vertices: [number, number][] = [];
  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const edgeLengths: number[] = [];
  const angleHistogram: Record<string, number> = {};
  let horizontalLength = 0;
  let verticalLength = 0;

  for (let y = 0; y <= candidate.rows; y++) {
    for (let x = 0; x <= candidate.cols; x++) {
      vertices.push([
        (x * candidate.verticalPleatInterval) / candidate.gridSizeX,
        (y * candidate.horizontalPleatInterval) / candidate.gridSizeY,
      ]);
    }
  }

  const vertexIndex = (x: number, y: number): number => y * (candidate.cols + 1) + x;
  const addEdge = (a: number, b: number, assignment: EdgeAssignment, length: number, angle: "0" | "90"): void => {
    edges.push([a, b]);
    assignments.push(assignment);
    edgeLengths.push(length);
    angleHistogram[angle] = (angleHistogram[angle] ?? 0) + 1;
    if (assignment !== "B") {
      if (angle === "0") horizontalLength += length;
      else verticalLength += length;
    }
  };

  for (let y = 0; y <= candidate.rows; y++) {
    for (let x = 0; x < candidate.cols; x++) {
      const assignment = y === 0 || y === candidate.rows
        ? "B"
        : horizontalAssignment(x, y, assignmentMode, primary, secondary);
      addEdge(vertexIndex(x, y), vertexIndex(x + 1, y), assignment, candidate.verticalPleatInterval / candidate.gridSizeX, "0");
    }
  }

  for (let y = 0; y < candidate.rows; y++) {
    for (let x = 0; x <= candidate.cols; x++) {
      const assignment = x === 0 || x === candidate.cols
        ? "B"
        : verticalAssignment(x, y, assignmentMode, primary, secondary);
      addEdge(vertexIndex(x, y), vertexIndex(x, y + 1), assignment, candidate.horizontalPleatInterval / candidate.gridSizeY, "90");
    }
  }

  const activeLength = Math.max(horizontalLength + verticalLength, 1e-9);
  const metadata: TessellationMetadata = {
    generator: "tessellation-fold-program",
    subfamily: "orthogonal-bp-grid",
    coordinateMode: "regular-grid-intervals",
    gridSizeX: candidate.gridSizeX,
    gridSizeY: candidate.gridSizeY,
    horizontalPleatInterval: candidate.horizontalPleatInterval,
    verticalPleatInterval: candidate.verticalPleatInterval,
    repeatX: candidate.cols,
    repeatY: candidate.rows,
    activeCreaseCount: activeCreaseCount(assignments),
    targetActiveCreaseRange: [minActive, maxActive],
    horizontalCreaseLengthFraction: roundRatio(horizontalLength / activeLength),
    verticalCreaseLengthFraction: roundRatio(verticalLength / activeLength),
    diagonalCreaseLengthFraction: 0,
    minRenderedSpacingPx1024: roundRatio(
      1024 * Math.min(
        candidate.verticalPleatInterval / candidate.gridSizeX,
        candidate.horizontalPleatInterval / candidate.gridSizeY,
      ),
    ),
    angleHistogram,
    assignmentMode,
    verticalBias,
    generatorSteps: [
      "orthogonal-repeat-grid",
      "segmented-border",
      "regular-grid-interval-pleats",
      "parity-mv-assignment",
    ],
  };
  const labelPolicy: LabelPolicy = {
    labelSource: "tessellation-fold-program",
    geometrySource: "tessellation-fold-program",
    assignmentSource: "tessellation-fold-program",
    trainingEligible: true,
    notes: [
      "Synthetic tessellation CP focused on dense horizontal and vertical crease evidence.",
      "Pleats lie on a regular BP grid, with independent horizontal and vertical active-line intervals.",
      "Interior grid assignments alternate full crease lines along one axis and connector segments preserve a 3-to-1 M/V split.",
    ],
  };
  const totalLength = edgeLengths.reduce((sum, value) => sum + value, 0);

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: "cp-synthetic-generator/tessellation-fold-program",
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
      density_metadata: {
        densityBucket: config.bucket,
        gridSize: Math.max(candidate.cols, candidate.rows),
        targetEdgeRange: [minActive, candidate.totalEdges],
        subfamily: "orthogonal-bp-grid",
        symmetry: verticalBias ? "vertical-heavy" : "balanced-orthogonal",
        generatorSteps: metadata.generatorSteps,
        moleculeCounts: {
          gridCell: candidate.cols * candidate.rows,
          horizontalActiveSegments: (candidate.rows - 1) * candidate.cols,
          verticalActiveSegments: (candidate.cols - 1) * candidate.rows,
        },
      },
      tessellation_metadata: metadata,
      label_policy: labelPolicy,
      tessellation_length_metadata: {
        totalEdgeLength: roundRatio(totalLength),
        horizontalActiveLength: roundRatio(horizontalLength),
        verticalActiveLength: roundRatio(verticalLength),
      },
    },
    "cp-synthetic-generator/tessellation-fold-program",
  );
}

function generateMiuraOri(config: GenerationConfig, rng: SeededRandom): FOLDFormat {
  const minActive = config.numCreases;
  const maxActive = Math.max(minActive, config.maxCreases ?? minActive);
  const sampler = config.tessellationSampler ?? {};
  const candidate = chooseMiuraCandidate(rng, minActive, maxActive, sampler);
  const primary: EdgeAssignment = rng.next() < 0.5 ? "M" : "V";
  const secondary: EdgeAssignment = primary === "M" ? "V" : "M";

  const vertices = miuraVertices(candidate);
  const edges: [number, number][] = [];
  const assignments: EdgeAssignment[] = [];
  const edgeLengths: number[] = [];
  const angleHistogram: Record<string, number> = {};
  let horizontalLength = 0;
  let diagonalLength = 0;

  const vertexIndex = (x: number, y: number): number => y * (candidate.cols + 1) + x;
  const addEdge = (a: number, b: number, assignment: EdgeAssignment, family: "horizontal" | "diagonal"): void => {
    edges.push([a, b]);
    assignments.push(assignment);
    const length = distance(vertices[a], vertices[b]);
    edgeLengths.push(length);
    const angle = angleBucket(vertices[a], vertices[b]);
    angleHistogram[angle] = (angleHistogram[angle] ?? 0) + 1;
    if (assignment !== "B") {
      if (family === "horizontal") horizontalLength += length;
      else diagonalLength += length;
    }
  };

  for (let y = 0; y <= candidate.rows; y++) {
    for (let x = 0; x < candidate.cols; x++) {
      const assignment = y === 0 || y === candidate.rows
        ? "B"
        : alternatingAssignment(x, primary, secondary);
      addEdge(vertexIndex(x, y), vertexIndex(x + 1, y), assignment, "horizontal");
    }
  }

  for (let y = 0; y < candidate.rows; y++) {
    for (let x = 1; x < candidate.cols; x++) {
      const assignment = alternatingAssignment(x, primary, secondary);
      addEdge(vertexIndex(x, y), vertexIndex(x, y + 1), assignment, "diagonal");
    }
  }

  for (let y = 0; y < candidate.rows; y++) {
    addEdge(vertexIndex(0, y), vertexIndex(0, y + 1), "B", "diagonal");
    addEdge(vertexIndex(candidate.cols, y), vertexIndex(candidate.cols, y + 1), "B", "diagonal");
  }

  const activeLength = Math.max(horizontalLength + diagonalLength, 1e-9);
  const minEdgeLength = Math.min(...edgeLengths);
  const metadata: TessellationMetadata = {
    generator: "tessellation-fold-program",
    subfamily: "miura-ori",
    coordinateMode: "miura-square-zigzag-grid",
    miuraSkewFactor: candidate.skewFactor,
    miuraCellAspectRatio: roundRatio(candidate.cols / candidate.rows),
    repeatX: candidate.cols,
    repeatY: candidate.rows,
    activeCreaseCount: activeCreaseCount(assignments),
    targetActiveCreaseRange: [minActive, maxActive],
    horizontalCreaseLengthFraction: roundRatio(horizontalLength / activeLength),
    verticalCreaseLengthFraction: 0,
    diagonalCreaseLengthFraction: roundRatio(diagonalLength / activeLength),
    minRenderedSpacingPx1024: roundRatio(1024 * minEdgeLength),
    angleHistogram,
    assignmentMode: "miura-column-alternating",
    verticalBias: false,
    generatorSteps: [
      "miura-square-zigzag-grid",
      "segmented-square-border",
      "alternating-offset-rows",
      "parity-mv-assignment",
    ],
  };
  const labelPolicy: LabelPolicy = {
    labelSource: "tessellation-fold-program",
    geometrySource: "tessellation-fold-program",
    assignmentSource: "tessellation-fold-program",
    trainingEligible: true,
    notes: [
      "Synthetic Miura-ori tessellation CP on a square sheet with regular alternating-offset rows.",
      "Density varies by repeat count while each sheet keeps a uniform cell size.",
      "Interior assignments alternate by column to preserve a 3-to-1 M/V split at every Miura vertex.",
    ],
  };
  const totalLength = edgeLengths.reduce((sum, value) => sum + value, 0);

  return normalizeFold(
    {
      file_spec: 1.1,
      file_creator: "cp-synthetic-generator/tessellation-fold-program",
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges,
      edges_assignment: assignments,
      density_metadata: {
        densityBucket: config.bucket,
        gridSize: Math.max(candidate.cols, candidate.rows),
        targetEdgeRange: [minActive, candidate.totalEdges],
        subfamily: "miura-ori",
        symmetry: "miura-zigzag",
        generatorSteps: metadata.generatorSteps,
        moleculeCounts: {
          miuraCell: candidate.cols * candidate.rows,
          horizontalActiveSegments: (candidate.rows - 1) * candidate.cols,
          diagonalActiveSegments: candidate.rows * (candidate.cols - 1),
          squareBorderSegments: 2 * candidate.cols + 2 * candidate.rows,
        },
      },
      tessellation_metadata: metadata,
      label_policy: labelPolicy,
      tessellation_length_metadata: {
        totalEdgeLength: roundRatio(totalLength),
        horizontalActiveLength: roundRatio(horizontalLength),
        diagonalActiveLength: roundRatio(diagonalLength),
      },
    },
    "cp-synthetic-generator/tessellation-fold-program",
  );
}

function chooseSubfamily(rng: SeededRandom, sampler?: TessellationSamplerConfig): TessellationSubfamily {
  const configuredWeights = sampler?.subfamilyWeights;
  const weights: Record<TessellationSubfamily, number> = configuredWeights
    ? {
        "orthogonal-bp-grid": configuredWeights["orthogonal-bp-grid"] ?? 0,
        "miura-ori": configuredWeights["miura-ori"] ?? 0,
      }
    : {
        "orthogonal-bp-grid": 1,
        "miura-ori": 0,
      };
  return rng.weightedChoice(weights);
}

function chooseGridCandidate(
  rng: SeededRandom,
  minActive: number,
  maxActive: number,
  sampler: TessellationSamplerConfig,
  verticalBias: boolean,
): GridCandidate {
  const minGridSize = Math.max(3, Math.floor(sampler.minRepeats ?? 6));
  const maxGridSize = Math.max(minGridSize, Math.floor(sampler.maxRepeats ?? 48));
  const gridSizes = (sampler.gridSizes?.length ? sampler.gridSizes : defaultGridSizes(minGridSize, maxGridSize))
    .map((value) => Math.floor(value))
    .filter((value) => value >= minGridSize && value <= maxGridSize);
  const intervalPairs = sampler.pleatIntervalPairs?.length
    ? sampler.pleatIntervalPairs
    : defaultPleatIntervalPairs();
  const targetActive = rng.int(minActive, maxActive);
  const candidates: GridCandidate[] = [];

  for (const gridSize of gridSizes) {
    for (const pair of intervalPairs) {
      const horizontalPleatInterval = Math.max(1, Math.floor(pair.horizontal));
      const verticalPleatInterval = Math.max(1, Math.floor(pair.vertical));
      if (gridSize % horizontalPleatInterval !== 0 || gridSize % verticalPleatInterval !== 0) continue;
      const rows = gridSize / horizontalPleatInterval;
      const cols = gridSize / verticalPleatInterval;
      if (rows < 2 || cols < 2) continue;
      const activeCreases = 2 * rows * cols - rows - cols;
      if (activeCreases < minActive || activeCreases > maxActive) continue;
      const totalEdges = 2 * rows * cols + rows + cols;
      const horizontalLength = rows - 1;
      const verticalLength = cols - 1;
      const activeLength = horizontalLength + verticalLength;
      candidates.push({
        gridSizeX: gridSize,
        gridSizeY: gridSize,
        horizontalPleatInterval,
        verticalPleatInterval,
        cols,
        rows,
        activeCreases,
        totalEdges,
        intervalWeight: Math.max(0.05, Number(pair.weight ?? 1)),
        horizontalCreaseLengthFraction: horizontalLength / activeLength,
        verticalCreaseLengthFraction: verticalLength / activeLength,
      });
    }
  }

  if (!candidates.length) {
    throw new Error(`No orthogonal grid can satisfy active crease range ${minActive}-${maxActive}`);
  }

  const preferredCandidates = choosePreferredCandidates(candidates, verticalBias);
  const weighted = preferredCandidates.map((candidate) => {
    const targetWeight = 1 / (1 + Math.abs(candidate.activeCreases - targetActive));
    const verticalWeight = verticalBias
      ? 0.2 + candidate.verticalCreaseLengthFraction ** 2
      : 0.2 + (1 - Math.abs(candidate.verticalCreaseLengthFraction - 0.5));
    return { candidate, weight: targetWeight * verticalWeight * candidate.intervalWeight };
  });
  const totalWeight = weighted.reduce((sum, item) => sum + item.weight, 0);
  let draw = rng.next() * totalWeight;
  for (const item of weighted) {
    draw -= item.weight;
    if (draw <= 0) return item.candidate;
  }
  return weighted[weighted.length - 1].candidate;
}

function chooseMiuraCandidate(
  rng: SeededRandom,
  minActive: number,
  maxActive: number,
  sampler: TessellationSamplerConfig,
): MiuraCandidate {
  const minRepeat = Math.max(3, Math.floor(sampler.minRepeats ?? 6));
  const maxRepeat = Math.max(minRepeat, Math.floor(sampler.maxRepeats ?? 48));
  const colsValues = (sampler.miuraCols?.length ? sampler.miuraCols : defaultMiuraRepeats(minRepeat, maxRepeat))
    .map((value) => Math.floor(value))
    .filter((value) => value >= minRepeat && value <= maxRepeat);
  const rowsValues = (sampler.miuraRows?.length ? sampler.miuraRows : defaultMiuraRepeats(minRepeat, maxRepeat))
    .map((value) => Math.floor(value))
    .filter((value) => value >= minRepeat && value <= maxRepeat);
  const skewFactors = (sampler.miuraSkewFactors?.length ? sampler.miuraSkewFactors : [0.24, 0.33, 0.45, 0.58])
    .filter((value) => Number.isFinite(value) && value > 0.05 && value < 0.9);
  const minCellAspectRatio = sampler.miuraMinCellAspectRatio ?? 0.5;
  const maxCellAspectRatio = sampler.miuraMaxCellAspectRatio ?? 3.0;
  const targetActive = rng.int(minActive, maxActive);
  const candidates: MiuraCandidate[] = [];

  for (const rows of rowsValues) {
    for (const cols of colsValues) {
      const cellAspectRatio = cols / rows;
      if (cellAspectRatio < minCellAspectRatio || cellAspectRatio > maxCellAspectRatio) continue;
      const activeCreases = 2 * rows * cols - rows - cols;
      if (activeCreases < minActive || activeCreases > maxActive) continue;
      const totalEdges = (rows + 1) * cols + rows * (cols + 1);
      for (const skewFactor of skewFactors) {
        candidates.push({
          cols,
          rows,
          skewFactor,
          activeCreases,
          totalEdges,
        });
      }
    }
  }

  if (!candidates.length) {
    throw new Error(`No Miura grid can satisfy active crease range ${minActive}-${maxActive}`);
  }

  const weighted = candidates.map((candidate) => {
    const targetWeight = 1 / (1 + Math.abs(candidate.activeCreases - targetActive));
    const aspectBalance = Math.min(candidate.cols, candidate.rows) / Math.max(candidate.cols, candidate.rows);
    const skewBalance = 1 - Math.abs(candidate.skewFactor - 0.42);
    return { candidate, weight: targetWeight * (0.25 + aspectBalance) * (0.4 + skewBalance) };
  });
  const totalWeight = weighted.reduce((sum, item) => sum + item.weight, 0);
  let draw = rng.next() * totalWeight;
  for (const item of weighted) {
    draw -= item.weight;
    if (draw <= 0) return item.candidate;
  }
  return weighted[weighted.length - 1].candidate;
}

function miuraVertices(candidate: MiuraCandidate): [number, number][] {
  const vertices: [number, number][] = [];
  for (let y = 0; y <= candidate.rows; y++) {
    const offset = y % 2 === 0 ? 0 : candidate.skewFactor;
    for (let x = 0; x <= candidate.cols; x++) {
      const normalizedX = x === 0 || x === candidate.cols
        ? x / candidate.cols
        : (x + offset) / candidate.cols;
      vertices.push([normalizedX, y / candidate.rows]);
    }
  }
  return vertices;
}

function defaultMiuraRepeats(minRepeat: number, maxRepeat: number): number[] {
  return Array.from({ length: maxRepeat - minRepeat + 1 }, (_, index) => minRepeat + index);
}

function defaultGridSizes(minGridSize: number, maxGridSize: number): number[] {
  return [8, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48]
    .filter((size) => size >= minGridSize && size <= maxGridSize);
}

function defaultPleatIntervalPairs(): Array<{ horizontal: number; vertical: number; weight: number }> {
  return [
    { horizontal: 1, vertical: 1, weight: 1.4 },
    { horizontal: 1, vertical: 2, weight: 1.2 },
    { horizontal: 2, vertical: 1, weight: 1.2 },
    { horizontal: 1, vertical: 3, weight: 0.9 },
    { horizontal: 3, vertical: 1, weight: 0.9 },
    { horizontal: 2, vertical: 3, weight: 0.6 },
    { horizontal: 3, vertical: 2, weight: 0.6 },
  ];
}

function choosePreferredCandidates(candidates: GridCandidate[], verticalBias: boolean): GridCandidate[] {
  if (verticalBias) {
    const verticalHeavy = candidates.filter((candidate) => candidate.verticalCreaseLengthFraction >= 0.58);
    if (verticalHeavy.length) return verticalHeavy;
  } else {
    const balanced = candidates.filter(
      (candidate) =>
        candidate.verticalCreaseLengthFraction >= 0.43 &&
        candidate.verticalCreaseLengthFraction <= 0.57,
    );
    if (balanced.length) return balanced;
  }
  return candidates;
}

function horizontalAssignment(
  x: number,
  y: number,
  mode: TessellationMetadata["assignmentMode"],
  primary: EdgeAssignment,
  secondary: EdgeAssignment,
): EdgeAssignment {
  return mode === "vertical-line-alternating"
    ? alternatingAssignment(x, primary, secondary)
    : alternatingAssignment(y, primary, secondary);
}

function verticalAssignment(
  x: number,
  y: number,
  mode: TessellationMetadata["assignmentMode"],
  primary: EdgeAssignment,
  secondary: EdgeAssignment,
): EdgeAssignment {
  return mode === "vertical-line-alternating"
    ? alternatingAssignment(x, primary, secondary)
    : alternatingAssignment(y, primary, secondary);
}

function alternatingAssignment(index: number, primary: EdgeAssignment, secondary: EdgeAssignment): EdgeAssignment {
  return index % 2 === 0 ? primary : secondary;
}

function activeCreaseCount(assignments: EdgeAssignment[]): number {
  return assignments.filter((assignment) => assignment === "M" || assignment === "V").length;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function angleBucket(a: [number, number], b: [number, number]): string {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const angle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
  const canonical = angle > 90 ? 180 - angle : angle;
  return String(Math.round(canonical / 5) * 5);
}

function distance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function roundRatio(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
