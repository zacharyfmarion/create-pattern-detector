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
  cols: number;
  rows: number;
  activeCreases: number;
  totalEdges: number;
  horizontalCreaseLengthFraction: number;
  verticalCreaseLengthFraction: number;
  minRenderedSpacingPx1024: number;
}

export function generateTessellationFoldProgram(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const subfamily = chooseSubfamily(rng, config.tessellationSampler);
  if (subfamily === "orthogonal-bp-grid") return generateOrthogonalBpGrid(config, rng);
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
      vertices.push([x / candidate.cols, y / candidate.rows]);
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
      addEdge(vertexIndex(x, y), vertexIndex(x + 1, y), assignment, 1 / candidate.cols, "0");
    }
  }

  for (let y = 0; y < candidate.rows; y++) {
    for (let x = 0; x <= candidate.cols; x++) {
      const assignment = x === 0 || x === candidate.cols
        ? "B"
        : verticalAssignment(x, y, assignmentMode, primary, secondary);
      addEdge(vertexIndex(x, y), vertexIndex(x, y + 1), assignment, 1 / candidate.rows, "90");
    }
  }

  const activeLength = Math.max(horizontalLength + verticalLength, 1e-9);
  const metadata: TessellationMetadata = {
    generator: "tessellation-fold-program",
    subfamily: "orthogonal-bp-grid",
    repeatX: candidate.cols,
    repeatY: candidate.rows,
    activeCreaseCount: activeCreaseCount(assignments),
    targetActiveCreaseRange: [minActive, maxActive],
    horizontalCreaseLengthFraction: roundRatio(horizontalLength / activeLength),
    verticalCreaseLengthFraction: roundRatio(verticalLength / activeLength),
    diagonalCreaseLengthFraction: 0,
    minRenderedSpacingPx1024: roundRatio(candidate.minRenderedSpacingPx1024),
    angleHistogram,
    assignmentMode,
    verticalBias,
    generatorSteps: [
      "orthogonal-repeat-grid",
      "segmented-border",
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

function chooseSubfamily(rng: SeededRandom, sampler?: TessellationSamplerConfig): TessellationSubfamily {
  const weights: Record<TessellationSubfamily, number> = {
    "orthogonal-bp-grid": sampler?.subfamilyWeights?.["orthogonal-bp-grid"] ?? 1,
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
  const minRepeats = Math.max(3, Math.floor(sampler.minRepeats ?? 6));
  const maxRepeats = Math.max(minRepeats, Math.floor(sampler.maxRepeats ?? 48));
  const targetActive = rng.int(minActive, maxActive);
  const candidates: GridCandidate[] = [];

  for (let rows = minRepeats; rows <= maxRepeats; rows++) {
    for (let cols = minRepeats; cols <= maxRepeats; cols++) {
      const activeCreases = 2 * rows * cols - rows - cols;
      if (activeCreases < minActive || activeCreases > maxActive) continue;
      const totalEdges = 2 * rows * cols + rows + cols;
      const horizontalLength = rows - 1;
      const verticalLength = cols - 1;
      const activeLength = horizontalLength + verticalLength;
      candidates.push({
        cols,
        rows,
        activeCreases,
        totalEdges,
        horizontalCreaseLengthFraction: horizontalLength / activeLength,
        verticalCreaseLengthFraction: verticalLength / activeLength,
        minRenderedSpacingPx1024: Math.min(1024 / cols, 1024 / rows),
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
    return { candidate, weight: targetWeight * verticalWeight };
  });
  const totalWeight = weighted.reduce((sum, item) => sum + item.weight, 0);
  let draw = rng.next() * totalWeight;
  for (const item of weighted) {
    draw -= item.weight;
    if (draw <= 0) return item.candidate;
  }
  return weighted[weighted.length - 1].candidate;
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

function roundRatio(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
