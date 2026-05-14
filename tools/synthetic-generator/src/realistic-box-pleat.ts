import ear from "rabbit-ear";
import { assignmentCounts, normalizeFold, roleCounts } from "./fold-utils.ts";
import { SeededRandom } from "./random.ts";
import type {
  BPRole,
  DesignTreeMetadata,
  EdgeAssignment,
  FOLDFormat,
  GenerationConfig,
  LayoutMetadata,
  MoleculeMetadata,
  RealismMetadata,
  RealisticBPArchetype,
} from "./types.ts";

type Point = [number, number];
type LineKind = "h" | "v" | "dMain" | "dAnti";
type Side = "top" | "right" | "bottom" | "left";
type CornerName = "ll" | "lr" | "ur" | "ul";

interface LineOp {
  kind: LineKind;
  n: number;
  assignment: EdgeAssignment;
  molecule: string;
}

interface GridRect {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface FlapTerminal {
  id: string;
  x: number;
  y: number;
  side: Side;
}

interface RealisticLayout {
  archetype: RealisticBPArchetype;
  gridSize: number;
  densityBucket: string;
  targetEdgeRange: [number, number];
  symmetry: string;
  bodyRegions: GridRect[];
  flapTerminals: FlapTerminal[];
  corridors: LayoutMetadata["corridors"];
  lineOps: LineOp[];
  moleculeCounts: Record<string, number>;
  portChecks: MoleculeMetadata["portChecks"];
}

interface MacroGridFoldResult {
  fold: FOLDFormat;
  activeCellCount: number;
  rowSelectors: number[];
  columnSelectors: number[];
  activeMode: string;
}

interface MacroGridEdge {
  a: number;
  b: number;
  assignment: EdgeAssignment;
  role: BPRole;
}

const ARCHETYPES: RealisticBPArchetype[] = ["insect", "quadruped", "bird", "object", "abstract"];
const CORNERS: CornerName[] = ["ll", "lr", "ur", "ul"];

export const REALISTIC_BP_MOLECULES = [
  "body-panel",
  "hinge-corridor",
  "axis-corridor",
  "appendage-fan",
  "corner-claw",
  "diamond-chain",
  "elias-like-stretch",
  "chevron-band",
] as const;

export function generateRealisticBoxPleatFold(config: GenerationConfig): FOLDFormat {
  const rng = new SeededRandom(config.seed);
  const archetype = config.realisticArchetype ?? rng.choice(ARCHETYPES);
  let lastFold: FOLDFormat | undefined;
  for (let attempt = 0; attempt < 10; attempt++) {
    const layout = makeRealisticLayout(archetype, config.bucket, config.numCreases, new SeededRandom(config.seed + attempt * 7919));
    for (const fold of [foldMacroGridProgram(layout, config.seed + attempt * 104729), foldLineProgram(layout)].filter(Boolean) as FOLDFormat[]) {
      lastFold = fold;
      if (
        fold.edges_vertices.length >= layout.targetEdgeRange[0]
        && fold.edges_vertices.length <= layout.targetEdgeRange[1]
        && isRealisticEnough(fold)
      ) {
        return fold;
      }
    }
  }
  if (lastFold) return lastFold;
  throw new Error("realistic box-pleat generator failed to build a line program");
}

export function makeRealisticLayout(
  archetype: RealisticBPArchetype,
  bucket: string,
  requestedCreases: number,
  rng: SeededRandom,
): RealisticLayout {
  const spec = realisticBucketSpec(bucket, requestedCreases, rng);
  const gridSize = spec.gridSize;
  const margin = Math.max(2, Math.round(gridSize * 0.06));
  const body = bodyRectFor(archetype, gridSize, rng);
  const layout: RealisticLayout = {
    archetype,
    gridSize,
    densityBucket: spec.bucket,
    targetEdgeRange: spec.targetEdgeRange,
    symmetry: symmetryFor(archetype, rng),
    bodyRegions: [{ id: "body", ...body }],
    flapTerminals: [],
    corridors: [],
    lineOps: [],
    moleculeCounts: {},
    portChecks: { checked: 0, rejected: 0 },
  };

  addBodyPanel(layout, body, rng);
  if (archetype === "insect") addInsectLayout(layout, body, margin, rng);
  else if (archetype === "quadruped") addQuadrupedLayout(layout, body, margin, rng);
  else if (archetype === "bird") addBirdLayout(layout, body, margin, rng);
  else if (archetype === "object") addObjectLayout(layout, body, margin, rng);
  else addAbstractLayout(layout, body, margin, rng);

  addDensityEnrichment(layout, body, requestedCreases, rng);
  layout.lineOps = dedupeLineOps(layout.lineOps, gridSize);
  layout.portChecks.checked = Math.max(0, layout.flapTerminals.length + layout.bodyRegions.length - 1);
  return layout;
}

export function scoreFoldRealism(fold: FOLDFormat, layout?: LayoutMetadata): RealismMetadata {
  const roles = fold.edges_bpRole ?? [];
  const bins = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0));
  const orientationHistogram: Record<string, number> = { horizontal: 0, vertical: 0, diagonalMain: 0, diagonalAnti: 0, other: 0 };
  const degrees = new Map<number, number>();
  let nonBorderEdges = 0;

  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    const assignment = fold.edges_assignment[edgeIndex];
    if (assignment === "B") continue;
    nonBorderEdges += 1;
    degrees.set(a, (degrees.get(a) ?? 0) + 1);
    degrees.set(b, (degrees.get(b) ?? 0) + 1);
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    orientationHistogram[orientationKey(dx, dy)] += 1;
    const mx = (p1[0] + p2[0]) / 2;
    const my = (p1[1] + p2[1]) / 2;
    const bx = Math.max(0, Math.min(7, Math.floor(mx * 8)));
    const by = Math.max(0, Math.min(7, Math.floor(my * 8)));
    const densityWeight = roles[edgeIndex] === "ridge" ? 2 : roles[edgeIndex] === "stretch" ? 0.8 : 0.18;
    bins[by][bx] += densityWeight;
  }

  const binValues = bins.flat();
  const mean = binValues.reduce((sum, value) => sum + value, 0) / binValues.length;
  const variance = binValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / binValues.length;
  const localDensityVariance = variance / (mean * mean + 1);
  const emptySpaceRatio = binValues.filter((value) => value <= Math.max(1, mean * 0.08)).length / binValues.length;
  const degreeHistogram: Record<string, number> = {};
  for (const degree of degrees.values()) degreeHistogram[String(degree)] = (degreeHistogram[String(degree)] ?? 0) + 1;
  const roleTotals = countValues(roles);
  const roleRatios: Record<string, number> = {};
  const roleTotal = Math.max(1, Object.values(roleTotals).reduce((sum, value) => sum + value, 0));
  for (const [role, count] of Object.entries(roleTotals)) roleRatios[role] = count / roleTotal;
  const orientationCounts = Object.values(orientationHistogram);
  const maxOrientationShare = Math.max(...orientationCounts) / Math.max(1, nonBorderEdges);
  const macroRegionDiversity = Math.min(
    1,
    ((layout?.bodyRegions.length ?? 0) * 2 + (layout?.flapTerminals.length ?? 0) + (layout?.corridors.length ?? 0)) / 22,
  );
  const repetitionPenalty = clamp01(
    (maxOrientationShare - 0.44) * 1.4
    + (localDensityVariance < 0.35 ? 0.25 : 0)
    + (emptySpaceRatio < 0.04 ? 0.18 : 0),
  );
  const orientationBalance = clamp01(1 - Math.max(0, maxOrientationShare - 0.35) / 0.45);
  const score = clamp01(
    0.25 * macroRegionDiversity
    + 0.25 * clamp01(localDensityVariance / 1.6)
    + 0.18 * clamp01(emptySpaceRatio / 0.25)
    + 0.17 * orientationBalance
    + 0.15 * (1 - repetitionPenalty),
  );

  return {
    score,
    emptySpaceRatio,
    localDensityVariance,
    repetitionPenalty,
    macroRegionDiversity,
    orientationHistogram,
    degreeHistogram,
    roleRatios,
    gates: {
      hasMacroRegions: macroRegionDiversity >= 0.25,
      hasDensityVariation: localDensityVariance >= 0.25,
      notUniformLattice: repetitionPenalty <= 0.75,
      enoughCreases: nonBorderEdges >= 120,
    },
  };
}

function foldLineProgram(layout: RealisticLayout): FOLDFormat {
  const graph = ear.graph.square() as FOLDFormat;
  for (const op of layout.lineOps) {
    const line = lineForOp(op, layout.gridSize);
    try {
      ear.graph.flatFold(graph, line.vector, line.origin, op.assignment);
    } catch {
      // Duplicate or boundary-tangent lines can be safely skipped; validation decides the sample.
    }
  }
  let fold = normalizeFold(graph, "cp-synthetic-generator/box-pleat/realistic-tree-base");
  fold.edges_bpRole = assignBPRoles(fold);
  fold.design_tree = makeDesignTree(layout);
  fold.layout_metadata = makeLayoutMetadata(layout);
  const counts = roleCounts(fold);
  fold.bp_metadata = {
    gridSize: layout.gridSize,
    bpSubfamily: "realistic-tree-base",
    flapCount: layout.flapTerminals.length,
    gadgetCount: Object.values(layout.moleculeCounts).reduce((sum, value) => sum + value, 0),
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  fold.density_metadata = {
    densityBucket: layout.densityBucket,
    gridSize: layout.gridSize,
    targetEdgeRange: layout.targetEdgeRange,
    subfamily: "realistic-tree-base",
    symmetry: layout.symmetry,
    generatorSteps: ["design-tree", "layout-search", "molecule-line-program", "rabbit-ear-flatfold"],
    moleculeCounts: layout.moleculeCounts,
  };
  fold.molecule_metadata = {
    libraryVersion: "realistic-bp-v3.0",
    molecules: layout.moleculeCounts,
    portChecks: layout.portChecks,
  };
  fold.realism_metadata = scoreFoldRealism(fold, fold.layout_metadata);
  return fold;
}

function foldMacroGridProgram(layout: RealisticLayout, seed: number): FOLDFormat | undefined {
  let closest: FOLDFormat | undefined;
  let closestDistance = Number.POSITIVE_INFINITY;
  for (let selectorAttempt = 0; selectorAttempt < 8; selectorAttempt++) {
    const selectors = selectMacroGridBands(layout, new SeededRandom(seed + selectorAttempt * 3571), selectorAttempt);
    for (const activeMode of macroGridModes(layout, selectorAttempt)) {
      const activeCells = macroGridActiveCells(layout.gridSize, selectors.rows, selectors.columns, activeMode, seed);
      if (activeCells.length === 0) continue;
      const edgeEstimate = 2 * layout.gridSize * (layout.gridSize + 1) + activeCells.length * 4;
      if (edgeEstimate > layout.targetEdgeRange[1] * 1.18) continue;
      for (let solveAttempt = 0; solveAttempt < 12; solveAttempt++) {
        const diagonalAssignments = solveDiagonalAssignments(
          layout.gridSize,
          activeCells,
          new SeededRandom(seed + selectorAttempt * 7919 + solveAttempt * 104729),
        );
        if (!diagonalAssignments) continue;
        const result = buildMacroGridFold(layout, {
          activeCells,
          diagonalAssignments,
          rowSelectors: selectors.rows,
          columnSelectors: selectors.columns,
          activeMode,
        });
        const fold = result.fold;
        if (!passesRabbitEarLocalChecks(fold)) continue;
        const targetMid = (layout.targetEdgeRange[0] + layout.targetEdgeRange[1]) / 2;
        const distance = Math.abs(fold.edges_vertices.length - targetMid);
        const realistic = isRealisticEnough(fold);
        if (realistic && distance < closestDistance) {
          closest = fold;
          closestDistance = distance;
        }
        if (
          realistic
          && fold.edges_vertices.length >= layout.targetEdgeRange[0]
          && fold.edges_vertices.length <= layout.targetEdgeRange[1]
        ) {
          return fold;
        }
      }
    }
  }
  return closest;
}

function isRealisticEnough(fold: FOLDFormat): boolean {
  const realism = fold.realism_metadata;
  return Boolean(
    realism
    && realism.score >= 0.35
    && realism.gates.hasMacroRegions
    && realism.gates.hasDensityVariation
    && realism.gates.notUniformLattice,
  );
}

function selectMacroGridBands(
  layout: RealisticLayout,
  rng: SeededRandom,
  attempt: number,
): { rows: number[]; columns: number[] } {
  const gridSize = layout.gridSize;
  const axisEdgeCount = 2 * gridSize * (gridSize + 1);
  const targetMid = (layout.targetEdgeRange[0] + layout.targetEdgeRange[1]) / 2;
  const desiredActiveCells = clampInt(Math.round((targetMid - axisEdgeCount) / 4), gridSize, Math.floor(gridSize * gridSize * 0.42));
  const balancedCount = clampInt(Math.round(desiredActiveCells / (2 * gridSize)) + (attempt % 2), 1, Math.max(2, Math.floor(gridSize / 3)));
  const rowBias = layout.archetype === "insect" || layout.archetype === "bird" ? 1 : 0;
  const columnBias = layout.archetype === "object" || layout.archetype === "abstract" ? 1 : 0;
  const rowCount = clampInt(balancedCount + rowBias, 1, Math.max(2, Math.floor(gridSize / 2.5)));
  const columnCount = clampInt(balancedCount + columnBias, 1, Math.max(2, Math.floor(gridSize / 2.5)));
  const rows = chooseBandCoordinates(rowCandidates(layout), rowCount, gridSize, rng);
  const columns = chooseBandCoordinates(columnCandidates(layout), columnCount, gridSize, rng);
  return { rows, columns };
}

function macroGridModes(layout: RealisticLayout, attempt: number): string[] {
  const preferred = layout.archetype === "object" ? "columns" : layout.archetype === "abstract" ? "xor" : "rows";
  const modes = ["xor", preferred, "rows", "columns", "checker"];
  if (attempt % 3 === 1) modes.unshift("columns");
  if (attempt % 3 === 2) modes.unshift("rows");
  return [...new Set(modes)];
}

function rowCandidates(layout: RealisticLayout): number[] {
  const gridSize = layout.gridSize;
  const body = layout.bodyRegions[0];
  const candidates = [
    body.y1,
    body.y2,
    Math.floor((body.y1 + body.y2) / 2),
    body.y1 - 1,
    body.y2 + 1,
    Math.floor(gridSize * 0.18),
    Math.floor(gridSize * 0.82),
    ...layout.flapTerminals.map((terminal) => terminal.y),
    ...layout.corridors.filter((corridor) => corridor.orientation === "horizontal").map((corridor) => corridor.coordinate),
  ];
  return candidates;
}

function columnCandidates(layout: RealisticLayout): number[] {
  const gridSize = layout.gridSize;
  const body = layout.bodyRegions[0];
  const candidates = [
    body.x1,
    body.x2,
    Math.floor((body.x1 + body.x2) / 2),
    body.x1 - 1,
    body.x2 + 1,
    Math.floor(gridSize * 0.18),
    Math.floor(gridSize * 0.82),
    ...layout.flapTerminals.map((terminal) => terminal.x),
    ...layout.corridors.filter((corridor) => corridor.orientation === "vertical").map((corridor) => corridor.coordinate),
  ];
  return candidates;
}

function chooseBandCoordinates(candidates: number[], count: number, gridSize: number, rng: SeededRandom): number[] {
  const normalized = [...new Set(candidates.map((value) => clampInt(value, 0, gridSize - 1)))].sort((a, b) => a - b);
  const selected: number[] = [];
  for (const candidate of normalized) {
    if (selected.length >= count) break;
    if (selected.some((value) => Math.abs(value - candidate) <= 1)) continue;
    selected.push(candidate);
  }
  while (selected.length < count) {
    const candidate = clampInt(rng.int(1, gridSize - 2), 0, gridSize - 1);
    if (selected.some((value) => Math.abs(value - candidate) <= 1)) continue;
    selected.push(candidate);
  }
  return selected.sort((a, b) => a - b);
}

function macroGridActiveCells(gridSize: number, rows: number[], columns: number[], mode: string, seed: number): Point[] {
  const rowSet = new Set(rows);
  const columnSet = new Set(columns);
  const active: Point[] = [];
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      let enabled = false;
      if (mode === "rows") enabled = rowSet.has(y);
      else if (mode === "columns") enabled = columnSet.has(x);
      else if (mode === "xor") enabled = rowSet.has(y) !== columnSet.has(x);
      else enabled = (x + y + Math.abs(seed)) % 2 === 0;
      if (enabled) active.push([x, y]);
    }
  }
  return active;
}

function solveDiagonalAssignments(
  gridSize: number,
  activeCells: Point[],
  rng: SeededRandom,
): Map<string, EdgeAssignment> | undefined {
  const active = new Set(activeCells.map(([x, y]) => cellKey(x, y)));
  const variableKeys: string[] = [];
  const variableIndex = new Map<string, number>();
  for (const [x, y] of activeCells) {
    for (const corner of CORNERS) {
      const key = diagonalVariableKey(x, y, corner);
      variableIndex.set(key, variableKeys.length);
      variableKeys.push(key);
    }
  }

  const equations: Array<{ indices: number[]; rhs: 0 | 1 }> = [];
  for (const [x, y] of activeCells) {
    equations.push({
      indices: CORNERS.map((corner) => variableIndex.get(diagonalVariableKey(x, y, corner))!),
      rhs: 1,
    });
  }
  for (let vy = 1; vy < gridSize; vy++) {
    for (let vx = 1; vx < gridSize; vx++) {
      const terms: number[] = [];
      for (const [cx, cy] of [
        [vx - 1, vy - 1],
        [vx, vy - 1],
        [vx - 1, vy],
        [vx, vy],
      ] as Point[]) {
        if (!active.has(cellKey(cx, cy))) continue;
        terms.push(variableIndex.get(diagonalVariableKey(cx, cy, cornerAtVertex(cx, cy, vx, vy)))!);
      }
      if (terms.length === 0) continue;
      if (terms.length % 2 !== 0) return undefined;
      equations.push({ indices: terms, rhs: terms.length === 2 ? 1 : 0 });
    }
  }

  const solution = solveBinarySystem(variableKeys.length, equations, rng);
  if (!solution) return undefined;
  const assignments = new Map<string, EdgeAssignment>();
  for (const [index, key] of variableKeys.entries()) {
    assignments.set(key, solution[index] === 1 ? "V" : "M");
  }
  return assignments;
}

function solveBinarySystem(
  variableCount: number,
  equations: Array<{ indices: number[]; rhs: 0 | 1 }>,
  rng: SeededRandom,
): Uint8Array | undefined {
  const width = variableCount + 1;
  const matrix = equations.map(({ indices, rhs }) => {
    const row = new Uint8Array(width);
    for (const index of indices) row[index] ^= 1;
    row[variableCount] = rhs;
    return row;
  });
  const pivotColumns: number[] = [];
  let pivotRow = 0;
  for (let column = 0; column < variableCount && pivotRow < matrix.length; column++) {
    let found = pivotRow;
    while (found < matrix.length && matrix[found][column] === 0) found += 1;
    if (found === matrix.length) continue;
    [matrix[pivotRow], matrix[found]] = [matrix[found], matrix[pivotRow]];
    for (let row = 0; row < matrix.length; row++) {
      if (row === pivotRow || matrix[row][column] === 0) continue;
      for (let j = column; j < width; j++) matrix[row][j] ^= matrix[pivotRow][j];
    }
    pivotColumns[pivotRow] = column;
    pivotRow += 1;
  }
  for (let row = pivotRow; row < matrix.length; row++) {
    let hasCoefficient = false;
    for (let column = 0; column < variableCount; column++) {
      if (matrix[row][column] === 1) {
        hasCoefficient = true;
        break;
      }
    }
    if (!hasCoefficient && matrix[row][variableCount] === 1) return undefined;
  }

  const pivotSet = new Set(pivotColumns);
  const solution = new Uint8Array(variableCount);
  for (let index = 0; index < variableCount; index++) {
    if (!pivotSet.has(index)) solution[index] = rng.next() < 0.5 ? 0 : 1;
  }
  for (let row = pivotRow - 1; row >= 0; row--) {
    const column = pivotColumns[row];
    let value = matrix[row][variableCount];
    for (let j = column + 1; j < variableCount; j++) {
      if (matrix[row][j] === 1) value ^= solution[j];
    }
    solution[column] = value;
  }
  return solution;
}

function buildMacroGridFold(
  layout: RealisticLayout,
  input: {
    activeCells: Point[];
    diagonalAssignments: Map<string, EdgeAssignment>;
    rowSelectors: number[];
    columnSelectors: number[];
    activeMode: string;
  },
): MacroGridFoldResult {
  const vertices: Point[] = [];
  const vertexKeys = new Map<string, number>();
  const edges: MacroGridEdge[] = [];
  const gridSize = layout.gridSize;
  const rowSet = new Set(input.rowSelectors);
  const columnSet = new Set(input.columnSelectors);
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
      const border = y === 0 || y === gridSize;
      const role: BPRole = border ? "border" : rowSet.has(y) || rowSet.has(y - 1) ? "stretch" : "hinge";
      addEdge(
        vertexIndex(x, y),
        vertexIndex(x + 1, y),
        border ? "B" : x % 2 === 0 ? "V" : "M",
        role,
      );
    }
  }
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      const border = x === 0 || x === gridSize;
      const role: BPRole = border ? "border" : columnSet.has(x) || columnSet.has(x - 1) ? "stretch" : "axis";
      addEdge(
        vertexIndex(x, y),
        vertexIndex(x, y + 1),
        border ? "B" : "M",
        role,
      );
    }
  }

  for (const [x, y] of input.activeCells) {
    const center = vertexIndex(x + 0.5, y + 0.5);
    for (const [cx, cy, corner] of [
      [x, y, "ll"],
      [x + 1, y, "lr"],
      [x + 1, y + 1, "ur"],
      [x, y + 1, "ul"],
    ] as Array<[number, number, CornerName]>) {
      addEdge(
        center,
        vertexIndex(cx, cy),
        input.diagonalAssignments.get(diagonalVariableKey(x, y, corner)) ?? "M",
        "ridge",
      );
    }
  }

  let fold = normalizeFold(
    {
      file_spec: 1.1,
      file_creator: "cp-synthetic-generator/box-pleat/realistic-tree-base/macro-grid",
      file_classes: ["singleModel"],
      frame_classes: ["creasePattern"],
      vertices_coords: vertices,
      edges_vertices: edges.map((edge) => [edge.a, edge.b]),
      edges_assignment: edges.map((edge) => edge.assignment),
      edges_bpRole: edges.map((edge) => edge.role),
    },
    "cp-synthetic-generator/box-pleat/realistic-tree-base/macro-grid",
  );
  fold.design_tree = makeDesignTree(layout);
  fold.layout_metadata = makeLayoutMetadata(layout);
  const moleculeCounts = {
    ...layout.moleculeCounts,
    "macro-grid-corridor": 2 * layout.gridSize - 2,
    "cell-x": input.activeCells.length,
    "row-chevron-band": input.rowSelectors.length,
    "column-chevron-band": input.columnSelectors.length,
  };
  const counts = roleCounts(fold);
  fold.bp_metadata = {
    gridSize: layout.gridSize,
    bpSubfamily: "realistic-tree-base",
    flapCount: layout.flapTerminals.length,
    gadgetCount: Object.values(moleculeCounts).reduce((sum, value) => sum + value, 0),
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  fold.density_metadata = {
    densityBucket: layout.densityBucket,
    gridSize: layout.gridSize,
    targetEdgeRange: layout.targetEdgeRange,
    subfamily: "realistic-tree-base",
    symmetry: `${layout.symmetry}/${input.activeMode}`,
    generatorSteps: ["design-tree", "layout-search", "macro-grid-band-selection", "gf2-diagonal-assignment"],
    moleculeCounts,
  };
  fold.molecule_metadata = {
    libraryVersion: "realistic-bp-v3.0",
    molecules: moleculeCounts,
    portChecks: {
      checked: Math.max(layout.portChecks.checked, input.rowSelectors.length + input.columnSelectors.length),
      rejected: layout.portChecks.rejected,
    },
  };
  fold.realism_metadata = scoreFoldRealism(fold, fold.layout_metadata);
  return {
    fold,
    activeCellCount: input.activeCells.length,
    rowSelectors: input.rowSelectors,
    columnSelectors: input.columnSelectors,
    activeMode: input.activeMode,
  };
}

function passesRabbitEarLocalChecks(fold: FOLDFormat): boolean {
  try {
    const graph = normalizeFold(fold);
    ear.graph.populate(graph);
    return ear.singleVertex.validateKawasaki(graph).length === 0 && ear.singleVertex.validateMaekawa(graph).length === 0;
  } catch {
    return false;
  }
}

function realisticBucketSpec(bucket: string, requestedCreases: number, rng: SeededRandom): { bucket: string; gridSize: number; targetEdgeRange: [number, number] } {
  const normalized = bucket === "small" || bucket === "medium" || bucket === "dense" || bucket === "superdense" ? bucket : "medium";
  const gridChoices: Record<string, number[]> = {
    small: [8, 10],
    medium: [10, 16],
    dense: [16, 20],
    superdense: [20, 25, 32],
  };
  const targets: Record<string, [number, number]> = {
    small: [150, 520],
    medium: [300, 950],
    dense: [650, 1800],
    superdense: [1100, 3600],
  };
  const choices = gridChoices[normalized];
  const requestedBias = requestedCreases > targets[normalized][0] ? 1 : 0;
  return {
    bucket: normalized,
    gridSize: choices[Math.min(choices.length - 1, (rng.int(0, choices.length - 1) + requestedBias) % choices.length)],
    targetEdgeRange: targets[normalized],
  };
}

function bodyRectFor(archetype: RealisticBPArchetype, gridSize: number, rng: SeededRandom): Omit<GridRect, "id"> {
  const jitter = rng.int(-2, 2);
  if (archetype === "bird") return rect(gridSize, 0.34, 0.24, 0.66, 0.72, jitter);
  if (archetype === "object") return rect(gridSize, 0.22, 0.30, 0.78, 0.70, jitter);
  if (archetype === "abstract") return rect(gridSize, 0.30, 0.28, 0.70, 0.68, jitter);
  return rect(gridSize, 0.28, 0.34, 0.72, 0.66, jitter);
}

function rect(gridSize: number, x1: number, y1: number, x2: number, y2: number, jitter: number): Omit<GridRect, "id"> {
  return {
    x1: clampGrid(Math.round(gridSize * x1) + jitter, 2, gridSize - 4),
    y1: clampGrid(Math.round(gridSize * y1) - jitter, 2, gridSize - 4),
    x2: clampGrid(Math.round(gridSize * x2) + jitter, 4, gridSize - 2),
    y2: clampGrid(Math.round(gridSize * y2) + jitter, 4, gridSize - 2),
  };
}

function addBodyPanel(layout: RealisticLayout, body: Omit<GridRect, "id">, rng: SeededRandom): void {
  addRect(layout, body, "body-panel");
  const panelStep = Math.max(2, Math.round((body.x2 - body.x1) / (layout.densityBucket === "small" ? 4 : 6)));
  for (let x = body.x1 + panelStep; x < body.x2; x += panelStep) addV(layout, x, x % 4 === 0 ? "M" : "V", "body-panel");
  for (let y = body.y1 + panelStep; y < body.y2; y += panelStep) addH(layout, y, y % 4 === 0 ? "V" : "M", "body-panel");
  addDiamondChain(layout, Math.floor((body.x1 + body.x2) / 2), body.y1, body.y2, Math.max(2, panelStep), rng);
  bump(layout, "body-panel");
}

function addInsectLayout(layout: RealisticLayout, body: Omit<GridRect, "id">, margin: number, rng: SeededRandom): void {
  const ys = spreadIntegers(body.y1 - 5, body.y2 + 5, 6, margin, layout.gridSize - margin);
  ys.forEach((y, index) => {
    const left = addTerminal(layout, `left-leg-${index + 1}`, margin, y, "left");
    const right = addTerminal(layout, `right-leg-${index + 1}`, layout.gridSize - margin, y, "right");
    addAppendageFan(layout, left, body.x1, y, index % 2 === 0 ? "appendage-fan" : "chevron-band");
    addAppendageFan(layout, right, body.x2, y, index % 2 === 0 ? "chevron-band" : "appendage-fan");
  });
  addTerminal(layout, "head", Math.floor((body.x1 + body.x2) / 2), margin, "top");
  addTerminal(layout, "tail", Math.floor((body.x1 + body.x2) / 2), layout.gridSize - margin, "bottom");
  addCornerClaws(layout, rng);
}

function addQuadrupedLayout(layout: RealisticLayout, body: Omit<GridRect, "id">, margin: number, rng: SeededRandom): void {
  const terminals = [
    addTerminal(layout, "front-left-leg", margin, body.y1, "left"),
    addTerminal(layout, "front-right-leg", layout.gridSize - margin, body.y1, "right"),
    addTerminal(layout, "back-left-leg", margin, body.y2, "left"),
    addTerminal(layout, "back-right-leg", layout.gridSize - margin, body.y2, "right"),
    addTerminal(layout, "head", body.x2 + Math.max(2, Math.floor((layout.gridSize - body.x2) / 2)), body.y1 - 2, "right"),
    addTerminal(layout, "tail", body.x1 - Math.max(2, Math.floor(body.x1 / 2)), body.y2 + 2, "left"),
  ];
  terminals.forEach((terminal, index) => addAppendageFan(layout, terminal, index % 2 === 0 ? body.x1 : body.x2, clampGrid(terminal.y, body.y1, body.y2), "appendage-fan"));
  addStretchBoxes(layout, body, rng);
}

function addBirdLayout(layout: RealisticLayout, body: Omit<GridRect, "id">, margin: number, rng: SeededRandom): void {
  const wingY = Math.floor((body.y1 + body.y2) / 2);
  const leftWing = addTerminal(layout, "left-wing", margin, wingY, "left");
  const rightWing = addTerminal(layout, "right-wing", layout.gridSize - margin, wingY, "right");
  addAppendageFan(layout, leftWing, body.x1, wingY, "chevron-band");
  addAppendageFan(layout, rightWing, body.x2, wingY, "chevron-band");
  addTerminal(layout, "head-neck", Math.floor((body.x1 + body.x2) / 2), margin, "top");
  addTerminal(layout, "tail-fan", Math.floor((body.x1 + body.x2) / 2), layout.gridSize - margin, "bottom");
  addDiamondChain(layout, Math.floor((body.x1 + body.x2) / 2), body.y2, layout.gridSize - margin, 3, rng);
}

function addObjectLayout(layout: RealisticLayout, body: Omit<GridRect, "id">, margin: number, rng: SeededRandom): void {
  addTerminal(layout, "handle-left", margin, Math.floor((body.y1 + body.y2) / 2), "left");
  addTerminal(layout, "handle-right", layout.gridSize - margin, Math.floor((body.y1 + body.y2) / 2), "right");
  addTerminal(layout, "top-lock", Math.floor((body.x1 + body.x2) / 2), margin, "top");
  addTerminal(layout, "bottom-lock", Math.floor((body.x1 + body.x2) / 2), layout.gridSize - margin, "bottom");
  for (let y = body.y1 - 4; y <= body.y2 + 4; y += 4) addH(layout, clampGrid(y, margin, layout.gridSize - margin), y % 8 === 0 ? "V" : "M", "hinge-corridor");
  for (let x = body.x1 - 4; x <= body.x2 + 4; x += 4) addV(layout, clampGrid(x, margin, layout.gridSize - margin), x % 8 === 0 ? "M" : "V", "axis-corridor");
  for (const terminal of layout.flapTerminals.slice(-4)) {
    addAppendageFan(layout, terminal, clampGrid(terminal.x, body.x1, body.x2), clampGrid(terminal.y, body.y1, body.y2), "chevron-band");
  }
  addStretchBoxes(layout, body, rng);
}

function addAbstractLayout(layout: RealisticLayout, body: Omit<GridRect, "id">, margin: number, rng: SeededRandom): void {
  const terminals = [
    addTerminal(layout, "top-left-flap", margin, margin, "top"),
    addTerminal(layout, "top-right-flap", layout.gridSize - margin, margin + rng.int(0, 4), "top"),
    addTerminal(layout, "bottom-left-flap", margin + rng.int(0, 4), layout.gridSize - margin, "bottom"),
    addTerminal(layout, "bottom-right-flap", layout.gridSize - margin, layout.gridSize - margin, "bottom"),
  ];
  terminals.forEach((terminal, index) => addAppendageFan(layout, terminal, index < 2 ? body.x1 : body.x2, index % 2 === 0 ? body.y1 : body.y2, "appendage-fan"));
  addDiamondChain(layout, body.x1, body.y1, body.y2, 3, rng);
  addDiamondChain(layout, body.x2, body.y1, body.y2, 3, rng);
}

function addDensityEnrichment(layout: RealisticLayout, body: Omit<GridRect, "id">, requestedCreases: number, rng: SeededRandom): void {
  const density = layout.densityBucket === "superdense" ? 3 : layout.densityBucket === "dense" ? 2 : layout.densityBucket === "medium" ? 1 : 0;
  const step = density >= 2 ? 2 : 4;
  for (let i = 0; i < density + 1; i++) {
    const inset = 2 + i * step;
    if (body.x1 - inset > 1) addV(layout, body.x1 - inset, i % 2 === 0 ? "M" : "V", "axis-corridor");
    if (body.x2 + inset < layout.gridSize - 1) addV(layout, body.x2 + inset, i % 2 === 0 ? "V" : "M", "axis-corridor");
    if (body.y1 - inset > 1) addH(layout, body.y1 - inset, i % 2 === 0 ? "V" : "M", "hinge-corridor");
    if (body.y2 + inset < layout.gridSize - 1) addH(layout, body.y2 + inset, i % 2 === 0 ? "M" : "V", "hinge-corridor");
  }
  const diagonalStep = requestedCreases > 1000 || layout.densityBucket === "superdense" ? 3 : requestedCreases > 500 ? 4 : 6;
  for (let c = body.x1 - body.y2; c <= body.x2 - body.y1; c += diagonalStep) addDMain(layout, c, "M", "diamond-chain");
  for (let c = body.x1 + body.y1; c <= body.x2 + body.y2; c += diagonalStep) addDAnti(layout, c, "V", "diamond-chain");

  if (density >= 1) {
    const globalStep = density === 3 ? 2 : density === 2 ? 3 : 5;
    const diagonalGlobalStep = density === 3 ? 2 : density === 2 ? 3 : 5;
    for (let x = 2; x < layout.gridSize - 1; x += globalStep) {
      const nearBody = x >= body.x1 - step * 3 && x <= body.x2 + step * 3;
      if (nearBody || density >= 2 || rng.next() < 0.35) {
        addV(layout, x, x % 4 === 0 ? "M" : "V", "axis-corridor");
      }
    }
    for (let y = 2; y < layout.gridSize - 1; y += globalStep) {
      const nearBody = y >= body.y1 - step * 3 && y <= body.y2 + step * 3;
      if (nearBody || density >= 2 || rng.next() < 0.35) {
        addH(layout, y, y % 4 === 0 ? "V" : "M", "hinge-corridor");
      }
    }
    for (let c = -layout.gridSize + diagonalGlobalStep; c < layout.gridSize; c += diagonalGlobalStep) {
      if (Math.abs(c) < layout.gridSize * 0.9 || density === 3 || rng.next() < 0.45) {
        addDMain(layout, c, "M", "chevron-band");
      }
    }
    for (let c = diagonalGlobalStep; c < layout.gridSize * 2; c += diagonalGlobalStep) {
      const centerDistance = Math.abs(c - layout.gridSize);
      if (centerDistance < layout.gridSize * 0.9 || density === 3 || rng.next() < 0.45) {
        addDAnti(layout, c, "V", "chevron-band");
      }
    }
  }
}

function addAppendageFan(layout: RealisticLayout, terminal: FlapTerminal, anchorX: number, anchorY: number, molecule: string): void {
  addCorridor(layout, "horizontal", terminal.y, terminal.id, terminal.y % 4 === 0 ? "V" : "M", "hinge-corridor");
  addCorridor(layout, "vertical", terminal.x, terminal.id, terminal.x % 4 === 0 ? "M" : "V", "axis-corridor");
  addDMain(layout, terminal.x - terminal.y, "M", molecule);
  addDMain(layout, anchorX - anchorY, "M", molecule);
  addDAnti(layout, terminal.x + terminal.y, "V", molecule);
  addDAnti(layout, anchorX + anchorY, "V", molecule);
  bump(layout, molecule);
}

function addCornerClaws(layout: RealisticLayout, rng: SeededRandom): void {
  const g = layout.gridSize;
  const offsets = [2, 4, 6 + (rng.int(0, 1) * 2)];
  for (const offset of offsets) {
    addH(layout, offset, offset % 4 === 0 ? "M" : "V", "corner-claw");
    addV(layout, offset, offset % 4 === 0 ? "V" : "M", "corner-claw");
    addH(layout, g - offset, offset % 4 === 0 ? "V" : "M", "corner-claw");
    addV(layout, g - offset, offset % 4 === 0 ? "M" : "V", "corner-claw");
  }
  bump(layout, "corner-claw", offsets.length * 4);
}

function addStretchBoxes(layout: RealisticLayout, body: Omit<GridRect, "id">, rng: SeededRandom): void {
  const centers = [
    [body.x1, body.y1],
    [body.x2, body.y1],
    [body.x2, body.y2],
    [body.x1, body.y2],
  ] as const;
  const radius = rng.choice([2, 3, 4]);
  centers.forEach(([cx, cy]) => {
    addDMain(layout, cx - cy - radius, "M", "elias-like-stretch");
    addDMain(layout, cx - cy + radius, "M", "elias-like-stretch");
    addDAnti(layout, cx + cy - radius, "V", "elias-like-stretch");
    addDAnti(layout, cx + cy + radius, "V", "elias-like-stretch");
  });
  bump(layout, "elias-like-stretch", centers.length);
}

function addDiamondChain(layout: RealisticLayout, x: number, y1: number, y2: number, radius: number, rng: SeededRandom): void {
  const step = Math.max(4, radius * 2);
  for (let y = y1; y <= y2; y += step) {
    const wobble = rng.int(-1, 1);
    addDMain(layout, x + wobble - y - radius, "M", "diamond-chain");
    addDMain(layout, x + wobble - y + radius, "M", "diamond-chain");
    addDAnti(layout, x + wobble + y - radius, "V", "diamond-chain");
    addDAnti(layout, x + wobble + y + radius, "V", "diamond-chain");
  }
  bump(layout, "diamond-chain");
}

function addRect(layout: RealisticLayout, rect: Omit<GridRect, "id">, molecule: string): void {
  addH(layout, rect.y1, "V", molecule);
  addH(layout, rect.y2, "V", molecule);
  addV(layout, rect.x1, "M", molecule);
  addV(layout, rect.x2, "M", molecule);
}

function addTerminal(layout: RealisticLayout, id: string, x: number, y: number, side: Side): FlapTerminal {
  const terminal = { id, x: clampGrid(x, 1, layout.gridSize - 1), y: clampGrid(y, 1, layout.gridSize - 1), side };
  layout.flapTerminals.push(terminal);
  return terminal;
}

function addCorridor(layout: RealisticLayout, orientation: "horizontal" | "vertical", coordinate: number, id: string, assignment: EdgeAssignment, molecule: string): void {
  if (orientation === "horizontal") addH(layout, coordinate, assignment, molecule);
  else addV(layout, coordinate, assignment, molecule);
  layout.corridors.push({
    id: `${id}-${orientation}-${coordinate}`,
    orientation,
    coordinate,
    role: orientation === "horizontal" ? "hinge" : "axis",
  });
}

function addH(layout: RealisticLayout, n: number, assignment: EdgeAssignment, molecule: string): void {
  if (n <= 0 || n >= layout.gridSize) return;
  layout.lineOps.push({ kind: "h", n, assignment, molecule });
  bump(layout, molecule);
}

function addV(layout: RealisticLayout, n: number, assignment: EdgeAssignment, molecule: string): void {
  if (n <= 0 || n >= layout.gridSize) return;
  layout.lineOps.push({ kind: "v", n, assignment, molecule });
  bump(layout, molecule);
}

function addDMain(layout: RealisticLayout, n: number, assignment: EdgeAssignment, molecule: string): void {
  if (n <= -layout.gridSize || n >= layout.gridSize) return;
  layout.lineOps.push({ kind: "dMain", n, assignment, molecule });
  bump(layout, molecule);
}

function addDAnti(layout: RealisticLayout, n: number, assignment: EdgeAssignment, molecule: string): void {
  if (n <= 0 || n >= layout.gridSize * 2) return;
  layout.lineOps.push({ kind: "dAnti", n, assignment, molecule });
  bump(layout, molecule);
}

function bump(layout: RealisticLayout, molecule: string, amount = 1): void {
  layout.moleculeCounts[molecule] = (layout.moleculeCounts[molecule] ?? 0) + amount;
}

function dedupeLineOps(ops: LineOp[], gridSize: number): LineOp[] {
  const seen = new Map<string, LineOp>();
  for (const op of ops) {
    if (op.kind === "h" || op.kind === "v") {
      if (op.n <= 0 || op.n >= gridSize) continue;
    } else if (op.kind === "dMain") {
      if (op.n <= -gridSize || op.n >= gridSize) continue;
    } else if (op.n <= 0 || op.n >= gridSize * 2) continue;
    const key = `${op.kind}:${op.n}`;
    if (!seen.has(key)) seen.set(key, op);
  }
  return [...seen.values()].sort((a, b) => lineSortKey(a) - lineSortKey(b));
}

function lineSortKey(op: LineOp): number {
  const order: Record<LineKind, number> = { h: 0, v: 1, dMain: 2, dAnti: 3 };
  return order[op.kind] * 10000 + op.n;
}

function lineForOp(op: LineOp, gridSize: number): { vector: Point; origin: Point } {
  const c = op.n / gridSize;
  if (op.kind === "h") return { vector: [1, 0], origin: [0, c] };
  if (op.kind === "v") return { vector: [0, 1], origin: [c, 0] };
  if (op.kind === "dMain") return { vector: [1, 1], origin: c >= 0 ? [c, 0] : [0, -c] };
  return { vector: [1, -1], origin: c <= 1 ? [0, c] : [c - 1, 1] };
}

function assignBPRoles(fold: FOLDFormat): BPRole[] {
  return fold.edges_vertices.map(([a, b], edgeIndex) => {
    if (fold.edges_assignment[edgeIndex] === "B") return "border";
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    if (isDiagonal45(p1, p2)) return "ridge";
    if (Math.abs(p1[0] - p2[0]) < 1e-8) return "axis";
    if (Math.abs(p1[1] - p2[1]) < 1e-8) return "hinge";
    return "ridge";
  });
}

function makeDesignTree(layout: RealisticLayout): DesignTreeMetadata {
  const nodes: DesignTreeMetadata["nodes"] = [
    { id: "body", kind: "body", label: `${layout.archetype} body` },
    ...layout.flapTerminals.map((terminal) => ({ id: terminal.id, kind: "flap" as const, label: terminal.id.replaceAll("-", " ") })),
  ];
  const edges: DesignTreeMetadata["edges"] = layout.flapTerminals.map((terminal) => ({
    from: "body",
    to: terminal.id,
    length: Math.max(1, Math.round(Math.hypot(terminal.x - layout.gridSize / 2, terminal.y - layout.gridSize / 2) / 2)),
    role: "flap",
  }));
  return { archetype: layout.archetype, rootId: "body", nodes, edges };
}

function makeLayoutMetadata(layout: RealisticLayout): LayoutMetadata {
  return {
    gridSize: layout.gridSize,
    symmetry: layout.symmetry,
    margin: Math.max(1, Math.round(layout.gridSize * 0.06)),
    bodyRegions: layout.bodyRegions,
    flapTerminals: layout.flapTerminals,
    corridors: layout.corridors,
    layoutScore: Math.min(1, (layout.flapTerminals.length + layout.bodyRegions.length * 2) / 14),
  };
}

function spreadIntegers(start: number, end: number, count: number, min: number, max: number): number[] {
  if (count <= 1) return [clampGrid(Math.round((start + end) / 2), min, max)];
  return Array.from({ length: count }, (_, index) => clampGrid(Math.round(start + ((end - start) * index) / (count - 1)), min, max));
}

function symmetryFor(archetype: RealisticBPArchetype, rng: SeededRandom): string {
  if (archetype === "insect" || archetype === "bird") return rng.next() < 0.8 ? "bilateral" : "near-bilateral";
  if (archetype === "quadruped") return rng.next() < 0.65 ? "bilateral" : "offset-bilateral";
  if (archetype === "object") return rng.next() < 0.75 ? "rectangular" : "offset-rectangular";
  return rng.choice(["asymmetric", "rotational", "near-bilateral"]);
}

function orientationKey(dx: number, dy: number): string {
  if (Math.abs(dy) < 1e-8) return "horizontal";
  if (Math.abs(dx) < 1e-8) return "vertical";
  if (Math.abs(Math.abs(dx) - Math.abs(dy)) < 1e-8) return dx * dy > 0 ? "diagonalMain" : "diagonalAnti";
  return "other";
}

function isDiagonal45(a: Point, b: Point): boolean {
  return Math.abs(Math.abs(a[0] - b[0]) - Math.abs(a[1] - b[1])) < 1e-8;
}

function countValues(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function cellKey(x: number, y: number): string {
  return `${x}:${y}`;
}

function diagonalVariableKey(x: number, y: number, corner: CornerName): string {
  return `${x}:${y}:${corner}`;
}

function cornerAtVertex(cellX: number, cellY: number, vertexX: number, vertexY: number): CornerName {
  if (vertexX === cellX && vertexY === cellY) return "ll";
  if (vertexX === cellX + 1 && vertexY === cellY) return "lr";
  if (vertexX === cellX + 1 && vertexY === cellY + 1) return "ur";
  return "ul";
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function clampInt(value: number, min: number, max: number): number {
  return Math.round(Math.min(max, Math.max(min, value)));
}

function clampGrid(value: number, min: number, max: number): number {
  return Math.round(Math.min(max, Math.max(min, value)));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function realismSummary(fold: FOLDFormat): {
  assignments: Record<string, number>;
  roles: Record<string, number>;
  realism: RealismMetadata;
} {
  return {
    assignments: assignmentCounts(fold),
    roles: roleCounts(fold),
    realism: fold.realism_metadata ?? scoreFoldRealism(fold, fold.layout_metadata),
  };
}
