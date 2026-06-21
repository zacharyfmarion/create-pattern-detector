import ear from "rabbit-ear";
import { normalizeFold } from "./fold-utils.ts";
import type { EdgeAssignment, FOLDFormat, GlobalValidationBackend, TessellationMetadata, ValidationConfig, ValidationResult } from "./types.ts";

const GEOMETRY_EPSILON = 1e-9;

export function preflightValidation(config: ValidationConfig): void {
  if (!config.strictGlobal) return;
  if (config.globalBackend === "rabbit-ear-solver") {
    if (typeof ear.layer?.solver !== "function") {
      throw new Error("Rabbit Ear layer solver is unavailable; cannot run strict global validation");
    }
    return;
  }
  if (config.globalBackend === "fold-cli") {
    const command = config.foldCliCommand;
    if (!command) {
      throw new Error("validation.foldCliCommand is required when globalBackend is fold-cli");
    }
    if (command === "fold" || command === "/usr/bin/fold") {
      throw new Error("Refusing to use system text-wrapping `fold`; configure an explicit FOLD validator command");
    }
    return;
  }
  throw new Error(`Unknown global validation backend: ${String(config.globalBackend)}`);
}

export async function validateFold(fold: FOLDFormat, config: ValidationConfig): Promise<ValidationResult> {
  const passed: string[] = [];
  const failed: string[] = [];
  const errors: string[] = [];
  const metrics: ValidationResult["metrics"] = {};
  const normalized = normalizeFold(fold);

  runCheck("schema", passed, failed, errors, () => checkSchema(normalized));
  runCheck("complete-border", passed, failed, errors, () => checkCompleteBorder(normalized));
  runCheck("edge-geometry", passed, failed, errors, () => checkEdgeGeometry(normalized));
  runCheck("no-self-intersections", passed, failed, errors, () => checkSelfIntersections(normalized));
  runCheck("complexity-bounds", passed, failed, errors, () => checkComplexity(normalized, config));
  if (config.requireDense) {
    runCheck("dense-structure", passed, failed, errors, () => checkDenseStructure(normalized));
  }
  if (config.requireTreeMaker || normalized.treemaker_metadata || normalized.tree_metadata) {
    runCheck("treemaker-structure", passed, failed, errors, () => checkTreeMakerStructure(normalized));
  }
  if (config.requireRabbitEarFoldProgram || normalized.rabbit_ear_metadata) {
    runCheck("rabbit-ear-fold-program-structure", passed, failed, errors, () => checkRabbitEarFoldProgramStructure(normalized));
  }
  if (config.requireTessellationFoldProgram || normalized.tessellation_metadata) {
    runCheck("tessellation-fold-program-structure", passed, failed, errors, () => checkTessellationFoldProgramStructure(normalized));
  }
  if (config.requireLocalFlatFoldability !== false) {
    runCheck("local-flat-foldability", passed, failed, errors, () => checkLocalFlatFoldability(normalized));
  }

  if (config.strictGlobal) {
    if (config.globalBackend === "rabbit-ear-solver") {
      runCheck("rabbit-ear-solver", passed, failed, errors, () => Object.assign(metrics, checkRabbitEarSolver(normalized)));
    } else if (config.globalBackend === "fold-cli") {
      const result = await checkFoldCli(normalized, config);
      if (result.ok) passed.push("fold-cli");
      else {
        failed.push("fold-cli");
        errors.push(...result.errors);
      }
    }
  }

  return {
    valid: failed.length === 0,
    passed,
    failed,
    errors,
    metrics,
  };
}

function runCheck(
  name: string,
  passed: string[],
  failed: string[],
  errors: string[],
  fn: () => void,
): void {
  try {
    fn();
    passed.push(name);
  } catch (error) {
    failed.push(name);
    errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function checkSchema(fold: FOLDFormat): void {
  if (!Array.isArray(fold.vertices_coords) || fold.vertices_coords.length < 4) {
    throw new Error("requires at least four vertices");
  }
  if (!Array.isArray(fold.edges_vertices) || fold.edges_vertices.length < 4) {
    throw new Error("requires at least four edges");
  }
  if (!Array.isArray(fold.edges_assignment) || fold.edges_assignment.length !== fold.edges_vertices.length) {
    throw new Error("edges_assignment must match edges_vertices length");
  }
}

function checkCompleteBorder(fold: FOLDFormat): void {
  const borderEdges = fold.edges_vertices.filter((_, i) => fold.edges_assignment[i] === "B");
  if (borderEdges.length < 4) {
    throw new Error(`expected at least 4 border edges, found ${borderEdges.length}`);
  }
  const degrees = new Map<number, number>();
  for (const [a, b] of borderEdges) {
    degrees.set(a, (degrees.get(a) ?? 0) + 1);
    degrees.set(b, (degrees.get(b) ?? 0) + 1);
  }
  const odd = [...degrees.values()].filter((degree) => degree !== 2);
  if (odd.length > 0) {
    throw new Error("border edges must form closed cycles with degree 2 at each border vertex");
  }
}

function checkEdgeGeometry(fold: FOLDFormat): void {
  const seen = new Set<string>();
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    if (a === b) throw new Error(`edge ${edgeIndex} is degenerate`);
    if (!fold.vertices_coords[a] || !fold.vertices_coords[b]) {
      throw new Error(`edge ${edgeIndex} references a missing vertex`);
    }
    const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
    if (seen.has(key)) throw new Error(`duplicate edge ${key}`);
    seen.add(key);
    if (distance(fold.vertices_coords[a], fold.vertices_coords[b]) < 1e-8) {
      throw new Error(`edge ${edgeIndex} has near-zero length`);
    }
  }
}

function checkSelfIntersections(fold: FOLDFormat): void {
  for (let i = 0; i < fold.edges_vertices.length; i++) {
    for (let j = i + 1; j < fold.edges_vertices.length; j++) {
      const e1 = fold.edges_vertices[i];
      const e2 = fold.edges_vertices[j];
      if (sharesVertex(e1, e2)) continue;
      const a = fold.vertices_coords[e1[0]];
      const b = fold.vertices_coords[e1[1]];
      const c = fold.vertices_coords[e2[0]];
      const d = fold.vertices_coords[e2[1]];
      if (segmentsIntersect(a, b, c, d)) {
        throw new Error(`edges ${i} and ${j} intersect away from vertices`);
      }
    }
  }
}

function checkComplexity(fold: FOLDFormat, config: ValidationConfig): void {
  if (fold.vertices_coords.length > config.maxVertices) {
    throw new Error(`too many vertices: ${fold.vertices_coords.length} > ${config.maxVertices}`);
  }
  if (fold.edges_vertices.length > config.maxEdges) {
    throw new Error(`too many edges: ${fold.edges_vertices.length} > ${config.maxEdges}`);
  }
  for (let i = 0; i < fold.vertices_coords.length; i++) {
    for (let j = i + 1; j < fold.vertices_coords.length; j++) {
      if (distance(fold.vertices_coords[i], fold.vertices_coords[j]) < config.minVertexDistance) {
        throw new Error(`vertices ${i} and ${j} are closer than ${config.minVertexDistance}`);
      }
    }
  }
}

function checkDenseStructure(fold: FOLDFormat): void {
  const metadata = fold.density_metadata;
  if (!metadata) throw new Error("density_metadata is required");
  const [minEdges, maxEdges] = metadata.targetEdgeRange;
  if (fold.edges_vertices.length < minEdges) {
    throw new Error(`too few edges for dense bucket: ${fold.edges_vertices.length} < ${minEdges}`);
  }
  if (fold.edges_vertices.length > maxEdges) {
    throw new Error(`too many edges for dense bucket: ${fold.edges_vertices.length} > ${maxEdges}`);
  }
  const assignmentTotals = countValues(fold.edges_assignment);
  if ((assignmentTotals.M ?? 0) + (assignmentTotals.V ?? 0) < Math.min(40, minEdges / 2)) {
    throw new Error("dense samples require substantial M/V crease assignments");
  }
  if (fold.treemaker_metadata) return;
  const graph = normalizeFold(fold);
  ear.graph.populate(graph);
  if (!graph.faces_vertices || graph.faces_vertices.length < 2) {
    throw new Error("dense samples require non-degenerate populated faces");
  }
}

function checkTreeMakerStructure(fold: FOLDFormat): void {
  const tree = fold.tree_metadata;
  const metadata = fold.treemaker_metadata;
  if (!tree) throw new Error("tree_metadata is required");
  if (!metadata) throw new Error("treemaker_metadata is required");
  if (tree.generator !== "treemaker-tree") throw new Error("tree_metadata.generator must be treemaker-tree");
  if (tree.terminalCount < 3) throw new Error("TreeMaker samples require at least three terminal flaps");
  if (!metadata.optimizationSuccess) throw new Error("TreeMaker optimization did not succeed");
  if (metadata.sourceCreaseCount < 4) throw new Error("TreeMaker output has too few source creases");
  if (!fold.edges_treemakerKind || fold.edges_treemakerKind.length !== fold.edges_vertices.length) {
    throw new Error("edges_treemakerKind must be present and match edges_vertices length");
  }
  const assignmentTotals = countValues(fold.edges_assignment);
  if ((assignmentTotals.M ?? 0) + (assignmentTotals.V ?? 0) < 2) {
    throw new Error("TreeMaker samples require active M/V creases");
  }
  if ((assignmentTotals.U ?? 0) + (assignmentTotals.F ?? 0) < 1) {
    throw new Error("TreeMaker full-CP samples should preserve flat/unfolded hinge lines");
  }
}

function checkRabbitEarFoldProgramStructure(fold: FOLDFormat): void {
  const metadata = fold.rabbit_ear_metadata;
  if (!metadata) throw new Error("rabbit_ear_metadata is required");
  if (metadata.generator !== "rabbit-ear-fold-program") {
    throw new Error("rabbit_ear_metadata.generator must be rabbit-ear-fold-program");
  }
  if (metadata.rabbitEarApi !== "ear.graph.flatFold") {
    throw new Error("rabbit_ear_metadata.rabbitEarApi must be ear.graph.flatFold");
  }
  if (metadata.appliedFoldCount < 1) throw new Error("Rabbit Ear fold-program samples require at least one applied fold");
  if (metadata.attemptedFoldCount < metadata.appliedFoldCount) {
    throw new Error("attemptedFoldCount cannot be less than appliedFoldCount");
  }
  if (!metadata.targetActiveCreaseRange || metadata.targetActiveCreaseRange.length !== 2) {
    throw new Error("targetActiveCreaseRange is required");
  }
  const [minActive, maxActive] = metadata.targetActiveCreaseRange;
  const activeCreases = fold.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length;
  if (activeCreases !== metadata.activeCreaseCount) {
    throw new Error(`active crease count ${activeCreases} does not match metadata ${metadata.activeCreaseCount}`);
  }
  if (activeCreases < minActive || activeCreases > maxActive) {
    throw new Error(`active crease count ${activeCreases} is outside requested range ${minActive}-${maxActive}`);
  }
  if (Object.keys(metadata.axiomUsage).length < 1) throw new Error("axiomUsage is required");
  const axiomTotal = Object.values(metadata.axiomUsage).reduce((sum, count) => sum + Number(count), 0);
  if (axiomTotal !== metadata.appliedFoldCount) {
    throw new Error(`axiomUsage total ${axiomTotal} does not match appliedFoldCount ${metadata.appliedFoldCount}`);
  }
  const labelPolicy = fold.label_policy;
  if (!labelPolicy?.trainingEligible) throw new Error("label_policy.trainingEligible must be true");
  if (
    labelPolicy.labelSource !== "rabbit-ear-fold-program" ||
    labelPolicy.geometrySource !== "rabbit-ear-fold-program" ||
    labelPolicy.assignmentSource !== "rabbit-ear-fold-program"
  ) {
    throw new Error("label_policy must mark Rabbit Ear fold-program provenance");
  }
}

function checkTessellationFoldProgramStructure(fold: FOLDFormat): void {
  const metadata = fold.tessellation_metadata;
  if (!metadata) throw new Error("tessellation_metadata is required");
  if (metadata.generator !== "tessellation-fold-program") {
    throw new Error("tessellation_metadata.generator must be tessellation-fold-program");
  }
  if (!metadata.targetActiveCreaseRange || metadata.targetActiveCreaseRange.length !== 2) {
    throw new Error("targetActiveCreaseRange is required");
  }
  const [minActive, maxActive] = metadata.targetActiveCreaseRange;
  const activeCreases = fold.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length;
  if (activeCreases !== metadata.activeCreaseCount) {
    throw new Error(`active crease count ${activeCreases} does not match metadata ${metadata.activeCreaseCount}`);
  }
  if (activeCreases < minActive || activeCreases > maxActive) {
    throw new Error(`active crease count ${activeCreases} is outside requested range ${minActive}-${maxActive}`);
  }
  const fractionSum =
    metadata.horizontalCreaseLengthFraction +
    metadata.verticalCreaseLengthFraction +
    metadata.diagonalCreaseLengthFraction;
  if (Math.abs(fractionSum - 1) > 0.001) {
    throw new Error(`crease length fractions must sum to 1, got ${fractionSum}`);
  }
  if (metadata.minRenderedSpacingPx1024 <= 0) {
    throw new Error("minRenderedSpacingPx1024 must be positive");
  }
  if (metadata.subfamily === "orthogonal-bp-grid") {
    checkOrthogonalTessellationStructure(fold, metadata);
  } else if (metadata.subfamily === "miura-ori") {
    checkMiuraTessellationStructure(fold, metadata);
  } else {
    throw new Error(`unsupported tessellation subfamily: ${String(metadata.subfamily)}`);
  }
  const labelPolicy = fold.label_policy;
  if (!labelPolicy?.trainingEligible) throw new Error("label_policy.trainingEligible must be true");
  if (
    labelPolicy.labelSource !== "tessellation-fold-program" ||
    labelPolicy.geometrySource !== "tessellation-fold-program" ||
    labelPolicy.assignmentSource !== "tessellation-fold-program"
  ) {
    throw new Error("label_policy must mark tessellation fold-program provenance");
  }
}

function checkOrthogonalTessellationStructure(fold: FOLDFormat, metadata: TessellationMetadata): void {
  if (metadata.coordinateMode !== "regular-grid-intervals") {
    throw new Error("orthogonal tessellation coordinateMode must be regular-grid-intervals");
  }
  const gridSizeX = requiredNumber(metadata.gridSizeX, "gridSizeX");
  const gridSizeY = requiredNumber(metadata.gridSizeY, "gridSizeY");
  const horizontalPleatInterval = requiredNumber(metadata.horizontalPleatInterval, "horizontalPleatInterval");
  const verticalPleatInterval = requiredNumber(metadata.verticalPleatInterval, "verticalPleatInterval");
  if (gridSizeX < metadata.repeatX || gridSizeY < metadata.repeatY) {
    throw new Error("grid size must be at least the active repeat count");
  }
  if (gridSizeX % verticalPleatInterval !== 0 || gridSizeY % horizontalPleatInterval !== 0) {
    throw new Error("pleat intervals must divide the base grid size");
  }
  if (
    metadata.repeatX !== gridSizeX / verticalPleatInterval ||
    metadata.repeatY !== gridSizeY / horizontalPleatInterval
  ) {
    throw new Error("active repeat counts must match grid size and pleat intervals");
  }
  if (metadata.repeatX < 3 || metadata.repeatY < 3) {
    throw new Error("tessellation repeat counts must be at least 3");
  }
  if ((metadata.angleHistogram["0"] ?? 0) <= 0 || (metadata.angleHistogram["90"] ?? 0) <= 0) {
    throw new Error("orthogonal tessellation samples require horizontal and vertical angle coverage");
  }
  checkTessellationRegularGridCoordinates(fold, metadata);
  checkTessellationLineAlternation(fold, metadata);
}

function checkMiuraTessellationStructure(fold: FOLDFormat, metadata: TessellationMetadata): void {
  if (metadata.coordinateMode !== "miura-square-zigzag-grid") {
    throw new Error("Miura tessellation coordinateMode must be miura-square-zigzag-grid");
  }
  if (metadata.repeatX < 3 || metadata.repeatY < 3) {
    throw new Error("Miura repeat counts must be at least 3");
  }
  requiredNumber(metadata.miuraSkewFactor, "miuraSkewFactor");
  requiredNumber(metadata.miuraCellAspectRatio, "miuraCellAspectRatio");
  if ((metadata.angleHistogram["0"] ?? 0) <= 0) {
    throw new Error("Miura tessellation samples require horizontal angle coverage");
  }
  const diagonalAngles = Object.entries(metadata.angleHistogram)
    .filter(([angle]) => angle !== "0" && angle !== "90")
    .reduce((sum, [, count]) => sum + count, 0);
  if (diagonalAngles <= 0) {
    throw new Error("Miura tessellation samples require diagonal angle coverage");
  }
  if (metadata.verticalCreaseLengthFraction !== 0 || metadata.diagonalCreaseLengthFraction <= 0) {
    throw new Error("Miura tessellation samples must report diagonal, not vertical, crease length");
  }
  if (metadata.assignmentMode !== "miura-column-alternating") {
    throw new Error(`unsupported Miura assignment mode: ${String(metadata.assignmentMode)}`);
  }
  checkMiuraGridCoordinates(fold, metadata);
  checkMiuraLineAlternation(fold, metadata);
}

function checkTessellationRegularGridCoordinates(fold: FOLDFormat, metadata: TessellationMetadata): void {
  const cols = metadata.repeatX;
  const rows = metadata.repeatY;
  const gridSizeX = requiredNumber(metadata.gridSizeX, "gridSizeX");
  const gridSizeY = requiredNumber(metadata.gridSizeY, "gridSizeY");
  const horizontalPleatInterval = requiredNumber(metadata.horizontalPleatInterval, "horizontalPleatInterval");
  const verticalPleatInterval = requiredNumber(metadata.verticalPleatInterval, "verticalPleatInterval");
  const expectedVertices = (cols + 1) * (rows + 1);
  if (fold.vertices_coords.length !== expectedVertices) {
    throw new Error(`orthogonal grid vertex layout mismatch: expected ${expectedVertices} vertices`);
  }

  const xCoords = Array.from({ length: cols + 1 }, (_, x) => fold.vertices_coords[x]?.[0]);
  const yCoords = Array.from({ length: rows + 1 }, (_, y) => fold.vertices_coords[y * (cols + 1)]?.[1]);
  axisWidths("x", xCoords);
  axisWidths("y", yCoords);
  const expectedDx = verticalPleatInterval / gridSizeX;
  const expectedDy = horizontalPleatInterval / gridSizeY;
  for (let x = 0; x <= cols; x++) {
    const expected = x * expectedDx;
    if (Math.abs(Number(xCoords[x]) - expected) > 1e-7) {
      throw new Error(`x coordinate ${x} must equal ${expected} on the regular grid`);
    }
  }
  for (let y = 0; y <= rows; y++) {
    const expected = y * expectedDy;
    if (Math.abs(Number(yCoords[y]) - expected) > 1e-7) {
      throw new Error(`y coordinate ${y} must equal ${expected} on the regular grid`);
    }
  }
  const minRenderedSpacing = roundRatio(1024 * Math.min(expectedDx, expectedDy));
  if (Math.abs(metadata.minRenderedSpacingPx1024 - minRenderedSpacing) > 0.01) {
    throw new Error(`minRenderedSpacingPx1024 ${metadata.minRenderedSpacingPx1024} does not match coordinates ${minRenderedSpacing}`);
  }
}

function axisWidths(axis: "x" | "y", coords: Array<number | undefined>): number[] {
  if (coords.some((coord) => coord === undefined || !Number.isFinite(coord))) {
    throw new Error(`${axis}-axis coordinate list is incomplete`);
  }
  if (Math.abs(Number(coords[0])) > 1e-7 || Math.abs(Number(coords[coords.length - 1]) - 1) > 1e-7) {
    throw new Error(`${axis}-axis coordinates must span 0 to 1`);
  }
  const widths: number[] = [];
  for (let index = 0; index < coords.length - 1; index++) {
    const width = Number(coords[index + 1]) - Number(coords[index]);
    if (width <= 0) throw new Error(`${axis}-axis coordinates must be strictly increasing`);
    widths.push(width);
  }
  return widths;
}

function checkTessellationLineAlternation(fold: FOLDFormat, metadata: TessellationMetadata): void {
  const cols = metadata.repeatX;
  const rows = metadata.repeatY;
  const expectedEdges = (rows + 1) * cols + (cols + 1) * rows;
  if (fold.edges_vertices.length !== expectedEdges || fold.edges_assignment.length !== expectedEdges) {
    throw new Error(`orthogonal grid edge layout mismatch: expected ${expectedEdges} edges`);
  }

  const horizontalIndex = (y: number, x: number): number => y * cols + x;
  const verticalOffset = (rows + 1) * cols;
  const verticalIndex = (y: number, x: number): number => verticalOffset + y * (cols + 1) + x;

  for (let y = 1; y < rows; y++) {
    for (let x = 1; x < cols; x++) {
      const left = fold.edges_assignment[horizontalIndex(y, x - 1)];
      const right = fold.edges_assignment[horizontalIndex(y, x)];
      const below = fold.edges_assignment[verticalIndex(y - 1, x)];
      const above = fold.edges_assignment[verticalIndex(y, x)];
      const assignments = [left, right, below, above];
      if (!assignments.every(isMountainOrValley)) {
        throw new Error(`interior grid vertex (${x}, ${y}) must have only M/V incident creases`);
      }
      const mountainCount = assignments.filter((assignment) => assignment === "M").length;
      if (mountainCount !== 1 && mountainCount !== 3) {
        throw new Error(`interior grid vertex (${x}, ${y}) must have a 3-to-1 M/V split`);
      }

      if (metadata.assignmentMode === "vertical-line-alternating") {
        if (below !== above || above !== right || left !== oppositeAssignment(above)) {
          throw new Error(`vertical-line assignment mismatch at grid vertex (${x}, ${y})`);
        }
      } else if (metadata.assignmentMode === "horizontal-line-alternating") {
        if (left !== right || right !== above || below !== oppositeAssignment(above)) {
          throw new Error(`horizontal-line assignment mismatch at grid vertex (${x}, ${y})`);
        }
      } else {
        throw new Error(`unsupported tessellation assignment mode: ${String(metadata.assignmentMode)}`);
      }
    }
  }
}

function checkMiuraGridCoordinates(fold: FOLDFormat, metadata: TessellationMetadata): void {
  const cols = metadata.repeatX;
  const rows = metadata.repeatY;
  const skewFactor = requiredNumber(metadata.miuraSkewFactor, "miuraSkewFactor");
  const expectedVertices = (cols + 1) * (rows + 1);
  if (fold.vertices_coords.length !== expectedVertices) {
    throw new Error(`Miura grid vertex layout mismatch: expected ${expectedVertices} vertices`);
  }

  const vertexIndex = (x: number, y: number): number => y * (cols + 1) + x;

  for (let y = 0; y <= rows; y++) {
    const offset = y % 2 === 0 ? 0 : skewFactor;
    for (let x = 0; x <= cols; x++) {
      const [actualX, actualY] = fold.vertices_coords[vertexIndex(x, y)] ?? [NaN, NaN];
      const expectedX = x === 0 || x === cols ? x / cols : (x + offset) / cols;
      const expectedY = y / rows;
      if (Math.abs(actualX - expectedX) > 1e-7 || Math.abs(actualY - expectedY) > 1e-7) {
        throw new Error(`Miura vertex (${x}, ${y}) must match square alternating-offset coordinates`);
      }
    }
  }

  const edgeLengths = fold.edges_vertices.map(([a, b]) => distance(fold.vertices_coords[a], fold.vertices_coords[b]));
  const minRenderedSpacing = roundRatio(1024 * Math.min(...edgeLengths));
  if (Math.abs(metadata.minRenderedSpacingPx1024 - minRenderedSpacing) > 0.01) {
    throw new Error(`minRenderedSpacingPx1024 ${metadata.minRenderedSpacingPx1024} does not match Miura coordinates ${minRenderedSpacing}`);
  }
}

function checkMiuraLineAlternation(fold: FOLDFormat, metadata: TessellationMetadata): void {
  const cols = metadata.repeatX;
  const rows = metadata.repeatY;
  const expectedEdges = (rows + 1) * cols + rows * (cols + 1);
  if (fold.edges_vertices.length !== expectedEdges || fold.edges_assignment.length !== expectedEdges) {
    throw new Error(`Miura grid edge layout mismatch: expected ${expectedEdges} edges`);
  }

  const horizontalIndex = (y: number, x: number): number => y * cols + x;
  const diagonalOffset = (rows + 1) * cols;
  const diagonalIndex = (y: number, x: number): number => diagonalOffset + y * (cols - 1) + (x - 1);
  const borderOffset = diagonalOffset + rows * (cols - 1);
  const leftBorderIndex = (y: number): number => borderOffset + y * 2;
  const rightBorderIndex = (y: number): number => borderOffset + y * 2 + 1;

  for (let y = 0; y <= rows; y++) {
    for (let x = 0; x < cols; x++) {
      const assignment = fold.edges_assignment[horizontalIndex(y, x)];
      if (y === 0 || y === rows) {
        if (assignment !== "B") throw new Error(`Miura top/bottom border edge (${x}, ${y}) must be B`);
      } else if (!isMountainOrValley(assignment)) {
        throw new Error(`Miura interior horizontal edge (${x}, ${y}) must be M/V`);
      }
    }
  }
  for (let y = 0; y < rows; y++) {
    if (fold.edges_assignment[leftBorderIndex(y)] !== "B" || fold.edges_assignment[rightBorderIndex(y)] !== "B") {
      throw new Error(`Miura side border edges at row ${y} must be B`);
    }
  }

  for (let y = 1; y < rows; y++) {
    for (let x = 1; x < cols; x++) {
      const left = fold.edges_assignment[horizontalIndex(y, x - 1)];
      const right = fold.edges_assignment[horizontalIndex(y, x)];
      const above = fold.edges_assignment[diagonalIndex(y - 1, x)];
      const below = fold.edges_assignment[diagonalIndex(y, x)];
      const assignments = [left, right, above, below];
      if (!assignments.every(isMountainOrValley)) {
        throw new Error(`interior Miura vertex (${x}, ${y}) must have only M/V incident creases`);
      }
      const mountainCount = assignments.filter((assignment) => assignment === "M").length;
      if (mountainCount !== 1 && mountainCount !== 3) {
        throw new Error(`interior Miura vertex (${x}, ${y}) must have a 3-to-1 M/V split`);
      }
      if (above !== below || below !== right || left !== oppositeAssignment(right)) {
        throw new Error(`Miura column assignment mismatch at vertex (${x}, ${y})`);
      }
    }
  }
}

function isMountainOrValley(assignment: EdgeAssignment): boolean {
  return assignment === "M" || assignment === "V";
}

function oppositeAssignment(assignment: EdgeAssignment): EdgeAssignment {
  if (assignment === "M") return "V";
  if (assignment === "V") return "M";
  throw new Error(`assignment has no M/V opposite: ${assignment}`);
}

function checkLocalFlatFoldability(fold: FOLDFormat): void {
  const graph = normalizeFold(fold);
  ear.graph.populate(graph);
  const kawasakiBad = ear.singleVertex.validateKawasaki(graph) as number[];
  const maekawaBad = ear.singleVertex.validateMaekawa(graph) as number[];
  const badVertices = [...new Set([...kawasakiBad, ...maekawaBad])].sort((a, b) => a - b);
  if (badVertices.length > 0) {
    throw new Error(`non-flat-foldable vertices: ${badVertices.slice(0, 12).join(", ")}`);
  }
}

function checkRabbitEarSolver(fold: FOLDFormat): NonNullable<ValidationResult["metrics"]> {
  const graph = normalizeFold(fold);
  ear.graph.populate(graph);
  const start = Date.now();
  const solverResult = ear.layer.solver(graph);
  const solverMs = Date.now() - start;
  if (!solverResult) {
    throw new Error("Rabbit Ear layer solver found no globally consistent layer ordering");
  }
  const folded = ear.graph.makeVerticesCoordsFlatFolded(graph);
  if (!Array.isArray(folded) || folded.length !== graph.vertices_coords.length) {
    throw new Error("Rabbit Ear could not compute folded vertex coordinates");
  }
  if (folded.some((coord) => !Array.isArray(coord) || coord.some((value) => !Number.isFinite(value)))) {
    throw new Error("Rabbit Ear folded coordinates contain non-finite values");
  }
  return { solverMs, faces: graph.faces_vertices?.length ?? 0 };
}

async function checkFoldCli(fold: FOLDFormat, config: ValidationConfig): Promise<{ ok: boolean; errors: string[] }> {
  const command = config.foldCliCommand;
  if (!command) return { ok: false, errors: ["foldCliCommand is not configured"] };
  const proc = Bun.spawn({
    cmd: [command],
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
  });
  proc.stdin.write(JSON.stringify(fold));
  proc.stdin.end();
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  const output = `${stdout}\n${stderr}`;
  if (exitCode !== 0 || /not flat|false|invalid|error/i.test(output)) {
    return { ok: false, errors: [`fold-cli rejected graph: ${output.trim()}`] };
  }
  return { ok: true, errors: [] };
}

function sharesVertex(a: [number, number], b: [number, number]): boolean {
  return a[0] === b[0] || a[0] === b[1] || a[1] === b[0] || a[1] === b[1];
}

function segmentsIntersect(
  a: [number, number],
  b: [number, number],
  c: [number, number],
  d: [number, number],
): boolean {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (o1 === 0 && onSegment(a, c, b)) return true;
  if (o2 === 0 && onSegment(a, d, b)) return true;
  if (o3 === 0 && onSegment(c, a, d)) return true;
  if (o4 === 0 && onSegment(c, b, d)) return true;
  return o1 !== o2 && o3 !== o4;
}

function orientation(a: [number, number], b: [number, number], c: [number, number]): number {
  const value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]);
  if (Math.abs(value) < GEOMETRY_EPSILON) return 0;
  return value > 0 ? 1 : 2;
}

function onSegment(a: [number, number], b: [number, number], c: [number, number]): boolean {
  return (
    b[0] <= Math.max(a[0], c[0]) + GEOMETRY_EPSILON &&
    b[0] >= Math.min(a[0], c[0]) - GEOMETRY_EPSILON &&
    b[1] <= Math.max(a[1], c[1]) + GEOMETRY_EPSILON &&
    b[1] >= Math.min(a[1], c[1]) - GEOMETRY_EPSILON
  );
}

function distance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function countValues(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function requiredNumber(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`${label} is required`);
  }
  return value;
}

function roundRatio(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
