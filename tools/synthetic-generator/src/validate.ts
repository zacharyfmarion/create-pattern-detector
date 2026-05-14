import ear from "rabbit-ear";
import { normalizeFold } from "./fold-utils.ts";
import type { FOLDFormat, GlobalValidationBackend, ValidationConfig, ValidationResult } from "./types.ts";

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
  if (config.requireRealistic) {
    runCheck("realistic-structure", passed, failed, errors, () => checkRealisticStructure(normalized, config.minRealismScore ?? 0));
  }
  if (config.requireBoxPleat || normalized.edges_bpRole || normalized.bp_metadata) {
    runCheck("box-pleat-structure", passed, failed, errors, () => checkBoxPleatStructure(normalized, config.boxPleatMode ?? "simple"));
  }
  runCheck("local-flat-foldability", passed, failed, errors, () => checkLocalFlatFoldability(normalized));

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
  const graph = normalizeFold(fold);
  ear.graph.populate(graph);
  if (!graph.faces_vertices || graph.faces_vertices.length < 2) {
    throw new Error("dense samples require non-degenerate populated faces");
  }
}

function checkRealisticStructure(fold: FOLDFormat, minScore: number): void {
  const designTree = fold.design_tree;
  const layout = fold.layout_metadata;
  const molecules = fold.molecule_metadata;
  const realism = fold.realism_metadata;
  if (!designTree) throw new Error("design_tree is required");
  if (!layout) throw new Error("layout_metadata is required");
  if (!molecules) throw new Error("molecule_metadata is required");
  if (!realism) throw new Error("realism_metadata is required");
  if (designTree.nodes.length < 4 || designTree.edges.length < 3) {
    throw new Error("realistic BP samples require a non-trivial design tree");
  }
  if (layout.bodyRegions.length < 1 || layout.flapTerminals.length < 3) {
    throw new Error("realistic BP samples require body regions and flap terminals");
  }
  if (Object.values(molecules.molecules).filter((count) => count > 0).length < 4) {
    throw new Error("realistic BP samples require multiple molecule types");
  }
  if (realism.score < minScore) {
    throw new Error(`realism score ${realism.score.toFixed(3)} < ${minScore}`);
  }
  if (!realism.gates.hasMacroRegions || !realism.gates.hasDensityVariation || !realism.gates.notUniformLattice) {
    throw new Error("realism gates indicate a uniform or under-structured pattern");
  }
}

function checkBoxPleatStructure(fold: FOLDFormat, mode: "simple" | "dense"): void {
  const roles = fold.edges_bpRole;
  const metadata = fold.bp_metadata;
  if (!roles || roles.length !== fold.edges_vertices.length) {
    throw new Error("edges_bpRole must be present and match edges_vertices length");
  }
  if (!metadata?.gridSize || metadata.gridSize < 2) {
    throw new Error("bp_metadata.gridSize is required");
  }

  const roleTotals = countValues(roles);
  if ((roleTotals.ridge ?? 0) < 1) throw new Error("requires at least one ridge crease");
  if ((roleTotals.hinge ?? 0) < 1) throw new Error("requires at least one hinge contour crease");
  if ((roleTotals.axis ?? 0) + (roleTotals.stretch ?? 0) < 1) {
    throw new Error("requires at least one axis or stretch crease");
  }

  const fullAxisLines = new Map<string, { min: number; max: number }>();
  const fullDiagonalLines = new Set<string>();
  let diagonalRidges = 0;
  let interiorEdges = 0;

  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    const assignment = fold.edges_assignment[edgeIndex];
    const role = roles[edgeIndex];
    const p1 = fold.vertices_coords[a];
    const p2 = fold.vertices_coords[b];
    if (role !== "border") interiorEdges += 1;

    for (const coordinate of [p1[0], p1[1], p2[0], p2[1]]) {
      if (!onHalfGrid(coordinate, metadata.gridSize)) {
        throw new Error(`edge ${edgeIndex} has non-grid coordinate ${coordinate}`);
      }
    }

    if (role === "border") {
      if (assignment !== "B") throw new Error(`edge ${edgeIndex} has border role without B assignment`);
      continue;
    }
    if (role === "ridge") {
      if (mode === "simple" && assignment !== "M") throw new Error(`ridge edge ${edgeIndex} must be mountain assigned`);
      if (mode === "dense" && assignment !== "M" && assignment !== "V") {
        throw new Error(`ridge edge ${edgeIndex} must be mountain or valley assigned`);
      }
      if (!isDiagonal45(p1, p2)) throw new Error(`ridge edge ${edgeIndex} is not a 45-degree crease`);
      diagonalRidges += 1;
      fullDiagonalLines.add(diagonalSignature(p1, p2));
      continue;
    }
    if (mode === "simple" && assignment !== "V") throw new Error(`${role} edge ${edgeIndex} must be valley assigned`);
    if (mode === "dense" && assignment !== "M" && assignment !== "V") {
      throw new Error(`${role} edge ${edgeIndex} must be mountain or valley assigned`);
    }
    if (!isAxisAligned(p1, p2)) throw new Error(`${role} edge ${edgeIndex} is not axis-aligned`);
    updateAxisLineCoverage(fullAxisLines, p1, p2);
  }

  if (interiorEdges === 0) throw new Error("requires interior BP creases");
  if (diagonalRidges / interiorEdges < 0.08) {
    throw new Error("diagonal ridge ratio is too low for box pleating");
  }
  if (mode === "simple") {
    const fullAxisLineCount = [...fullAxisLines.values()].filter(({ min, max }) => min < 1e-8 && max > 1 - 1e-8).length;
    if (fullAxisLineCount > Math.max(6, fullDiagonalLines.size * 3 + 2)) {
      throw new Error("too many full-sheet axis-aligned lines relative to ridge structure");
    }
  }
  checkConnected(fold);
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

function onHalfGrid(value: number, gridSize: number): boolean {
  return Math.abs(value * gridSize * 2 - Math.round(value * gridSize * 2)) < 1e-6;
}

function isAxisAligned(a: [number, number], b: [number, number]): boolean {
  return Math.abs(a[0] - b[0]) < 1e-8 || Math.abs(a[1] - b[1]) < 1e-8;
}

function isDiagonal45(a: [number, number], b: [number, number]): boolean {
  return Math.abs(Math.abs(a[0] - b[0]) - Math.abs(a[1] - b[1])) < 1e-8;
}

function diagonalSignature(a: [number, number], b: [number, number]): string {
  const slope = Math.sign((b[1] - a[1]) / (b[0] - a[0]));
  const value = slope >= 0 ? a[1] - a[0] : a[1] + a[0];
  return `${slope >= 0 ? "p" : "n"}:${value.toFixed(6)}`;
}

function updateAxisLineCoverage(lines: Map<string, { min: number; max: number }>, a: [number, number], b: [number, number]): void {
  const vertical = Math.abs(a[0] - b[0]) < 1e-8;
  const key = vertical ? `v:${a[0].toFixed(6)}` : `h:${a[1].toFixed(6)}`;
  const min = vertical ? Math.min(a[1], b[1]) : Math.min(a[0], b[0]);
  const max = vertical ? Math.max(a[1], b[1]) : Math.max(a[0], b[0]);
  const current = lines.get(key);
  lines.set(key, current ? { min: Math.min(current.min, min), max: Math.max(current.max, max) } : { min, max });
}

function checkConnected(fold: FOLDFormat): void {
  const adjacency = Array.from({ length: fold.vertices_coords.length }, () => [] as number[]);
  for (const [a, b] of fold.edges_vertices) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const start = adjacency.findIndex((neighbors) => neighbors.length > 0);
  if (start < 0) throw new Error("graph has no connected edges");
  const seen = new Set<number>([start]);
  const stack = [start];
  while (stack.length) {
    const vertex = stack.pop()!;
    for (const next of adjacency[vertex]) {
      if (seen.has(next)) continue;
      seen.add(next);
      stack.push(next);
    }
  }
  const disconnected = adjacency.findIndex((neighbors, index) => neighbors.length > 0 && !seen.has(index));
  if (disconnected >= 0) throw new Error(`graph is disconnected at vertex ${disconnected}`);
}
