import ear from "rabbit-ear";
import { normalizeFold } from "./fold-utils.ts";
import { normalizeBPStudioFold } from "./bp-studio-validation.ts";
import type { AdapterMetadata } from "./bp-studio-realistic.ts";
import type { BPStudioAdapterSpec } from "./bp-studio-spec.ts";
import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

interface SourceLine {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
  order: number;
}

export interface BPStudioFoldProgramCompletionOptions {
  id: string;
  spec: BPStudioAdapterSpec;
  adapterMetadata: AdapterMetadata;
  gridSize: number;
}

export interface BPStudioFoldProgramCompletionResult {
  ok: boolean;
  fold?: FOLDFormat;
  foldCount: number;
  skippedCount: number;
  errors: string[];
}

export function completeBPStudioScaffoldByFoldProgram(
  scaffoldFold: FOLDFormat,
  options: BPStudioFoldProgramCompletionOptions,
): BPStudioFoldProgramCompletionResult {
  const normalizedScaffold = normalizeBPStudioFold(scaffoldFold);
  const graph = ear.graph.square();
  let foldCount = 0;
  let skippedCount = 0;
  const errors: string[] = [];

  for (const line of sourceFoldProgramLines(normalizedScaffold)) {
    try {
      ear.graph.flatFold(graph, vector(line.p1, line.p2), line.p1, foldProgramAssignment(line.assignment));
      foldCount += 1;
    } catch (error) {
      skippedCount += 1;
      if (errors.length < 8) errors.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (foldCount === 0) {
    return { ok: false, foldCount, skippedCount, errors: ["bp-studio fold program emitted no folds"] };
  }

  const fold = normalizeFold(graph);
  fold.file_creator = "cp-synthetic-generator/bp-studio-source-line-program";
  fold.file_title = `${options.id} BP Studio source-line completion`;
  fold.file_description = "Strict CP completed by folding along BP Studio scaffold line families in a deterministic source-role order.";
  fold.edges_bpRole = fold.edges_assignment.map((assignment, edgeIndex) =>
    inferredRoleForEdge(fold, edgeIndex, assignment)
  );
  const roles = fold.edges_bpRole;
  fold.edges_compilerSource = fold.edges_assignment.map((assignment, edgeIndex) => ({
    kind: assignment === "B" ? "sheet-border" : "bp-studio-source-line-program",
    mandatory: true,
    role: roles[edgeIndex],
  }));
  fold.bp_metadata = {
    gridSize: options.gridSize,
    bpSubfamily: "bp-studio-source-line-program",
    flapCount: options.spec.layout.flaps.length,
    gadgetCount: options.adapterMetadata.stretches?.filter((stretch) => stretch.active !== false).length ?? 0,
    ridgeCount: roles.filter((role) => role === "ridge").length,
    hingeCount: roles.filter((role) => role === "hinge").length,
    axisCount: roles.filter((role) => role === "axis").length,
  };
  fold.density_metadata = {
    densityBucket: "source-line",
    gridSize: options.gridSize,
    targetEdgeRange: [80, 3000],
    subfamily: "bp-studio-source-line-program",
    symmetry: options.spec.layout.symmetry,
    generatorSteps: [
      "normalize-bp-studio-scaffold",
      "sort-source-lines-by-role-and-length",
      `rabbit-ear-flat-fold-program:${foldCount}`,
      "infer-output-edge-roles",
    ],
    moleculeCounts: {
      "bp-studio-source-line": foldCount,
      "bp-studio-source-line-skipped": skippedCount,
    },
  };
  fold.completion_metadata = {
    engine: "bp-studio-source-line-program",
    version: "v0.1.0",
    source: "bp-studio-optimized-layout",
    scaffoldSummary: {
      adapterLineCount: options.adapterMetadata.cp?.lineCount ?? scaffoldFold.edges_vertices.length,
      adapterVertexCount: options.adapterMetadata.cp?.vertexCount ?? scaffoldFold.vertices_coords.length,
      adapterEdgeCount: options.adapterMetadata.cp?.edgeCount ?? scaffoldFold.edges_vertices.length,
      optimizedFlapCount: options.adapterMetadata.optimizedLayout?.flaps?.length ?? options.spec.layout.flaps.length,
      optimizedTreeEdgeCount: options.adapterMetadata.optimizedLayout?.edges?.length ?? options.spec.tree.edges.length,
    },
    selectedCenter: [0.5, 0.5],
    selectedFlapIds: options.spec.layout.flaps.map((flap) => Number(flap.nodeId)).filter(Number.isFinite),
    portJoinCount: 0,
    rejectedCandidateCount: skippedCount,
    compilerSteps: [
      "normalize-bp-studio-scaffold",
      "sort-source-lines-by-role-and-length",
      `rabbit-ear-flat-fold-program:${foldCount}`,
      "infer-output-edge-roles",
    ],
  };
  fold.label_policy = {
    labelSource: "compiler",
    geometrySource: "compiler",
    assignmentSource: "compiler",
    trainingEligible: true,
    notes: [
      "BP Studio raw export is used only as source fold-line geometry; final M/V/B labels are emitted by Rabbit Ear flat-fold operations.",
      "This is a source-line completion checkpoint. It should be replaced by bounded molecule completion once hub/terminal molecules are certified.",
    ],
  };
  return { ok: true, fold, foldCount, skippedCount, errors };
}

function sourceFoldProgramLines(fold: FOLDFormat): SourceLine[] {
  return fold.edges_vertices
    .map(([a, b], edgeIndex): SourceLine => ({
      p1: fold.vertices_coords[a],
      p2: fold.vertices_coords[b],
      assignment: fold.edges_assignment[edgeIndex],
      role: fold.edges_bpRole?.[edgeIndex] ?? inferredRoleForEdge(fold, edgeIndex, fold.edges_assignment[edgeIndex]),
      order: edgeIndex,
    }))
    .filter((line) => line.assignment !== "B" && distance(line.p1, line.p2) > 1e-9)
    .sort((left, right) =>
      sourceRoleRank(left.role) - sourceRoleRank(right.role) ||
      assignmentRank(left.assignment) - assignmentRank(right.assignment) ||
      distance(right.p1, right.p2) - distance(left.p1, left.p2) ||
      left.order - right.order
    );
}

function sourceRoleRank(role: BPRole): number {
  if (role === "hinge") return 0;
  if (role === "axis") return 1;
  if (role === "stretch") return 2;
  if (role === "ridge") return 3;
  return 4;
}

function assignmentRank(assignment: EdgeAssignment): number {
  if (assignment === "V") return 0;
  if (assignment === "M") return 1;
  return 2;
}

function foldProgramAssignment(assignment: EdgeAssignment): Extract<EdgeAssignment, "M" | "V"> {
  return assignment === "M" ? "M" : "V";
}

function inferredRoleForEdge(fold: FOLDFormat, edgeIndex: number, assignment: EdgeAssignment): BPRole {
  if (assignment === "B") return "border";
  const [a, b] = fold.edges_vertices[edgeIndex];
  return roleForSegment(fold.vertices_coords[a], fold.vertices_coords[b]);
}

function roleForSegment(a: Point, b: Point): BPRole {
  const dx = Math.abs(a[0] - b[0]);
  const dy = Math.abs(a[1] - b[1]);
  if (dx > 1e-9 && dy > 1e-9) return "ridge";
  if (dx < 1e-9) return "axis";
  return "hinge";
}

function vector(a: Point, b: Point): Point {
  return [b[0] - a[0], b[1] - a[1]];
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
