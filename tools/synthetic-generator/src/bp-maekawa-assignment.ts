import ear from "rabbit-ear";
import { normalizeFold } from "./fold-utils.ts";
import type { EdgeAssignment, FOLDFormat } from "./types.ts";

export interface AssignmentSolveResult {
  ok: boolean;
  fold?: FOLDFormat;
  steps: number;
  errors: string[];
}

interface VertexConstraint {
  vertex: number;
  edges: number[];
  allowedMountainCounts: number[];
}

const MAX_SOLVER_STEPS = 5_000_000;

export function solveMaekawaAssignments(fold: FOLDFormat): AssignmentSolveResult {
  const working = normalizeFold(fold);
  const borderVertices = new Set<number>();
  const incidentFoldedEdges = Array.from({ length: working.vertices_coords.length }, () => [] as number[]);

  for (const [edgeIndex, [a, b]] of working.edges_vertices.entries()) {
    if (working.edges_assignment[edgeIndex] === "B") {
      borderVertices.add(a);
      borderVertices.add(b);
      continue;
    }
    incidentFoldedEdges[a].push(edgeIndex);
    incidentFoldedEdges[b].push(edgeIndex);
  }

  const kawasakiProbe = {
    ...working,
    edges_assignment: working.edges_assignment.map((assignment): EdgeAssignment => assignment === "B" ? "B" : "M"),
  };
  ear.graph.populate(kawasakiProbe);
  const kawasakiBad = ear.singleVertex.validateKawasaki(kawasakiProbe) as number[];
  if (kawasakiBad.length > 0) {
    return {
      ok: false,
      steps: 0,
      errors: [`kawasaki-failed-before-assignment:${kawasakiBad.slice(0, 16).join(",")}`],
    };
  }

  const constraints: VertexConstraint[] = [];
  for (const [vertex, edges] of incidentFoldedEdges.entries()) {
    if (borderVertices.has(vertex) || edges.length === 0) continue;
    if (edges.length < 2 || edges.length % 2 !== 0) {
      return {
        ok: false,
        steps: 0,
        errors: [`dangling-or-odd-active-degree:v${vertex}:degree${edges.length}`],
      };
    }
    constraints.push({
      vertex,
      edges,
      allowedMountainCounts: [(edges.length + 2) / 2, (edges.length - 2) / 2],
    });
  }

  const edgeToConstraints = Array.from({ length: working.edges_vertices.length }, () => [] as number[]);
  for (const [constraintIndex, constraint] of constraints.entries()) {
    for (const edge of constraint.edges) edgeToConstraints[edge].push(constraintIndex);
  }

  const variables = working.edges_vertices
    .map((_, edgeIndex) => edgeIndex)
    .filter((edgeIndex) => working.edges_assignment[edgeIndex] !== "B")
    .sort((a, b) => edgeToConstraints[b].length - edgeToConstraints[a].length || a - b);

  const assignments = Array.from({ length: working.edges_vertices.length }, () => -1);
  let steps = 0;

  const feasible = (constraint: VertexConstraint): boolean => {
    let assignedMountains = 0;
    let unassigned = 0;
    for (const edge of constraint.edges) {
      if (assignments[edge] === 1) assignedMountains += 1;
      else if (assignments[edge] < 0) unassigned += 1;
    }
    return constraint.allowedMountainCounts.some((target) =>
      assignedMountains <= target && target <= assignedMountains + unassigned
    );
  };

  const exact = (constraint: VertexConstraint): boolean => {
    let assignedMountains = 0;
    for (const edge of constraint.edges) assignedMountains += assignments[edge] === 1 ? 1 : 0;
    return constraint.allowedMountainCounts.includes(assignedMountains);
  };

  const search = (variableIndex: number): boolean => {
    steps += 1;
    if (steps > MAX_SOLVER_STEPS) return false;
    if (variableIndex === variables.length) return constraints.every(exact);

    const edge = variables[variableIndex];
    for (const assignment of preferredAssignmentOrder(working, edge)) {
      assignments[edge] = assignment;
      let ok = true;
      for (const constraintIndex of edgeToConstraints[edge]) {
        if (!feasible(constraints[constraintIndex])) {
          ok = false;
          break;
        }
      }
      if (ok && search(variableIndex + 1)) return true;
    }
    assignments[edge] = -1;
    return false;
  };

  if (!search(0)) {
    return {
      ok: false,
      steps,
      errors: [`maekawa-assignment-unsat-or-step-limit:${steps}`],
    };
  }

  working.edges_assignment = working.edges_assignment.map((assignment, edgeIndex): EdgeAssignment => {
    if (assignment === "B") return "B";
    return assignments[edgeIndex] === 1 ? "M" : "V";
  });
  working.edges_foldAngle = working.edges_assignment.map((assignment) => {
    if (assignment === "M") return -180;
    if (assignment === "V") return 180;
    return 0;
  });
  return { ok: true, fold: working, steps, errors: [] };
}

function preferredAssignmentOrder(fold: FOLDFormat, edgeIndex: number): number[] {
  const role = fold.edges_bpRole?.[edgeIndex];
  if (role === "ridge" || role === "stretch") return [1, 0];
  return [0, 1];
}
