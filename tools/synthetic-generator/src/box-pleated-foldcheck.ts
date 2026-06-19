// Global flat-foldability check via rabbit-ear's layer solver.
//
// Our pipeline guarantees the local conditions - Kawasaki (the grid geometry is
// 45/90 by construction) and Maekawa (assignBoxPleated drives conflicts to 0).
// Those are necessary but not sufficient: a crease pattern can satisfy every
// vertex locally yet still have no valid global stacking of layers. This module
// runs rabbit-ear's layer solver on the flat-folded geometry to confirm a valid
// layer ordering exists (the global condition, equivalent to a CAMV pass).

import ear from "rabbit-ear";
import type { AssignedEdge } from "./box-pleated-assignment.ts";

export interface FoldCheck {
  foldable: boolean;
  /** Number of valid global layer orderings found (0 when not foldable). */
  layerSolutions: number;
  /** Set when the solver rejected the folded state (e.g. impossible layer order). */
  reason?: string;
}

interface FoldGraph {
  vertices_coords: number[][];
  edges_vertices: number[][];
  edges_assignment: string[];
}

/** Build a FOLD graph (crease-pattern coords) from assigned unit edges. */
export function toFoldGraph(edges: AssignedEdge[]): FoldGraph {
  const index = new Map<string, number>();
  const vertices_coords: number[][] = [];
  const idx = (p: { x: number; y: number }): number => {
    const k = `${p.x},${p.y}`;
    const existing = index.get(k);
    if (existing !== undefined) return existing;
    index.set(k, vertices_coords.length);
    vertices_coords.push([p.x, p.y]);
    return vertices_coords.length - 1;
  };
  const edges_vertices: number[][] = [];
  const edges_assignment: string[] = [];
  for (const e of edges) {
    edges_vertices.push([idx(e.a), idx(e.b)]);
    edges_assignment.push(e.mv ?? "U");
  }
  return { vertices_coords, edges_vertices, edges_assignment };
}

/**
 * Verify a global flat-folded layer ordering exists. Folds the pattern to flat
 * coordinates, then runs the layer solver on the overlapping faces. A degenerate
 * fold (non-finite coordinates) or a solver rejection means it does not fold
 * flat globally.
 */
export function verifyFlatFoldable(edges: AssignedEdge[]): FoldCheck {
  const graph = ear.graph.populate(structuredClone(toFoldGraph(edges)));
  if (!graph.faces_vertices || graph.faces_vertices.length === 0) {
    return { foldable: false, layerSolutions: 0, reason: "no faces" };
  }
  let folded: number[][];
  try {
    folded = ear.graph.makeVerticesCoordsFlatFolded(graph);
  } catch (error) {
    return { foldable: false, layerSolutions: 0, reason: `fold failed: ${errorMessage(error)}` };
  }
  if (folded.some((p) => !Number.isFinite(p[0]) || !Number.isFinite(p[1]))) {
    return { foldable: false, layerSolutions: 0, reason: "degenerate fold" };
  }
  const foldedGraph = { ...graph, vertices_coords: folded };
  try {
    const solver = ear.layer.solver(foldedGraph);
    const count = solver.allSolutions().length;
    return count > 0
      ? { foldable: true, layerSolutions: count }
      : { foldable: false, layerSolutions: 0, reason: "no valid layer ordering" };
  } catch (error) {
    return { foldable: false, layerSolutions: 0, reason: `layer solver: ${errorMessage(error)}` };
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
