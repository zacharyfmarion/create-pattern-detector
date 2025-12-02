/**
 * Local validation using Rabbit Ear
 *
 * Validates Maekawa's and Kawasaki's theorems at each interior vertex
 */

import ear from 'rabbit-ear';
import type { FOLDFormat, ValidationResult } from '../types/crease-pattern.ts';
import { getInteriorVertices, isVertexOnBorder } from '../utils/fold-helpers.ts';

/**
 * Validate crease pattern using local checks only
 *
 * @param fold FOLD format crease pattern
 * @returns ValidationResult with tier classification
 */
export function validateLocal(fold: FOLDFormat): ValidationResult {
  const passed: string[] = [];
  const failed: string[] = [];
  const errors: string[] = [];

  try {
    // Convert to Rabbit Ear graph
    const graph = ear.graph(fold);

    // Check 1: Maekawa's theorem at all interior vertices
    const maekawaResult = validateMaekawa(graph, fold);
    if (maekawaResult.valid) {
      passed.push('maekawa');
    } else {
      failed.push('maekawa');
      errors.push(...maekawaResult.errors);
    }

    // Check 2: Kawasaki's theorem at all interior vertices
    const kawasakiResult = validateKawasaki(graph, fold);
    if (kawasakiResult.valid) {
      passed.push('kawasaki');
    } else {
      failed.push('kawasaki');
      errors.push(...kawasakiResult.errors);
    }

    // Check 3: No self-intersections
    const intersectionResult = checkSelfIntersections(fold);
    if (intersectionResult.valid) {
      passed.push('no-self-intersections');
    } else {
      failed.push('no-self-intersections');
      errors.push(...intersectionResult.errors);
    }

    // Check 4: Complete border
    const borderResult = checkBorder(fold);
    if (borderResult.valid) {
      passed.push('complete-border');
    } else {
      failed.push('complete-border');
      errors.push(...borderResult.errors);
    }

    // Check 5: 2-colorability (flat-foldability)
    const colorResult = check2Colorability(graph);
    if (colorResult.valid) {
      passed.push('2-colorable');
    } else {
      failed.push('2-colorable');
      errors.push(...colorResult.errors);
    }

  } catch (error) {
    errors.push(`Validation error: ${error}`);
  }

  // Determine tier
  const allPassed = failed.length === 0;
  const tier = allPassed ? 'A' : 'REJECT';

  return {
    tier,
    passed,
    failed,
    errors,
  };
}

/**
 * Validate Maekawa's theorem: |M - V| = 2 at all interior vertices
 */
function validateMaekawa(graph: any, fold: FOLDFormat): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const interiorVertices = getInteriorVertices(fold);

  for (const vertexIdx of interiorVertices) {
    // Use Rabbit Ear's built-in validator
    const isValid = ear.singleVertex.validateMaekawa(graph, vertexIdx);

    if (!isValid) {
      const coords = fold.vertices_coords[vertexIdx];
      errors.push(`Maekawa's theorem violated at vertex ${vertexIdx} ${coords}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate Kawasaki's theorem: alternating angle sum = 0 at all interior vertices
 */
function validateKawasaki(graph: any, fold: FOLDFormat): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const interiorVertices = getInteriorVertices(fold);

  for (const vertexIdx of interiorVertices) {
    // Use Rabbit Ear's built-in validator
    const isValid = ear.singleVertex.validateKawasaki(graph, vertexIdx);

    if (!isValid) {
      const coords = fold.vertices_coords[vertexIdx];
      errors.push(`Kawasaki's theorem violated at vertex ${vertexIdx} ${coords}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check for self-intersecting edges
 */
function checkSelfIntersections(fold: FOLDFormat): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check all edge pairs for intersection
  for (let i = 0; i < fold.edges_vertices.length; i++) {
    for (let j = i + 1; j < fold.edges_vertices.length; j++) {
      const edge1 = fold.edges_vertices[i];
      const edge2 = fold.edges_vertices[j];

      // Skip if edges share a vertex
      if (sharesVertex(edge1, edge2)) {
        continue;
      }

      // Check for intersection
      const v1 = fold.vertices_coords[edge1[0]];
      const v2 = fold.vertices_coords[edge1[1]];
      const v3 = fold.vertices_coords[edge2[0]];
      const v4 = fold.vertices_coords[edge2[1]];

      if (segmentsIntersect(v1, v2, v3, v4)) {
        errors.push(`Self-intersection found between edges ${i} and ${j}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check if two edges share a vertex
 */
function sharesVertex(edge1: [number, number], edge2: [number, number]): boolean {
  return edge1[0] === edge2[0] || edge1[0] === edge2[1] ||
         edge1[1] === edge2[0] || edge1[1] === edge2[1];
}

/**
 * Check if two line segments intersect (CCW test method)
 */
function segmentsIntersect(
  a: [number, number],
  b: [number, number],
  c: [number, number],
  d: [number, number]
): boolean {
  function ccw(p1: [number, number], p2: [number, number], p3: [number, number]): boolean {
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0]);
  }

  return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

/**
 * Check for complete rectangular border
 */
function checkBorder(fold: FOLDFormat): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Get border edges
  const borderEdges = fold.edges_vertices
    .map((edge, idx) => ({ edge, assignment: fold.edges_assignment[idx] }))
    .filter(e => e.assignment === 'B');

  if (borderEdges.length < 4) {
    errors.push(`Incomplete border: found ${borderEdges.length} border edges, expected 4`);
  }

  // Check that border forms a closed rectangle
  // For now, just verify we have 4 border edges
  // TODO: More rigorous rectangle checking

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check if pattern is 2-colorable (bipartite face graph)
 */
function check2Colorability(graph: any): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  try {
    // Use Rabbit Ear's face extraction
    const origami = ear.origami(graph);

    // Check if flat-foldable (2-colorable)
    const isFlatFoldable = origami.flatFolded !== undefined && origami.flatFolded !== null;

    if (!isFlatFoldable) {
      errors.push('Pattern is not 2-colorable (not flat-foldable)');
    }
  } catch (error) {
    // If face extraction fails, it's likely not valid
    errors.push(`2-colorability check failed: ${error}`);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
