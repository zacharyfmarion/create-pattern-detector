/**
 * Symmetry Utilities
 *
 * Apply symmetry transformations to crease patterns:
 * - 2-fold (mirror symmetry)
 * - 4-fold (rotational symmetry)
 */

import type { FOLDFormat, SymmetryType } from '../types/crease-pattern.ts';
import { distance } from '../utils/fold-helpers.ts';

/**
 * Apply symmetry to a crease pattern
 *
 * @param fold Original FOLD pattern
 * @param symmetry Symmetry type to apply
 * @returns Symmetric FOLD pattern
 */
export function applySymmetry(fold: FOLDFormat, symmetry: SymmetryType): FOLDFormat {
  switch (symmetry) {
    case 'none':
      return fold;
    case '2-fold':
      return apply2FoldSymmetry(fold);
    case '4-fold':
      return apply4FoldSymmetry(fold);
    default:
      return fold;
  }
}

/**
 * Apply 2-fold mirror symmetry
 *
 * Reflects pattern across vertical center line.
 */
function apply2FoldSymmetry(fold: FOLDFormat): FOLDFormat {
  const width = Math.max(...fold.vertices_coords.map(v => v[0]));
  const height = Math.max(...fold.vertices_coords.map(v => v[1]));
  const centerX = width / 2;

  // Create a copy
  const newFold: FOLDFormat = {
    ...fold,
    vertices_coords: [...fold.vertices_coords],
    edges_vertices: [...fold.edges_vertices],
    edges_assignment: [...fold.edges_assignment],
  };

  // Get non-border edges
  const nonBorderEdges = fold.edges_vertices
    .map((edge, idx) => ({ edge, assignment: fold.edges_assignment[idx], idx }))
    .filter(e => e.assignment !== 'B');

  // For each non-border edge, add its mirror
  for (const { edge, assignment } of nonBorderEdges) {
    const [v1Idx, v2Idx] = edge;
    const v1 = fold.vertices_coords[v1Idx];
    const v2 = fold.vertices_coords[v2Idx];

    // Skip if edge crosses center line (already symmetric)
    if ((v1[0] <= centerX && v2[0] >= centerX) || (v1[0] >= centerX && v2[0] <= centerX)) {
      continue;
    }

    // Mirror vertices across center line
    const v1Mirror: [number, number] = [2 * centerX - v1[0], v1[1]];
    const v2Mirror: [number, number] = [2 * centerX - v2[0], v2[1]];

    // Find or add mirrored vertices
    const v1MirrorIdx = findOrAddVertex(newFold, v1Mirror);
    const v2MirrorIdx = findOrAddVertex(newFold, v2Mirror);

    // Add mirrored edge
    newFold.edges_vertices.push([v1MirrorIdx, v2MirrorIdx]);
    newFold.edges_assignment.push(assignment);
  }

  return newFold;
}

/**
 * Apply 4-fold rotational symmetry
 *
 * Rotates pattern 90°, 180°, and 270° around center.
 */
function apply4FoldSymmetry(fold: FOLDFormat): FOLDFormat {
  const width = Math.max(...fold.vertices_coords.map(v => v[0]));
  const height = Math.max(...fold.vertices_coords.map(v => v[1]));
  const centerX = width / 2;
  const centerY = height / 2;

  // Create a copy
  const newFold: FOLDFormat = {
    ...fold,
    vertices_coords: [...fold.vertices_coords],
    edges_vertices: [...fold.edges_vertices],
    edges_assignment: [...fold.edges_assignment],
  };

  // Get non-border edges
  const nonBorderEdges = fold.edges_vertices
    .map((edge, idx) => ({ edge, assignment: fold.edges_assignment[idx], idx }))
    .filter(e => e.assignment !== 'B');

  // For each non-border edge, add 90°, 180°, 270° rotations
  for (const { edge, assignment } of nonBorderEdges) {
    const [v1Idx, v2Idx] = edge;
    const v1 = fold.vertices_coords[v1Idx];
    const v2 = fold.vertices_coords[v2Idx];

    // Rotate 90°, 180°, 270°
    for (let rotation = 1; rotation <= 3; rotation++) {
      const v1Rotated = rotatePoint(v1, centerX, centerY, rotation * 90);
      const v2Rotated = rotatePoint(v2, centerX, centerY, rotation * 90);

      // Find or add rotated vertices
      const v1RotatedIdx = findOrAddVertex(newFold, v1Rotated);
      const v2RotatedIdx = findOrAddVertex(newFold, v2Rotated);

      // Add rotated edge
      newFold.edges_vertices.push([v1RotatedIdx, v2RotatedIdx]);
      newFold.edges_assignment.push(assignment);
    }
  }

  return newFold;
}

/**
 * Rotate a point around center by angle (in degrees)
 */
function rotatePoint(
  point: [number, number],
  centerX: number,
  centerY: number,
  angleDeg: number
): [number, number] {
  const angleRad = (angleDeg * Math.PI) / 180;

  // Translate to origin
  const x = point[0] - centerX;
  const y = point[1] - centerY;

  // Rotate
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  const xRotated = x * cos - y * sin;
  const yRotated = x * sin + y * cos;

  // Translate back
  return [xRotated + centerX, yRotated + centerY];
}

/**
 * Find vertex index or add if not exists
 */
function findOrAddVertex(
  fold: FOLDFormat,
  vertex: [number, number],
  tolerance: number = 1e-6
): number {
  // Find existing vertex
  for (let i = 0; i < fold.vertices_coords.length; i++) {
    if (distance(fold.vertices_coords[i], vertex) < tolerance) {
      return i;
    }
  }

  // Add new vertex
  fold.vertices_coords.push(vertex);
  return fold.vertices_coords.length - 1;
}

/**
 * Generate a symmetric pattern from a seed pattern
 *
 * @param generateSeed Function that generates the seed pattern (1/2 or 1/4 of final)
 * @param symmetry Symmetry type
 * @param width Full pattern width
 * @param height Full pattern height
 * @returns Symmetric FOLD pattern
 */
export function generateSymmetricPattern(
  generateSeed: (width: number, height: number) => FOLDFormat,
  symmetry: SymmetryType,
  width: number,
  height: number
): FOLDFormat {
  if (symmetry === 'none') {
    return generateSeed(width, height);
  }

  // Generate seed pattern (half or quarter size)
  const seedWidth = symmetry === '2-fold' ? width / 2 : width;
  const seedHeight = symmetry === '4-fold' ? height / 2 : height;

  const seed = generateSeed(seedWidth, seedHeight);

  // Apply symmetry
  return applySymmetry(seed, symmetry);
}
