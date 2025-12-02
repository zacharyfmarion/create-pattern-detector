/**
 * Utility functions for working with FOLD format
 */

import type { FOLDFormat, BorderRect, VertexInfo } from '../types/crease-pattern.ts';

/**
 * Create an empty FOLD object with border
 */
export function createEmptyFOLD(
  width: number,
  height: number,
  creator: string = 'cp-synthetic-generator'
): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: creator,
    file_classes: ['singleModel'],
    vertices_coords: [
      [0, 0],
      [width, 0],
      [width, height],
      [0, height],
    ],
    edges_vertices: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
    ],
    edges_assignment: ['B', 'B', 'B', 'B'],
  };
}

/**
 * Get border rectangle from FOLD
 */
export function getBorderRect(fold: FOLDFormat): BorderRect {
  const xs = fold.vertices_coords.map(v => v[0]);
  const ys = fold.vertices_coords.map(v => v[1]);

  return {
    left: Math.min(...xs),
    right: Math.max(...xs),
    top: Math.min(...ys),
    bottom: Math.max(...ys),
  };
}

/**
 * Check if vertex is on border
 */
export function isVertexOnBorder(
  vertexIdx: number,
  fold: FOLDFormat,
  tolerance: number = 1.0
): boolean {
  const [x, y] = fold.vertices_coords[vertexIdx];
  const border = getBorderRect(fold);

  const onLeft = Math.abs(x - border.left) < tolerance;
  const onRight = Math.abs(x - border.right) < tolerance;
  const onTop = Math.abs(y - border.top) < tolerance;
  const onBottom = Math.abs(y - border.bottom) < tolerance;

  return onLeft || onRight || onTop || onBottom;
}

/**
 * Get all edges incident to a vertex
 */
export function getIncidentEdges(vertexIdx: number, fold: FOLDFormat): number[] {
  const incidents: number[] = [];

  fold.edges_vertices.forEach((edge, edgeIdx) => {
    if (edge[0] === vertexIdx || edge[1] === vertexIdx) {
      incidents.push(edgeIdx);
    }
  });

  return incidents;
}

/**
 * Get vertex info with incident edges
 */
export function getVertexInfo(vertexIdx: number, fold: FOLDFormat): VertexInfo {
  return {
    index: vertexIdx,
    coords: fold.vertices_coords[vertexIdx],
    incidentEdges: getIncidentEdges(vertexIdx, fold),
    isInterior: !isVertexOnBorder(vertexIdx, fold),
  };
}

/**
 * Get all interior vertices
 */
export function getInteriorVertices(fold: FOLDFormat): number[] {
  const interior: number[] = [];

  for (let i = 0; i < fold.vertices_coords.length; i++) {
    if (!isVertexOnBorder(i, fold)) {
      interior.push(i);
    }
  }

  return interior;
}

/**
 * Add a crease to the FOLD
 */
export function addCrease(
  fold: FOLDFormat,
  v1: [number, number],
  v2: [number, number],
  assignment: 'M' | 'V'
): FOLDFormat {
  // Find or create vertices
  const v1Idx = findOrAddVertex(fold, v1);
  const v2Idx = findOrAddVertex(fold, v2);

  // Add edge
  fold.edges_vertices.push([v1Idx, v2Idx]);
  fold.edges_assignment.push(assignment);

  return fold;
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
    const [x, y] = fold.vertices_coords[i];
    const dist = Math.sqrt((x - vertex[0]) ** 2 + (y - vertex[1]) ** 2);
    if (dist < tolerance) {
      return i;
    }
  }

  // Add new vertex
  fold.vertices_coords.push(vertex);
  return fold.vertices_coords.length - 1;
}

/**
 * Calculate distance between two points
 */
export function distance(p1: [number, number], p2: [number, number]): number {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}

/**
 * Calculate angle from point p1 to p2 (in radians)
 */
export function angleFromPoint(p1: [number, number], p2: [number, number]): number {
  return Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
}

/**
 * Save FOLD to JSON file
 */
export async function saveFOLD(fold: FOLDFormat, filepath: string): Promise<void> {
  await Bun.write(filepath, JSON.stringify(fold, null, 2));
}

/**
 * Load FOLD from JSON file
 */
export async function loadFOLD(filepath: string): Promise<FOLDFormat> {
  const file = Bun.file(filepath);
  return await file.json();
}
