/**
 * TypeScript types for crease patterns and FOLD format
 */

/**
 * FOLD format specification v1.1
 * See: https://github.com/edemaine/fold
 */
export interface FOLDFormat {
  file_spec: number;
  file_creator: string;
  file_author?: string;
  file_title?: string;
  file_description?: string;
  file_classes?: string[];

  // Vertices
  vertices_coords: [number, number][];

  // Edges
  edges_vertices: [number, number][];
  edges_assignment: EdgeAssignment[];
  edges_foldAngle?: number[];

  // Faces (optional)
  faces_vertices?: number[][];
  faces_edges?: number[][];

  // Metadata
  frame_attributes?: string[];
  frame_classes?: string[];
}

/**
 * Edge assignment types
 */
export type EdgeAssignment = 'B' | 'M' | 'V' | 'F' | 'U' | 'C';

/**
 * Validation tier classification
 */
export type ValidationTier = 'S' | 'A' | 'REJECT';

/**
 * Symmetry types for generation
 */
export type SymmetryType = 'none' | '2-fold' | '4-fold' | 'diagonal';

/**
 * Validation result
 */
export interface ValidationResult {
  tier: ValidationTier;
  passed: string[];
  failed: string[];
  errors: string[];
}

/**
 * Generation configuration
 */
export interface GenerationConfig {
  // Target complexity
  numCreases: number;

  // Size
  width: number;
  height: number;

  // Symmetry
  symmetry: SymmetryType;

  // Randomness
  seed?: number;

  // Generation method
  method: 'tree' | 'box-pleating' | 'classic-base';

  // For classic bases
  baseName?: 'bird' | 'frog' | 'waterbomb' | 'preliminary' | 'fish';
}

/**
 * Dataset entry
 */
export interface DatasetEntry {
  id: string;
  fold: FOLDFormat;
  validation: ValidationResult;
  config: GenerationConfig;
  timestamp: string;
}

/**
 * Border rectangle bounds
 */
export interface BorderRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Vertex with incident edges
 */
export interface VertexInfo {
  index: number;
  coords: [number, number];
  incidentEdges: number[];
  isInterior: boolean;
}
