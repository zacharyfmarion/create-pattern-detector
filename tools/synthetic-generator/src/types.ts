export type EdgeAssignment = "B" | "M" | "V" | "F" | "U" | "C";
export type BPRole = "border" | "hinge" | "ridge" | "axis" | "stretch";
export type BPSubfamily = "two-flap-stretch" | "uniaxial-chain" | "symmetric-insect-lite";

export interface BPMetadata {
  gridSize: number;
  bpSubfamily: BPSubfamily;
  flapCount: number;
  gadgetCount: number;
  ridgeCount: number;
  hingeCount: number;
  axisCount: number;
}

export interface FOLDFormat {
  file_spec: number;
  file_creator: string;
  file_classes?: string[];
  frame_classes?: string[];
  vertices_coords: [number, number][];
  edges_vertices: [number, number][];
  edges_assignment: EdgeAssignment[];
  edges_foldAngle?: number[];
  edges_bpRole?: BPRole[];
  bp_metadata?: BPMetadata;
  faces_vertices?: number[][];
  faces_edges?: number[][];
  [key: string]: unknown;
}

export const GENERATOR_FAMILIES = ["axiom", "classic", "single-vertex", "box-pleat", "grid-baseline"] as const;
export type GeneratorFamily = (typeof GENERATOR_FAMILIES)[number];
export type GlobalValidationBackend = "rabbit-ear-solver" | "fold-cli";

export interface ComplexityBucket {
  name: string;
  minCreases: number;
  maxCreases: number;
  weight: number;
}

export interface ValidationConfig {
  strictGlobal: boolean;
  globalBackend: GlobalValidationBackend;
  foldCliCommand?: string;
  minVertexDistance: number;
  maxVertices: number;
  maxEdges: number;
  requireBoxPleat?: boolean;
}

export interface RenderVariantConfig {
  name: string;
  assignmentVisibility: "visible" | "hidden";
  count: number;
}

export interface SyntheticRecipe {
  name: string;
  seed: number;
  imageSize: number;
  padding: number;
  splits: Record<"train" | "val" | "test", number>;
  families: Record<GeneratorFamily, number>;
  complexityBuckets: ComplexityBucket[];
  validation: ValidationConfig;
  renderVariants: RenderVariantConfig[];
}

export interface GenerationConfig {
  id: string;
  family: GeneratorFamily;
  seed: number;
  numCreases: number;
  bucket: string;
}

export interface ValidationResult {
  valid: boolean;
  passed: string[];
  failed: string[];
  errors: string[];
}

export interface RawManifestRow {
  id: string;
  seed: number;
  family: GeneratorFamily;
  bucket: string;
  split: "train" | "val" | "test";
  foldPath: string;
  metadataPath: string;
  vertices: number;
  edges: number;
  assignments: Record<string, number>;
  roleCounts?: Record<string, number>;
  bpMetadata?: BPMetadata;
  validation: ValidationResult;
}
