export type EdgeAssignment = "B" | "M" | "V" | "F" | "U" | "C";
export type BPRole = "border" | "hinge" | "ridge" | "axis" | "stretch";
export type BPSubfamily = "two-flap-stretch" | "uniaxial-chain" | "symmetric-insect-lite" | "dense-molecule-tessellation" | "realistic-tree-base";
export type DenseNonBPSubfamily = "recursive-axiom" | "expanded-classic" | "radial-multi-vertex" | "tessellation-like";
export type RealisticBPArchetype = "insect" | "quadruped" | "bird" | "object" | "abstract";
export type BoxPleatMode = "simple" | "dense";

export interface BPMetadata {
  gridSize: number;
  bpSubfamily: BPSubfamily;
  flapCount: number;
  gadgetCount: number;
  ridgeCount: number;
  hingeCount: number;
  axisCount: number;
}

export interface DensityMetadata {
  densityBucket: string;
  gridSize: number;
  targetEdgeRange: [number, number];
  subfamily: BPSubfamily | DenseNonBPSubfamily | string;
  symmetry: string;
  generatorSteps: string[];
  moleculeCounts: Record<string, number>;
  solverMs?: number;
}

export interface DesignTreeMetadata {
  archetype: RealisticBPArchetype;
  rootId: string;
  nodes: Array<{
    id: string;
    kind: "body" | "flap" | "river" | "hub";
    label: string;
  }>;
  edges: Array<{
    from: string;
    to: string;
    length: number;
    role: "flap" | "river" | "body";
  }>;
}

export interface LayoutMetadata {
  gridSize: number;
  symmetry: string;
  margin: number;
  bodyRegions: Array<{ id: string; x1: number; y1: number; x2: number; y2: number }>;
  flapTerminals: Array<{ id: string; x: number; y: number; side: "top" | "right" | "bottom" | "left" }>;
  corridors: Array<{ id: string; orientation: "horizontal" | "vertical"; coordinate: number; role: "axis" | "hinge" | "stretch" }>;
  layoutScore: number;
}

export interface MoleculeMetadata {
  libraryVersion: string;
  molecules: Record<string, number>;
  portChecks: {
    checked: number;
    rejected: number;
  };
}

export interface RealismMetadata {
  score: number;
  emptySpaceRatio: number;
  localDensityVariance: number;
  repetitionPenalty: number;
  macroRegionDiversity: number;
  orientationHistogram: Record<string, number>;
  degreeHistogram: Record<string, number>;
  roleRatios: Record<string, number>;
  gates: Record<string, boolean>;
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
  density_metadata?: DensityMetadata;
  design_tree?: DesignTreeMetadata;
  layout_metadata?: LayoutMetadata;
  molecule_metadata?: MoleculeMetadata;
  realism_metadata?: RealismMetadata;
  faces_vertices?: number[][];
  faces_edges?: number[][];
  [key: string]: unknown;
}

export const GENERATOR_FAMILIES = ["axiom", "classic", "single-vertex", "box-pleat", "realistic-box-pleat", "dense-non-bp", "grid-baseline"] as const;
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
  boxPleatMode?: BoxPleatMode;
  requireDense?: boolean;
  requireRealistic?: boolean;
  minRealismScore?: number;
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
  dense?: boolean;
  denseSubfamily?: string;
  realisticArchetype?: RealisticBPArchetype;
}

export interface ValidationResult {
  valid: boolean;
  passed: string[];
  failed: string[];
  errors: string[];
  metrics?: {
    solverMs?: number;
    faces?: number;
  };
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
  densityMetadata?: DensityMetadata;
  designTree?: DesignTreeMetadata;
  layoutMetadata?: LayoutMetadata;
  moleculeMetadata?: MoleculeMetadata;
  realismMetadata?: RealismMetadata;
  validation: ValidationResult;
}
