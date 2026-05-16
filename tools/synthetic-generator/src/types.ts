export type EdgeAssignment = "B" | "M" | "V" | "F" | "U" | "C";
export type BPRole = "border" | "hinge" | "ridge" | "axis" | "stretch";
export interface BPStudioEdgeSource {
  kind: string;
  creaseType?: number;
  mandatory?: boolean;
  ownerId?: number | string;
  stretchId?: string;
  deviceIndex?: number;
  lineIndex?: number;
  clippedSegmentIndex?: number;
}
export interface CompilerEdgeSource {
  kind: string;
  mandatory: boolean;
  moleculeKind?: string;
  role?: BPRole;
}
export type BPSubfamily =
  | "two-flap-stretch"
  | "dense-molecule-tessellation"
  | "realistic-tree-base"
  | "bp-studio-export"
  | "bp-studio-source-line-program"
  | "bp-studio-completed-uniaxial"
  | "diagonal-staircase-cap-primitive"
  | "staircase-bridge-primitive";
export type DenseNonBPSubfamily = "recursive-axiom" | "expanded-classic" | "radial-multi-vertex" | "tessellation-like";
export type RealisticBPArchetype = "insect" | "quadruped" | "bird" | "object" | "abstract";
export type BoxPleatMode = "simple" | "dense" | "bp-studio-source";
export type TreeMakerArchetype = "insect" | "quadruped" | "bird" | "creature" | "object" | "abstract";
export type TreeMakerSymmetryClass = "diagonal" | "middle-axis" | "asymmetric";
export type TreeMakerSymmetryVariant = "main-diagonal" | "anti-diagonal" | "vertical" | "horizontal" | "none";
export type TreeMakerCreaseKind =
  | "BORDER"
  | "AXIAL"
  | "RIDGE"
  | "GUSSET"
  | "FOLDED_HINGE"
  | "UNFOLDED_HINGE"
  | "PSEUDOHINGE"
  | "CONSTRUCTION"
  | "UNKNOWN";

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

export interface TreeMetadata {
  generator: "treemaker-tree";
  archetype: TreeMakerArchetype;
  symmetryClass: TreeMakerSymmetryClass;
  symmetryVariant: TreeMakerSymmetryVariant;
  rootId: string;
  nodeCount: number;
  terminalCount: number;
  branchDepth: number;
  edgeLengths: number[];
  nodes: Array<{
    id: string;
    kind: "root" | "hub" | "terminal";
    label: string;
    x: number;
    y: number;
  }>;
  edges: Array<{
    id: string;
    from: string;
    to: string;
    length: number;
  }>;
}

export interface TreeMakerMetadata {
  adapterVersion: string;
  toolVersion?: string;
  externalCommand?: string;
  optimizationSuccess: boolean;
  foldedFormSuccess?: boolean;
  warnings: string[];
  creaseKindCounts: Record<string, number>;
  sourceCreaseCount: number;
}

export interface CompletionMetadata {
  engine: string;
  version: string;
  source: "fixture" | "bp-studio-optimized-layout";
  scaffoldSummary: {
    adapterLineCount: number;
    adapterVertexCount: number;
    adapterEdgeCount: number;
    optimizedFlapCount: number;
    optimizedTreeEdgeCount: number;
  };
  selectedCenter: [number, number];
  selectedFlapIds: number[];
  portJoinCount: number;
  rejectedCandidateCount: number;
  compilerSteps: string[];
}

export interface LabelPolicy {
  labelSource: "compiler" | "bp-studio-raw" | "treemaker-external";
  geometrySource: "compiler" | "bp-studio-raw" | "treemaker-external";
  assignmentSource: "compiler" | "bp-studio-raw" | "treemaker-external";
  trainingEligible: boolean;
  notes: string[];
}

export interface BPStudioSummary {
  adapterVersion?: string;
  bpStudioVersion?: string;
  optimizerLayout?: string;
  optimizerSeed?: number | null;
  exportMode?: string;
  scaffoldEdges?: number;
  scaffoldVertices?: number;
  scaffoldAssignments?: Record<string, number>;
  optimizedFlapCount?: number;
  optimizedTreeEdgeCount?: number;
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
  edges_bpStudioSource?: BPStudioEdgeSource[];
  edges_compilerSource?: CompilerEdgeSource[];
  bp_metadata?: BPMetadata;
  density_metadata?: DensityMetadata;
  design_tree?: DesignTreeMetadata;
  layout_metadata?: LayoutMetadata;
  molecule_metadata?: MoleculeMetadata;
  realism_metadata?: RealismMetadata;
  tree_metadata?: TreeMetadata;
  treemaker_metadata?: TreeMakerMetadata;
  edges_treemakerKind?: TreeMakerCreaseKind[];
  completion_metadata?: CompletionMetadata;
  label_policy?: LabelPolicy;
  bp_studio_summary?: BPStudioSummary;
  faces_vertices?: number[][];
  faces_edges?: number[][];
  [key: string]: unknown;
}

export const GENERATOR_FAMILIES = [
  "bp-studio-realistic",
  "bp-studio-completed",
  "treemaker-tree",
] as const;
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
  requireTreeMaker?: boolean;
  requireLocalFlatFoldability?: boolean;
}

export interface RenderVariantConfig {
  name: string;
  assignmentVisibility: "visible" | "hidden" | "active-only";
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
  bpStudioSampler?: Record<string, unknown>;
  treeMakerSampler?: TreeMakerSamplerConfig;
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
  treeMakerSampler?: TreeMakerSamplerConfig;
}

export interface TreeMakerSamplerConfig {
  symmetryWeights?: Partial<Record<TreeMakerSymmetryClass, number>>;
  middleAxisWeights?: Partial<Record<Extract<TreeMakerSymmetryVariant, "vertical" | "horizontal">, number>>;
  diagonalWeights?: Partial<Record<Extract<TreeMakerSymmetryVariant, "main-diagonal" | "anti-diagonal">, number>>;
  archetypeWeights?: Partial<Record<TreeMakerArchetype, number>>;
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
  treeMetadata?: TreeMetadata;
  treeMakerMetadata?: TreeMakerMetadata;
  completionMetadata?: CompletionMetadata;
  labelPolicy?: LabelPolicy;
  bpStudioSummary?: BPStudioSummary;
  validation: ValidationResult;
}
