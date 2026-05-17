export type EdgeAssignment = "B" | "M" | "V" | "F" | "U" | "C";
export type TreeMakerArchetype = "insect" | "quadruped" | "bird" | "creature" | "object" | "abstract";
export type TreeMakerSymmetryClass = "diagonal" | "middle-axis" | "asymmetric";
export type TreeMakerSymmetryVariant = "main-diagonal" | "anti-diagonal" | "vertical" | "horizontal" | "none";
export type TreeMakerTopology = "radial-star" | "hubbed-limbs" | "spine-chain" | "branched-hybrid";
export type RabbitEarFoldProgramAxiom = "axiom1" | "axiom2" | "axiom3" | "axiom4" | "axiom7";
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

export interface DensityMetadata {
  densityBucket: string;
  gridSize: number;
  targetEdgeRange: [number, number];
  subfamily: string;
  symmetry: string;
  generatorSteps: string[];
  moleculeCounts: Record<string, number>;
  solverMs?: number;
}

export interface TreeMetadata {
  generator: "treemaker-tree";
  archetype: TreeMakerArchetype;
  symmetryClass: TreeMakerSymmetryClass;
  symmetryVariant: TreeMakerSymmetryVariant;
  topology: TreeMakerTopology;
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

export interface RabbitEarFoldProgramMetadata {
  generator: "rabbit-ear-fold-program";
  rabbitEarApi: "ear.graph.flatFold";
  appliedFoldCount: number;
  attemptedFoldCount: number;
  axiomUsage: Partial<Record<RabbitEarFoldProgramAxiom, number>>;
  activeCreaseCount: number;
  targetActiveCreases: number;
  targetActiveCreaseRange: [number, number];
  requestedBucket: string;
}

export interface LabelPolicy {
  labelSource: "treemaker-external" | "rabbit-ear-fold-program";
  geometrySource: "treemaker-external" | "rabbit-ear-fold-program";
  assignmentSource: "treemaker-external" | "rabbit-ear-fold-program";
  trainingEligible: boolean;
  notes: string[];
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
  density_metadata?: DensityMetadata;
  tree_metadata?: TreeMetadata;
  treemaker_metadata?: TreeMakerMetadata;
  rabbit_ear_metadata?: RabbitEarFoldProgramMetadata;
  edges_treemakerKind?: TreeMakerCreaseKind[];
  label_policy?: LabelPolicy;
  faces_vertices?: number[][];
  faces_edges?: number[][];
  [key: string]: unknown;
}

export const GENERATOR_FAMILIES = [
  "treemaker-tree",
  "rabbit-ear-fold-program",
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
  requireDense?: boolean;
  requireTreeMaker?: boolean;
  requireRabbitEarFoldProgram?: boolean;
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
  treeMakerSampler?: TreeMakerSamplerConfig;
}

export interface GenerationConfig {
  id: string;
  family: GeneratorFamily;
  seed: number;
  numCreases: number;
  maxCreases?: number;
  bucket: string;
  dense?: boolean;
  treeMakerSampler?: TreeMakerSamplerConfig;
}

export interface TreeMakerSamplerConfig {
  symmetryWeights?: Partial<Record<TreeMakerSymmetryClass, number>>;
  middleAxisWeights?: Partial<Record<Extract<TreeMakerSymmetryVariant, "vertical" | "horizontal">, number>>;
  diagonalWeights?: Partial<Record<Extract<TreeMakerSymmetryVariant, "main-diagonal" | "anti-diagonal">, number>>;
  archetypeWeights?: Partial<Record<TreeMakerArchetype, number>>;
  topologyWeights?: Partial<Record<TreeMakerTopology, number>>;
  acceptedMix?: TreeMakerAcceptedMixConfig;
}

export interface TreeMakerAcceptedMixConfig {
  enabled?: boolean;
  symmetryWeights?: Partial<Record<TreeMakerSymmetryClass, number>>;
  archetypeWeights?: Partial<Record<TreeMakerArchetype, number>>;
  topologyWeights?: Partial<Record<TreeMakerTopology, number>>;
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
  densityMetadata?: DensityMetadata;
  treeMetadata?: TreeMetadata;
  treeMakerMetadata?: TreeMakerMetadata;
  rabbitEarMetadata?: RabbitEarFoldProgramMetadata;
  labelPolicy?: LabelPolicy;
  validation: ValidationResult;
}
