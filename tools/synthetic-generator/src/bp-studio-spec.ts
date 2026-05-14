export const BP_STUDIO_SPEC_SCHEMA_VERSION = "bp-studio-adapter-spec/v1" as const;

export const BP_STUDIO_ARCHETYPES = ["insect", "quadruped", "bird", "object", "abstract"] as const;
export type BPStudioArchetype = (typeof BP_STUDIO_ARCHETYPES)[number];

export const BP_STUDIO_COMPLEXITY_BUCKETS = ["small", "medium", "dense", "superdense"] as const;
export type BPStudioComplexityBucket = (typeof BP_STUDIO_COMPLEXITY_BUCKETS)[number];

export type BPStudioCoordinateSystem = "integer-grid";
export type BPStudioSide = "top" | "right" | "bottom" | "left" | "interior";
export type BPStudioAxis = "horizontal" | "vertical" | "diagonal" | "polyline";
export type BPStudioRiverBend = "direct" | "dogleg" | "staircase" | "fan";
export type BPStudioSymmetry =
  | "asymmetric"
  | "bilateral-x"
  | "bilateral-y"
  | "rotational-2"
  | "radial-4";

export type BPStudioTreeNodeKind = "root" | "body" | "hub" | "flap";
export type BPStudioTreeEdgeRole = "body" | "river" | "appendage" | "hinge" | "support";
export type BPStudioFlapClass = "terminal" | "wide" | "elevation" | "body";

export interface BPStudioPoint {
  x: number;
  y: number;
}

export interface BPStudioSheetSpec {
  width: number;
  height: number;
  gridSize: number;
  margin: number;
  coordinateSystem: BPStudioCoordinateSystem;
  unit: "bp-grid";
}

export interface BPStudioTreeNode {
  id: string;
  kind: BPStudioTreeNodeKind;
  label: string;
  width?: number;
  height?: number;
  elevation?: number;
  tags?: string[];
}

export interface BPStudioTreeEdge {
  id: string;
  from: string;
  to: string;
  length: number;
  role: BPStudioTreeEdgeRole;
  width?: number;
  tags?: string[];
}

export interface BPStudioTreeSpec {
  rootId: string;
  nodes: BPStudioTreeNode[];
  edges: BPStudioTreeEdge[];
}

export interface BPStudioBodyPlacement {
  nodeId: string;
  label: string;
  center: BPStudioPoint;
  width: number;
  height: number;
  elevation: number;
  tags?: string[];
}

export interface BPStudioFlapPlacement {
  nodeId: string;
  label: string;
  class: BPStudioFlapClass;
  terminal: BPStudioPoint;
  side: BPStudioSide;
  width: number;
  height: number;
  elevation: number;
  terminalRadius: number;
  priority: number;
  mirroredWith?: string;
  tags?: string[];
}

export interface BPStudioRiverHint {
  edgeId: string;
  from: string;
  to: string;
  width: number;
  preferredAxis: BPStudioAxis;
  bend: BPStudioRiverBend;
  clearance: number;
  tags?: string[];
}

export interface BPStudioOptimizerHints {
  objective: "utilization-balanced" | "compact-flaps" | "river-clearance";
  keepSymmetry: boolean;
  allowTerminalRelaxation: boolean;
  maxIterations: number;
  topK: number;
}

export interface BPStudioLayoutSpec {
  symmetry: BPStudioSymmetry;
  bodies: BPStudioBodyPlacement[];
  flaps: BPStudioFlapPlacement[];
  rivers: BPStudioRiverHint[];
  optimizerHints: BPStudioOptimizerHints;
}

export interface BPStudioExpectedComplexity {
  bucket: BPStudioComplexityBucket;
  targetFlaps: [number, number];
  targetTreeEdges: [number, number];
  targetCreases: [number, number];
  expectedGridSize: [number, number];
}

export interface BPStudioSamplerMetadata {
  samplerVersion: "bp-studio-sampler/v1";
  grammar: string;
  seed: number;
  variation: number;
  symmetry: BPStudioSymmetry;
  notes: string[];
}

export interface BPStudioAdapterSpec {
  schemaVersion: typeof BP_STUDIO_SPEC_SCHEMA_VERSION;
  id: string;
  seed: number;
  archetype: BPStudioArchetype;
  expectedComplexity: BPStudioExpectedComplexity;
  sheet: BPStudioSheetSpec;
  tree: BPStudioTreeSpec;
  layout: BPStudioLayoutSpec;
  sampler: BPStudioSamplerMetadata;
}
