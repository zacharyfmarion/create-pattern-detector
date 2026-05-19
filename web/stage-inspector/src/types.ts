export type StageStatus = "implemented" | "scaffolded";

export interface StageInfo {
  id: string;
  label: string;
  title: string;
  status: StageStatus;
}

export interface Stage4ExampleRow {
  key: string;
  id: string;
  sampleIndex: number;
  family: string;
  bucket: string;
  profile: string;
  status: string;
  warnings: string[];
  repairs: string[];
  edgePrecision: number;
  edgeRecall: number;
  vertexPrecision: number;
  vertexRecall: number;
  assignmentAccuracy: number;
  structuralValid: boolean;
  predEdges: number;
  gtEdges: number;
  unknownEdges: number;
  observedEdges: number;
}

export interface Stage4ExamplesResponse {
  summary: Record<string, unknown>;
  rows: Stage4ExampleRow[];
  filters: {
    profiles: string[];
    families: string[];
    statuses: string[];
    warnings: string[];
    repairs: string[];
  };
  counts: {
    status: Record<string, number>;
    warnings: Record<string, number>;
    profiles: Record<string, number>;
    families: Record<string, number>;
  };
  metricRanges: Record<string, { min: number; max: number }>;
}

export interface WarningEntry {
  code: string;
  message: string;
  severity: string;
  edge_indices: number[];
  vertex_indices: number[];
  details: Record<string, unknown>;
}

export interface RepairEntry {
  code: string;
  message: string;
  edge_indices: number[];
  vertex_indices: number[];
  details: Record<string, unknown>;
}

export interface GraphVertex {
  id: number;
  x: number;
  y: number;
  degree: number;
  incidentEdges: number[];
  issues?: string[];
  repairs?: string[];
  matchedGtVertex?: number | null;
  matchedPredVertex?: number | null;
  matchErrorPx?: number | null;
  kawasakiResidual?: number | null;
}

export interface EdgeMatch {
  state: "matched" | "missing" | "extra";
  gtEdge?: number | null;
  predEdge?: number | null;
  assignmentCorrect: boolean;
}

export interface GraphEdge {
  id: number;
  vertices: [number, number];
  assignment: "M" | "V" | "B" | "U" | string;
  assignmentIndex: number;
  issues: string[];
  repairs?: string[];
  support?: number;
  confidence?: number;
  margin?: number;
  source?: string;
  match: EdgeMatch;
}

export interface Stage4Diagnostic {
  key: string;
  stage: "stage4";
  imageUrl: string;
  imageSize: number;
  row: Record<string, unknown>;
  metrics: Record<string, unknown>;
  status: string;
  structuralValidity: Record<string, unknown>;
  warnings: WarningEntry[];
  warningCounts: Record<string, number>;
  repairs: RepairEntry[];
  repairCounts: Record<string, number>;
  whatIfStatuses: Record<string, string>;
  graph: {
    groundTruth: {
      vertices: GraphVertex[];
      edges: GraphEdge[];
    };
    prediction: {
      vertices: GraphVertex[];
      edges: GraphEdge[];
    };
    matches: {
      vertexTolerancePx: number;
      matchedVertices: number;
      meanVertexErrorPx: number;
      matchedPredEdges: number[];
      matchedGtEdges: number[];
      missingGtEdges: number[];
      extraPredEdges: number[];
    };
  };
  cache?: {
    hit: boolean;
    key: string;
  };
  recomputeParams?: Record<string, unknown>;
}

export type EntitySelection =
  | { kind: "pred-edge"; id: number }
  | { kind: "gt-edge"; id: number }
  | { kind: "pred-vertex"; id: number }
  | { kind: "gt-vertex"; id: number }
  | null;

export type LayerKey =
  | "gtGraph"
  | "predGraph"
  | "missingEdges"
  | "extraEdges"
  | "ambiguousEdges"
  | "weakEdges"
  | "shortEdges"
  | "crowdedVertices"
  | "evenDegree"
  | "kawasaki"
  | "maekawa"
  | "repairs"
  | "labels";

export type Layers = Record<LayerKey, boolean>;
