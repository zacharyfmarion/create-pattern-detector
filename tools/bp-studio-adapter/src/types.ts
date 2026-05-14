export interface BpStudioAdapterSpec {
  title?: string;
  description?: string;
  sheet: SheetSpec;
  useAuxiliary?: boolean;
  completeRepositories?: boolean;
  tree?: TreeSpec;
  edges?: EdgeSpec[];
  flaps?: FlapSpec[];
}

export interface SheetSpec {
  width: number;
  height: number;
}

export interface TreeSpec {
  edges: EdgeSpec[];
  flaps: FlapSpec[];
}

export interface EdgeSpec {
  n1: number;
  n2: number;
  length: number;
}

export interface FlapSpec {
  id: number;
  x: number;
  y: number;
  width?: number;
  height?: number;
}

export interface FoldDocument {
  file_spec: 1.1;
  file_creator: string;
  file_title: string;
  file_description: string;
  vertices_coords: [number, number][];
  edges_vertices: [number, number][];
  edges_assignment: Assignment[];
  edges_foldAngle: number[];
}

export type Assignment = "B" | "M" | "V" | "F" | "U";

export interface AdapterMetadata {
  adapter: {
    name: string;
    version: string;
  };
  bpStudio: {
    version: string;
    source: string;
  };
  spec: {
    title: string;
    sheet: SheetSpec;
    useAuxiliary: boolean;
    completeRepositories: boolean;
    edgeCount: number;
    flapCount: number;
  };
  cp: {
    lineCount: number;
    vertexCount: number;
    edgeCount: number;
    assignmentCounts: Record<Assignment, number>;
  };
  stretches: StretchMetadata[];
}

export interface StretchMetadata {
  id: string;
  active: boolean;
  repository: {
    signature: string;
    isValid: boolean;
    complete: boolean;
    configurationCount: number;
    selectedConfigurationIndex: number | null;
    selectedPatternIndex: number | null;
    selectedPatternCount: number | null;
    selectedDeviceCount: number | null;
    selectedGadgetCount: number | null;
    selectedAddOnCount: number | null;
    quadrantCount: number;
    junctionCount: number;
    serialized?: unknown;
  };
}

export interface GenerationResult {
  fold: FoldDocument;
  metadata: AdapterMetadata;
}
