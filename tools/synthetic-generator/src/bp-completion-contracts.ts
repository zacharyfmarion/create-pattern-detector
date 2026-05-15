import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

export type CompletionAxis = "horizontal" | "vertical";
export type CompletionSource = "fixture" | "bp-studio-optimized-layout";
export type MoleculeKind =
  | "sheet-border"
  | "flap-contour"
  | "river-corridor"
  | "hinge-corridor"
  | "corner-fan"
  | "diagonal-staircase"
  | "diamond-connector"
  | "stretch-gadget"
  | "body-panel";

export interface CompletionPoint {
  x: number;
  y: number;
}

export interface CompletionRegion {
  id: string;
  kind: "body" | "flap" | "river" | "hub";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface CompletionTerminal {
  id: string;
  nodeId: string;
  x: number;
  y: number;
  side: "top" | "right" | "bottom" | "left" | "interior";
  width: number;
  height: number;
  priority: number;
}

export interface CompletionCorridor {
  id: string;
  from: string;
  to: string;
  orientation: CompletionAxis;
  coordinate: number;
  width: number;
}

export interface CompletionLayout {
  id: string;
  source: CompletionSource;
  gridSize: number;
  axis: CompletionAxis;
  spineCoordinate: number;
  regions: CompletionRegion[];
  terminals: CompletionTerminal[];
  corridors: CompletionCorridor[];
  scaffoldSummary: {
    adapterLineCount: number;
    adapterVertexCount: number;
    adapterEdgeCount: number;
    optimizedFlapCount: number;
    optimizedTreeEdgeCount: number;
  };
}

export interface Port {
  id: string;
  moleculeId: string;
  orientation: CompletionAxis | "diagonal-positive" | "diagonal-negative";
  side: "top" | "right" | "bottom" | "left" | "interior";
  coordinate: number;
  width: number;
  parity: "integer" | "half";
  role: BPRole;
}

export interface PortJoin {
  from: string;
  to: string;
  orientation: Port["orientation"];
  width: number;
  accepted: boolean;
  reason?: string;
}

export interface MoleculeTemplate {
  id: string;
  kind: MoleculeKind;
  version: string;
  ports: Port[];
  localProof: {
    kawasaki: boolean;
    maekawa: boolean;
    fixture: string;
  };
}

export interface CompletionFoldLine {
  id: string;
  moleculeId: string;
  moleculeKind: MoleculeKind;
  vector: [number, number];
  origin: [number, number];
  assignment: Extract<EdgeAssignment, "M" | "V">;
  role: Exclude<BPRole, "border">;
}

export interface CompletionRejection {
  code: string;
  message: string;
  moleculeId?: string;
}

export interface CompletionResult {
  ok: boolean;
  fold?: FOLDFormat;
  layout: CompletionLayout;
  foldLines: CompletionFoldLine[];
  molecules: MoleculeTemplate[];
  portJoins: PortJoin[];
  rejected: CompletionRejection[];
}
