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
  allocationRadius?: number;
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
  position?: CompletionPoint;
  expectedPartnerRole?: BPRole;
}

export interface PortJoin {
  from: string;
  to: string;
  orientation: Port["orientation"];
  width: number;
  accepted: boolean;
  reason?: string;
  fromPosition?: CompletionPoint;
  toPosition?: CompletionPoint;
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

export interface MoleculePatchVertex {
  id: string;
  x: number;
  y: number;
}

export interface MoleculePatchSegment {
  id: string;
  from: string;
  to: string;
  assignment: Extract<EdgeAssignment, "M" | "V">;
  role: Exclude<BPRole, "border">;
}

export interface MoleculePatch {
  id: string;
  kind: MoleculeKind;
  version: string;
  vertices: MoleculePatchVertex[];
  segments: MoleculePatchSegment[];
  ports: Port[];
  bounds: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

export interface MoleculeTransform {
  translate: CompletionPoint;
  rotateQuarterTurns?: 0 | 1 | 2 | 3;
  mirrorX?: boolean;
  mirrorY?: boolean;
  scale?: number;
}

export interface MoleculeInstance {
  id: string;
  templateId: string;
  kind: MoleculeKind;
  transform: MoleculeTransform;
  ports: Port[];
  patch?: MoleculePatch;
}

export interface CompletionSegment {
  id: string;
  moleculeId: string;
  moleculeKind: MoleculeKind;
  p1: [number, number];
  p2: [number, number];
  assignment: Extract<EdgeAssignment, "M" | "V" | "B">;
  role: BPRole;
}

export interface CompositionFixture {
  id: string;
  description: string;
  requiredMoleculeKinds: MoleculeKind[];
}

export interface MoleculeCertificationReport {
  fixtureId: string;
  ok: boolean;
  moleculeKinds: MoleculeKind[];
  checkedPortJoins: number;
  rejectedPortJoins: number;
  danglingEndpointCount: number;
  validationPassed: string[];
  validationFailed: string[];
  errors: string[];
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
  moleculeInstances?: MoleculeInstance[];
  segments?: CompletionSegment[];
  portJoins: PortJoin[];
  rejected: CompletionRejection[];
}

export interface RegionRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type PleatStripOrientation = "vertical" | "horizontal";

export interface FlapRegion {
  id: string;
  terminalId: string;
  nodeId: string;
  side: CompletionTerminal["side"];
  rect: RegionRect;
  center: CompletionPoint;
  allocationRadius?: number;
}

export interface BodyPanelRegion {
  id: string;
  rect: RegionRect;
  center: CompletionPoint;
}

export interface PleatStripRegion {
  id: string;
  from: string;
  to: string;
  rect: RegionRect;
  orientation: PleatStripOrientation;
  pitch: number;
  phase: number;
  startAssignment: Extract<EdgeAssignment, "M" | "V">;
  treeEdgeId?: string;
}

export interface BoundaryPort {
  id: string;
  regionId: string;
  side: "top" | "right" | "bottom" | "left";
  orientation: PleatStripOrientation | "diagonal-positive" | "diagonal-negative";
  position: CompletionPoint;
  width: number;
}

export interface PleatStripPortConstraint {
  id: string;
  aStripId: string;
  aSide: "start" | "end";
  bStripId: string;
  bSide: "start" | "end";
  sequenceOrder?: "same" | "reversed";
}

export interface StairBoundary {
  id: string;
  stripId: string;
  side: "start" | "end";
  lines: Array<{
    p1: [number, number];
    p2: [number, number];
    assignment: Extract<EdgeAssignment, "M" | "V">;
    role: Exclude<BPRole, "border">;
  }>;
}

export type CompletionCandidateValidity =
  | "layout-valid"
  | "candidate-complete"
  | "locally-valid"
  | "globally-valid"
  | "training-eligible"
  | "rejected";

export interface RegionLayout {
  id: string;
  sourceLayoutId: string;
  gridSize: number;
  axis: CompletionAxis;
  bodies: BodyPanelRegion[];
  flaps: FlapRegion[];
  pleatStrips: PleatStripRegion[];
  boundaryPorts: BoundaryPort[];
  portConstraints?: PleatStripPortConstraint[];
}

export interface RegionCandidateSegment {
  id: string;
  regionId: string;
  kind: "border" | "body-boundary" | "flap-boundary" | "strip-pleat" | "stair-boundary" | "turn-closure";
  p1: [number, number];
  p2: [number, number];
  assignment: Extract<EdgeAssignment, "M" | "V" | "B">;
  role: BPRole;
}

export interface RegionLocalFailurePoint {
  x: number;
  y: number;
  kawasaki: boolean;
  maekawa: boolean;
}

export interface RegionLocalFlatFoldProbe {
  activeSegmentKinds: RegionCandidateSegment["kind"][];
  arrangedVertices: number;
  arrangedEdges: number;
  kawasakiBad: number;
  maekawaBad: number;
  badVertices: number;
  locallyFlatFoldable: boolean;
  failureReasons: Record<string, number>;
  failurePoints: RegionLocalFailurePoint[];
}

export interface RegionCompletionCandidate {
  id: string;
  layout: RegionLayout;
  validity: CompletionCandidateValidity;
  segments: RegionCandidateSegment[];
  stairBoundaries: StairBoundary[];
  localProbe?: RegionLocalFlatFoldProbe;
  rejectionReasons: string[];
}
