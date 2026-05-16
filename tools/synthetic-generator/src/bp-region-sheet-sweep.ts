import { arrangeSegments } from "./line-arrangement.ts";
import { solveMaekawaAssignments } from "./bp-maekawa-assignment.ts";
import type { RegionCandidateSegment, RegionCompletionCandidate } from "./bp-completion-contracts.ts";
import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

interface SheetSweepSegment {
  p1: Point;
  p2: Point;
  assignment: Extract<EdgeAssignment, "M" | "V" | "B">;
  role: BPRole;
  source: {
    kind: string;
    mandatory: boolean;
    ownerId: string;
  };
}

export interface RegionSheetSweepCompletionResult {
  ok: boolean;
  fold?: FOLDFormat;
  assignmentSteps: number;
  errors: string[];
}

export function completeRegionCandidateBySheetSweep(candidate: RegionCompletionCandidate): RegionSheetSweepCompletionResult {
  const segments = sheetSweepSegments(candidate);
  const arranged = arrangeSegments(
    segments,
    "cp-synthetic-generator/bp-region/sheet-sweep-lab",
    {
      gridSize: candidate.layout.gridSize,
      bpSubfamily: "bp-studio-completed-uniaxial",
      flapCount: candidate.layout.flaps.length,
      gadgetCount: 0,
      ridgeCount: 1,
      hingeCount: 1,
      axisCount: 1,
    },
  );
  const solved = solveMaekawaAssignments(arranged);
  if (!solved.ok || !solved.fold) {
    return {
      ok: false,
      assignmentSteps: solved.steps,
      errors: solved.errors,
    };
  }

  const fold = solved.fold;
  fold.file_creator = "cp-synthetic-generator/bp-region/sheet-sweep-lab";
  fold.file_description = "Lab-only region completion probe. Extends BP Studio-guided corridor pleat lanes to the sheet border before assignment solving; useful for proving foldability, not production distribution.";
  delete fold.bp_metadata;
  delete fold.edges_bpRole;
  delete fold.edges_bpStudioSource;
  fold.completion_metadata = {
    engine: "bp-region-sheet-sweep-lab",
    version: "v0.1.0",
    source: "bp-studio-optimized-layout",
    scaffoldSummary: {
      adapterLineCount: 0,
      adapterVertexCount: 0,
      adapterEdgeCount: 0,
      optimizedFlapCount: candidate.layout.flaps.length,
      optimizedTreeEdgeCount: candidate.layout.pleatStrips.length,
    },
    selectedCenter: [0.5, 0.5],
    selectedFlapIds: candidate.layout.flaps.map((flap) => Number(flap.nodeId)).filter(Number.isFinite),
    portJoinCount: 0,
    rejectedCandidateCount: candidate.rejectionReasons.length,
    compilerSteps: [
      "take-bp-studio-region-candidate",
      "extend-pleat-lanes-to-sheet-border",
      "arrange-and-split-swept-lines",
      `solve-maekawa-assignments:${solved.steps}`,
    ],
  };
  fold.label_policy = {
    labelSource: "compiler",
    geometrySource: "compiler",
    assignmentSource: "compiler",
    trainingEligible: false,
    notes: [
      "Lab-only completion probe. Sheet-wide lane extension can invade flap allocation territories and must not be promoted to production.",
      "Used to verify that BP Studio-guided long-axis lane geometry can be made strictly flat-foldable before bounded terminal/hub closure molecules are implemented.",
    ],
  };
  return { ok: true, fold, assignmentSteps: solved.steps, errors: [] };
}

function sheetSweepSegments(candidate: RegionCompletionCandidate): SheetSweepSegment[] {
  const result = new Map<string, SheetSweepSegment>();
  for (const segment of candidate.segments) {
    if (segment.kind === "border") {
      add(result, {
        p1: segment.p1,
        p2: segment.p2,
        assignment: "B",
        role: "border",
        source: { kind: "region-border", mandatory: true, ownerId: segment.regionId },
      });
      continue;
    }
    if (segment.kind !== "strip-pleat") continue;
    add(result, extendPleatToSheet(segment));
  }
  return [...result.values()];
}

function extendPleatToSheet(segment: RegionCandidateSegment): SheetSweepSegment {
  const vertical = Math.abs(segment.p1[0] - segment.p2[0]) < 1e-9;
  const coordinate = vertical ? segment.p1[0] : segment.p1[1];
  return {
    p1: vertical ? [coordinate, 0] : [0, coordinate],
    p2: vertical ? [coordinate, 1] : [1, coordinate],
    assignment: segment.assignment,
    role: vertical ? "axis" : "hinge",
    source: {
      kind: "region-sheet-sweep-pleat",
      mandatory: true,
      ownerId: segment.regionId,
    },
  };
}

function add(target: Map<string, SheetSweepSegment>, segment: SheetSweepSegment): void {
  target.set(segmentKey(segment.p1, segment.p2), segment);
}

function segmentKey(a: Point, b: Point): string {
  const left = `${a[0].toFixed(9)},${a[1].toFixed(9)}`;
  const right = `${b[0].toFixed(9)},${b[1].toFixed(9)}`;
  return left < right ? `${left}:${right}` : `${right}:${left}`;
}
