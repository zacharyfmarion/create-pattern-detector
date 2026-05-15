import { roleCounts } from "./fold-utils.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import { solveMaekawaAssignments } from "./bp-maekawa-assignment.ts";
import { buildDiagonalStaircaseCapPrimitive, type StaircaseCapCorner } from "./bp-staircase-cap.ts";
import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

interface BridgeSegment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
}

export interface StaircaseBridgeOptions {
  laneCount: number;
  orientation: "diagonal-positive" | "diagonal-negative";
}

export interface StaircaseBridgeResult {
  ok: boolean;
  fold?: FOLDFormat;
  assignmentSteps: number;
  errors: string[];
}

export function buildStaircaseBridgePrimitive(options: StaircaseBridgeOptions): StaircaseBridgeResult {
  const laneCount = Math.max(1, Math.floor(options.laneCount));
  const corners: [StaircaseCapCorner, StaircaseCapCorner] = options.orientation === "diagonal-positive"
    ? ["bottom-left", "top-right"]
    : ["bottom-right", "top-left"];
  const segments: BridgeSegment[] = sheetBorderSegments();

  for (const corner of corners) {
    const cap = buildDiagonalStaircaseCapPrimitive({
      laneCount,
      startAxisAssignment: "V",
      corner,
    });
    segments.push(...nonBorderSegments(cap));
  }

  const arranged = arrangeSegments(segments, "cp-synthetic-generator/bp-staircase-bridge");
  const arrangedCounts = roleCounts(arranged);
  arranged.bp_metadata = {
    gridSize: laneCount + 3,
    bpSubfamily: "staircase-bridge-primitive",
    flapCount: 0,
    gadgetCount: 2,
    ridgeCount: arrangedCounts.ridge ?? 0,
    hingeCount: arrangedCounts.hinge ?? 0,
    axisCount: arrangedCounts.axis ?? 0,
  };

  const solved = solveMaekawaAssignments(arranged);
  if (!solved.ok || !solved.fold) {
    return {
      ok: false,
      assignmentSteps: solved.steps,
      errors: solved.errors,
    };
  }

  const counts = roleCounts(solved.fold);
  solved.fold.file_creator = "cp-synthetic-generator/bp-staircase-bridge";
  solved.fold.bp_metadata = {
    ...arranged.bp_metadata,
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  return {
    ok: true,
    fold: solved.fold,
    assignmentSteps: solved.steps,
    errors: [],
  };
}

function nonBorderSegments(fold: FOLDFormat): BridgeSegment[] {
  return fold.edges_vertices.flatMap(([a, b], edgeIndex): BridgeSegment[] => {
    if (fold.edges_assignment[edgeIndex] === "B") return [];
    return [{
      p1: fold.vertices_coords[a],
      p2: fold.vertices_coords[b],
      assignment: "M",
      role: fold.edges_bpRole?.[edgeIndex] ?? roleForEdge(fold.vertices_coords[a], fold.vertices_coords[b]),
    }];
  });
}

function sheetBorderSegments(): BridgeSegment[] {
  return [
    border([0, 0], [1, 0]),
    border([1, 0], [1, 1]),
    border([1, 1], [0, 1]),
    border([0, 1], [0, 0]),
  ];
}

function border(p1: Point, p2: Point): BridgeSegment {
  return { p1, p2, assignment: "B", role: "border" };
}

function roleForEdge(a: Point, b: Point): Exclude<BPRole, "border"> {
  const dx = Math.abs(a[0] - b[0]);
  const dy = Math.abs(a[1] - b[1]);
  if (dx > 1e-8 && dy > 1e-8) return "ridge";
  if (dx < 1e-8) return "axis";
  return "hinge";
}
