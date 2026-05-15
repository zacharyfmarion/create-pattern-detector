import { roleCounts } from "./fold-utils.ts";
import { arrangeSegments } from "./line-arrangement.ts";
import type { BPRole, EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

export type StaircaseCapCorner = "bottom-left" | "bottom-right" | "top-left" | "top-right";

interface StaircaseCapSegment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: BPRole;
}

export interface DiagonalStaircaseCapOptions {
  laneCount: number;
  startAxisAssignment: Extract<EdgeAssignment, "M" | "V">;
  corner?: StaircaseCapCorner;
}

export function buildDiagonalStaircaseCapPrimitive(options: DiagonalStaircaseCapOptions): FOLDFormat {
  const laneCount = Math.max(1, Math.floor(options.laneCount));
  const gridSize = laneCount + 3;
  const corner = options.corner ?? "bottom-left";
  const canonicalSegments: StaircaseCapSegment[] = [
    border([0, 0], [1, 0]),
    border([1, 0], [1, 1]),
    border([1, 1], [0, 1]),
    border([0, 1], [0, 0]),
    { p1: [0, 0], p2: [1, 1], assignment: "M", role: "ridge" },
  ];

  for (let index = 0; index < laneCount; index += 1) {
    const t = (index + 2) / gridSize;
    const axisAssignment = alternate(options.startAxisAssignment, index);
    const hingeAssignment = opposite(axisAssignment);
    canonicalSegments.push({
      p1: [t, 0],
      p2: [t, t],
      assignment: axisAssignment,
      role: "axis",
    });
    canonicalSegments.push({
      p1: [0, t],
      p2: [t, t],
      assignment: hingeAssignment,
      role: "hinge",
    });
  }

  const segments = canonicalSegments.map((segment) => transformSegment(segment, corner));
  const fold = arrangeSegments(segments, "cp-synthetic-generator/bp-staircase-cap");
  const counts = roleCounts(fold);
  fold.bp_metadata = {
    gridSize,
    bpSubfamily: "diagonal-staircase-cap-primitive",
    flapCount: 0,
    gadgetCount: 1,
    ridgeCount: counts.ridge ?? 0,
    hingeCount: counts.hinge ?? 0,
    axisCount: counts.axis ?? 0,
  };
  return fold;
}

function transformSegment(segment: StaircaseCapSegment, corner: StaircaseCapCorner): StaircaseCapSegment {
  return {
    ...segment,
    p1: transformPoint(segment.p1, corner),
    p2: transformPoint(segment.p2, corner),
  };
}

function transformPoint(point: Point, corner: StaircaseCapCorner): Point {
  const [x, y] = point;
  if (corner === "bottom-right") return [1 - x, y];
  if (corner === "top-left") return [x, 1 - y];
  if (corner === "top-right") return [1 - x, 1 - y];
  return point;
}

function border(p1: Point, p2: Point): StaircaseCapSegment {
  return { p1, p2, assignment: "B", role: "border" };
}

function alternate(start: Extract<EdgeAssignment, "M" | "V">, index: number): Extract<EdgeAssignment, "M" | "V"> {
  if (index % 2 === 0) return start;
  return opposite(start);
}

function opposite(assignment: Extract<EdgeAssignment, "M" | "V">): Extract<EdgeAssignment, "M" | "V"> {
  return assignment === "M" ? "V" : "M";
}
