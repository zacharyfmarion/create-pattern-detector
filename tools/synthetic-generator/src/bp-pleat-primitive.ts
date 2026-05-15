import { arrangeSegments } from "./line-arrangement.ts";
import type { EdgeAssignment, FOLDFormat } from "./types.ts";

type Point = [number, number];

interface PrimitiveSegment {
  p1: Point;
  p2: Point;
  assignment: EdgeAssignment;
  role: "border" | "axis";
}

export interface SheetPleatPrimitiveOptions {
  laneCount: number;
  orientation: "vertical" | "horizontal";
  startAssignment: Extract<EdgeAssignment, "M" | "V">;
}

export function buildSheetPleatPrimitive(options: SheetPleatPrimitiveOptions): FOLDFormat {
  const laneCount = Math.max(1, Math.floor(options.laneCount));
  const segments: PrimitiveSegment[] = [
    border("border-bottom", [0, 0], [1, 0]),
    border("border-right", [1, 0], [1, 1]),
    border("border-top", [1, 1], [0, 1]),
    border("border-left", [0, 1], [0, 0]),
  ];
  for (let index = 0; index < laneCount; index += 1) {
    const coordinate = (index + 1) / (laneCount + 1);
    const assignment = alternate(options.startAssignment, index);
    segments.push({
      p1: options.orientation === "vertical" ? [coordinate, 0] as Point : [0, coordinate] as Point,
      p2: options.orientation === "vertical" ? [coordinate, 1] as Point : [1, coordinate] as Point,
      assignment,
      role: "axis" as const,
    });
  }

  const fold = arrangeSegments(segments, "cp-synthetic-generator/bp-pleat-primitive");
  delete fold.edges_bpRole;
  delete fold.edges_bpStudioSource;
  delete fold.bp_metadata;
  return fold;
}

function border(id: string, p1: Point, p2: Point): PrimitiveSegment {
  void id;
  return { p1, p2, assignment: "B" as const, role: "border" as const };
}

function alternate(start: Extract<EdgeAssignment, "M" | "V">, index: number): Extract<EdgeAssignment, "M" | "V"> {
  if (index % 2 === 0) return start;
  return start === "M" ? "V" : "M";
}
