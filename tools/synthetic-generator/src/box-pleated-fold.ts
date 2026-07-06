// Serialize an assigned box-pleat crease pattern (PackingCP) to the FOLD format
// the training pipeline consumes: coordinates normalized to the [0,1]x[0,1] square
// domain, one FOLD edge per assigned unit crease, and edges_assignment in
// M/V/B/U. Box-pleat-specific provenance + quality (conflicts, Kawasaki failures,
// unassigned creases) rides along under `box_pleated_metadata` so downstream can
// tier/filter without re-deriving it.

import type { PackingCP } from "./box-pleated-cp.ts";
import type { EdgeAssignment, FOLDFormat } from "./types.ts";

export interface BoxPleatedFoldMeta {
  id: string;
  seed: number;
  leafCount: number;
}

/** Quality summary of a box-pleat CP (also embedded in the FOLD metadata). */
export interface BoxPleatedQuality {
  /** Interior vertices still failing Maekawa after assignment + repair. */
  conflicts: number;
  /** Interior junctions failing Kawasaki / even-degree. */
  kawasakiFailing: number;
  /** Non-border creases left without an M/V label. */
  unassigned: number;
  /** True when conflicts, Kawasaki failures and unassigned creases are all zero. */
  clean: boolean;
}

const MV: Record<string, EdgeAssignment> = { M: "M", V: "V", B: "B" };

/** Non-border creases (M/V/U) that never received an M or V label. */
function unassignedCount(cp: PackingCP): number {
  return cp.assignedEdges.filter((e) => e.type !== "boundary" && e.mv !== "M" && e.mv !== "V").length;
}

/** Quality summary for a (valid) box-pleat CP. */
export function boxPleatedQuality(cp: PackingCP): BoxPleatedQuality {
  const conflicts = cp.mvConflicts.length;
  const kawasakiFailing = cp.failing.length;
  const unassigned = unassignedCount(cp);
  return { conflicts, kawasakiFailing, unassigned, clean: conflicts === 0 && kawasakiFailing === 0 && unassigned === 0 };
}

/**
 * FOLD crease pattern for a box-pleat CP, coords normalized to [0,1]^2. Vertices
 * are deduplicated at 1e-6 after normalization (off-grid stretch points included).
 */
export function boxPleatedCpToFold(cp: PackingCP, meta: BoxPleatedFoldMeta): FOLDFormat {
  const { width: W, height: H } = cp.sheet;
  const index = new Map<string, number>();
  const vertices_coords: [number, number][] = [];
  const idx = (p: { x: number; y: number }): number => {
    const nx = p.x / W;
    const ny = p.y / H;
    const k = `${Math.round(nx * 1e6)},${Math.round(ny * 1e6)}`;
    const existing = index.get(k);
    if (existing !== undefined) return existing;
    index.set(k, vertices_coords.length);
    vertices_coords.push([nx, ny]);
    return vertices_coords.length - 1;
  };

  const edges_vertices: [number, number][] = [];
  const edges_assignment: EdgeAssignment[] = [];
  for (const e of cp.assignedEdges) {
    const a = idx(e.a);
    const b = idx(e.b);
    if (a === b) continue; // degenerate (sub-1e-6 after normalization)
    edges_vertices.push([a, b]);
    edges_assignment.push(MV[e.mv ?? ""] ?? "U");
  }

  const quality = boxPleatedQuality(cp);
  return {
    file_spec: 1.1,
    file_creator: "cp-detector/box-pleated",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords,
    edges_vertices,
    edges_assignment,
    box_pleated_metadata: {
      id: meta.id,
      seed: meta.seed,
      leafCount: meta.leafCount,
      sheet: { width: W, height: H },
      quality,
    },
  };
}
