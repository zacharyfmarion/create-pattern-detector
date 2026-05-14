import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

import { generate } from "../src/index";

import type { Assignment } from "../src/types";

describe("BP Studio adapter", () => {
  test("generates a non-empty FOLD document from the two-flap fixture", async () => {
    const fixture = JSON.parse(await readFile(new URL("../fixtures/two-flap.json", import.meta.url), "utf8"));
    const { fold, metadata } = await generate(fixture);
    const assignments = new Set<Assignment>(fold.edges_assignment);

    expect(fold.vertices_coords.length).toBeGreaterThan(0);
    expect(fold.edges_vertices.length).toBeGreaterThan(0);
    expect(assignments.has("B")).toBe(true);
    expect(assignments.has("M")).toBe(true);
    expect(assignments.has("V")).toBe(true);
    expect(fold.edges_bpRole).toHaveLength(fold.edges_vertices.length);
    expect(fold.edges_bpStudioSource).toHaveLength(fold.edges_vertices.length);
    expect(new Set(fold.edges_bpRole).has("ridge")).toBe(true);
    expect(new Set(fold.edges_bpRole).has("axis")).toBe(true);
    expect(fold.edges_bpStudioSource.some((source) => source.kind === "device-draw-ridge")).toBe(true);
    expect(fold.edges_bpStudioSource.some((source) => source.kind === "device-axis-parallel")).toBe(true);
    expect(metadata.stretches.length).toBeGreaterThan(0);
    expect(metadata.cp.assignmentCounts.B).toBeGreaterThan(0);
    expect(metadata.cp.assignmentCounts.M).toBeGreaterThan(0);
    expect(metadata.cp.assignmentCounts.V).toBeGreaterThan(0);
    expect(metadata.cp.roleCounts.ridge).toBeGreaterThan(0);
  });

  test("supports final-rendered export mode with BP Studio source ancestry", async () => {
    const fixture = JSON.parse(await readFile(new URL("../fixtures/two-flap.json", import.meta.url), "utf8"));
    const { fold, metadata } = await generate({ ...fixture, exportMode: "final", useAuxiliary: true });

    expect(metadata.spec.exportMode).toBe("final");
    expect(fold.edges_assignment).toContain("F");
    expect(fold.edges_bpRole).toHaveLength(fold.edges_vertices.length);
    expect(fold.edges_bpStudioSource).toHaveLength(fold.edges_vertices.length);
    expect(fold.edges_bpStudioSource.every((source) => typeof source.kind === "string")).toBe(true);
    expect(metadata.cp.roleCounts.hinge).toBeGreaterThan(0);
  });

  test("rejects unflapped leaves before BP Studio junction processing", async () => {
    await expect(generate({
      title: "bad dummy root",
      sheet: { width: 16, height: 16 },
      tree: {
        edges: [
          { n1: 0, n2: 1, length: 4 },
          { n1: 1, n2: 2, length: 4 },
        ],
        flaps: [
          { id: 2, x: 8, y: 8, width: 0, height: 0 },
        ],
      },
    })).rejects.toThrow("missing flap ids: 0");
  });
});
