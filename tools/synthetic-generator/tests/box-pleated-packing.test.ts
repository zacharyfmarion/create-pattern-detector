import { expect, test } from "bun:test";
import { canLoadBpStudioOptimizer } from "../src/bp-studio-optimizer.ts";
import {
  generateBoxPleatedPacking,
  renderBoxPleatedPackingSvg,
  validateBoxPleatedPacking,
} from "../src/box-pleated-packing.ts";

const BP_STUDIO_ROOT = process.env.BP_STUDIO_ROOT ?? "/tmp/bp-studio-source";
const bpOptimizerAvailable = await canLoadBpStudioOptimizer(BP_STUDIO_ROOT);
const bpTest = bpOptimizerAvailable ? test : test.skip;

bpTest("box-pleated packing generation is deterministic by seed", async () => {
  const config = {
    id: "bp-deterministic",
    seed: 4242,
    numCreases: 260,
    bucket: "medium",
    symmetry: "none" as const,
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  };

  const first = await generateBoxPleatedPacking(config);
  const second = await generateBoxPleatedPacking(config);

  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(validateBoxPleatedPacking(first)).toEqual([]);
  expect(first.schemaVersion).toBe("box-pleated-packing/v3");
  expect(first.optimizer.source).toBe("bp-studio");
  expect(first.layout.source).toBe("bp-studio-core");
  expect(first.layout.patternNotFound).toBe(false);
});

bpTest("generated packings include BP Studio layout primitives", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-layout",
    seed: 13013,
    numCreases: 180,
    bucket: "small",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  });

  expect(validateBoxPleatedPacking(packing)).toEqual([]);
  expect(packing.layout.objects.some((object) => object.kind === "flap")).toBe(true);
  expect(packing.layout.objects.some((object) => object.kind === "river")).toBe(true);
  expect(packing.stats.hingeContours).toBeGreaterThan(0);
  expect(packing.stats.ridgeCreases).toBeGreaterThan(0);
});

bpTest("optimizer flap positions satisfy tree-distance constraints", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-constraints",
    seed: 5151,
    numCreases: 260,
    bucket: "medium",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  });

  expect(validateBoxPleatedPacking(packing)).toEqual([]);
  for (const flap of packing.flaps) {
    expect(flap.x).toBeGreaterThanOrEqual(0);
    expect(flap.y).toBeGreaterThanOrEqual(0);
    expect(flap.x).toBeLessThanOrEqual(packing.sheet.width);
    expect(flap.y).toBeLessThanOrEqual(packing.sheet.height);
  }
});

bpTest("validation rejects overlapping flap radius outlines", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-overlap-check",
    seed: 7171,
    numCreases: 180,
    bucket: "small",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  });

  packing.flaps[1] = {
    ...packing.flaps[1],
    x: packing.flaps[0].x,
    y: packing.flaps[0].y,
  };

  expect(validateBoxPleatedPacking(packing).some((error) => error.includes("flap radius outlines"))).toBe(true);
});

bpTest("no-stretch mode rejects BP Studio stretch devices and off-grid ridges", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-no-stretch",
    seed: 500000,
    numCreases: 300,
    bucket: "no-stretch",
    symmetry: "horizontal",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 24,
    targetLeafCount: 6,
    noStretches: true,
  });

  expect(validateBoxPleatedPacking(packing)).toEqual([]);
  expect(packing.constraints.noStretches).toBe(true);
  expect(packing.stats.stretchDevices).toBe(0);
  expect(packing.stats.axisParallelCreases).toBe(0);
  expect(packing.stats.offGridRidgeCreases).toBe(0);
  expect(packing.layout.objects.some((object) => object.kind === "stretch-device")).toBe(false);
});

bpTest("packing SVG renderer draws BP Studio layers", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-svg",
    seed: 9191,
    numCreases: 180,
    bucket: "small",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  });

  const svg = renderBoxPleatedPackingSvg(packing, { cellSize: 10 });
  const debugSvg = renderBoxPleatedPackingSvg(packing, { cellSize: 10, includeLayoutContours: true });
  expect(svg).toContain("<svg");
  expect(svg).toContain("bp-svg");
  expect(svg).toContain("blue flap radius outlines");
  expect(svg).toContain("red ridge creases");
  expect(svg).toContain("green stretch lines");
  expect(svg).not.toContain("blue BP layout contours");
  expect(debugSvg).toContain("blue BP layout contours");
  expect(svg).not.toContain("fabricated filler");
});
