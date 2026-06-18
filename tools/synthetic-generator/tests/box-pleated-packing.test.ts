import { expect, test } from "bun:test";
import { canLoadBpStudioOptimizer } from "../src/bp-studio-optimizer.ts";
import {
  generateBoxPleatedPacking,
  renderBoxPleatedPackingSvg,
  validateBoxPleatedPacking,
} from "../src/box-pleated-packing.ts";
import {
  completeBoxPleatedCreaseScaffold,
  renderBoxPleatedCreaseScaffoldSvg,
} from "../src/box-pleated-scaffold.ts";

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

bpTest("tight mode optimizes a fixed tree across restarts and keeps the tightest layout", async () => {
  // Tight mode mirrors the app's "Optimize Layout": basin-hopping restarts of a
  // single tree, keeping the smallest sheet. Pythagorean stretch devices are
  // allowed, so the result should be markedly tighter than the stretch-free
  // banded layout for the same leaf count.
  const tight = await generateBoxPleatedPacking({
    id: "bp-tight",
    seed: 4242,
    numCreases: 320,
    bucket: "tight",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    targetLeafCount: 8,
    tight: true,
    tightRestarts: 8,
  });

  expect(validateBoxPleatedPacking(tight)).toEqual([]);
  expect(tight.stats.leaves).toBe(8);
  expect(tight.layout.patternNotFound).toBe(false);
  expect(tight.constraints.noStretches).toBe(false);
  // Deterministic by seed.
  const again = await generateBoxPleatedPacking({
    id: "bp-tight",
    seed: 4242,
    numCreases: 320,
    bucket: "tight",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    targetLeafCount: 8,
    tight: true,
    tightRestarts: 8,
  });
  expect(again.sheet.width).toBe(tight.sheet.width);

  // Far tighter than the stretch-free banded layout for the same leaf count.
  const banded = await generateBoxPleatedPacking({
    id: "bp-banded",
    seed: 4242,
    numCreases: 320,
    bucket: "no-stretch",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 24,
    targetLeafCount: 8,
    noStretches: true,
  });
  expect(tight.sheet.width).toBeLessThan(banded.sheet.width);
});

bpTest("asymmetric no-stretch packings are grid-native and need no rejection cliff", async () => {
  // Regression: the symmetry="none" no-stretch path used to flip each flap
  // between horizontal/vertical placement at random, scattering anchors off any
  // common lattice so BP Studio almost always inserted stretch devices. The
  // grid-native banded placement should produce a clean box-pleated packing on
  // the first few attempts for a range of leaf counts.
  for (const targetLeafCount of [6, 8, 10]) {
    const packing = await generateBoxPleatedPacking({
      id: `bp-none-no-stretch-${targetLeafCount}`,
      seed: 24680,
      numCreases: 320,
      bucket: "no-stretch",
      symmetry: "none",
      bpStudioRoot: BP_STUDIO_ROOT,
      maxAttempts: 6,
      targetLeafCount,
      noStretches: true,
    });

    expect(validateBoxPleatedPacking(packing)).toEqual([]);
    expect(packing.symmetry).toBe("none");
    expect(packing.stats.leaves).toBe(targetLeafCount);
    expect(packing.stats.stretchDevices).toBe(0);
    expect(packing.stats.offGridRidgeCreases).toBe(0);
    expect(packing.layout.patternNotFound).toBe(false);
  }
});

bpTest("crease scaffold fills remaining empty cells with unassigned ridge candidates", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-scaffold",
    seed: 500000,
    numCreases: 300,
    bucket: "scaffold",
    symmetry: "horizontal",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 24,
    targetLeafCount: 6,
    noStretches: true,
  });

  const scaffold = completeBoxPleatedCreaseScaffold(packing);
  expect(scaffold.schemaVersion).toBe("box-pleated-crease-scaffold/v1");
  expect(scaffold.stats.bpRidges).toBeGreaterThan(0);
  expect(scaffold.stats.bpRidges).toBeLessThanOrEqual(packing.stats.ridgeCreases);
  expect(scaffold.stats.computedAxials).toBeGreaterThan(0);
  expect(scaffold.stats.gapRidges).toBeGreaterThan(0);
  expect(scaffold.stats.unfilledGapCells).toBe(0);
  for (const line of scaffold.lines.filter((candidate) => candidate.kind === "gap-ridge")) {
    const dx = Math.abs(line.line[1].x - line.line[0].x);
    const dy = Math.abs(line.line[1].y - line.line[0].y);
    expect(dx).toBe(dy);
  }
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
  // Contour boxes (flap fold-region boxes + river contours, i.e. FOLD aux
  // lines) only render in the contour-enabled view.
  expect(svg).not.toContain("amber flap boxes");
  expect(debugSvg).toContain("amber flap boxes (aux lines)");
  expect(debugSvg).toContain("cyan river contours (aux lines)");
  expect(debugSvg).toContain("flap-box");
  expect(debugSvg).toContain("river-box");
  expect(svg).not.toContain("fabricated filler");
});

bpTest("crease scaffold SVG renderer labels inferred geometry", async () => {
  const packing = await generateBoxPleatedPacking({
    id: "bp-scaffold-svg",
    seed: 9191,
    numCreases: 180,
    bucket: "small",
    symmetry: "none",
    bpStudioRoot: BP_STUDIO_ROOT,
    maxAttempts: 8,
    targetLeafCount: 4,
  });

  const scaffold = completeBoxPleatedCreaseScaffold(packing);
  const svg = renderBoxPleatedCreaseScaffoldSvg(packing, scaffold, { cellSize: 10 });
  expect(svg).toContain("<svg");
  expect(svg).toContain("gap-fill ridge candidates");
  expect(svg).toContain("computed axial candidates");
  expect(svg).toContain("unassigned");
});
