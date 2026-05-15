import { expect, test } from "bun:test";
import { compilerGridSizeForSheet, regularizeBPStudioLayout } from "../src/bp-completion.ts";
import { runBPStudioAdapter, toAdapterSpec } from "../src/bp-studio-realistic.ts";
import { simpleQuadrupedBPStudioSpec } from "../src/bp-studio-fixtures.ts";
import {
  bpStudioPackingCircles,
  validateBPStudioPacking,
  validateBPStudioPackingLayout,
} from "../src/bp-studio-packing-validity.ts";

test("BP Studio optimized simple quadruped packing has no overlapping flap circles", () => {
  const spec = simpleQuadrupedBPStudioSpec(7);
  const adapterSpec = toAdapterSpec(spec);
  adapterSpec.optimizeLayout = true;
  adapterSpec.optimizerLayout = "view";
  adapterSpec.optimizerSeed = 7;
  adapterSpec.optimizerUseBH = true;

  const { metadata } = runBPStudioAdapter(adapterSpec);
  const validation = validateBPStudioPacking(spec, metadata);

  expect(validation.ok).toBe(true);
  expect(validation.errors).toHaveLength(0);
  expect(validation.metrics.circleCount).toBe(6);
  expect(validation.metrics.overlapCount).toBe(0);
  expect(validation.metrics.minGap ?? 0).toBeGreaterThanOrEqual(-1e-8);
  expect(validation.circles.find((circle) => circle.nodeId === "tail")?.radius).toBe(3);
});

test("regularized compiler grid is a multiple of BP Studio optimized sheet units", () => {
  const spec = simpleQuadrupedBPStudioSpec(7);
  const adapterSpec = toAdapterSpec(spec);
  adapterSpec.optimizeLayout = true;
  adapterSpec.optimizerLayout = "view";
  adapterSpec.optimizerSeed = 7;
  adapterSpec.optimizerUseBH = true;

  const { metadata } = runBPStudioAdapter(adapterSpec);
  const layout = regularizeBPStudioLayout(spec, { adapterSpec, adapterMetadata: metadata });

  expect(compilerGridSizeForSheet(7, 7)).toBe(28);
  expect(layout.gridSize).toBe(28);
  expect(layout.terminals.find((terminal) => terminal.id === "head")).toMatchObject({
    x: 4 / 7,
    y: 6 / 7,
  });
  expect(layout.terminals.find((terminal) => terminal.id === "tail")).toMatchObject({
    x: 3 / 7,
    y: 0,
  });
});

test("packing validity rejects overlapping flap allocation circles", () => {
  const spec = simpleQuadrupedBPStudioSpec(11);
  const validation = validateBPStudioPackingLayout(spec, {
    sheet: { width: 4, height: 4 },
    flaps: [
      { id: 2, x: 1, y: 1, width: 0, height: 0 },
      { id: 4, x: 1.5, y: 1, width: 0, height: 0 },
    ],
  });

  expect(validation.ok).toBe(false);
  expect(validation.metrics.circleCount).toBe(2);
  expect(validation.metrics.overlapCount).toBe(1);
  expect(validation.errors[0]).toStartWith("flap-circle-overlap:head:front-left-leg");
});

test("packing validity reports boundary overflow separately from overlap", () => {
  const spec = simpleQuadrupedBPStudioSpec(13);
  const validation = validateBPStudioPackingLayout(spec, {
    sheet: { width: 7, height: 7 },
    flaps: [
      { id: 2, x: 0, y: 6, width: 0, height: 0 },
    ],
  });

  expect(validation.ok).toBe(true);
  expect(validation.metrics.outsideCount).toBe(1);
  expect(validation.warnings[0]).toStartWith("flap-circle-outside-sheet:head");
});

test("packing circles use tree edge lengths instead of rendered point size", () => {
  const spec = simpleQuadrupedBPStudioSpec(17);
  const circles = bpStudioPackingCircles(spec, {
    sheet: { width: 7, height: 7 },
    flaps: [
      { id: 3, x: 3, y: 0, width: 0, height: 0 },
    ],
  });

  expect(circles).toHaveLength(1);
  expect(circles[0]).toMatchObject({
    nodeId: "tail",
    label: "tail",
    radius: 3,
    radiusSource: "tree-edge-length",
  });
});
