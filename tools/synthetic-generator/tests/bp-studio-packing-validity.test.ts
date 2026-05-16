import { expect, test } from "bun:test";
import { compilerGridSizeForSheet, regularizeBPStudioLayout } from "../src/bp-completion.ts";
import { compileRegionCandidate, regionLayoutFromCompletionLayout } from "../src/bp-region-compiler.ts";
import { buildBPStudioLayoutGraph } from "../src/bp-studio-layout-graph.ts";
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
    sourceContour: {
      x1: 3 / 7,
      y1: 5 / 7,
      x2: 5 / 7,
      y2: 1,
      source: "bp-studio-final-contour",
    },
  });
  expect(layout.terminals.find((terminal) => terminal.id === "tail")).toMatchObject({
    x: 3 / 7,
    y: 0,
    sourceContour: {
      x1: 0,
      y1: 0,
      x2: 6 / 7,
      y2: 3 / 7,
      source: "bp-studio-final-contour",
    },
  });
  expect(layout.regions.find((body) => body.id === "front-hub")?.x1).not.toBe(15 / 32);
});

test("region layout carries BP Studio terminal contour bounds as source metadata", () => {
  const spec = simpleQuadrupedBPStudioSpec(7);
  const adapterSpec = toAdapterSpec(spec);
  adapterSpec.optimizeLayout = true;
  adapterSpec.optimizerLayout = "view";
  adapterSpec.optimizerSeed = 7;
  adapterSpec.optimizerUseBH = true;

  const { metadata } = runBPStudioAdapter(adapterSpec);
  const completionLayout = regularizeBPStudioLayout(spec, { adapterSpec, adapterMetadata: metadata });
  const regionLayout = regionLayoutFromCompletionLayout(completionLayout);
  const head = regionLayout.flaps.find((flap) => flap.terminalId === "head");

  expect(head?.sourceContourRect?.x1).toBeCloseTo(3 / 7, 8);
  expect(head?.sourceContourRect?.y1).toBeCloseTo(5 / 7, 8);
  expect(head?.sourceContourRect?.x2).toBeCloseTo(5 / 7, 8);
  expect(head?.sourceContourRect?.y2).toBe(1);
  expect(head?.rect).not.toEqual(head?.sourceContourRect);
});

test("regularized simple quadruped uses tangent lanes and downgrades inferred-body overlap", () => {
  const spec = simpleQuadrupedBPStudioSpec(7);
  const adapterSpec = toAdapterSpec(spec);
  adapterSpec.optimizeLayout = true;
  adapterSpec.optimizerLayout = "view";
  adapterSpec.optimizerSeed = 7;
  adapterSpec.optimizerUseBH = true;

  const { metadata } = runBPStudioAdapter(adapterSpec);
  const completionLayout = regularizeBPStudioLayout(spec, { adapterSpec, adapterMetadata: metadata });
  const candidate = compileRegionCandidate(regionLayoutFromCompletionLayout(completionLayout));

  expect(candidate.validity).toBe("layout-valid");
  expect(candidate.localProbe?.locallyFlatFoldable).toBe(false);
  expect(candidate.localProbe?.kawasakiBad).toBe(0);
  expect(candidate.segments.filter((segment) => segment.kind === "turn-closure").length).toBeGreaterThan(0);
  expect(candidate.rejectionReasons.filter((reason) => reason.startsWith("flap-allocation-overlap"))).toHaveLength(0);
  expect(candidate.rejectionReasons.filter((reason) => reason.startsWith("pleat-strip-body-overlap"))).toHaveLength(0);
  expect(candidate.warnings?.filter((reason) => reason.startsWith("inferred-pleat-strip-body-overlap"))).not.toHaveLength(0);
  expect(candidate.layout.pleatStrips.length).toBeGreaterThan(7);
  expect(candidate.layout.pleatStrips.every((strip) =>
    (strip.rect.x2 - strip.rect.x1) > 0 && (strip.rect.y2 - strip.rect.y1) > 0
  )).toBe(true);
});

test("BP Studio layout graph derives internal hubs from optimized BP graph only", () => {
  const spec = simpleQuadrupedBPStudioSpec(7);
  const adapterSpec = toAdapterSpec(spec);
  adapterSpec.optimizeLayout = true;
  adapterSpec.optimizerLayout = "view";
  adapterSpec.optimizerSeed = 7;
  adapterSpec.optimizerUseBH = true;

  const { metadata } = runBPStudioAdapter(adapterSpec);
  const graph = buildBPStudioLayoutGraph(spec, { adapterSpec, adapterMetadata: metadata });
  const frontHub = graph?.nodes.find((node) => node.nodeId === "front-hub");
  const head = graph?.nodes.find((node) => node.nodeId === "head");

  expect(graph?.sheet).toEqual({ width: 7, height: 7 });
  expect(frontHub?.source).toBe("bp-studio-inferred-internal");
  expect(frontHub?.point.x).toBeGreaterThan(0);
  expect(frontHub?.point.y).toBeGreaterThan(0);
  expect(head).toMatchObject({
    source: "bp-studio-optimized-flap",
    point: { x: 4, y: 6 },
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
