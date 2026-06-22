import { expect, test } from "bun:test";
import { evenShorterFlap } from "../src/box-pleated-packing.ts";

// A flap whose rectangular cross-section has an odd shorter side >= 1 would put
// its straight-skeleton spine on a half-grid line (a sub-grid crease), which box
// pleating forbids. evenShorterFlap reduces the odd shorter side to the next even
// value so every sampled flap molecule converges on the grid.

const shorter = (d: { width: number; height: number }): number => Math.min(d.width, d.height);

test("odd shorter side is reduced to even", () => {
  expect(evenShorterFlap(1, 2)).toEqual({ width: 0, height: 2 });
  expect(evenShorterFlap(2, 1)).toEqual({ width: 2, height: 0 });
  expect(evenShorterFlap(1, 3)).toEqual({ width: 0, height: 3 });
  expect(evenShorterFlap(3, 3)).toEqual({ width: 2, height: 3 });
  expect(evenShorterFlap(1, 1)).toEqual({ width: 0, height: 1 });
});

test("valid rectangles are left untouched", () => {
  // Even shorter side (or degenerate) - already on-grid.
  for (const [w, h] of [[0, 0], [2, 2], [2, 3], [3, 2], [0, 1], [3, 0], [0, 3], [2, 0]] as const) {
    expect(evenShorterFlap(w, h)).toEqual({ width: w, height: h });
  }
});

test("every sampleable dimension pair yields an even (or zero) shorter side", () => {
  const values = [0, 1, 2, 3];
  for (const w of values) {
    for (const h of values) {
      expect(shorter(evenShorterFlap(w, h)) % 2).toBe(0);
    }
  }
});
