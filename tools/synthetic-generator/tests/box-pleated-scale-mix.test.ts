import { describe, expect, test } from "bun:test";
import { parseScaleMix, scaleForSeed } from "../src/box-pleated-store";

describe("scale mix", () => {
  test("parses a spec and rejects invalid entries", () => {
    expect(parseScaleMix("1:0.25,2:0.35,4:0.4")).toEqual([
      { scale: 1, weight: 0.25 },
      { scale: 2, weight: 0.35 },
      { scale: 4, weight: 0.4 },
    ]);
    // 3x preserves gap parity (odd stays odd) and probed invalid on every
    // packing — deliberately unrepresentable.
    expect(() => parseScaleMix("3:1")).toThrow();
    expect(() => parseScaleMix("2:-1")).toThrow();
    expect(() => parseScaleMix("2:0")).toThrow(); // no positive weight
  });

  test("scaleForSeed is deterministic and roughly proportional", () => {
    const mix = parseScaleMix("1:0.25,2:0.35,4:0.4");
    const counts: Record<number, number> = { 1: 0, 2: 0, 4: 0 };
    for (let seed = 0; seed < 20000; seed++) {
      const a = scaleForSeed(seed, mix);
      expect(scaleForSeed(seed, mix)).toBe(a); // deterministic
      counts[a]!++;
    }
    expect(counts[1]! / 20000).toBeGreaterThan(0.22);
    expect(counts[1]! / 20000).toBeLessThan(0.28);
    expect(counts[2]! / 20000).toBeGreaterThan(0.32);
    expect(counts[2]! / 20000).toBeLessThan(0.38);
    expect(counts[4]! / 20000).toBeGreaterThan(0.37);
    expect(counts[4]! / 20000).toBeLessThan(0.43);
  });
});
