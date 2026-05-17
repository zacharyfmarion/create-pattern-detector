export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    // Mulberry32: compact, deterministic, and good enough for dataset recipes.
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(minInclusive: number, maxInclusive: number): number {
    return Math.floor(this.next() * (maxInclusive - minInclusive + 1)) + minInclusive;
  }

  float(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  choice<T>(items: readonly T[]): T {
    if (items.length === 0) {
      throw new Error("Cannot choose from an empty array");
    }
    return items[this.int(0, items.length - 1)];
  }

  weightedChoice<T extends string>(weights: Record<T, number>): T {
    const entries = Object.entries(weights).filter(([, weight]) => Number(weight) > 0) as [T, number][];
    const total = entries.reduce((sum, [, weight]) => sum + weight, 0);
    if (total <= 0) {
      throw new Error("Weighted choice requires at least one positive weight");
    }
    let draw = this.next() * total;
    for (const [key, weight] of entries) {
      draw -= weight;
      if (draw <= 0) return key;
    }
    return entries[entries.length - 1][0];
  }
}
