// Disk cache for generated box-pleat packings. Packing generation (the tight-pack
// search) costs ~700ms/seed and is fully deterministic in (seed, params), while the
// packing itself is plain JSON-serialisable data. Caching it makes re-running the
// debug/render tools on the same seeds near-instant (buildPackingMolecule is ~30ms).
//
// Cache dir defaults to /tmp/bp-packing-cache; override with BP_CACHE_DIR.

import { generateBoxPleatedPacking, type BoxPleatedPacking } from "./box-pleated-packing.ts";

const CACHE_DIR = process.env.BP_CACHE_DIR ?? "/tmp/bp-packing-cache";

export interface SeedOpts {
  numCreases?: number;
  tightRestarts?: number;
}

/** Leaf-count convention shared across the box-pleat debug tools. */
export function leafCountForSeed(seed: number): number {
  return [4, 5, 6][seed % 3];
}

/** Generate the packing for a seed, or load it from the disk cache if present. */
export async function cachedPacking(seed: number, opts: SeedOpts = {}): Promise<BoxPleatedPacking> {
  const numCreases = opts.numCreases ?? 300;
  const tightRestarts = opts.tightRestarts ?? 14;
  const leafCount = leafCountForSeed(seed);
  const file = `${CACHE_DIR}/${seed}_n${numCreases}_r${tightRestarts}_l${leafCount}.json`;

  const cached = Bun.file(file);
  if (await cached.exists()) {
    try {
      return JSON.parse(await cached.text()) as BoxPleatedPacking;
    } catch {
      // fall through and regenerate on a corrupt cache entry
    }
  }

  const packing = await generateBoxPleatedPacking({
    id: `s${seed}`, seed, numCreases, bucket: "s", symmetry: "none",
    targetLeafCount: leafCount, tight: true, tightRestarts,
  });
  await Bun.write(file, JSON.stringify(packing));
  return packing;
}
