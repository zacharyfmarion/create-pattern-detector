// The box-pleat PACKING STORE: the single source of truth for generated packings,
// living outside the work tree so the corpus is generated once and reused. Stage A
// (box-pleated-pack) writes packings here; Stage B (assignment -> FOLD) and the
// debug tools read from here. Packings are the expensive, stable artifact - only
// the ASSIGNMENT downstream keeps changing - so decoupling storage lets assignment
// improve without regenerating packings.
//
// Location: $BP_PACKING_STORE, defaulting to the path below. Each packing is a
// gzipped JSON at <store>/<shard>/<key>.json.gz, sharded by seed/1000 so no
// directory holds a huge number of files.

import { mkdir, rename } from "node:fs/promises";
import { dirname, join } from "node:path";
import { generateBoxPleatedPacking, type BoxPleatedPacking, type BoxPleatedPackingConfig } from "./box-pleated-packing.ts";

/** Root of the packing store. Override with BP_PACKING_STORE. */
export const PACKING_STORE =
  process.env.BP_PACKING_STORE ?? "/Users/zacharymarion/Documents/datasets/create-pattern-detector/bp-packings";

// Fingerprint of the packing GENERATOR (not assignment). Bump only if the tight-pack
// search / packing params change, so a stale corpus is detectable. Assignment changes
// never touch this.
export const PACKER_VERSION = "v1-n300-r14";

/** Standard leaf-count convention shared across the box-pleat tools. */
export const leafCountForSeed = (seed: number): number => [4, 5, 6][seed % 3];

/** The canonical, fully-determined packing config for a seed. */
export function configForSeed(seed: number): BoxPleatedPackingConfig {
  return {
    id: `bp-${seed}`,
    seed,
    numCreases: 300,
    bucket: "s",
    symmetry: "none",
    targetLeafCount: leafCountForSeed(seed),
    tight: true,
    tightRestarts: 14,
  };
}

const shardOf = (seed: number): string => String(Math.floor(seed / 1000));

/** Absolute path of a seed's packing in the store. */
export function packingPath(seed: number, store: string = PACKING_STORE): string {
  return join(store, shardOf(seed), `bp-${seed}.json.gz`);
}

/** Is this seed's packing already stored? */
export async function hasPacking(seed: number, store: string = PACKING_STORE): Promise<boolean> {
  return Bun.file(packingPath(seed, store)).exists();
}

/** Write a packing to the store (gzipped, atomic via temp+rename). */
export async function writePacking(seed: number, packing: BoxPleatedPacking, store: string = PACKING_STORE): Promise<number> {
  const path = packingPath(seed, store);
  await mkdir(dirname(path), { recursive: true });
  const gz = Bun.gzipSync(new TextEncoder().encode(JSON.stringify(packing)), { level: 9 });
  const tmp = `${path}.${process.pid}.tmp`;
  await Bun.write(tmp, gz);
  await rename(tmp, path);
  return gz.byteLength;
}

/** Read a seed's packing from the store, or null if absent. */
export async function readPacking(seed: number, store: string = PACKING_STORE): Promise<BoxPleatedPacking | null> {
  const file = Bun.file(packingPath(seed, store));
  if (!(await file.exists())) return null;
  const gz = new Uint8Array(await file.arrayBuffer());
  return JSON.parse(new TextDecoder().decode(Bun.gunzipSync(gz))) as BoxPleatedPacking;
}

/** Generate a seed's packing (Stage A primitive). */
export async function packForSeed(seed: number): Promise<BoxPleatedPacking> {
  return generateBoxPleatedPacking(configForSeed(seed));
}

/** All seeds currently in the store (from the sharded file names). */
export async function* storedSeeds(store: string = PACKING_STORE): AsyncGenerator<number> {
  const glob = new Bun.Glob("*/bp-*.json.gz");
  for await (const rel of glob.scan({ cwd: store })) {
    const m = rel.match(/bp-(\d+)\.json\.gz$/);
    if (m) yield Number(m[1]);
  }
}
