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
  const tmp = `${path}.${process.pid}-${Math.random().toString(36).slice(2)}.tmp`;
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

/**
 * Scale a packing's geometry by an integer factor (the "grid doubling" lever). All
 * coordinates and lengths multiply by k, so the packing stays valid (integers stay
 * integers) but its molecule fills the finer grid with more pleats - pushing it up
 * the grid/complexity axis for near-free (assignment is budget-bound, ~1x slower).
 */
export function scalePacking(packing: BoxPleatedPacking, k: number): BoxPleatedPacking {
  const sp = <T extends { x: number; y: number }>(p: T): T => ({ ...p, x: p.x * k, y: p.y * k });
  const sl = (l: [{ x: number; y: number }, { x: number; y: number }]): unknown => [sp(l[0]), sp(l[1])];
  return {
    ...packing,
    id: `${packing.id}-${k}x`,
    sheet: { ...packing.sheet, width: packing.sheet.width * k, height: packing.sheet.height * k },
    flaps: packing.flaps.map((f) => ({ ...f, x: f.x * k, y: f.y * k, width: f.width * k, height: f.height * k, radius: f.radius * k })),
    tree: {
      ...packing.tree,
      nodes: packing.tree.nodes.map((n) => {
        const wh = n as unknown as { width?: number; height?: number };
        return { ...n, lengthToParent: n.lengthToParent * k, ...(wh.width !== undefined ? { width: wh.width * k, height: wh.height! * k } : {}) };
      }),
    },
    layout: {
      ...packing.layout,
      objects: packing.layout.objects.map((o) => ({
        ...o,
        ridges: o.ridges.map(sl) as typeof o.ridges,
        axisParallel: (o.axisParallel ?? []).map(sl) as typeof o.axisParallel,
        contours: o.contours.map((c) => ({ outer: c.outer.map(sp), inner: c.inner?.map((ring) => ring.map(sp)) })),
      })),
    },
  };
}

/** Deterministic [0,1) hash of a seed (uncorrelated with seed%3 leaf-count). */
function seedHash(seed: number): number {
  return (Math.imul(seed ^ 0x9e3779b9, 2654435761) >>> 0) / 4294967296;
}

/** Deterministically decide whether to emit this seed's packing doubled (Stage B). */
export function shouldDouble(seed: number, fraction: number): boolean {
  return fraction > 0 && seedHash(seed) < fraction;
}

/** Deterministic train/val/test split by seed (85/10/5), independent of workers. */
export function splitForSeed(seed: number): "train" | "val" | "test" {
  const h = (Math.imul(seed ^ 0x85ebca6b, 0xc2b2ae35) >>> 0) / 4294967296;
  return h < 0.85 ? "train" : h < 0.95 ? "val" : "test";
}

/** All seeds currently in the store (from the sharded file names). */
export async function* storedSeeds(store: string = PACKING_STORE): AsyncGenerator<number> {
  const glob = new Bun.Glob("*/bp-*.json.gz");
  for await (const rel of glob.scan({ cwd: store })) {
    const m = rel.match(/bp-(\d+)\.json\.gz$/);
    if (m) yield Number(m[1]);
  }
}
