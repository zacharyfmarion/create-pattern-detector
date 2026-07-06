// Stage B: read packings from the store, assign M/V, and write box-pleat crease
// patterns as FOLD training data. This is the RE-RUNNABLE step - improving the
// assignment/geometry never requires regenerating packings (Stage A owns those).
//
//   bun run src/box-pleated-generate.ts --out data/output/box-pleated
//   bun run src/box-pleated-generate.ts --out … --double-fraction 0.5 --limit 5000
//
// For each stored packing it deterministically decides (by seed hash) whether to
// emit a DOUBLED variant (--double-fraction, default 0.5) - the grid-doubling lever
// that pushes a packing up the complexity axis while staying valid. Valid CPs go to
// <out>/folds/<id>.fold (id gains a -2x suffix when doubled) with per-CP quality in
// the manifest; invalid ones are counted by reason. Store: $BP_PACKING_STORE.

import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { buildPackingCP } from "./box-pleated-cp.ts";
import { boxPleatedCpToFold, boxPleatedQuality } from "./box-pleated-fold.ts";
import { PACKING_STORE, leafCountForSeed, readPacking, scalePacking, shouldDouble, storedSeeds } from "./box-pleated-store.ts";
import type { EdgeAssignment } from "./types.ts";

interface Args {
  store: string;
  out: string;
  doubleFraction: number;
  limit: number | null;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const limitRaw = get("--limit");
  return {
    store: get("--store") ?? PACKING_STORE,
    out: get("--out") ?? "data/output/box-pleated",
    doubleFraction: Number(get("--double-fraction") ?? 0.5),
    limit: limitRaw !== undefined ? Number(limitRaw) : null,
  };
}

const splitForIndex = (i: number): "train" | "val" | "test" => {
  const r = i % 20;
  return r < 17 ? "train" : r < 19 ? "val" : "test";
};

async function writeJson(path: string, value: unknown): Promise<void> {
  await Bun.write(path, JSON.stringify(value));
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  console.log(`Stage B: store ${args.store} -> ${args.out} (double-fraction ${args.doubleFraction})`);
  await mkdir(join(args.out, "folds"), { recursive: true });
  await mkdir(join(args.out, "metadata"), { recursive: true });

  const seeds: number[] = [];
  for await (const s of storedSeeds(args.store)) seeds.push(s);
  seeds.sort((a, b) => a - b);
  if (seeds.length === 0) {
    console.log("No packings in the store. Run box-pleated-pack first.");
    return;
  }

  const manifest: string[] = [];
  const rejections: Record<string, number> = { incomplete: 0, "off-grid": 0, throw: 0 };
  const conflictHist: Record<number, number> = {};
  let accepted = 0;
  let doubled = 0;
  let clean = 0;
  const t0 = performance.now();
  let lastLog = t0;

  for (const seed of seeds) {
    if (args.limit !== null && accepted >= args.limit) break;
    const raw = await readPacking(seed, args.store);
    if (!raw) continue;
    const scale = shouldDouble(seed, args.doubleFraction) ? 2 : 1;
    const packing = scale === 2 ? scalePacking(raw, 2) : raw;

    let cp;
    try {
      cp = buildPackingCP(packing);
    } catch {
      rejections.throw++;
      continue;
    }
    if (!cp.valid) {
      rejections[cp.complete ? "off-grid" : "incomplete"]++;
      continue;
    }

    const id = `bp-${seed}${scale === 2 ? "-2x" : ""}`;
    const fold = boxPleatedCpToFold(cp, { id, seed, leafCount: leafCountForSeed(seed), scale });
    const q = boxPleatedQuality(cp);
    const assignments = fold.edges_assignment.reduce<Record<EdgeAssignment, number>>((acc, a) => {
      acc[a] = (acc[a] ?? 0) + 1;
      return acc;
    }, {} as Record<EdgeAssignment, number>);

    const foldPath = join("folds", `${id}.fold`);
    await writeJson(join(args.out, foldPath), fold);
    await writeJson(join(args.out, "metadata", `${id}.json`), { id, seed, scale, quality: q });

    manifest.push(JSON.stringify({
      id, seed, scale, leafCount: leafCountForSeed(seed),
      split: splitForIndex(accepted), foldPath,
      grid: Math.round(cp.sheet.width),
      vertices: fold.vertices_coords.length,
      edges: fold.edges_vertices.length,
      assignments, quality: q,
    }));
    conflictHist[q.conflicts] = (conflictHist[q.conflicts] ?? 0) + 1;
    accepted++;
    if (scale === 2) doubled++;
    if (q.clean) clean++;

    if (performance.now() - lastLog > 5000) {
      console.log(`  seed ${seed} · accepted ${accepted} (${doubled} doubled) · clean ${clean}`);
      lastLog = performance.now();
    }
  }

  await Bun.write(join(args.out, "manifest.jsonl"), manifest.join("\n") + (manifest.length ? "\n" : ""));
  const secs = (performance.now() - t0) / 1000;
  const summary = {
    storedPackings: seeds.length,
    accepted,
    doubled,
    doubleFraction: args.doubleFraction,
    clean,
    rejections,
    conflictHistogram: conflictHist,
    seconds: +secs.toFixed(1),
    cpsPerSec: +(accepted / Math.max(secs, 1e-6)).toFixed(2),
  };
  await writeJson(join(args.out, "summary.json"), summary);
  console.log(`Stage B done -> ${args.out}`);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
