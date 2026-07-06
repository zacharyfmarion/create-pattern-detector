// Batch generator for box-pleat crease-pattern training data.
//
//   bun run src/box-pleated-generate.ts --count 1000 --out data/output/box-pleated [--seed 60000]
//
// Deterministically walks seeds from --seed, builds each packing's CP, and for
// every VALID one writes a normalized FOLD to <out>/folds/<id>.fold plus a
// per-CP metadata JSON, appending a row to <out>/manifest.jsonl. Invalid
// candidates are counted by rejection reason (incomplete / off-grid). A run
// summary (yield, rejection breakdown, quality distribution) is written to
// <out>/summary.json. Crash-safe: a throwing candidate is counted and skipped.

import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { generateBoxPleatedPacking, type BoxPleatedPackingConfig } from "./box-pleated-packing.ts";
import { buildPackingCP } from "./box-pleated-cp.ts";
import { boxPleatedCpToFold, boxPleatedQuality } from "./box-pleated-fold.ts";
import type { EdgeAssignment } from "./types.ts";

interface Args {
  count: number;
  out: string;
  seed: number;
  maxAttempts: number;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const count = Number(get("--count") ?? 100);
  return {
    count,
    out: get("--out") ?? "data/output/box-pleated",
    seed: Number(get("--seed") ?? 60000),
    maxAttempts: Number(get("--max-attempts") ?? count * 20),
  };
}

/** Standard leaf-count convention (shared with the debug tooling). */
const leafCountForSeed = (seed: number): number => [4, 5, 6][seed % 3];

const configForSeed = (seed: number): BoxPleatedPackingConfig => ({
  id: `bp-${seed}`,
  seed,
  numCreases: 300,
  bucket: "s",
  symmetry: "none",
  targetLeafCount: leafCountForSeed(seed),
  tight: true,
  tightRestarts: 14,
});

const splitForIndex = (i: number): "train" | "val" | "test" => {
  const r = i % 20;
  return r < 17 ? "train" : r < 19 ? "val" : "test";
};

async function writeJson(path: string, value: unknown): Promise<void> {
  await Bun.write(path, JSON.stringify(value));
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  await mkdir(join(args.out, "folds"), { recursive: true });
  await mkdir(join(args.out, "metadata"), { recursive: true });

  const manifest: string[] = [];
  const rejections: Record<string, number> = { incomplete: 0, "off-grid": 0, throw: 0 };
  let accepted = 0;
  let clean = 0;
  let attempts = 0;
  const conflictHist: Record<number, number> = {};

  const t0 = performance.now();
  for (let seed = args.seed; accepted < args.count && attempts < args.seed + args.maxAttempts; seed++) {
    attempts++;
    const config = configForSeed(seed);
    let cp;
    try {
      cp = buildPackingCP(await generateBoxPleatedPacking(config));
    } catch (e) {
      rejections.throw++;
      continue;
    }
    if (!cp.valid) {
      rejections[cp.complete ? "off-grid" : "incomplete"]++;
      continue;
    }

    const id = config.id;
    const fold = boxPleatedCpToFold(cp, { id, seed, leafCount: config.targetLeafCount ?? leafCountForSeed(seed) });
    const q = boxPleatedQuality(cp);
    const assignments = fold.edges_assignment.reduce<Record<EdgeAssignment, number>>((acc, a) => {
      acc[a] = (acc[a] ?? 0) + 1;
      return acc;
    }, {} as Record<EdgeAssignment, number>);

    const foldPath = join("folds", `${id}.fold`);
    await writeJson(join(args.out, foldPath), fold);
    await writeJson(join(args.out, "metadata", `${id}.json`), { id, seed, config, quality: q });

    manifest.push(JSON.stringify({
      id,
      seed,
      leafCount: config.targetLeafCount,
      split: splitForIndex(accepted),
      foldPath,
      vertices: fold.vertices_coords.length,
      edges: fold.edges_vertices.length,
      assignments,
      quality: q,
    }));
    conflictHist[q.conflicts] = (conflictHist[q.conflicts] ?? 0) + 1;
    accepted++;
    if (q.clean) clean++;
  }

  await Bun.write(join(args.out, "manifest.jsonl"), manifest.join("\n") + (manifest.length ? "\n" : ""));
  const ms = performance.now() - t0;
  const summary = {
    requested: args.count,
    accepted,
    clean,
    attempts,
    yield: attempts ? +(accepted / attempts).toFixed(3) : 0,
    rejections,
    conflictHistogram: conflictHist,
    seconds: +(ms / 1000).toFixed(1),
    cpsPerSec: +(accepted / (ms / 1000)).toFixed(2),
  };
  await writeJson(join(args.out, "summary.json"), summary);
  console.log(`box-pleat generation -> ${args.out}`);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
