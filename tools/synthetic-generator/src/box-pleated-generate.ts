// Stage B: read packings from the store, assign M/V, and write box-pleat crease
// patterns as FOLD training data. This is the RE-RUNNABLE step - improving the
// assignment/geometry never requires regenerating packings (Stage A owns those).
//
//   bun run src/box-pleated-generate.ts --out data/output/box-pleated
//   bun run src/box-pleated-generate.ts --out … --double-fraction 0.5 --workers 8 --limit 5000
//
// Runs a Bun worker pool over the stored seeds (default = CPU count). Each worker
// assigns its stride and writes FOLDs + a manifest shard; the main thread merges
// shards into manifest.jsonl and writes summary.json. Doubling (--double-fraction)
// and rescue-fallback (retry a 1x-invalid packing at 2x) happen in the workers.
// Store: $BP_PACKING_STORE.

import { mkdir, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { PACKING_STORE, parseScaleMix, type ScaleMix } from "./box-pleated-store.ts";

interface Args {
  store: string;
  out: string;
  doubleFraction: number;
  scaleMix: ScaleMix | null;
  scaleMixSpec: string | null;
  limit: number | null;
  workers: number;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const limitRaw = get("--limit");
  const scaleMixSpec = get("--scale-mix") ?? null;
  return {
    store: get("--store") ?? PACKING_STORE,
    out: get("--out") ?? "data/output/box-pleated",
    doubleFraction: Number(get("--double-fraction") ?? 0.5),
    scaleMix: scaleMixSpec ? parseScaleMix(scaleMixSpec) : null,
    scaleMixSpec,
    limit: limitRaw !== undefined ? Number(limitRaw) : null,
    workers: Number(get("--workers") ?? navigator.hardwareConcurrency ?? 8),
  };
}

interface Stats {
  accepted: number;
  doubled: number;
  rescued: number;
  clean: number;
  incomplete: number;
  offgrid: number;
  thrown: number;
  hist: Record<number, number>;
  scaleCounts: Record<number, number>;
}
const zero = (): Stats => ({ accepted: 0, doubled: 0, rescued: 0, clean: 0, incomplete: 0, offgrid: 0, thrown: 0, hist: {}, scaleCounts: {} });
function add(a: Stats, b: Stats): Stats {
  const hist = { ...a.hist };
  for (const [k, v] of Object.entries(b.hist)) hist[Number(k)] = (hist[Number(k)] ?? 0) + v;
  const scaleCounts = { ...a.scaleCounts };
  for (const [k, v] of Object.entries(b.scaleCounts ?? {})) scaleCounts[Number(k)] = (scaleCounts[Number(k)] ?? 0) + v;
  return { accepted: a.accepted + b.accepted, doubled: a.doubled + b.doubled, rescued: a.rescued + b.rescued, clean: a.clean + b.clean, incomplete: a.incomplete + b.incomplete, offgrid: a.offgrid + b.offgrid, thrown: a.thrown + b.thrown, hist, scaleCounts };
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  await mkdir(join(args.out, "folds"), { recursive: true });
  await mkdir(join(args.out, "metadata"), { recursive: true });
  const workers = Math.max(1, args.workers);
  console.log(`Stage B: store ${args.store} -> ${args.out} · ${args.scaleMixSpec ? `scale-mix ${args.scaleMixSpec}` : `double-fraction ${args.doubleFraction}`} · ${workers} workers`);

  const perWorkerLimit = args.limit !== null ? Math.ceil(args.limit / workers) : null;
  const latest: Stats[] = Array.from({ length: workers }, zero);
  const t0 = performance.now();
  let done = 0;

  await new Promise<void>((resolve) => {
    for (let w = 0; w < workers; w++) {
      const worker = new Worker(new URL("./box-pleated-generate-worker.ts", import.meta.url).href, { type: "module" });
      worker.onmessage = (e: MessageEvent<Stats & { type: string }>): void => {
        const { type, ...s } = e.data;
        latest[w] = s;
        if (type === "done") {
          worker.terminate();
          if (++done === workers) resolve();
        }
      };
      worker.postMessage({ store: args.store, out: args.out, doubleFraction: args.doubleFraction, scaleMix: args.scaleMix, workers, workerId: w, limit: perWorkerLimit });
    }
    const timer = setInterval(() => {
      const s = latest.reduce(add, zero());
      const secs = (performance.now() - t0) / 1000;
      console.log(`  accepted ${s.accepted} (${s.doubled} doubled, ${s.rescued} rescued) · clean ${s.clean} · ${(s.accepted / secs).toFixed(2)}/s`);
      if (done === workers) clearInterval(timer);
    }, 5000);
  });

  // Merge manifest shards.
  const shards = (await readdir(args.out)).filter((f) => /^manifest\.w\d+\.jsonl$/.test(f)).sort();
  const parts: string[] = [];
  for (const shard of shards) {
    parts.push((await Bun.file(join(args.out, shard)).text()).trimEnd());
    await rm(join(args.out, shard));
  }
  const manifest = parts.filter((p) => p.length).join("\n");
  await Bun.write(join(args.out, "manifest.jsonl"), manifest + (manifest.length ? "\n" : ""));

  const total = latest.reduce(add, zero());
  const secs = (performance.now() - t0) / 1000;
  const summary = {
    accepted: total.accepted,
    doubled: total.doubled,
    rescued: total.rescued,
    doubleFraction: args.doubleFraction,
    clean: total.clean,
    rejections: { incomplete: total.incomplete, "off-grid": total.offgrid, throw: total.thrown },
    conflictHistogram: total.hist,
    scaleCounts: total.scaleCounts,
    seconds: +secs.toFixed(1),
    cpsPerSec: +(total.accepted / Math.max(secs, 1e-6)).toFixed(2),
  };
  await Bun.write(join(args.out, "summary.json"), JSON.stringify(summary));
  console.log(`Stage B done -> ${args.out}`);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
