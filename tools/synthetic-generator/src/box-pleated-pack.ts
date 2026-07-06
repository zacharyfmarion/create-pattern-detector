// Stage A: generate box-pleat PACKINGS into the store (the expensive, stable
// artifact). Deterministic per seed, resumable (skips packings already stored),
// and crash-safe (atomic writes). Run it in the background to build the corpus;
// Stage B (assignment -> FOLD) reads from the store and can be re-run freely as
// the assignment logic improves, without regenerating packings.
//
//   bun run src/box-pleated-pack.ts --from 0 --to 100000          (range, parallel)
//   bun run src/box-pleated-pack.ts --from 0 --to 100000 --workers 4
//   bun run src/box-pleated-pack.ts --from 0 --count 10000        (incremental, single)
//
// Range mode (--to) runs a Bun worker pool (default = CPU count); --count mode is
// single-threaded (its running "N new" target needs no cross-worker coordination).
// Store: $BP_PACKING_STORE (see box-pleated-store.ts for the default).

import { PACKING_STORE, hasPacking, packForSeed, writePacking } from "./box-pleated-store.ts";

interface Args {
  from: number;
  count: number;
  to: number | null;
  workers: number;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const toRaw = get("--to");
  return {
    from: Number(get("--from") ?? 0),
    count: Number(get("--count") ?? 1000),
    to: toRaw !== undefined ? Number(toRaw) : null,
    workers: Number(get("--workers") ?? navigator.hardwareConcurrency ?? 8),
  };
}

interface Stats {
  generated: number;
  skipped: number;
  errors: number;
  bytes: number;
}

function report(prefix: string, s: Stats, secs: number): void {
  console.log(
    `${prefix} generated ${s.generated}, skipped ${s.skipped}, errors ${s.errors} in ${secs.toFixed(0)}s ` +
      `(${(s.generated / Math.max(secs, 1e-6)).toFixed(2)}/s), ${(s.bytes / 1e6).toFixed(1)}MB ` +
      `(${s.generated ? (s.bytes / s.generated / 1024).toFixed(1) : 0}KB/packing avg).`,
  );
}

/** Single-threaded incremental generation: N new packings from --from. */
async function runSingle(args: Args): Promise<void> {
  const s: Stats = { generated: 0, skipped: 0, errors: 0, bytes: 0 };
  const t0 = performance.now();
  let lastLog = t0;
  for (let seed = args.from; s.generated < args.count; seed++) {
    if (await hasPacking(seed)) {
      s.skipped++;
      continue;
    }
    try {
      s.bytes += await writePacking(seed, await packForSeed(seed));
      s.generated++;
    } catch (e) {
      s.errors++;
      console.error(`seed ${seed}: ${(e as Error).message}`);
    }
    if (performance.now() - lastLog > 5000) {
      console.log(`  seed ${seed} · generated ${s.generated} · skipped ${s.skipped} · errors ${s.errors}`);
      lastLog = performance.now();
    }
  }
  report("\nDONE:", s, (performance.now() - t0) / 1000);
}

/** Parallel range generation across a Bun worker pool. */
async function runParallel(args: Args): Promise<void> {
  const to = args.to!;
  const workers = Math.max(1, Math.min(args.workers, to - args.from));
  console.log(`workers: ${workers}`);
  const latest: Stats[] = Array.from({ length: workers }, () => ({ generated: 0, skipped: 0, errors: 0, bytes: 0 }));
  const t0 = performance.now();
  let done = 0;

  await new Promise<void>((resolve) => {
    const pool = Array.from({ length: workers }, (_, w) => {
      const worker = new Worker(new URL("./box-pleated-pack-worker.ts", import.meta.url).href, { type: "module" });
      worker.onmessage = (e: MessageEvent<Stats & { type: string }>): void => {
        latest[w] = e.data;
        if (e.data.type === "done") {
          worker.terminate();
          if (++done === workers) resolve();
        }
      };
      worker.postMessage({ from: args.from, to, workers, workerId: w });
      return worker;
    });

    const timer = setInterval(() => {
      const sum = latest.reduce<Stats>((a, x) => ({ generated: a.generated + x.generated, skipped: a.skipped + x.skipped, errors: a.errors + x.errors, bytes: a.bytes + x.bytes }), { generated: 0, skipped: 0, errors: 0, bytes: 0 });
      const secs = (performance.now() - t0) / 1000;
      console.log(`  generated ${sum.generated} · skipped ${sum.skipped} · errors ${sum.errors} · ${(sum.generated / secs).toFixed(1)}/s · ${(sum.bytes / 1e6).toFixed(1)}MB`);
      if (done === workers) clearInterval(timer);
    }, 5000);
    void pool;
  });

  const total = latest.reduce<Stats>((a, x) => ({ generated: a.generated + x.generated, skipped: a.skipped + x.skipped, errors: a.errors + x.errors, bytes: a.bytes + x.bytes }), { generated: 0, skipped: 0, errors: 0, bytes: 0 });
  report("\nDONE:", total, (performance.now() - t0) / 1000);
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  console.log(`Stage A: packing store -> ${PACKING_STORE}`);
  if (args.to !== null && args.workers > 1) {
    console.log(`range [${args.from}, ${args.to})`);
    await runParallel(args);
  } else {
    console.log(`from seed ${args.from}, up to ${args.count} new packings (single-threaded)`);
    await runSingle(args);
  }
}

await main();
