// Stage A: generate box-pleat PACKINGS into the store (the expensive, stable
// artifact). Deterministic per seed, resumable (skips packings already stored),
// and crash-safe (atomic writes). Run it in the background to build the corpus;
// Stage B (assignment -> FOLD) reads from the store and can be re-run freely as
// the assignment logic improves, without regenerating packings.
//
//   bun run src/box-pleated-pack.ts --from 0 --count 10000
//   bun run src/box-pleated-pack.ts --from 0 --to 100000        (whole range)
//
// Store: $BP_PACKING_STORE (see box-pleated-store.ts for the default).

import { PACKING_STORE, hasPacking, packForSeed, writePacking } from "./box-pleated-store.ts";

interface Args {
  from: number;
  count: number;
  to: number | null;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const from = Number(get("--from") ?? 0);
  const toRaw = get("--to");
  return {
    from,
    count: Number(get("--count") ?? 1000),
    to: toRaw !== undefined ? Number(toRaw) : null,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  console.log(`Stage A: packing store -> ${PACKING_STORE}`);
  console.log(`from seed ${args.from}${args.to !== null ? `, to ${args.to}` : `, up to ${args.count} new packings`}`);

  let generated = 0;
  let skipped = 0;
  let errors = 0;
  let bytes = 0;
  const t0 = performance.now();
  let lastLog = t0;

  for (let seed = args.from; ; seed++) {
    if (args.to !== null && seed >= args.to) break;
    if (args.to === null && generated >= args.count) break;

    if (await hasPacking(seed)) {
      skipped++;
      continue;
    }
    try {
      const packing = await packForSeed(seed);
      bytes += await writePacking(seed, packing);
      generated++;
    } catch (e) {
      errors++;
      console.error(`seed ${seed}: ${(e as Error).message}`);
    }

    const now = performance.now();
    if (now - lastLog > 5000) {
      const rate = generated / ((now - t0) / 1000);
      console.log(
        `  seed ${seed} · generated ${generated} · skipped ${skipped} · errors ${errors} · ` +
          `${rate.toFixed(2)}/s · ${(bytes / 1e6).toFixed(1)}MB`,
      );
      lastLog = now;
    }
  }

  const secs = (performance.now() - t0) / 1000;
  console.log(
    `\nDONE: generated ${generated}, skipped ${skipped}, errors ${errors} in ${secs.toFixed(0)}s ` +
      `(${(generated / secs).toFixed(2)}/s), ${(bytes / 1e6).toFixed(1)}MB written ` +
      `(${generated ? (bytes / generated / 1024).toFixed(1) : 0}KB/packing avg).`,
  );
}

await main();
