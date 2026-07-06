#!/usr/bin/env bun
// Debug a BP Studio packing: render the river/hole/ridge view for one or more
// seeds. Each river (tree node) gets its own colour with yellow boundaries between
// adjacent rivers; holes are magenta, the stretch device green, flaps grey, and
// the straight-skeleton ridges red.
//
//   bun run src/box-pleated-debug.ts <seed...> [--leaves N] [--restarts N] [--out DIR]
//
// Example: bun run src/box-pleated-debug.ts 50444 60202 --leaves 4 --out /tmp/dbg
// Writes <out>/<seed>.svg for each seed and prints hole/validity stats.

import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { generateBoxPleatedPacking, findPackingHoles, fillPackingGaps } from "./box-pleated-packing.ts";
import { renderPackingDebugSvg } from "./box-pleated-debug-render.ts";
import { buildPackingCP } from "./box-pleated-cp.ts";

interface Args {
  seeds: number[];
  leaves?: number;
  restarts: number;
  out: string;
  cellSize?: number;
}

function parseArgs(argv: string[]): Args {
  const seeds: number[] = [];
  let leaves: number | undefined;
  let restarts = 14;
  let out = "/tmp/bp-debug";
  let cellSize: number | undefined;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--leaves") leaves = Number(argv[++i]);
    else if (a === "--restarts") restarts = Number(argv[++i]);
    else if (a === "--out") out = argv[++i];
    else if (a === "--cell") cellSize = Number(argv[++i]);
    else if (/^\d+$/.test(a)) seeds.push(Number(a));
    else throw new Error(`unknown argument: ${a}`);
  }
  if (seeds.length === 0) throw new Error("usage: box-pleated-debug.ts <seed...> [--leaves N] [--restarts N] [--out DIR]");
  return { seeds, leaves, restarts, out, cellSize };
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  await mkdir(args.out, { recursive: true });
  for (const seed of args.seeds) {
    const leaves = args.leaves ?? [4, 5, 6][seed % 3];
    let packing;
    try {
      packing = await generateBoxPleatedPacking({
        id: `debug-${seed}`,
        seed,
        numCreases: 300,
        bucket: "s",
        symmetry: "none",
        targetLeafCount: leaves,
        tight: true,
        tightRestarts: args.restarts,
      });
    } catch (error) {
      console.log(`${seed}: generation failed - ${error instanceof Error ? error.message : error}`);
      continue;
    }
    const svg = renderPackingDebugSvg(packing, { cellSize: args.cellSize });
    const file = join(args.out, `${seed}.svg`);
    await Bun.write(file, svg);

    const holes = findPackingHoles(packing);
    const gap = fillPackingGaps(packing);
    const cp = buildPackingCP(packing);
    const status = cp.valid ? "VALID" : !cp.complete ? "reject:unfillable-hole" : cp.offGrid.length ? "reject:off-grid" : "reject:failing";
    console.log(
      `${seed} (lv${leaves}): holes=${holes.length} gapComplete=${gap.complete} offGrid=${cp.offGrid.length} -> ${status}  ${file}`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
