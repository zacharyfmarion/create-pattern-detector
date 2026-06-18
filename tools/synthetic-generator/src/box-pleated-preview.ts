#!/usr/bin/env bun
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import {
  generateBoxPleatedPacking,
  renderBoxPleatedPackingSvg,
  validateBoxPleatedPacking,
  type BoxPleatedPacking,
  type BoxPleatedPackingSymmetry,
} from "./box-pleated-packing.ts";
import {
  completeBoxPleatedCreaseScaffold,
  renderBoxPleatedCreaseScaffoldSvg,
  type BoxPleatedCreaseScaffold,
} from "./box-pleated-scaffold.ts";

interface CliArgs {
  count: number;
  out: string;
  seed: number;
  symmetry: BoxPleatedPackingSymmetry;
  bucket: string;
  numCreases: number;
  bpStudioRoot?: string;
  maxAttempts: number;
  targetLeafCount?: number;
  optimizerDistanceScale?: number;
  noStretches: boolean;
  tight: boolean;
  tightRestarts?: number;
  showContours: boolean;
  scaffold: boolean;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  await mkdir(args.out, { recursive: true });
  await mkdir(join(args.out, "json"), { recursive: true });
  await mkdir(join(args.out, "svg"), { recursive: true });
  if (args.scaffold) await mkdir(join(args.out, "scaffold-json"), { recursive: true });

  const packings: BoxPleatedPacking[] = [];
  const scaffolds: BoxPleatedCreaseScaffold[] = [];
  for (let index = 0; index < args.count; index++) {
    const seed = args.seed + index * 1009;
    const packing = await generateBoxPleatedPacking({
      id: `box-pleated-${args.seed.toString(36)}-${index.toString().padStart(3, "0")}`,
      seed,
      numCreases: args.numCreases,
      bucket: args.bucket,
      symmetry: args.symmetry,
      bpStudioRoot: args.bpStudioRoot,
      maxAttempts: args.maxAttempts,
      targetLeafCount: args.targetLeafCount,
      optimizerDistanceScale: args.optimizerDistanceScale,
      noStretches: args.noStretches,
      tight: args.tight,
      tightRestarts: args.tightRestarts,
    });
    const errors = validateBoxPleatedPacking(packing);
    if (errors.length) {
      throw new Error(`invalid generated packing ${packing.id}: ${errors.join("; ")}`);
    }
    packings.push(packing);
    await Bun.write(join(args.out, "json", `${packing.id}.json`), JSON.stringify(packing, null, 2) + "\n");
    let scaffold: BoxPleatedCreaseScaffold | undefined;
    if (args.scaffold) {
      scaffold = completeBoxPleatedCreaseScaffold(packing);
      scaffolds.push(scaffold);
      await Bun.write(join(args.out, "scaffold-json", `${packing.id}.json`), JSON.stringify(scaffold, null, 2) + "\n");
      await Bun.write(join(args.out, "svg", `${packing.id}.svg`), renderBoxPleatedCreaseScaffoldSvg(packing, scaffold));
    } else {
      await Bun.write(join(args.out, "svg", `${packing.id}.svg`), renderBoxPleatedPackingSvg(packing, {
        includeLayoutContours: args.showContours,
      }));
    }
    const scaffoldText = scaffold ? `, ${scaffold.stats.gapRidges} gap-fill ridges, ${scaffold.stats.computedAxials} axial candidates` : "";
    console.log(`[${index + 1}/${args.count}] wrote ${packing.id} (${packing.stats.leaves} leaves, ${packing.stats.ridgeCreases} ridges, ${packing.stats.stretchDevices} stretch devices, ${packing.stats.offGridRidgeCreases} off-grid ridges${scaffoldText}, noStretches=${packing.constraints.noStretches})`);
  }

  await Bun.write(join(args.out, "index.html"), renderIndex(packings, args.showContours, args.scaffold));
  await Bun.write(join(args.out, "summary.json"), JSON.stringify(summary(packings, scaffolds), null, 2) + "\n");
  console.log(`Wrote ${packings.length} BP-style box-pleated packing previews to ${args.out}`);
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    count: 8,
    out: "/tmp/box-pleated-packings",
    seed: 13013,
    symmetry: "none",
    bucket: "preview",
    numCreases: 260,
    maxAttempts: 24,
    noStretches: false,
    tight: false,
    showContours: false,
    scaffold: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--count") args.count = Number(argv[++index]);
    else if (arg === "--out") args.out = argv[++index];
    else if (arg === "--seed") args.seed = Number(argv[++index]);
    else if (arg === "--symmetry") args.symmetry = parseSymmetry(argv[++index]);
    else if (arg === "--bucket") args.bucket = argv[++index];
    else if (arg === "--num-creases") args.numCreases = Number(argv[++index]);
    else if (arg === "--bp-studio-root") args.bpStudioRoot = argv[++index];
    else if (arg === "--max-attempts") args.maxAttempts = Number(argv[++index]);
    else if (arg === "--target-leaf-count") args.targetLeafCount = Number(argv[++index]);
    else if (arg === "--optimizer-distance-scale") args.optimizerDistanceScale = Number(argv[++index]);
    else if (arg === "--no-stretches" || arg === "--pure-45-90") args.noStretches = true;
    else if (arg === "--tight") args.tight = true;
    else if (arg === "--restarts") args.tightRestarts = Number(argv[++index]);
    else if (arg === "--show-contours") args.showContours = true;
    else if (arg === "--scaffold") args.scaffold = true;
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run box-pleated-preview -- --count 8 --out /tmp/box-pleated-packings --bp-studio-root /tmp/bp-studio-source [--tight [--restarts 16]] [--no-stretches] [--show-contours] [--scaffold]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isFinite(args.count) || args.count < 1) throw new Error("--count must be positive");
  if (!Number.isFinite(args.seed)) throw new Error("--seed must be numeric");
  if (!Number.isFinite(args.numCreases) || args.numCreases < 1) throw new Error("--num-creases must be positive");
  if (!Number.isFinite(args.maxAttempts) || args.maxAttempts < 1) throw new Error("--max-attempts must be positive");
  if (args.tightRestarts !== undefined && (!Number.isFinite(args.tightRestarts) || args.tightRestarts < 1)) {
    throw new Error("--restarts must be at least 1");
  }
  if (args.targetLeafCount !== undefined && (!Number.isFinite(args.targetLeafCount) || args.targetLeafCount < 2)) {
    throw new Error("--target-leaf-count must be at least 2");
  }
  if (
    args.optimizerDistanceScale !== undefined &&
    (!Number.isFinite(args.optimizerDistanceScale) || args.optimizerDistanceScale < 1)
  ) {
    throw new Error("--optimizer-distance-scale must be at least 1");
  }
  return args;
}

function parseSymmetry(value: string): BoxPleatedPackingSymmetry {
  if (value === "vertical" || value === "horizontal" || value === "none") return value;
  throw new Error(`Unsupported symmetry: ${value}`);
}

function renderIndex(packings: BoxPleatedPacking[], showContours: boolean, scaffold: boolean): string {
  const items = packings.map((packing) => `
      <figure>
        <a href="svg/${packing.id}.svg"><img src="svg/${packing.id}.svg" alt="${packing.id}"/></a>
        <figcaption>${packing.id}: ${packing.sheet.width}x${packing.sheet.height}, ${packing.symmetry}, ${packing.stats.leaves} leaves, ${packing.stats.ridgeCreases} ridges, ${packing.stats.axisParallelCreases} axis-parallel, ${packing.stats.stretchDevices} stretch devices, ${packing.stats.offGridRidgeCreases} off-grid ridges, noStretches=${packing.constraints.noStretches}, layoutContours=${showContours ? "shown" : "hidden"}, scaffold=${scaffold}, patternNotFound=${packing.layout.patternNotFound}</figcaption>
      </figure>
  `).join("\n");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>BP-Style Box-Pleated Packing Previews</title>
  <style>
    body { margin: 24px; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; }
    main { display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 24px; align-items: start; }
    figure { margin: 0; border: 1px solid #d7dde2; padding: 12px; background: #fff; }
    img { width: 100%; height: auto; display: block; }
    figcaption { margin-top: 8px; font-size: 13px; color: #475569; }
  </style>
</head>
<body>
  <h1>BP-Style Box-Pleated Packing Previews</h1>
  <main>
    ${items}
  </main>
</body>
</html>
`;
}

function summary(packings: BoxPleatedPacking[], scaffolds: BoxPleatedCreaseScaffold[]): unknown {
  const result: Record<string, unknown> = {
    count: packings.length,
    noStretches: packings.filter((packing) => packing.constraints.noStretches).length,
    symmetries: countBy(packings.map((packing) => packing.symmetry)),
    leaves: summarize(packings.map((packing) => packing.stats.leaves)),
    ridgeCreases: summarize(packings.map((packing) => packing.stats.ridgeCreases)),
    axisParallelCreases: summarize(packings.map((packing) => packing.stats.axisParallelCreases)),
    stretchDevices: summarize(packings.map((packing) => packing.stats.stretchDevices)),
    offGridRidgeCreases: summarize(packings.map((packing) => packing.stats.offGridRidgeCreases)),
    patternNotFound: packings.filter((packing) => packing.layout.patternNotFound).length,
  };
  if (scaffolds.length) {
    result.scaffold = {
      count: scaffolds.length,
      bpRidges: summarize(scaffolds.map((scaffold) => scaffold.stats.bpRidges)),
      bpContours: summarize(scaffolds.map((scaffold) => scaffold.stats.bpContours)),
      computedAxials: summarize(scaffolds.map((scaffold) => scaffold.stats.computedAxials)),
      gapRidges: summarize(scaffolds.map((scaffold) => scaffold.stats.gapRidges)),
      filledGapCells: summarize(scaffolds.map((scaffold) => scaffold.stats.filledGapCells)),
      skippedFlapCells: summarize(scaffolds.map((scaffold) => scaffold.stats.skippedFlapCells)),
      unfilledGapCells: summarize(scaffolds.map((scaffold) => scaffold.stats.unfilledGapCells)),
    };
  }
  return result;
}

function countBy(values: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function summarize(values: number[]): { min: number; max: number; mean: number } {
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
