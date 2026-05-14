#!/usr/bin/env bun
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { scoreFoldRealism } from "./realistic-box-pleat.ts";
import type { FOLDFormat, RealismMetadata } from "./types.ts";

interface CliArgs {
  root: string;
  refs?: string;
  out?: string;
}

interface GeneratedRow {
  id: string;
  foldPath: string;
  family: string;
  bpMetadata?: { bpSubfamily?: string };
  designTree?: { archetype?: string };
  realismMetadata?: RealismMetadata;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const rawRows = await readJsonl<GeneratedRow>(join(args.root, "raw-manifest.jsonl"));
  const rows = [];
  for (const row of rawRows) {
    const fold = await Bun.file(join(args.root, row.foldPath)).json() as FOLDFormat;
    const realism = row.realismMetadata ?? fold.realism_metadata ?? scoreFoldRealism(fold, fold.layout_metadata);
    rows.push({
      id: row.id,
      family: row.family,
      archetype: row.designTree?.archetype ?? fold.design_tree?.archetype ?? "unknown",
      bpSubfamily: row.bpMetadata?.bpSubfamily ?? fold.bp_metadata?.bpSubfamily ?? "unknown",
      score: realism.score,
      emptySpaceRatio: realism.emptySpaceRatio,
      localDensityVariance: realism.localDensityVariance,
      repetitionPenalty: realism.repetitionPenalty,
      gates: realism.gates,
    });
  }

  const refRows = args.refs && await fileExists(args.refs) ? await readJsonl<Record<string, unknown>>(args.refs) : [];
  const report = {
    root: args.root,
    generatedRows: rows.length,
    referenceRows: refRows.length,
    archetypes: countBy(rows, "archetype"),
    bpSubfamilies: countBy(rows, "bpSubfamily"),
    realismScore: summarize(rows.map((row) => row.score)),
    emptySpaceRatio: summarize(rows.map((row) => row.emptySpaceRatio)),
    localDensityVariance: summarize(rows.map((row) => row.localDensityVariance)),
    repetitionPenalty: summarize(rows.map((row) => row.repetitionPenalty)),
    failedGates: countFailedGates(rows),
    rows,
    referenceManifest: args.refs ?? null,
    referenceNote: refRows.length
      ? "Reference rows are calibration-only; this report does not use them as training labels."
      : "No reference rows loaded. Add clean digital BP references to compare future generated distributions.",
  };

  const out = args.out ?? join(args.root, "qa", "bp-realism-report.json");
  await mkdir(dirname(out), { recursive: true });
  await Bun.write(out, JSON.stringify(report, null, 2) + "\n");
  console.log(`Wrote BP realism report to ${out}`);
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { root: "" };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--root") args.root = argv[++i];
    else if (arg === "--refs") args.refs = argv[++i];
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run bp-realism-report -- --root <dataset-root> [--refs refs.jsonl] [--out report.json]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.root) throw new Error("--root is required");
  return args;
}

async function readJsonl<T>(path: string): Promise<T[]> {
  const text = await Bun.file(path).text();
  return text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

async function fileExists(path: string): Promise<boolean> {
  return Bun.file(path).exists();
}

function countBy<T extends Record<string, unknown>>(rows: T[], key: keyof T): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    const value = String(row[key] ?? "unknown");
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function countFailedGates(rows: Array<{ gates: Record<string, boolean> }>): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    for (const [gate, passed] of Object.entries(row.gates)) {
      if (!passed) counts[gate] = (counts[gate] ?? 0) + 1;
    }
  }
  return counts;
}

function summarize(values: number[]): { min: number; max: number; mean: number } | null {
  if (!values.length) return null;
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
