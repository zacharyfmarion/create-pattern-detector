#!/usr/bin/env bun
import ear from "rabbit-ear";
import { mkdir } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { normalizeFold, relativePath } from "./fold-utils.ts";
import type { FOLDFormat } from "./types.ts";

interface CliArgs {
  root: string;
  limit: number;
  out?: string;
  family?: string;
  skipFailures: boolean;
}

interface RawRow {
  id: string;
  family: string;
  foldPath: string;
  validation?: { passed?: string[] };
}

interface FoldedPreviewRow {
  id: string;
  family: string;
  originalFoldPath: string;
  foldedFoldPath: string;
  vertices: number;
  edges: number;
  faces: number;
  solverRootOrders: number;
  solverBranchCount: number;
  solverSolutionCount: number;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const outDir = args.out ?? join(args.root, "qa", "folded");
  const foldsDir = join(outDir, "folds");
  await mkdir(foldsDir, { recursive: true });

  const rawRows = await readRawManifest(join(args.root, "raw-manifest.jsonl"));
  const selectedRows = rawRows.filter((row) => !args.family || row.family === args.family).slice(0, args.limit);
  if (selectedRows.length === 0) {
    throw new Error(`No rows found in ${args.root} matching family=${args.family ?? "*"}`);
  }

  const rows: FoldedPreviewRow[] = [];
  const failures: Array<{ id: string; family: string; originalFoldPath: string; error: string }> = [];
  for (const row of selectedRows) {
    const fold = await readJson<FOLDFormat>(join(args.root, row.foldPath));
    let preview: ReturnType<typeof makeFlatFoldedPreview>;
    try {
      preview = makeFlatFoldedPreview(fold);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push({ id: row.id, family: row.family, originalFoldPath: row.foldPath, error: message });
      if (!args.skipFailures) throw error;
      console.warn(`[${rows.length + failures.length}/${selectedRows.length}] skipped ${row.id}: ${message}`);
      continue;
    }
    const foldedFoldPath = join(foldsDir, `${row.id}--flat-folded.fold`);
    await writeJson(foldedFoldPath, preview.foldedFold);
    rows.push({
      id: row.id,
      family: row.family,
      originalFoldPath: row.foldPath,
      foldedFoldPath: relativePath(foldedFoldPath, args.root),
      vertices: fold.vertices_coords.length,
      edges: fold.edges_vertices.length,
      faces: preview.faces,
      solverRootOrders: preview.solverRootOrders,
      solverBranchCount: preview.solverBranchCount,
      solverSolutionCount: preview.solverSolutionCount,
    });
    console.log(
      `[${rows.length}/${selectedRows.length}] folded ${row.id} (${row.family}, faces=${preview.faces}, solver branches=${preview.solverBranchCount})`,
    );
  }

  await Bun.write(join(outDir, "folded-manifest.jsonl"), rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
  await writeJson(join(outDir, "folded-qa.json"), {
    root: args.root,
    rows: rows.length,
    failures,
    families: countBy(rows, (row) => row.family),
    solverRootOrders: summarize(rows.map((row) => row.solverRootOrders)),
    solverBranchCounts: summarize(rows.map((row) => row.solverBranchCount)),
    solverSolutionCounts: summarize(rows.map((row) => row.solverSolutionCount)),
  });
  console.log(`Wrote folded preview manifest to ${join(outDir, "folded-manifest.jsonl")}`);
}

export function makeFlatFoldedPreview(fold: FOLDFormat): {
  foldedFold: FOLDFormat;
  faces: number;
  solverRootOrders: number;
  solverBranchCount: number;
  solverSolutionCount: number;
} {
  const graph = normalizeFold(fold, fold.file_creator);
  ear.graph.populate(graph);
  const solverResult = ear.layer.solver(graph) as
    | {
        root?: Record<string, number>;
        branches?: Array<Array<Record<string, number>>>;
      }
    | undefined;
  if (!solverResult) {
    throw new Error("Rabbit Ear layer solver found no globally consistent layer ordering");
  }

  const foldedCoords = ear.graph.makeVerticesCoordsFlatFolded(graph) as [number, number][];
  if (!Array.isArray(foldedCoords) || foldedCoords.length !== graph.vertices_coords.length) {
    throw new Error("Rabbit Ear could not compute folded vertex coordinates");
  }
  if (foldedCoords.some((coord) => !Array.isArray(coord) || coord.some((value) => !Number.isFinite(value)))) {
    throw new Error("Rabbit Ear folded coordinates contain non-finite values");
  }

  const branchCount = solverResult.branches?.length ?? 0;
  const solutionCount = solverResult.branches?.reduce((sum, branch) => sum + branch.length, 0) ?? 0;
  const foldedFold = normalizeFold(
    {
      ...graph,
      file_creator: `${fold.file_creator}/flat-folded-preview`,
      frame_classes: ["foldedForm"],
      vertices_coords: foldedCoords,
    },
    `${fold.file_creator}/flat-folded-preview`,
  );

  return {
    foldedFold,
    faces: graph.faces_vertices?.length ?? 0,
    solverRootOrders: Object.keys(solverResult.root ?? {}).length,
    solverBranchCount: branchCount,
    solverSolutionCount: solutionCount,
  };
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { root: "", limit: 12, skipFailures: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--root") args.root = argv[++i];
    else if (arg === "--limit") args.limit = Number(argv[++i]);
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--family") args.family = argv[++i];
    else if (arg === "--skip-failures") args.skipFailures = true;
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run folded-preview -- --root <dataset-root> [--limit 12] [--family treemaker-tree] [--skip-failures]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.root) throw new Error("--root is required");
  if (!Number.isFinite(args.limit) || args.limit <= 0) throw new Error("--limit must be a positive number");
  return args;
}

async function readRawManifest(path: string): Promise<RawRow[]> {
  const text = await Bun.file(path).text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as RawRow);
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await Bun.file(path).text()) as T;
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, JSON.stringify(value, null, 2) + "\n");
}

function countBy<T>(values: T[], fn: (value: T) => string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const key = basename(fn(value));
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function summarize(values: number[]): { min: number; max: number; mean: number } {
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
