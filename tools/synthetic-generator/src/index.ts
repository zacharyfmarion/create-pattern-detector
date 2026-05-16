#!/usr/bin/env bun
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { assignmentCounts, countCreases, relativePath, roleCounts, splitForIndex, stableId } from "./fold-utils.ts";
import { generateFold } from "./generators.ts";
import { SeededRandom } from "./random.ts";
import { loadRecipe } from "./recipe.ts";
import { samplerForAcceptedTreeMakerMix } from "./treemaker-mix-scheduler.ts";
import type { ComplexityBucket, GenerationConfig, GeneratorFamily, RawManifestRow } from "./types.ts";
import { preflightValidation, validateFold } from "./validate.ts";

interface CliArgs {
  recipe?: string;
  count: number;
  out: string;
  seed?: number;
  maxAttempts?: number;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  const recipe = await loadRecipe(args.recipe);
  if (args.seed !== undefined) recipe.seed = args.seed;

  preflightValidation(recipe.validation);

  await mkdir(args.out, { recursive: true });
  await mkdir(join(args.out, "folds"), { recursive: true });
  await mkdir(join(args.out, "metadata"), { recursive: true });
  await mkdir(join(args.out, "rejected"), { recursive: true });

  const rng = new SeededRandom(recipe.seed);
  const rows: RawManifestRow[] = [];
  const rejectionCounts: Record<string, number> = {};
  const familyCounts: Record<string, number> = {};
  const bpSubfamilyCounts: Record<string, number> = {};
  const bpSubfamilyRejectionCounts: Record<string, number> = {};
  const aggregateRoleCounts: Record<string, number> = {};
  const gridSizeCounts: Record<string, number> = {};
  const densityBucketCounts: Record<string, number> = {};
  const denseSubfamilyCounts: Record<string, number> = {};
  const solverMsValues: number[] = [];
  const faceCountValues: number[] = [];
  const realismScoreValues: number[] = [];
  const archetypeCounts: Record<string, number> = {};
  const treeMakerSymmetryCounts: Record<string, number> = {};
  const treeMakerVariantCounts: Record<string, number> = {};
  const treeMakerArchetypeCounts: Record<string, number> = {};
  const treeMakerTopologyCounts: Record<string, number> = {};
  const acceptedTreeMetadata = (): NonNullable<RawManifestRow["treeMetadata"]>[] =>
    rows.map((row) => row.treeMetadata).filter((metadata): metadata is NonNullable<RawManifestRow["treeMetadata"]> => metadata !== undefined);
  const maxAttempts = args.maxAttempts ?? args.count * 40;
  let attempts = 0;

  while (rows.length < args.count && attempts < maxAttempts) {
    attempts += 1;
    const family = rng.weightedChoice(recipe.families as Record<GeneratorFamily, number>);
    const bucket = chooseBucket(rng, recipe.complexityBuckets);
    const sampleSeed = recipe.seed + attempts * 1009;
    const id = stableId(recipe.name, recipe.seed, attempts - 1);
    const config: GenerationConfig = {
      id,
      family,
      seed: sampleSeed,
      numCreases: rng.int(bucket.minCreases, bucket.maxCreases),
      bucket: bucket.name,
      dense: recipe.validation.requireDense === true,
      treeMakerSampler: family === "treemaker-tree"
        ? samplerForAcceptedTreeMakerMix(recipe.treeMakerSampler, acceptedTreeMetadata(), args.count)
        : recipe.treeMakerSampler,
    };

    try {
      const fold = generateFold(config);
      const validation = await validateFold(fold, recipe.validation);
      if (!validation.valid) {
        incrementRejections(rejectionCounts, validation.failed[0] ?? "invalid");
        if (fold.bp_metadata?.bpSubfamily) incrementRejections(bpSubfamilyRejectionCounts, fold.bp_metadata.bpSubfamily);
        await writeJson(join(args.out, "rejected", `${id}.json`), {
          id,
          config,
          validation,
          bpMetadata: fold.bp_metadata,
          designTree: fold.design_tree,
        realismMetadata: fold.realism_metadata,
        treeMetadata: fold.tree_metadata,
        treeMakerMetadata: fold.treemaker_metadata,
      });
        continue;
      }

      const foldPath = join(args.out, "folds", `${id}.fold`);
      const metadataPath = join(args.out, "metadata", `${id}.json`);
      await writeJson(foldPath, fold);
      await writeJson(metadataPath, { id, config, validation, fold });

      const row: RawManifestRow = {
        id,
        seed: sampleSeed,
        family,
        bucket: bucket.name,
        split: splitForIndex(rows.length, recipe.splits),
        foldPath: relativePath(foldPath, args.out),
        metadataPath: relativePath(metadataPath, args.out),
        vertices: fold.vertices_coords.length,
        edges: fold.edges_vertices.length,
        assignments: assignmentCounts(fold),
        roleCounts: roleCounts(fold),
        bpMetadata: fold.bp_metadata,
        densityMetadata: fold.density_metadata,
        designTree: fold.design_tree,
        layoutMetadata: fold.layout_metadata,
        moleculeMetadata: fold.molecule_metadata,
        realismMetadata: fold.realism_metadata,
        treeMetadata: fold.tree_metadata,
        treeMakerMetadata: fold.treemaker_metadata,
        completionMetadata: fold.completion_metadata,
        labelPolicy: fold.label_policy,
        bpStudioSummary: fold.bp_studio_summary,
        validation,
      };
      rows.push(row);
      familyCounts[family] = (familyCounts[family] ?? 0) + 1;
      for (const [role, count] of Object.entries(row.roleCounts ?? {})) {
        aggregateRoleCounts[role] = (aggregateRoleCounts[role] ?? 0) + count;
      }
      if (fold.bp_metadata) {
        bpSubfamilyCounts[fold.bp_metadata.bpSubfamily] = (bpSubfamilyCounts[fold.bp_metadata.bpSubfamily] ?? 0) + 1;
        const gridKey = String(fold.bp_metadata.gridSize);
        gridSizeCounts[gridKey] = (gridSizeCounts[gridKey] ?? 0) + 1;
      }
      if (fold.density_metadata) {
        densityBucketCounts[fold.density_metadata.densityBucket] = (densityBucketCounts[fold.density_metadata.densityBucket] ?? 0) + 1;
        denseSubfamilyCounts[fold.density_metadata.subfamily] = (denseSubfamilyCounts[fold.density_metadata.subfamily] ?? 0) + 1;
      }
      if (fold.design_tree?.archetype) {
        archetypeCounts[fold.design_tree.archetype] = (archetypeCounts[fold.design_tree.archetype] ?? 0) + 1;
      }
      if (fold.tree_metadata) {
        treeMakerSymmetryCounts[fold.tree_metadata.symmetryClass] = (treeMakerSymmetryCounts[fold.tree_metadata.symmetryClass] ?? 0) + 1;
        treeMakerVariantCounts[fold.tree_metadata.symmetryVariant] = (treeMakerVariantCounts[fold.tree_metadata.symmetryVariant] ?? 0) + 1;
        treeMakerArchetypeCounts[fold.tree_metadata.archetype] = (treeMakerArchetypeCounts[fold.tree_metadata.archetype] ?? 0) + 1;
        treeMakerTopologyCounts[fold.tree_metadata.topology] = (treeMakerTopologyCounts[fold.tree_metadata.topology] ?? 0) + 1;
      }
      if (fold.realism_metadata?.score !== undefined) realismScoreValues.push(fold.realism_metadata.score);
      if (validation.metrics?.solverMs !== undefined) solverMsValues.push(validation.metrics.solverMs);
      if (validation.metrics?.faces !== undefined) faceCountValues.push(validation.metrics.faces);
      console.log(`[${rows.length}/${args.count}] accepted ${id} (${family}, ${countCreases(fold)} creases)`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      incrementRejections(rejectionCounts, "exception");
      await writeJson(join(args.out, "rejected", `${id}.json`), { id, config, error: message });
    }
  }

  if (rows.length < args.count) {
    throw new Error(`Only generated ${rows.length}/${args.count} accepted patterns after ${attempts} attempts`);
  }

  await Bun.write(
    join(args.out, "raw-manifest.jsonl"),
    rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
  );
  await writeJson(join(args.out, "recipe.json"), recipe);
  await writeJson(join(args.out, "qa.json"), {
    recipe: recipe.name,
    requested: args.count,
    accepted: rows.length,
    attempts,
    acceptanceRate: rows.length / attempts,
    familyCounts,
    bpSubfamilyCounts,
    bpSubfamilyRejectionCounts,
    densityBucketCounts,
    denseSubfamilyCounts,
    archetypeCounts,
    treeMakerSymmetryCounts,
    treeMakerVariantCounts,
    treeMakerArchetypeCounts,
    treeMakerTopologyCounts,
    realismScore: summarizeOrNull(realismScoreValues),
    roleCounts: aggregateRoleCounts,
    gridSizeCounts,
    solverMs: summarizeOrNull(solverMsValues),
    faces: summarizeOrNull(faceCountValues),
    rabbitEarStrictPassRate: recipe.validation.strictGlobal
      ? rows.filter((row) => row.validation.passed.includes("rabbit-ear-solver")).length / rows.length
      : null,
    treeMakerAcceptedMix: recipe.treeMakerSampler?.acceptedMix,
    rejectionCounts,
    vertices: summarize(rows.map((row) => row.vertices)),
    edges: summarize(rows.map((row) => row.edges)),
  });

  console.log(`Wrote ${rows.length} accepted folds to ${args.out}`);
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    count: 32,
    out: "data/generated/synthetic/bp_studio_realistic_v1",
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--recipe") args.recipe = argv[++i];
    else if (arg === "--count") args.count = Number(argv[++i]);
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--seed") args.seed = Number(argv[++i]);
    else if (arg === "--max-attempts") args.maxAttempts = Number(argv[++i]);
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bun run generate -- --recipe <recipe.yaml> --count 32 --out <output-dir>");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function chooseBucket(rng: SeededRandom, buckets: ComplexityBucket[]): ComplexityBucket {
  const weights: Record<string, number> = {};
  for (const bucket of buckets) weights[bucket.name] = bucket.weight;
  const name = rng.weightedChoice(weights);
  const bucket = buckets.find((item) => item.name === name);
  if (!bucket) throw new Error(`Missing complexity bucket: ${name}`);
  return bucket;
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await Bun.write(path, JSON.stringify(value, null, 2) + "\n");
}

function incrementRejections(rejectionCounts: Record<string, number>, reason: string): void {
  rejectionCounts[reason] = (rejectionCounts[reason] ?? 0) + 1;
}

function summarize(values: number[]): { min: number; max: number; mean: number } {
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

function summarizeOrNull(values: number[]): { min: number; max: number; mean: number } | null {
  return values.length ? summarize(values) : null;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
