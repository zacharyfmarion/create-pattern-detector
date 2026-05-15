import { completeBoxPleat } from "./bp-completion.ts";
import { assertValidBPStudioSpec, generateBPStudioSpec } from "./bp-studio-sampler.ts";
import {
  bucketFor,
  ensureAdapterAvailable,
  makeDesignTree,
  makeMoleculeCounts,
  runBPStudioAdapter,
  summarizeAdapterMetadata,
  toAdapterSpec,
} from "./bp-studio-realistic.ts";
import type { BPStudioArchetype } from "./bp-studio-spec.ts";
import type { FOLDFormat, GenerationConfig } from "./types.ts";

export function generateBPStudioCompletedFold(config: GenerationConfig): FOLDFormat {
  ensureAdapterAvailable();
  const bucket = bucketFor(config);
  const archetype = config.realisticArchetype as BPStudioArchetype | undefined;
  let lastError = "unknown error";
  const attemptErrors: string[] = [];

  for (let attempt = 0; attempt < 8; attempt++) {
    const attemptSeed = config.seed + attempt * 104729;
    try {
      const spec = generateBPStudioSpec({
        id: config.id,
        seed: attemptSeed,
        bucket,
        archetype,
        variation: Math.abs(attemptSeed) % 1_000_000,
      });
      assertValidBPStudioSpec(spec);

      const adapterSpec = toAdapterSpec(spec);
      const { fold: scaffoldFold, metadata: adapterMetadata } = runBPStudioAdapter(adapterSpec);
      const completion = completeBoxPleat(spec, {
        adapterSpec,
        adapterMetadata,
        layoutId: config.id,
        maxFoldLines: bucket === "small" ? 96 : 128,
      });
      if (!completion.ok || !completion.fold) {
        throw new Error(`completion failed: ${completion.rejected.map((item) => `${item.code}:${item.message}`).join("; ")}`);
      }

      const fold = completion.fold;
      const foldedCreases = fold.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length;
      if (foldedCreases < 80) throw new Error(`completion too sparse: ${foldedCreases} folded creases < 80`);
      fold.file_creator = "cp-synthetic-generator/bp-studio-completed";
      fold.file_title = `${config.id} BP Studio completed`;
      fold.file_description = "Strict box-pleat CP compiled from a BP Studio optimized tree/layout scaffold.";
      fold.design_tree = makeDesignTree(spec);
      fold.bp_studio_metadata = {
        samplerSpec: spec,
        adapterSpec,
        adapterMetadata,
        scaffoldSummary: {
          vertices: scaffoldFold.vertices_coords.length,
          edges: scaffoldFold.edges_vertices.length,
          assignments: adapterMetadata.cp?.assignmentCounts ?? {},
        },
      };
      fold.bp_studio_summary = summarizeAdapterMetadata(adapterMetadata);
      fold.density_metadata = {
        densityBucket: bucket,
        gridSize: fold.bp_metadata?.gridSize ?? spec.sheet.gridSize,
        targetEdgeRange: [80, bucket === "small" ? 1500 : 3000],
        subfamily: "bp-studio-completed-uniaxial",
        symmetry: spec.layout.symmetry,
        generatorSteps: [
          "bp-studio-spec-sampler",
          "bp-studio-adapter-optimizer",
          "regularize-layout",
          "certified-molecule-fold-program",
          "pending-strict-validation",
        ],
        moleculeCounts: {
          ...makeMoleculeCounts(spec, adapterMetadata),
          ...(fold.molecule_metadata?.molecules ?? {}),
        },
      };
      return fold;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      attemptErrors.push(`attempt ${attempt}: ${truncate(lastError, 1200)}`);
    }
  }

  throw new Error(
    `BP Studio completion generator failed after retries: ${lastError}\nRecent attempts:\n${attemptErrors.slice(-5).join("\n")}`,
  );
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength)}...`;
}
