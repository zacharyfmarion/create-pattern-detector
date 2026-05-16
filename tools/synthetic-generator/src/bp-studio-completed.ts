import ear from "rabbit-ear";
import { compilerGridSizeForSheet } from "./bp-completion.ts";
import { completeBPStudioScaffoldByFoldProgram } from "./bp-studio-fold-program-completion.ts";
import { normalizeFold } from "./fold-utils.ts";
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
      const sheet = adapterMetadata.optimizedLayout?.sheet ?? adapterMetadata.layout?.sheet ?? adapterSpec.sheet;
      const completion = completeBPStudioScaffoldByFoldProgram(scaffoldFold, {
        id: config.id,
        spec,
        adapterMetadata,
        gridSize: compilerGridSizeForSheet(sheet.width, sheet.height),
      });
      if (!completion.ok || !completion.fold) {
        throw new Error(`completion failed: ${completion.errors.join("; ")}`);
      }

      const fold = completion.fold;
      const foldedCreases = fold.edges_assignment.filter((assignment) => assignment === "M" || assignment === "V").length;
      if (foldedCreases < 60) throw new Error(`completion too sparse: ${foldedCreases} folded creases < 60`);
      fold.bp_metadata = {
        ...(fold.bp_metadata ?? {
          gridSize: compilerGridSizeForSheet(sheet.width, sheet.height),
          bpSubfamily: "bp-studio-completed-uniaxial",
          flapCount: spec.layout.flaps.length,
          gadgetCount: 0,
          ridgeCount: 0,
          hingeCount: 0,
          axisCount: 0,
        }),
        bpSubfamily: "bp-studio-completed-uniaxial",
      };
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
          "bp-studio-source-line-program",
          `rabbit-ear-flat-fold-program:${completion.foldCount}`,
          "pending-strict-validation",
        ],
        moleculeCounts: {
          ...makeMoleculeCounts(spec, adapterMetadata),
          ...(fold.molecule_metadata?.molecules ?? {}),
        },
      };
      strictRabbitEarPrecheck(fold);
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

function strictRabbitEarPrecheck(fold: FOLDFormat): void {
  const graph = normalizeFold(fold);
  ear.graph.populate(graph);
  const kawasakiBad = ear.singleVertex.validateKawasaki(graph) as number[];
  const maekawaBad = ear.singleVertex.validateMaekawa(graph) as number[];
  if (kawasakiBad.length || maekawaBad.length) {
    throw new Error(
      `source-line completion failed local flat-foldability: kawasaki=${kawasakiBad.length}, maekawa=${maekawaBad.length}`,
    );
  }
  const solverResult = ear.layer.solver(graph);
  if (!solverResult) {
    throw new Error("source-line completion failed Rabbit Ear layer solver");
  }
  const folded = ear.graph.makeVerticesCoordsFlatFolded(graph);
  if (!Array.isArray(folded) || folded.length !== graph.vertices_coords.length) {
    throw new Error("source-line completion could not compute folded coordinates");
  }
  if (folded.some((coord) => !Array.isArray(coord) || coord.some((value) => !Number.isFinite(value)))) {
    throw new Error("source-line completion produced non-finite folded coordinates");
  }
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength)}...`;
}
