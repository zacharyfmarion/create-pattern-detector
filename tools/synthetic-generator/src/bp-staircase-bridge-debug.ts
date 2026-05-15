import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { foldComparisonSvg } from "./bp-debug-svg.ts";
import { buildStaircaseBridgePrimitive } from "./bp-staircase-bridge.ts";
import { makeFlatFoldedPreview } from "./folded-preview.ts";
import { validateFold } from "./validate.ts";

interface Options {
  out: string;
  laneCount: number;
  orientation: "diagonal-positive" | "diagonal-negative";
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const result = buildStaircaseBridgePrimitive(options);
  if (!result.ok || !result.fold) throw new Error(`staircase bridge failed assignment solve: ${result.errors.join("; ")}`);
  const validation = await validateFold(result.fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-9,
    maxVertices: 2000,
    maxEdges: 2000,
    requireBoxPleat: true,
    boxPleatMode: "dense",
  });
  if (!validation.valid) throw new Error(`staircase bridge failed validation: ${validation.errors.join("; ")}`);
  const preview = makeFlatFoldedPreview(result.fold);
  await mkdir(dirname(options.out), { recursive: true });
  await writeFile(options.out, foldComparisonSvg({
    title: "Strict staircase bridge primitive",
    subtitle: "Two diagonal caps are composed first, then final M/V labels are solved at the completed molecule boundary.",
    leftTitle: "crease pattern",
    rightTitle: "Rabbit Ear folded coordinates",
    cp: result.fold,
    folded: preview.foldedFold,
    footer: `Strict checks passed after ${result.assignmentSteps} assignment-search steps. Faces: ${validation.metrics?.faces ?? preview.faces}.`,
  }));
  await writeFile(options.out.replace(/\.svg$/u, ".fold"), JSON.stringify(result.fold, null, 2) + "\n");
  await writeFile(options.out.replace(/\.svg$/u, ".folded.fold"), JSON.stringify(preview.foldedFold, null, 2) + "\n");
  console.log(JSON.stringify({
    out: options.out,
    vertices: result.fold.vertices_coords.length,
    edges: result.fold.edges_vertices.length,
    faces: preview.faces,
    assignmentSteps: result.assignmentSteps,
    validation: validation.passed,
  }, null, 2));
}

function parseArgs(args: string[]): Options {
  const options: Options = {
    out: "/tmp/bp-staircase-bridge-debug/staircase-bridge.svg",
    laneCount: 5,
    orientation: "diagonal-positive",
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--out") {
      options.out = requiredValue(args[++index], "--out");
    } else if (arg === "--lane-count") {
      options.laneCount = Number(requiredValue(args[++index], "--lane-count"));
    } else if (arg === "--orientation") {
      options.orientation = parseOrientation(requiredValue(args[++index], "--orientation"));
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(options.laneCount) || options.laneCount < 1) {
    throw new Error("--lane-count must be a positive integer");
  }
  return options;
}

function parseOrientation(value: string): Options["orientation"] {
  if (value === "diagonal-positive" || value === "diagonal-negative") return value;
  throw new Error(`--orientation must be diagonal-positive or diagonal-negative; got ${value}`);
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`${flag} requires a value`);
  return value;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
