import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { foldComparisonSvg } from "./bp-debug-svg.ts";
import { buildDiagonalStaircaseCapPrimitive } from "./bp-staircase-cap.ts";
import { makeFlatFoldedPreview } from "./folded-preview.ts";
import { validateFold } from "./validate.ts";

interface Options {
  out: string;
  laneCount: number;
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const fold = buildDiagonalStaircaseCapPrimitive({
    laneCount: options.laneCount,
    startAxisAssignment: "V",
  });
  const validation = await validateFold(fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-9,
    maxVertices: 1000,
    maxEdges: 1000,
    requireBoxPleat: true,
    boxPleatMode: "dense",
  });
  if (!validation.valid) throw new Error(`staircase cap failed validation: ${validation.errors.join("; ")}`);
  const preview = makeFlatFoldedPreview(fold);
  await mkdir(dirname(options.out), { recursive: true });
  await writeFile(options.out, foldComparisonSvg({
    title: "Strict staircase cap primitive",
    subtitle: "A diagonal ridge caps long alternating pleats with perpendicular partner creases at every endpoint.",
    leftTitle: "crease pattern",
    rightTitle: "Rabbit Ear folded coordinates",
    cp: fold,
    folded: preview.foldedFold,
    footer: `Strict checks passed: local Kawasaki/Maekawa, Rabbit Ear layer solver, finite folded coordinates. Faces: ${validation.metrics?.faces ?? preview.faces}.`,
  }));
  await writeFile(options.out.replace(/\.svg$/u, ".fold"), JSON.stringify(fold, null, 2) + "\n");
  await writeFile(options.out.replace(/\.svg$/u, ".folded.fold"), JSON.stringify(preview.foldedFold, null, 2) + "\n");
  console.log(JSON.stringify({
    out: options.out,
    vertices: fold.vertices_coords.length,
    edges: fold.edges_vertices.length,
    faces: preview.faces,
    validation: validation.passed,
  }, null, 2));
}

function parseArgs(args: string[]): Options {
  const options: Options = {
    out: "/tmp/bp-staircase-cap-debug/staircase-cap.svg",
    laneCount: 7,
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--out") {
      options.out = requiredValue(args[++index], "--out");
    } else if (arg === "--lane-count") {
      options.laneCount = Number(requiredValue(args[++index], "--lane-count"));
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(options.laneCount) || options.laneCount < 1) {
    throw new Error("--lane-count must be a positive integer");
  }
  return options;
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`${flag} requires a value`);
  return value;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
