import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { buildSheetPleatPrimitive } from "./bp-pleat-primitive.ts";
import { makeFlatFoldedPreview } from "./folded-preview.ts";
import { validateFold } from "./validate.ts";
import type { EdgeAssignment, FOLDFormat } from "./types.ts";

interface Options {
  out: string;
  laneCount: number;
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const fold = buildSheetPleatPrimitive({
    laneCount: options.laneCount,
    orientation: "vertical",
    startAssignment: "M",
  });
  const validation = await validateFold(fold, {
    strictGlobal: true,
    globalBackend: "rabbit-ear-solver",
    minVertexDistance: 1e-9,
    maxVertices: 1000,
    maxEdges: 1000,
  });
  if (!validation.valid) throw new Error(`pleat primitive failed validation: ${validation.errors.join("; ")}`);
  const preview = makeFlatFoldedPreview(fold);
  await mkdir(dirname(options.out), { recursive: true });
  await writeFile(options.out, debugSvg(fold, preview.foldedFold, validation.metrics?.faces ?? preview.faces));
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
    out: "/tmp/bp-pleat-primitive-debug/pleat-primitive.svg",
    laneCount: 9,
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

function debugSvg(cp: FOLDFormat, folded: FOLDFormat, faces: number): string {
  const width = 1220;
  const height = 760;
  const panel = 520;
  const x1 = 48;
  const x2 = 650;
  const y = 150;
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="${x1}" y="44" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="#0f172a">Strict pleat primitive: flat-foldable reference</text>`,
    `<text x="${x1}" y="76" font-family="Inter, Arial, sans-serif" font-size="15" fill="#475569">This is a certified accordion pleat field. It is intentionally sheet-spanning; the next step is local caps/hubs so strips can terminate inside realistic BP layouts.</text>`,
    legend(x1, 108),
    panelSvg(x1, y, panel, "crease pattern", renderFold(cp, panel, false)),
    panelSvg(x2, y, panel, "Rabbit Ear folded coordinates", renderFold(folded, panel, true)),
    `<text x="${x1}" y="${height - 44}" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">Strict checks passed: local Kawasaki/Maekawa, Rabbit Ear layer solver, finite folded coordinates. Faces: ${faces}.</text>`,
    `</svg>`,
  ].join("\n");
}

function panelSvg(x: number, y: number, size: number, title: string, body: string): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<text x="0" y="-20" font-family="Inter, Arial, sans-serif" font-size="19" font-weight="800" fill="#0f172a">${escapeXml(title)}</text>`,
    `<rect x="-8" y="-8" width="${size + 16}" height="${size + 16}" rx="8" fill="#ffffff" stroke="#cbd5e1"/>`,
    body,
    `</g>`,
  ].join("\n");
}

function renderFold(fold: FOLDFormat, size: number, normalize: boolean): string {
  const coords = normalize ? normalizedCoords(fold.vertices_coords, size) : fold.vertices_coords.map(([x, y]) => [x * size, (1 - y) * size]);
  const edges = fold.edges_vertices.map(([a, b], index) => {
    const p1 = coords[a];
    const p2 = coords[b];
    const assignment = fold.edges_assignment[index] ?? "U";
    const color = assignmentColor(assignment);
    const width = assignment === "B" ? 2.8 : 3.2;
    return `<line x1="${p1[0]}" y1="${p1[1]}" x2="${p2[0]}" y2="${p2[1]}" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-opacity="0.96"/>`;
  });
  return `<g>${grid(size)}${edges.join("\n")}</g>`;
}

function normalizedCoords(coords: [number, number][], size: number): [number, number][] {
  const xs = coords.map(([x]) => x);
  const ys = coords.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const padding = size * 0.12;
  const scale = (size - padding * 2) / span;
  return coords.map(([x, y]) => [
    padding + (x - minX) * scale + (size - padding * 2 - (maxX - minX) * scale) / 2,
    size - (padding + (y - minY) * scale + (size - padding * 2 - (maxY - minY) * scale) / 2),
  ]);
}

function grid(size: number): string {
  const lines: string[] = [];
  for (let index = 0; index <= 16; index += 1) {
    const p = (index / 16) * size;
    lines.push(`<line x1="${p}" y1="0" x2="${p}" y2="${size}" stroke="#e2e8f0" stroke-width="0.7"/>`);
    lines.push(`<line x1="0" y1="${p}" x2="${size}" y2="${p}" stroke="#e2e8f0" stroke-width="0.7"/>`);
  }
  return lines.join("\n");
}

function legend(x: number, y: number): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect x="0" y="-18" width="408" height="40" rx="8" fill="#ffffff" stroke="#e2e8f0"/>`,
    `<line x1="18" y1="2" x2="56" y2="2" stroke="#ff1f1f" stroke-width="4" stroke-linecap="round"/>`,
    `<text x="70" y="7" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">mountain</text>`,
    `<line x1="160" y1="2" x2="198" y2="2" stroke="#0057ff" stroke-width="4" stroke-linecap="round"/>`,
    `<text x="212" y="7" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">valley</text>`,
    `<line x1="286" y1="2" x2="324" y2="2" stroke="#111827" stroke-width="4" stroke-linecap="round"/>`,
    `<text x="338" y="7" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">border</text>`,
    `</g>`,
  ].join("\n");
}

function assignmentColor(assignment: EdgeAssignment): string {
  if (assignment === "M") return "#ff1f1f";
  if (assignment === "V") return "#0057ff";
  if (assignment === "B") return "#111827";
  return "#94a3b8";
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`${flag} requires a value`);
  return value;
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
