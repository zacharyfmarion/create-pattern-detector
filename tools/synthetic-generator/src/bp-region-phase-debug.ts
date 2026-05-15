import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  compileRegionCandidate,
  regionCandidateToSvg,
  regionPhaseProblem,
  solveRegionPleatStripPhases,
} from "./bp-region-compiler.ts";
import { sequenceToString } from "./bp-port-assignment-solver.ts";
import type { RegionLayout } from "./bp-completion-contracts.ts";

interface Options {
  out: string;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const layout = constrainedPhaseLayout();
  const raw = compileRegionCandidate(layout, { solvePortPhases: false });
  const solved = compileRegionCandidate(layout);
  const phase = solveRegionPleatStripPhases(layout);
  const problem = regionPhaseProblem(layout);
  await mkdir(dirname(options.out), { recursive: true });
  await writeFile(options.out, debugSvg(layout, raw, solved));
  console.log(JSON.stringify({
    out: options.out,
    phaseOk: phase.ok,
    assignments: phase.solver?.assignments ?? {},
    joins: phase.solver?.joins ?? [],
    regionSequences: Object.fromEntries(problem.regions.map((region) => [
      region.id,
      region.states.map((state) => ({
        state: state.id,
        sequence: sequenceToString(state.ports[0]?.sequence ?? []),
      })),
    ])),
  }, null, 2));
}

function parseArgs(args: string[]): Options {
  const options: Options = { out: "/tmp/bp-region-phase-debug/region-phase-debug.svg" };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--out") {
      options.out = requiredValue(args[++index], "--out");
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return options;
}

function debugSvg(
  layout: RegionLayout,
  raw: ReturnType<typeof compileRegionCandidate>,
  solved: ReturnType<typeof compileRegionCandidate>,
): string {
  const size = 620;
  const width = 1580;
  const height = 980;
  const gap = 44;
  const x1 = 56;
  const x2 = x1 + size + gap;
  const y = 220;
  const rawStripB = raw.layout.pleatStrips.find((strip) => strip.id === "strip-b");
  const solvedStripB = solved.layout.pleatStrips.find((strip) => strip.id === "strip-b");
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="${x1}" y="48" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="#0f172a">Region strip phase solve - real compiler slice</text>`,
    `<text x="${x1}" y="78" font-family="Inter, Arial, sans-serif" font-size="15" fill="#475569">This uses actual RegionLayout pleat strips. The only change between panels is solving the strip-port phase constraint before emitting creases.</text>`,
    legend(x1, 112),
    panel(x1, y, size, "1. Before phase solve", "strip-a end exposes VMVMV, strip-b start exposes MVMVM, so the body-facing ports disagree.", raw, [
      `strip-b phase: ${rawStripB?.phase ?? "missing"}`,
      `strip-b start: ${rawStripB?.startAssignment ?? "missing"}`,
      `constraint: strip-a:end = strip-b:start`,
    ]),
    panel(x2, y, size, "2. After phase solve", "The solver flips strip-b as a whole region state, so the exposed sequence becomes VMVMV.", solved, [
      `strip-b phase: ${solvedStripB?.phase ?? "missing"}`,
      `strip-b start: ${solvedStripB?.startAssignment ?? "missing"}`,
      `accepted join: direct`,
    ]),
    `<g transform="translate(${x2 + size + 30},${y + 110})">`,
    `<rect width="160" height="230" rx="8" fill="#ffffff" stroke="#cbd5e1"/>`,
    `<text x="18" y="34" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="800" fill="#0f172a">What changed?</text>`,
    `<text x="18" y="72" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">No geometry moved.</text>`,
    `<text x="18" y="100" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">No single crease</text>`,
    `<text x="18" y="120" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">was recolored.</text>`,
    `<text x="18" y="158" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">The selected</text>`,
    `<text x="18" y="178" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">strip-b state</text>`,
    `<text x="18" y="198" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">changed phase.</text>`,
    `</g>`,
    `</svg>`,
  ].join("\n");
}

function panel(
  x: number,
  y: number,
  size: number,
  title: string,
  subtitle: string,
  candidate: ReturnType<typeof compileRegionCandidate>,
  notes: string[],
): string {
  const imageSvg = regionCandidateToSvg(candidate, size, {
    showLegend: false,
    showFlapTargets: false,
    showFlapBoundaries: false,
  });
  const href = Buffer.from(imageSvg).toString("base64");
  return [
    `<g transform="translate(${x},${y})">`,
    `<text x="0" y="-44" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="800" fill="#0f172a">${escapeXml(title)}</text>`,
    `<text x="0" y="-18" font-family="Inter, Arial, sans-serif" font-size="13" fill="#475569">${escapeXml(subtitle)}</text>`,
    `<rect x="-10" y="-10" width="${size + 20}" height="${size + 128}" rx="10" fill="#ffffff" stroke="#cbd5e1"/>`,
    `<image href="data:image/svg+xml;base64,${href}" x="0" y="0" width="${size}" height="${size}"/>`,
    ...notes.map((note, index) => `<text x="18" y="${size + 36 + index * 24}" font-family="Menlo, Consolas, monospace" font-size="13" fill="#334155">${escapeXml(note)}</text>`),
    `</g>`,
  ].join("\n");
}

function legend(x: number, y: number): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect x="0" y="-18" width="612" height="42" rx="8" fill="#ffffff" stroke="#e2e8f0"/>`,
    `<line x1="20" y1="3" x2="58" y2="3" stroke="#ff1f1f" stroke-width="4" stroke-linecap="round"/>`,
    `<text x="72" y="8" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">mountain crease</text>`,
    `<line x1="206" y1="3" x2="244" y2="3" stroke="#0057ff" stroke-width="4" stroke-linecap="round"/>`,
    `<text x="258" y="8" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">valley crease</text>`,
    `<rect x="390" y="-8" width="28" height="20" fill="#facc15" fill-opacity="0.42" stroke="#ca8a04"/>`,
    `<text x="432" y="8" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">pleat strip region</text>`,
    `</g>`,
  ].join("\n");
}

function constrainedPhaseLayout(): RegionLayout {
  return {
    id: "region-phase-debug-layout",
    sourceLayoutId: "region-phase-debug-layout",
    gridSize: 16,
    axis: "horizontal",
    bodies: [{
      id: "body",
      rect: { x1: 0.4375, y1: 0.375, x2: 0.5625, y2: 0.625 },
      center: { x: 0.5, y: 0.5 },
    }],
    flaps: [],
    boundaryPorts: [],
    pleatStrips: [
      {
        id: "strip-a",
        from: "flap-a",
        to: "body",
        rect: { x1: 2 / 16, y1: 5 / 16, x2: 8 / 16, y2: 13 / 16 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "V",
      },
      {
        id: "strip-b",
        from: "body",
        to: "flap-b",
        rect: { x1: 8 / 16, y1: 5 / 16, x2: 14 / 16, y2: 13 / 16 },
        orientation: "vertical",
        pitch: 1 / 16,
        phase: 0,
        startAssignment: "M",
      },
    ],
    portConstraints: [{
      id: "body-port-phase",
      aStripId: "strip-a",
      aSide: "end",
      bStripId: "strip-b",
      bSide: "start",
    }],
  };
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
