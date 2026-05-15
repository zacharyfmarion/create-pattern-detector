import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { regularizeBPStudioLayout } from "./bp-completion.ts";
import {
  compileRegionCandidate,
  fixtureRegionLayout,
  regionLayoutFromCompletionLayout,
  regionCandidateToSvg,
  type RegionFixtureName,
} from "./bp-region-compiler.ts";
import { generateBPStudioSpec } from "./bp-studio-sampler.ts";
import { runBPStudioAdapter, toAdapterSpec } from "./bp-studio-realistic.ts";
import {
  BP_STUDIO_ARCHETYPES,
  BP_STUDIO_COMPLEXITY_BUCKETS,
  type BPStudioAdapterSpec,
  type BPStudioArchetype,
  type BPStudioComplexityBucket,
} from "./bp-studio-spec.ts";
import type { AdapterMetadata } from "./bp-studio-realistic.ts";
import type { RegionCompletionCandidate } from "./bp-completion-contracts.ts";

const FIXTURES = ["two-flap-stretch", "three-flap-relay", "five-flap-uniaxial", "insect-lite"] as const;

interface Options {
  allFixtures: boolean;
  bpStudioSample: boolean;
  archetype: BPStudioArchetype;
  bucket: BPStudioComplexityBucket;
  fixture: RegionFixtureName;
  out: string;
  seed: number;
  size: number;
}

interface RenderedCandidate {
  label: string;
  svg: string;
  segmentCount: number;
  stripCount: number;
  validity: string;
  rejectionReasons: string[];
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  await mkdir(options.out, { recursive: true });

  const rendered: RenderedCandidate[] = [];
  if (options.bpStudioSample) {
    const spec = generateBPStudioSpec({
      seed: options.seed,
      archetype: options.archetype,
      bucket: options.bucket,
    });
    const adapterSpec = toAdapterSpec(spec);
    const { metadata: adapterMetadata } = runBPStudioAdapter(adapterSpec);
    const completionLayout = regularizeBPStudioLayout(spec, {
      adapterSpec,
      adapterMetadata,
      layoutId: `${spec.id}-optimized`,
    });
    const layout = regionLayoutFromCompletionLayout(completionLayout);
    const candidate = compileRegionCandidate(layout);
    const svg = regionCandidateToSvg(candidate, options.size);
    const label = `bp-studio-${spec.archetype}-${spec.expectedComplexity.bucket}-${options.seed}`;
    rendered.push({
      label,
      svg,
      segmentCount: candidate.segments.length,
      stripCount: candidate.layout.pleatStrips.length,
      validity: candidate.validity,
      rejectionReasons: candidate.rejectionReasons,
    });
    await writeFile(join(options.out, `${label}.svg`), svg);
    await writeFile(join(options.out, `${label}.json`), JSON.stringify(candidate, null, 2));
    await writeFile(join(options.out, `${label}.spec.json`), JSON.stringify(spec, null, 2));
    await writeFile(join(options.out, `${label}.three_panel.svg`), threePanelSvg(spec, adapterMetadata, candidate, options.size));
  } else {
    const fixtures = options.allFixtures ? [...FIXTURES] : [options.fixture];
    for (const fixture of fixtures) {
      const layout = fixtureRegionLayout(fixture);
      const candidate = compileRegionCandidate(layout);
      const svg = regionCandidateToSvg(candidate, options.size);
      rendered.push({
        label: fixture,
        svg,
        segmentCount: candidate.segments.length,
        stripCount: candidate.layout.pleatStrips.length,
        validity: candidate.validity,
        rejectionReasons: candidate.rejectionReasons,
      });
      await writeFile(join(options.out, `${fixture}.svg`), svg);
      await writeFile(join(options.out, `${fixture}.json`), JSON.stringify(candidate, null, 2));
    }
  }

  await writeFile(join(options.out, "contact_sheet.svg"), contactSheetSvg(rendered, options.size));
  console.log(JSON.stringify({
    out: options.out,
    candidates: rendered.map((item) => ({
      label: item.label,
      segmentCount: item.segmentCount,
      stripCount: item.stripCount,
      validity: item.validity,
      rejectionReasons: item.rejectionReasons,
    })),
  }, null, 2));
}

function parseArgs(args: string[]): Options {
  const options: Options = {
    allFixtures: false,
    bpStudioSample: false,
    archetype: "insect",
    bucket: "small",
    fixture: "insect-lite",
    out: "/tmp/bp-region-examples",
    seed: 1,
    size: 720,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--all-fixtures") {
      options.allFixtures = true;
    } else if (arg === "--bp-studio-sample") {
      options.bpStudioSample = true;
    } else if (arg === "--archetype") {
      options.archetype = parseArchetype(args[++index]);
    } else if (arg === "--bucket") {
      options.bucket = parseBucket(args[++index]);
    } else if (arg === "--fixture") {
      options.fixture = parseFixture(args[++index]);
    } else if (arg === "--out") {
      options.out = requiredValue(args[++index], "--out");
    } else if (arg === "--seed") {
      const value = Number(requiredValue(args[++index], "--seed"));
      if (!Number.isInteger(value)) throw new Error("--seed must be an integer");
      options.seed = value;
    } else if (arg === "--size") {
      const value = Number(requiredValue(args[++index], "--size"));
      if (!Number.isFinite(value) || value < 200) throw new Error("--size must be a number >= 200");
      options.size = value;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function parseArchetype(value: string | undefined): BPStudioArchetype {
  const archetype = requiredValue(value, "--archetype");
  if ((BP_STUDIO_ARCHETYPES as readonly string[]).includes(archetype)) return archetype as BPStudioArchetype;
  throw new Error(`Unknown archetype '${archetype}'. Expected one of: ${BP_STUDIO_ARCHETYPES.join(", ")}`);
}

function parseBucket(value: string | undefined): BPStudioComplexityBucket {
  const bucket = requiredValue(value, "--bucket");
  if ((BP_STUDIO_COMPLEXITY_BUCKETS as readonly string[]).includes(bucket)) return bucket as BPStudioComplexityBucket;
  throw new Error(`Unknown bucket '${bucket}'. Expected one of: ${BP_STUDIO_COMPLEXITY_BUCKETS.join(", ")}`);
}

function parseFixture(value: string | undefined): RegionFixtureName {
  const fixture = requiredValue(value, "--fixture");
  if ((FIXTURES as readonly string[]).includes(fixture)) return fixture as RegionFixtureName;
  throw new Error(`Unknown fixture '${fixture}'. Expected one of: ${FIXTURES.join(", ")}`);
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`${flag} requires a value`);
  return value;
}

function contactSheetSvg(items: RenderedCandidate[], size: number): string {
  const columns = Math.min(2, Math.max(1, items.length));
  const labelHeight = 56;
  const gap = 24;
  const tileWidth = size;
  const tileHeight = size + labelHeight;
  const width = columns * tileWidth + (columns + 1) * gap;
  const rows = Math.ceil(items.length / columns);
  const height = rows * tileHeight + (rows + 1) * gap;
  const tiles = items.map((item, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const x = gap + column * (tileWidth + gap);
    const y = gap + row * (tileHeight + gap);
    const data = Buffer.from(item.svg).toString("base64");
    return [
      `<rect x="${x}" y="${y}" width="${tileWidth}" height="${tileHeight}" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>`,
      `<image href="data:image/svg+xml;base64,${data}" x="${x}" y="${y}" width="${tileWidth}" height="${tileWidth}"/>`,
      `<text x="${x + 12}" y="${y + tileWidth + 24}" font-family="Inter, Arial, sans-serif" font-size="17" fill="#111827">${escapeXml(item.label)}</text>`,
      `<text x="${x + 12}" y="${y + tileWidth + 46}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#4b5563">${item.validity}, ${item.stripCount} strips, ${item.segmentCount} candidate segments</text>`,
    ].join("\n");
  });
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#f3f4f6"/>`,
    ...tiles,
    `</svg>`,
  ].join("\n");
}

function threePanelSvg(
  spec: BPStudioAdapterSpec,
  adapterMetadata: AdapterMetadata,
  candidate: RegionCompletionCandidate,
  panelSize: number,
): string {
  const gutter = Math.max(24, Math.round(panelSize * 0.035));
  const titleHeight = Math.max(74, Math.round(panelSize * 0.095));
  const footerHeight = Math.max(44, Math.round(panelSize * 0.055));
  const width = panelSize * 3 + gutter * 4;
  const height = panelSize + titleHeight + footerHeight + gutter;
  const panels = [
    {
      x: gutter,
      title: "1. Source Tree",
      subtitle: `${spec.archetype}, ${spec.tree.nodes.length} nodes, ${spec.tree.edges.length} edges`,
      body: sourceTreePanel(spec, panelSize),
    },
    {
      x: gutter * 2 + panelSize,
      title: "2. BP Studio Optimized Packing",
      subtitle: adapterMetadata.layout?.optimized ? "optimizer output: packed flap squares over tree edges" : "adapter did not report optimized layout",
      body: packingPanel(spec, adapterMetadata, panelSize),
    },
    {
      x: gutter * 3 + panelSize * 2,
      title: "3. Compiler Candidate Overlay",
      subtitle: `${candidate.validity}, ${candidate.layout.pleatStrips.length} pleat corridors`,
      body: candidateOverlayPanel(spec, adapterMetadata, candidate, panelSize),
    },
  ];
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    ...panels.map((panel) => [
      `<g transform="translate(${panel.x},${gutter})">`,
      `<text x="0" y="24" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700" fill="#0f172a">${escapeXml(panel.title)}</text>`,
      `<text x="0" y="48" font-family="Inter, Arial, sans-serif" font-size="14" fill="#475569">${escapeXml(panel.subtitle)}</text>`,
      `<g transform="translate(0,${titleHeight})">${panel.body}</g>`,
      `</g>`,
    ].join("\n")),
    `<text x="${gutter}" y="${height - gutter * 0.85}" font-family="Inter, Arial, sans-serif" font-size="14" fill="#475569">Panel 3 overlays BP Studio's optimized flap squares/circles with our region compiler scaffold and candidate crease content. Fills/dashed outlines are debug layers, not final training labels.</text>`,
    `</svg>`,
  ].join("\n");
}

function sourceTreePanel(spec: BPStudioAdapterSpec, size: number): string {
  const graph = treeLayout(spec);
  const edgeItems = spec.tree.edges.map((edge) => {
    const a = graph.get(edge.from);
    const b = graph.get(edge.to);
    if (!a || !b) return "";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#64748b" stroke-width="2.2" stroke-linecap="round"/>`;
  });
  const nodeItems = spec.tree.nodes.map((node) => {
    const point = graph.get(node.id);
    if (!point) return "";
    const color = node.kind === "flap" ? "#4ade80" : node.kind === "body" ? "#93c5fd" : "#facc15";
    const stroke = node.kind === "flap" ? "#16a34a" : node.kind === "body" ? "#2563eb" : "#ca8a04";
    const radius = node.kind === "flap" ? 11 : 14;
    return [
      `<circle cx="${point.x}" cy="${point.y}" r="${radius}" fill="${color}" fill-opacity="0.85" stroke="${stroke}" stroke-width="2"/>`,
      `<text x="${point.x + radius + 4}" y="${point.y + 4}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#0f172a">${escapeXml(node.label)}</text>`,
    ].join("\n");
  });
  return panelFrame(size, [
    ...edgeItems,
    ...nodeItems,
    treeLegend(size),
  ]);
}

function packingPanel(spec: BPStudioAdapterSpec, adapterMetadata: AdapterMetadata, size: number): string {
  const layout = adapterMetadata.layout ?? adapterMetadata.optimizedLayout ?? adapterMetadata.inputLayout;
  const sheet = layout?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const toPanel = sheetProjector(sheet.width, sheet.height, size);
  const edges = (layout?.edges ?? []).map((edge) => {
    const flaps = layout?.flaps ?? [];
    const a = flaps.find((flap) => flap.id === edge.n1);
    const b = flaps.find((flap) => flap.id === edge.n2);
    if (!a || !b) return "";
    const pa = toPanel(centerOfAdapterFlap(a));
    const pb = toPanel(centerOfAdapterFlap(b));
    return `<line x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>`;
  });
  const flaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, "#4ade80", "#16a34a", true));
  return panelFrame(size, [
    sheetGrid(size, 16),
    ...edges,
    ...flaps,
    packingLegend(size),
  ]);
}

function candidateOverlayPanel(
  spec: BPStudioAdapterSpec,
  adapterMetadata: AdapterMetadata,
  candidate: RegionCompletionCandidate,
  size: number,
): string {
  const layout = adapterMetadata.layout ?? adapterMetadata.optimizedLayout ?? adapterMetadata.inputLayout;
  const sheet = layout?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const toPanel = sheetProjector(sheet.width, sheet.height, size);
  const optimizedFlaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, "#22c55e", "#15803d", false));
  const candidateSvg = regionCandidateToSvg(candidate, size)
    .replace(/<svg[^>]*>/, `<g>`)
    .replace(/<\/svg>\s*$/, `</g>`);
  return panelFrame(size, [
    candidateSvg,
    `<g opacity="0.72">${optimizedFlaps.join("\n")}</g>`,
  ], false);
}

function panelFrame(size: number, body: string[], includeBorder = true): string {
  return [
    `<rect x="0" y="0" width="${size}" height="${size}" fill="white" stroke="#cbd5e1" stroke-width="1.2"/>`,
    ...body,
    includeBorder ? `<rect x="0" y="0" width="${size}" height="${size}" fill="none" stroke="#0f172a" stroke-width="2"/>` : "",
  ].join("\n");
}

function sheetGrid(size: number, divisions: number): string {
  return Array.from({ length: divisions + 1 }, (_, index) => {
    const v = index * size / divisions;
    const major = index % 4 === 0;
    const stroke = major ? "#94a3b8" : "#cbd5e1";
    const opacity = major ? 0.22 : 0.10;
    return [
      `<line x1="${v}" y1="0" x2="${v}" y2="${size}" stroke="${stroke}" stroke-opacity="${opacity}" stroke-width="${major ? 0.8 : 0.45}"/>`,
      `<line x1="0" y1="${v}" x2="${size}" y2="${v}" stroke="${stroke}" stroke-opacity="${opacity}" stroke-width="${major ? 0.8 : 0.45}"/>`,
    ].join("\n");
  }).join("\n");
}

function treeLayout(spec: BPStudioAdapterSpec): Map<string, { x: number; y: number }> {
  const children = new Map<string, string[]>();
  const parent = new Map<string, string>();
  for (const edge of spec.tree.edges) {
    const list = children.get(edge.from) ?? [];
    list.push(edge.to);
    children.set(edge.from, list);
    parent.set(edge.to, edge.from);
  }
  const roots = spec.tree.nodes.filter((node) => !parent.has(node.id));
  const rootId = spec.tree.rootId || roots[0]?.id || spec.tree.nodes[0]?.id;
  const levels = new Map<string, number>();
  const queue = rootId ? [rootId] : [];
  levels.set(rootId, 0);
  while (queue.length) {
    const node = queue.shift()!;
    const level = levels.get(node) ?? 0;
    for (const child of children.get(node) ?? []) {
      levels.set(child, level + 1);
      queue.push(child);
    }
  }
  const byLevel = new Map<number, string[]>();
  for (const node of spec.tree.nodes) {
    const level = levels.get(node.id) ?? 0;
    const list = byLevel.get(level) ?? [];
    list.push(node.id);
    byLevel.set(level, list);
  }
  const maxLevel = Math.max(...levels.values(), 1);
  const size = 720;
  const margin = 52;
  const result = new Map<string, { x: number; y: number }>();
  for (const [level, ids] of byLevel) {
    ids.forEach((id, index) => {
      result.set(id, {
        x: margin + level * ((size - margin * 2) / Math.max(1, maxLevel)),
        y: margin + (index + 1) * ((size - margin * 2) / (ids.length + 1)),
      });
    });
  }
  return result;
}

function sheetProjector(width: number, height: number, size: number): (point: { x: number; y: number }) => { x: number; y: number } {
  const maxDimension = Math.max(width, height);
  const scale = size / maxDimension;
  const offsetX = (size - width * scale) / 2;
  const offsetY = (size - height * scale) / 2;
  return (point) => ({
    x: offsetX + point.x * scale,
    y: offsetY + (height - point.y) * scale,
  });
}

function centerOfAdapterFlap(flap: { x: number; y: number; width?: number; height?: number }): { x: number; y: number } {
  return {
    x: flap.x + (flap.width ?? 0) / 2,
    y: flap.y + (flap.height ?? 0) / 2,
  };
}

function adapterFlapMark(
  flap: { id: number; x: number; y: number; width?: number; height?: number },
  project: (point: { x: number; y: number }) => { x: number; y: number },
  fill: string,
  stroke: string,
  showLabel: boolean,
): string {
  const p1 = project({ x: flap.x, y: flap.y });
  const p2 = project({ x: flap.x + (flap.width ?? 0), y: flap.y + (flap.height ?? 0) });
  const x = Math.min(p1.x, p2.x);
  const y = Math.min(p1.y, p2.y);
  const width = Math.max(10, Math.abs(p2.x - p1.x));
  const height = Math.max(10, Math.abs(p2.y - p1.y));
  const center = project(centerOfAdapterFlap(flap));
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}" fill-opacity="0.22" stroke="${stroke}" stroke-width="1.8" stroke-dasharray="5 4"/>`,
    `<circle cx="${center.x}" cy="${center.y}" r="${Math.max(width, height) * 0.52}" fill="none" stroke="${stroke}" stroke-width="1.4" stroke-opacity="0.75"/>`,
    showLabel ? `<text x="${center.x + 6}" y="${center.y - 6}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#064e3b">id ${flap.id}</text>` : "",
  ].join("\n");
}

function treeLegend(size: number): string {
  return [
    `<g transform="translate(${size - 190},18)">`,
    `<rect width="172" height="86" rx="8" fill="white" fill-opacity="0.9" stroke="#cbd5e1"/>`,
    `<circle cx="18" cy="24" r="8" fill="#93c5fd" stroke="#2563eb" stroke-width="2"/><text x="34" y="28" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">body / hub node</text>`,
    `<circle cx="18" cy="50" r="7" fill="#4ade80" stroke="#16a34a" stroke-width="2"/><text x="34" y="54" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap node</text>`,
    `<line x1="10" y1="72" x2="27" y2="72" stroke="#64748b" stroke-width="2"/><text x="34" y="76" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">tree edge</text>`,
    `</g>`,
  ].join("\n");
}

function packingLegend(size: number): string {
  return [
    `<g transform="translate(${size - 214},18)">`,
    `<rect width="196" height="74" rx="8" fill="white" fill-opacity="0.9" stroke="#cbd5e1"/>`,
    `<rect x="12" y="16" width="22" height="18" fill="#4ade80" fill-opacity="0.22" stroke="#16a34a" stroke-dasharray="5 4"/><text x="44" y="30" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">optimized flap square</text>`,
    `<circle cx="23" cy="52" r="11" fill="none" stroke="#16a34a" stroke-width="1.4"/><text x="44" y="56" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap circle/radius guide</text>`,
    `</g>`,
  ].join("\n");
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
