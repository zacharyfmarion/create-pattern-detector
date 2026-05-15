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
  BP_STUDIO_SPEC_SCHEMA_VERSION,
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
  simpleAnimal: boolean;
  archetype: BPStudioArchetype;
  bucket: BPStudioComplexityBucket;
  fixture: RegionFixtureName;
  optimizerLayout: "view" | "random";
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
    const spec = options.simpleAnimal
      ? simpleAnimalSpec(options.seed)
      : generateBPStudioSpec({
        seed: options.seed,
        archetype: options.archetype,
        bucket: options.bucket,
      });
    const adapterSpec = toAdapterSpec(spec);
    adapterSpec.optimizeLayout = true;
    adapterSpec.optimizerLayout = options.optimizerLayout;
    adapterSpec.optimizerSeed = options.seed;
    const { metadata: adapterMetadata } = runBPStudioAdapter(adapterSpec);
    const completionLayout = regularizeBPStudioLayout(spec, {
      adapterSpec,
      adapterMetadata,
      layoutId: `${spec.id}-optimized`,
    });
    const layout = regionLayoutFromCompletionLayout(completionLayout);
    const candidate = compileRegionCandidate(layout);
    const svg = regionCandidateToSvg(candidate, options.size);
    const label = `${options.simpleAnimal ? "simple-animal" : `bp-studio-${spec.archetype}`}-${spec.expectedComplexity.bucket}-${options.optimizerLayout}-${options.seed}`;
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
    await writeFile(join(options.out, `${label}.adapter_metadata.json`), JSON.stringify(adapterMetadata, null, 2));
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
    simpleAnimal: false,
    archetype: "insect",
    bucket: "small",
    fixture: "insect-lite",
    optimizerLayout: "view",
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
    } else if (arg === "--simple-animal") {
      options.simpleAnimal = true;
      options.bpStudioSample = true;
    } else if (arg === "--archetype") {
      options.archetype = parseArchetype(args[++index]);
    } else if (arg === "--bucket") {
      options.bucket = parseBucket(args[++index]);
    } else if (arg === "--optimizer-layout") {
      options.optimizerLayout = parseOptimizerLayout(args[++index]);
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

function parseOptimizerLayout(value: string | undefined): "view" | "random" {
  const layout = requiredValue(value, "--optimizer-layout");
  if (layout === "view" || layout === "random") return layout;
  throw new Error("--optimizer-layout must be either 'view' or 'random'");
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

function simpleAnimalSpec(seed: number): BPStudioAdapterSpec {
  const nodes: BPStudioAdapterSpec["tree"]["nodes"] = [
    { id: "root", kind: "root", label: "root" },
    { id: "body", kind: "body", label: "body", width: 8, height: 4, tags: ["torso"] },
    { id: "head", kind: "flap", label: "head", width: 4, height: 4, tags: ["terminal"] },
    { id: "tail", kind: "flap", label: "tail", width: 3, height: 3, tags: ["terminal"] },
    { id: "front-left-leg", kind: "flap", label: "front left leg", width: 3, height: 5, tags: ["terminal", "leg"] },
    { id: "front-right-leg", kind: "flap", label: "front right leg", width: 3, height: 5, tags: ["terminal", "leg"] },
    { id: "rear-left-leg", kind: "flap", label: "rear left leg", width: 3, height: 5, tags: ["terminal", "leg"] },
    { id: "rear-right-leg", kind: "flap", label: "rear right leg", width: 3, height: 5, tags: ["terminal", "leg"] },
  ];
  const appendages = [
    ["body-head", "head", 5, "horizontal"],
    ["body-tail", "tail", 5, "horizontal"],
    ["body-front-left-leg", "front-left-leg", 4, "vertical"],
    ["body-front-right-leg", "front-right-leg", 4, "vertical"],
    ["body-rear-left-leg", "rear-left-leg", 4, "vertical"],
    ["body-rear-right-leg", "rear-right-leg", 4, "vertical"],
  ] as const;
  const flapPlacements: BPStudioAdapterSpec["layout"]["flaps"] = [
    { nodeId: "head", label: "head", class: "terminal", terminal: { x: 26, y: 16 }, side: "right", width: 4, height: 4, elevation: 0, terminalRadius: 8, priority: 1 },
    { nodeId: "tail", label: "tail", class: "terminal", terminal: { x: 6, y: 16 }, side: "left", width: 3, height: 3, elevation: 0, terminalRadius: 8, priority: 2 },
    { nodeId: "front-left-leg", label: "front left leg", class: "terminal", terminal: { x: 12, y: 25 }, side: "top", width: 3, height: 5, elevation: 0, terminalRadius: 7, priority: 3, mirroredWith: "front-right-leg" },
    { nodeId: "front-right-leg", label: "front right leg", class: "terminal", terminal: { x: 20, y: 25 }, side: "top", width: 3, height: 5, elevation: 0, terminalRadius: 7, priority: 4, mirroredWith: "front-left-leg" },
    { nodeId: "rear-left-leg", label: "rear left leg", class: "terminal", terminal: { x: 12, y: 7 }, side: "bottom", width: 3, height: 5, elevation: 0, terminalRadius: 7, priority: 5, mirroredWith: "rear-right-leg" },
    { nodeId: "rear-right-leg", label: "rear right leg", class: "terminal", terminal: { x: 20, y: 7 }, side: "bottom", width: 3, height: 5, elevation: 0, terminalRadius: 7, priority: 6, mirroredWith: "rear-left-leg" },
  ];
  return {
    schemaVersion: BP_STUDIO_SPEC_SCHEMA_VERSION,
    id: `simple-animal-${seed}`,
    seed,
    archetype: "quadruped",
    expectedComplexity: {
      bucket: "small",
      targetFlaps: [6, 6],
      targetTreeEdges: [7, 7],
      targetCreases: [80, 220],
      expectedGridSize: [16, 32],
    },
    sheet: {
      width: 32,
      height: 32,
      gridSize: 32,
      margin: 1,
      coordinateSystem: "integer-grid",
      unit: "bp-grid",
    },
    tree: {
      rootId: "root",
      nodes,
      edges: [
        { id: "root-body", from: "root", to: "body", length: 0, role: "body", width: 4 },
        ...appendages.map(([id, to, length]) => ({
          id,
          from: "body",
          to,
          length,
          role: "appendage" as const,
          width: 2,
        })),
      ],
    },
    layout: {
      symmetry: "bilateral-x",
      bodies: [
        { nodeId: "body", label: "body", center: { x: 16, y: 16 }, width: 8, height: 4, elevation: 0, tags: ["torso"] },
      ],
      flaps: flapPlacements,
      rivers: appendages.map(([id, , , preferredAxis]) => ({
        edgeId: id,
        from: "body",
        to: id.replace("body-", ""),
        width: 2,
        preferredAxis,
        bend: preferredAxis === "horizontal" ? "direct" : "dogleg",
        clearance: 1,
      })),
      optimizerHints: {
        objective: "compact-flaps",
        keepSymmetry: false,
        allowTerminalRelaxation: true,
        maxIterations: 5000,
        topK: 5,
      },
    },
    sampler: {
      samplerVersion: "bp-studio-sampler/v1",
      grammar: "debug-simple-animal",
      seed,
      variation: seed,
      symmetry: "bilateral-x",
      notes: ["Debug fixture: body, head, tail, and four legs only."],
    },
  };
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
      subtitle: adapterMetadata.layout?.optimized
        ? `${adapterMetadata.spec?.optimizerLayout ?? "unknown"} optimizer output, sheet ${formatSheetSize(chosenLayout(adapterMetadata)?.sheet)}`
        : "adapter did not report optimized layout",
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
  const graph = treeLayout(spec, size);
  const edgeItems = spec.tree.edges.map((edge) => {
    const a = graph.get(edge.from);
    const b = graph.get(edge.to);
    if (!a || !b) return "";
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    return [
      `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#64748b" stroke-width="2.2" stroke-linecap="round"/>`,
      `<g transform="translate(${mid.x},${mid.y})">`,
      `<rect x="-16" y="-12" width="32" height="17" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="0.7"/>`,
      `<text x="0" y="1" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="10.5" font-weight="700" fill="#334155">L ${edge.length}</text>`,
      `</g>`,
    ].join("\n");
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
  const layout = chosenLayout(adapterMetadata) ?? adapterMetadata.inputLayout;
  const sheet = layout?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const toPanel = sheetProjector(sheet.width, sheet.height, size);
  const edges = packedTreeOverlay(spec, layout, toPanel);
  const nodes = (layout?.nodes ?? [])
    .filter((node) => !node.isLeaf)
    .map((node) => adapterNodeBoundsMark(node, toPanel, String(node.id)));
  const flaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, "#4ade80", "#16a34a", true));
  return panelFrame(size, [
    sheetGrid(size, 16),
    ...edges,
    ...nodes,
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
  const layout = chosenLayout(adapterMetadata) ?? adapterMetadata.inputLayout;
  const sheet = layout?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const toPanel = sheetProjector(sheet.width, sheet.height, size);
  const optimizedNodes = (layout?.nodes ?? [])
    .filter((node) => !node.isLeaf)
    .map((node) => adapterNodeBoundsMark(node, toPanel, String(node.id)));
  const optimizedFlaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, "#22c55e", "#15803d", false));
  const candidateSvg = regionCandidateToSvg(candidate, size)
    .replace(/<svg[^>]*>/, `<g>`)
    .replace(/<\/svg>\s*$/, `</g>`);
  return panelFrame(size, [
    candidateSvg,
    `<g opacity="0.56">${optimizedNodes.join("\n")}</g>`,
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

function packedTreeOverlay(
  spec: BPStudioAdapterSpec,
  layout: AdapterMetadata["layout"] | AdapterMetadata["optimizedLayout"] | AdapterMetadata["inputLayout"] | undefined,
  project: (point: { x: number; y: number }) => { x: number; y: number },
): string[] {
  if (!layout) return [];
  const adapterIds = adapterNodeIds(spec);
  const nodeByAdapterId = new Map([...adapterIds.entries()].map(([nodeId, adapterId]) => [adapterId, nodeId]));
  const points = new Map<string, { x: number; y: number }>();
  for (const flap of layout.flaps ?? []) {
    const nodeId = nodeByAdapterId.get(flap.id);
    if (nodeId) points.set(nodeId, centerOfAdapterFlap(flap));
  }
  const edges = spec.tree.edges.filter((edge) => edge.from !== spec.tree.rootId && edge.to !== spec.tree.rootId);
  for (let pass = 0; pass < spec.tree.nodes.length; pass += 1) {
    let changed = false;
    for (const node of spec.tree.nodes) {
      if (points.has(node.id) || node.id === spec.tree.rootId) continue;
      const neighbors = edges
        .flatMap((edge) => edge.from === node.id ? [edge.to] : edge.to === node.id ? [edge.from] : [])
        .map((neighborId) => points.get(neighborId))
        .filter((point): point is { x: number; y: number } => Boolean(point));
      if (!neighbors.length) continue;
      points.set(node.id, {
        x: mean(neighbors.map((point) => point.x)),
        y: mean(neighbors.map((point) => point.y)),
      });
      changed = true;
    }
    if (!changed) break;
  }
  return edges.map((edge) => {
    const a = points.get(edge.from);
    const b = points.get(edge.to);
    if (!a || !b) return "";
    const pa = project(a);
    const pb = project(b);
    const mid = { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2 };
    return [
      `<line x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}" stroke="#64748b" stroke-width="1.8" stroke-linecap="round" stroke-opacity="0.72"/>`,
      `<g transform="translate(${mid.x},${mid.y})">`,
      `<rect x="-14" y="-11" width="28" height="16" rx="4" fill="#ffffff" fill-opacity="0.90" stroke="#cbd5e1" stroke-width="0.7"/>`,
      `<text x="0" y="1" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="10" font-weight="700" fill="#334155">L ${edge.length}</text>`,
      `</g>`,
    ].join("\n");
  });
}

function adapterNodeIds(spec: BPStudioAdapterSpec): Map<string, number> {
  const referenced = new Set<string>();
  for (const edge of spec.tree.edges) {
    if (edge.from === spec.tree.rootId || edge.to === spec.tree.rootId) continue;
    referenced.add(edge.from);
    referenced.add(edge.to);
  }
  for (const flap of spec.layout.flaps) referenced.add(flap.nodeId);
  return new Map(
    spec.tree.nodes
      .map((node) => node.id)
      .filter((id) => id !== spec.tree.rootId && referenced.has(id))
      .map((id, index) => [id, index]),
  );
}

function chosenLayout(adapterMetadata: AdapterMetadata): NonNullable<AdapterMetadata["layout"]> | NonNullable<AdapterMetadata["optimizedLayout"]> | undefined {
  return adapterMetadata.optimizedLayout ?? adapterMetadata.layout;
}

function formatSheetSize(sheet: { width?: number; height?: number } | undefined): string {
  if (!sheet?.width || !sheet?.height) return "unknown";
  return `${round2(sheet.width)} x ${round2(sheet.height)}`;
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

function treeLayout(spec: BPStudioAdapterSpec, size: number): Map<string, { x: number; y: number }> {
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
  const circleRadius = Math.max(4, Math.min(width, height) * 0.45);
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}" fill-opacity="0.22" stroke="${stroke}" stroke-width="1.8" stroke-dasharray="5 4"/>`,
    `<circle cx="${center.x}" cy="${center.y}" r="${circleRadius}" fill="none" stroke="${stroke}" stroke-width="1.4" stroke-opacity="0.75"/>`,
    showLabel ? `<text x="${center.x + 6}" y="${center.y - 6}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#064e3b">id ${flap.id}</text>` : "",
  ].join("\n");
}

function adapterNodeBoundsMark(
  node: { id: number; bounds: { top: number; right: number; bottom: number; left: number } },
  project: (point: { x: number; y: number }) => { x: number; y: number },
  label: string,
): string {
  const p1 = project({ x: node.bounds.left, y: node.bounds.bottom });
  const p2 = project({ x: node.bounds.right, y: node.bounds.top });
  const x = Math.min(p1.x, p2.x);
  const y = Math.min(p1.y, p2.y);
  const width = Math.max(0, Math.abs(p2.x - p1.x));
  const height = Math.max(0, Math.abs(p2.y - p1.y));
  if (width < 2 || height < 2) return "";
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="#60a5fa" fill-opacity="0.10" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="8 5"/>`,
    `<text x="${x + 5}" y="${Math.max(13, y + 14)}" font-family="Inter, Arial, sans-serif" font-size="10.5" fill="#1d4ed8">node ${escapeXml(label)} bounds</text>`,
  ].join("\n");
}

function treeLegend(size: number): string {
  return [
    `<g transform="translate(${size - 190},18)">`,
    `<rect width="172" height="108" rx="8" fill="white" fill-opacity="0.9" stroke="#cbd5e1"/>`,
    `<circle cx="18" cy="24" r="8" fill="#93c5fd" stroke="#2563eb" stroke-width="2"/><text x="34" y="28" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">body / hub node</text>`,
    `<circle cx="18" cy="50" r="7" fill="#4ade80" stroke="#16a34a" stroke-width="2"/><text x="34" y="54" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap node</text>`,
    `<line x1="10" y1="72" x2="27" y2="72" stroke="#64748b" stroke-width="2"/><text x="34" y="76" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">tree edge</text>`,
    `<rect x="10" y="88" width="26" height="14" rx="4" fill="#ffffff" stroke="#cbd5e1"/><text x="42" y="99" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">edge length</text>`,
    `</g>`,
  ].join("\n");
}

function packingLegend(size: number): string {
  return [
    `<g transform="translate(${size - 224},18)">`,
    `<rect width="206" height="116" rx="8" fill="white" fill-opacity="0.9" stroke="#cbd5e1"/>`,
    `<rect x="12" y="16" width="22" height="18" fill="#4ade80" fill-opacity="0.22" stroke="#16a34a" stroke-dasharray="5 4"/><text x="44" y="30" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">optimized flap square</text>`,
    `<circle cx="23" cy="52" r="11" fill="none" stroke="#16a34a" stroke-width="1.4"/><text x="44" y="56" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap circle/radius guide</text>`,
    `<rect x="12" y="72" width="22" height="14" fill="#60a5fa" fill-opacity="0.10" stroke="#2563eb" stroke-dasharray="8 5"/><text x="44" y="84" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">BP Studio node bounds</text>`,
    `<line x1="12" y1="98" x2="34" y2="98" stroke="#64748b" stroke-width="1.8"/><text x="44" y="102" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">inferred tree edge</text>`,
    `</g>`,
  ].join("\n");
}

function round2(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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
