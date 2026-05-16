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
import { simpleQuadrupedBPStudioSpec } from "./bp-studio-fixtures.ts";
import {
  validateBPStudioPacking,
  type BPStudioPackingValidation,
} from "./bp-studio-packing-validity.ts";
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
  optimizerUseBH: boolean;
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
      ? simpleQuadrupedBPStudioSpec(options.seed)
      : generateBPStudioSpec({
        seed: options.seed,
        archetype: options.archetype,
        bucket: options.bucket,
      });
    const adapterSpec = toAdapterSpec(spec);
    adapterSpec.optimizeLayout = true;
    adapterSpec.optimizerLayout = options.optimizerLayout;
    adapterSpec.optimizerSeed = options.seed;
    adapterSpec.optimizerUseBH = options.optimizerUseBH;
    const { metadata: adapterMetadata } = runBPStudioAdapter(adapterSpec);
    const packingValidity = validateBPStudioPacking(spec, adapterMetadata);
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
    await writeFile(join(options.out, `${label}.packing_validity.json`), JSON.stringify(packingValidity, null, 2));
    await writeFile(join(options.out, `${label}.three_panel.svg`), threePanelSvg(spec, adapterMetadata, candidate, packingValidity, options.size));
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
    optimizerUseBH: true,
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
    } else if (arg === "--no-optimizer-use-bh") {
      options.optimizerUseBH = false;
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
  packingValidity: BPStudioPackingValidation,
  panelSize: number,
): string {
  const gutter = Math.max(24, Math.round(panelSize * 0.035));
  const titleHeight = Math.max(74, Math.round(panelSize * 0.095));
  const legendGap = Math.max(12, Math.round(panelSize * 0.018));
  const legendHeight = Math.max(118, Math.round(panelSize * 0.17));
  const footerHeight = Math.max(44, Math.round(panelSize * 0.055));
  const width = panelSize * 3 + gutter * 4;
  const height = panelSize + titleHeight + legendGap + legendHeight + footerHeight + gutter;
  const panels = [
    {
      x: gutter,
      title: "1. Source Tree",
      subtitle: `${spec.archetype}, ${spec.tree.nodes.length} nodes, ${spec.tree.edges.length} edges`,
      body: sourceTreePanel(spec, panelSize),
      legend: treeLegendOutside(panelSize),
    },
    {
      x: gutter * 2 + panelSize,
      title: "2. BP Studio Optimized Packing",
      subtitle: chosenLayout(adapterMetadata)?.optimized
        ? `${adapterMetadata.spec?.optimizerLayout ?? "unknown"} optimizer${adapterMetadata.spec?.optimizerUseBH ? " + variations" : ""}, sheet ${formatSheetSize(chosenLayout(adapterMetadata)?.sheet)}, ${packingSubtitle(packingValidity)}`
        : "adapter did not report optimized layout",
      body: packingPanel(spec, adapterMetadata, panelSize),
      legend: packingLegendOutside(panelSize),
    },
    {
      x: gutter * 3 + panelSize * 2,
      title: "3. Compiler Candidate Overlay",
      subtitle: `${candidate.validity}, ${candidate.layout.pleatStrips.length} pleat corridors, active K/M ${candidate.localProbe?.kawasakiBad ?? "?"}/${candidate.localProbe?.maekawaBad ?? "?"}`,
      body: candidateOverlayPanel(spec, adapterMetadata, candidate, panelSize),
      legend: compilerLegendOutside(panelSize),
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
      `<g transform="translate(0,${titleHeight + panelSize + legendGap})">${panel.legend}</g>`,
      `</g>`,
    ].join("\n")),
    `<text x="${gutter}" y="${height - gutter * 0.85}" font-family="Inter, Arial, sans-serif" font-size="14" fill="#475569">Panel 3 overlays BP Studio's optimized flap targets and tree-length circles with our region compiler scaffold and candidate crease content. Fills/dashed outlines are debug layers, not final training labels.</text>`,
    `</svg>`,
  ].join("\n");
}

function sourceTreePanel(spec: BPStudioAdapterSpec, size: number): string {
  const graph = treeLayout(spec, size);
  const edgeItems = spec.tree.edges.filter((edge) => edge.from !== spec.tree.rootId).map((edge) => {
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
  const nodeItems = spec.tree.nodes.filter((node) => node.id !== spec.tree.rootId).map((node) => {
    const point = graph.get(node.id);
    if (!point) return "";
    const isHub = node.kind === "body" || node.kind === "hub";
    const color = node.kind === "flap" ? "#4ade80" : isHub ? "#93c5fd" : "#facc15";
    const stroke = node.kind === "flap" ? "#16a34a" : isHub ? "#2563eb" : "#ca8a04";
    const radius = node.kind === "flap" ? 11 : 14;
    return [
      `<circle cx="${point.x}" cy="${point.y}" r="${radius}" fill="${color}" fill-opacity="0.85" stroke="${stroke}" stroke-width="2"/>`,
      `<text x="${point.x + radius + 4}" y="${point.y + 4}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#0f172a">${escapeXml(node.label)}</text>`,
    ].join("\n");
  });
  return panelFrame(size, [
    ...edgeItems,
    ...nodeItems,
  ]);
}

function packingPanel(spec: BPStudioAdapterSpec, adapterMetadata: AdapterMetadata, size: number): string {
  const layout = chosenLayout(adapterMetadata) ?? adapterMetadata.inputLayout;
  const sheet = layout?.sheet ?? { width: spec.sheet.width, height: spec.sheet.height };
  const toPanel = sheetProjector(sheet.width, sheet.height, size);
  const scale = sheetScale(sheet.width, sheet.height, size);
  const terminalLengths = terminalLengthByAdapterId(spec);
  const flaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, scale, terminalLengths.get(flap.id), "#4ade80", "#16a34a", true));
  return panelFrame(size, [
    sheetGridForSheet(sheet.width, sheet.height, size),
    ...flaps,
  ], true, "bp-packing-clip");
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
  const scale = sheetScale(sheet.width, sheet.height, size);
  const terminalLengths = terminalLengthByAdapterId(spec);
  const optimizedFlaps = (layout?.flaps ?? []).map((flap) => adapterFlapMark(flap, toPanel, scale, terminalLengths.get(flap.id), "#22c55e", "#15803d", false));
  const routeOverlay = compilerRouteOverlay(candidate, size);
  const failureOverlay = compilerLocalFailureOverlay(candidate, size);
  const candidateSvg = regionCandidateToSvg(candidate, size, {
    showGrid: false,
    showLegend: false,
    showFlapTargets: false,
    showFlapBoundaries: false,
  })
    .replace(/<svg[^>]*>/, `<g>`)
    .replace(/<\/svg>\s*$/, `</g>`);
  return panelFrame(size, [
    sheetGridForSheet(sheet.width, sheet.height, size),
    candidateSvg,
    routeOverlay,
    failureOverlay,
    `<g opacity="0.72">${optimizedFlaps.join("\n")}</g>`,
  ], false, "bp-candidate-overlay-clip");
}

function compilerRouteOverlay(candidate: RegionCompletionCandidate, size: number): string {
  const toPx = (point: { x: number; y: number }): { x: number; y: number } => ({
    x: Math.round(point.x * size),
    y: Math.round((1 - point.y) * size),
  });
  const endpoints = new Map<string, { x: number; y: number; kind: "body" | "flap"; label: string }>();
  for (const body of candidate.layout.bodies) {
    endpoints.set(body.id, { ...body.center, kind: "body", label: body.id });
  }
  for (const flap of candidate.layout.flaps) {
    endpoints.set(flap.terminalId, { ...flap.center, kind: "flap", label: flap.terminalId });
    endpoints.set(flap.id, { ...flap.center, kind: "flap", label: flap.terminalId });
  }
  const anchors = [...endpoints.entries()]
    .filter(([, endpoint]) => endpoint.kind === "body")
    .map(([id, endpoint]) => {
      const p = toPx(endpoint);
      return [
        `<circle cx="${p.x}" cy="${p.y}" r="5.2" fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5"/>`,
        `<text x="${p.x + 7}" y="${p.y - 7}" font-family="Inter, Arial, sans-serif" font-size="9.5" fill="#1e3a8a">${escapeXml(id)}</text>`,
      ].join("\n");
    });
  const routes = candidate.layout.pleatStrips.map((strip) => {
    const from = endpoints.get(strip.from);
    const to = endpoints.get(strip.to);
    if (!from || !to) return "";
    const [a, b] = stripAxisEndpoints(strip);
    const direct = distanceSquared(from, a) + distanceSquared(to, b);
    const swapped = distanceSquared(from, b) + distanceSquared(to, a);
    const fromPort = direct <= swapped ? a : b;
    const toPort = direct <= swapped ? b : a;
    const fromPx = toPx(from);
    const toPxPoint = toPx(to);
    const fromPortPx = toPx(fromPort);
    const toPortPx = toPx(toPort);
    const mid = toPx({ x: (fromPort.x + toPort.x) / 2, y: (fromPort.y + toPort.y) / 2 });
    const label = strip.treeEdgeId ?? strip.id.replace(/^strip-\d+-/, "");
    return [
      `<line x1="${fromPx.x}" y1="${fromPx.y}" x2="${fromPortPx.x}" y2="${fromPortPx.y}" stroke="#64748b" stroke-width="1.2" stroke-opacity="0.52" stroke-dasharray="3 4"/>`,
      `<line x1="${toPxPoint.x}" y1="${toPxPoint.y}" x2="${toPortPx.x}" y2="${toPortPx.y}" stroke="#64748b" stroke-width="1.2" stroke-opacity="0.52" stroke-dasharray="3 4"/>`,
      `<line x1="${fromPortPx.x}" y1="${fromPortPx.y}" x2="${toPortPx.x}" y2="${toPortPx.y}" stroke="#7c3aed" stroke-width="2.2" stroke-opacity="0.82" stroke-dasharray="7 5" stroke-linecap="round"/>`,
      routePortDot(fromPortPx, from.kind),
      routePortDot(toPortPx, to.kind),
      `<g transform="translate(${mid.x},${mid.y})">`,
      `<rect x="-24" y="-10" width="48" height="16" rx="4" fill="#ffffff" fill-opacity="0.88" stroke="#ddd6fe" stroke-width="0.8"/>`,
      `<text x="0" y="2" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="8.8" font-weight="700" fill="#5b21b6">${escapeXml(shortRouteLabel(label))}</text>`,
      `</g>`,
    ].join("\n");
  });
  return `<g data-debug-overlay="compiler-routes">${anchors.join("\n")}${routes.join("\n")}</g>`;
}

function stripAxisEndpoints(strip: RegionCompletionCandidate["layout"]["pleatStrips"][number]): [{ x: number; y: number }, { x: number; y: number }] {
  const rect = strip.rect;
  if (strip.orientation === "vertical") {
    const y = (rect.y1 + rect.y2) / 2;
    return [{ x: rect.x1, y }, { x: rect.x2, y }];
  }
  const x = (rect.x1 + rect.x2) / 2;
  return [{ x, y: rect.y1 }, { x, y: rect.y2 }];
}

function routePortDot(point: { x: number; y: number }, kind: "body" | "flap"): string {
  const fill = kind === "flap" ? "#f97316" : "#2563eb";
  return `<circle cx="${point.x}" cy="${point.y}" r="4.4" fill="${fill}" fill-opacity="0.96" stroke="#ffffff" stroke-width="1.2"/>`;
}

function compilerLocalFailureOverlay(candidate: RegionCompletionCandidate, size: number): string {
  const points = candidate.localProbe?.failurePoints.slice(0, 160) ?? [];
  const dots = points.flatMap((point) => {
    const x = Math.round(point.x * size);
    const y = Math.round((1 - point.y) * size);
    if (point.kawasaki) {
      return [
        `<circle cx="${x}" cy="${y}" r="5.8" fill="#d946ef" fill-opacity="0.22" stroke="#a21caf" stroke-width="1.5"/>`,
        `<line x1="${x - 4}" y1="${y - 4}" x2="${x + 4}" y2="${y + 4}" stroke="#a21caf" stroke-width="1.2"/>`,
        `<line x1="${x - 4}" y1="${y + 4}" x2="${x + 4}" y2="${y - 4}" stroke="#a21caf" stroke-width="1.2"/>`,
      ];
    }
    return [
      `<circle cx="${x}" cy="${y}" r="4.7" fill="#fb923c" fill-opacity="0.22" stroke="#ea580c" stroke-width="1.4"/>`,
      `<line x1="${x - 4}" y1="${y}" x2="${x + 4}" y2="${y}" stroke="#ea580c" stroke-width="1.2"/>`,
      `<line x1="${x}" y1="${y - 4}" x2="${x}" y2="${y + 4}" stroke="#ea580c" stroke-width="1.2"/>`,
    ];
  });
  if (dots.length === 0) return "";
  return `<g data-debug-overlay="local-failures">${dots.join("\n")}</g>`;
}

function shortRouteLabel(label: string): string {
  return label
    .replace(/^front-/, "f-")
    .replace(/^rear-/, "r-")
    .replace("-right-", "-R-")
    .replace("-left-", "-L-")
    .replace("-hub", "");
}

function distanceSquared(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function panelFrame(size: number, body: string[], includeBorder = true, clipId?: string): string {
  return [
    clipId ? `<defs><clipPath id="${clipId}"><rect x="0" y="0" width="${size}" height="${size}"/></clipPath></defs>` : "",
    `<rect x="0" y="0" width="${size}" height="${size}" fill="white" stroke="#cbd5e1" stroke-width="1.2"/>`,
    clipId ? `<g clip-path="url(#${clipId})">` : "",
    ...body,
    clipId ? `</g>` : "",
    includeBorder ? `<rect x="0" y="0" width="${size}" height="${size}" fill="none" stroke="#0f172a" stroke-width="2"/>` : "",
  ].join("\n");
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

function terminalLengthByAdapterId(spec: BPStudioAdapterSpec): Map<number, number> {
  const adapterIds = adapterNodeIds(spec);
  const incomingLength = new Map(spec.tree.edges.map((edge) => [edge.to, edge.length]));
  const result = new Map<number, number>();
  for (const flap of spec.layout.flaps) {
    const adapterId = adapterIds.get(flap.nodeId);
    if (adapterId === undefined) continue;
    result.set(adapterId, incomingLength.get(flap.nodeId) ?? flap.terminalRadius ?? 0);
  }
  return result;
}

function chosenLayout(adapterMetadata: AdapterMetadata): NonNullable<AdapterMetadata["layout"]> | NonNullable<AdapterMetadata["optimizedLayout"]> | undefined {
  return adapterMetadata.optimizedLayout ?? adapterMetadata.layout;
}

function formatSheetSize(sheet: { width?: number; height?: number } | undefined): string {
  if (!sheet?.width || !sheet?.height) return "unknown";
  return `${round2(sheet.width)} x ${round2(sheet.height)}`;
}

function packingSubtitle(validity: BPStudioPackingValidation): string {
  const status = validity.ok ? "no circle overlaps" : `${validity.metrics.overlapCount} circle overlaps`;
  const outside = validity.metrics.outsideCount ? `, ${validity.metrics.outsideCount} boundary warnings` : "";
  const minGap = validity.metrics.minGap === null ? "" : `, min gap ${round2(validity.metrics.minGap)}`;
  return `${status}${outside}${minGap}`;
}

function sheetGridForSheet(width: number, height: number, size: number): string {
  const project = sheetProjector(width, height, size);
  const xStep = gridDisplayStep(width);
  const yStep = gridDisplayStep(height);
  const vertical: string[] = [];
  const horizontal: string[] = [];
  for (let x = 0; x <= width + 1e-9; x += xStep) {
    const p1 = project({ x, y: 0 });
    const p2 = project({ x, y: height });
    const major = isMultiple(x, Math.max(1, xStep * 4));
    vertical.push(`<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${major ? "#64748b" : "#94a3b8"}" stroke-opacity="${major ? 0.30 : 0.24}" stroke-width="${major ? 0.95 : 0.7}"/>`);
  }
  for (let y = 0; y <= height + 1e-9; y += yStep) {
    const p1 = project({ x: 0, y });
    const p2 = project({ x: width, y });
    const major = isMultiple(y, Math.max(1, yStep * 4));
    horizontal.push(`<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${major ? "#64748b" : "#94a3b8"}" stroke-opacity="${major ? 0.30 : 0.24}" stroke-width="${major ? 0.95 : 0.7}"/>`);
  }
  return [...vertical, ...horizontal].join("\n");
}

function gridDisplayStep(span: number): number {
  if (span <= 32) return 1;
  if (span <= 64) return 2;
  if (span <= 128) return 4;
  return Math.max(8, Math.ceil(span / 32));
}

function isMultiple(value: number, step: number): boolean {
  return Math.abs(value / step - Math.round(value / step)) < 1e-8;
}

function treeLayout(spec: BPStudioAdapterSpec, size: number): Map<string, { x: number; y: number }> {
  if (spec.sampler.grammar === "debug-simple-animal") return simpleAnimalTreeLayout(size);

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

function simpleAnimalTreeLayout(size: number): Map<string, { x: number; y: number }> {
  const cx = size * 0.5;
  const topHubY = size * 0.32;
  const rearHubY = size * 0.56;
  const sideDx = size * 0.115;
  return new Map([
    ["root", { x: cx, y: topHubY }],
    ["front-hub", { x: cx, y: topHubY }],
    ["head", { x: cx, y: topHubY - size * 0.12 }],
    ["front-left-leg", { x: cx - sideDx, y: topHubY }],
    ["front-right-leg", { x: cx + sideDx, y: topHubY }],
    ["rear-hub", { x: cx, y: rearHubY }],
    ["rear-left-leg", { x: cx - sideDx, y: rearHubY }],
    ["rear-right-leg", { x: cx + sideDx, y: rearHubY }],
    ["tail", { x: cx, y: rearHubY + size * 0.22 }],
  ]);
}

function sheetProjector(width: number, height: number, size: number): (point: { x: number; y: number }) => { x: number; y: number } {
  const maxDimension = Math.max(width, height);
  const scale = sheetScale(width, height, size);
  const offsetX = (size - width * scale) / 2;
  const offsetY = (size - height * scale) / 2;
  return (point) => ({
    x: offsetX + point.x * scale,
    y: offsetY + (height - point.y) * scale,
  });
}

function sheetScale(width: number, height: number, size: number): number {
  return size / Math.max(width, height);
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
  scale: number,
  flapLength: number | undefined,
  fill: string,
  stroke: string,
  showLabel: boolean,
): string {
  const center = project(centerOfAdapterFlap(flap));
  const lengthRadius = Math.max(0, (flapLength ?? 0) * scale);
  const pointRadius = 4.2;
  const pointArm = 8;
  return [
    lengthRadius > 0 ? `<circle cx="${center.x}" cy="${center.y}" r="${lengthRadius}" fill="none" stroke="${stroke}" stroke-width="1.6" stroke-opacity="0.78"/>` : "",
    `<circle cx="${center.x}" cy="${center.y}" r="${pointRadius}" fill="${fill}" fill-opacity="0.9" stroke="${stroke}" stroke-width="1.6"/>`,
    `<line x1="${center.x - pointArm}" y1="${center.y}" x2="${center.x + pointArm}" y2="${center.y}" stroke="${stroke}" stroke-width="1.4" stroke-linecap="round"/>`,
    `<line x1="${center.x}" y1="${center.y - pointArm}" x2="${center.x}" y2="${center.y + pointArm}" stroke="${stroke}" stroke-width="1.4" stroke-linecap="round"/>`,
    lengthRadius > 0 && showLabel ? `<text x="${center.x + lengthRadius + 4}" y="${center.y + 4}" font-family="Inter, Arial, sans-serif" font-size="10.5" fill="#064e3b">L ${flapLength}</text>` : "",
    showLabel ? `<text x="${center.x + 6}" y="${center.y - 6}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#064e3b">id ${flap.id}</text>` : "",
  ].join("\n");
}

function treeLegendOutside(size: number): string {
  const y = 34;
  return [
    legendBand(size, "Tree legend"),
    `<circle cx="18" cy="${y}" r="8" fill="#93c5fd" stroke="#2563eb" stroke-width="2"/><text x="34" y="${y + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">body / hub node</text>`,
    `<circle cx="190" cy="${y}" r="7" fill="#4ade80" stroke="#16a34a" stroke-width="2"/><text x="206" y="${y + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap node</text>`,
    `<line x1="340" y1="${y}" x2="362" y2="${y}" stroke="#64748b" stroke-width="2"/><text x="372" y="${y + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">tree edge</text>`,
    `<rect x="486" y="${y - 8}" width="28" height="16" rx="4" fill="#ffffff" stroke="#cbd5e1"/><text x="524" y="${y + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">edge length</text>`,
  ].join("\n");
}

function packingLegendOutside(size: number): string {
  const y1 = 34;
  const y2 = 72;
  return [
    legendBand(size, "Packing legend"),
    `<circle cx="20" cy="${y1}" r="4.2" fill="#4ade80" fill-opacity="0.9" stroke="#16a34a" stroke-width="1.6"/><line x1="12" y1="${y1}" x2="28" y2="${y1}" stroke="#16a34a" stroke-width="1.4"/><line x1="20" y1="${y1 - 8}" x2="20" y2="${y1 + 8}" stroke="#16a34a" stroke-width="1.4"/><text x="42" y="${y1 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">optimized flap terminal point</text>`,
    `<circle cx="274" cy="${y1}" r="17" fill="none" stroke="#16a34a" stroke-width="1.6"/><text x="302" y="${y1 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">flap length circle</text>`,
    `<text x="18" y="${y2 + 4}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#64748b">Panel shows only BP Studio optimized flap packing, not inferred hub points.</text>`,
  ].join("\n");
}

function compilerLegendOutside(size: number): string {
  const y1 = 34;
  const y2 = 72;
  const y3 = 102;
  return [
    legendBand(size, "Compiler overlay legend"),
    `<rect x="16" y="${y1 - 9}" width="22" height="18" fill="#ffdf4d" fill-opacity="0.72" stroke="#ca8a04" stroke-width="0.8"/><text x="48" y="${y1 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">pleat corridor</text>`,
    `<rect x="170" y="${y1 - 9}" width="22" height="18" fill="#bfdbfe" fill-opacity="0.76" stroke="#2563eb" stroke-width="0.8"/><text x="202" y="${y1 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">body panel</text>`,
    `<line x1="342" y1="${y1}" x2="374" y2="${y1}" stroke="#ff1f1f" stroke-width="3" stroke-linecap="round"/><text x="384" y="${y1 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">mountain</text>`,
    `<line x1="16" y1="${y2}" x2="48" y2="${y2}" stroke="#0057ff" stroke-width="3" stroke-linecap="round"/><text x="58" y="${y2 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">valley</text>`,
    `<line x1="170" y1="${y2}" x2="202" y2="${y2}" stroke="#0057ff" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round"/><text x="212" y="${y2 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">debug boundary</text>`,
    `<text x="350" y="${y2 + 4}" font-family="Inter, Arial, sans-serif" font-size="11" fill="#64748b">Fills are scaffold/debug only</text>`,
    `<line x1="16" y1="${y3}" x2="48" y2="${y3}" stroke="#7c3aed" stroke-width="2.2" stroke-dasharray="7 5" stroke-linecap="round"/><text x="58" y="${y3 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">selected route lane</text>`,
    `<circle cx="184" cy="${y3}" r="4.4" fill="#f97316" stroke="#ffffff" stroke-width="1.2"/><text x="198" y="${y3 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">circle contact</text>`,
    `<line x1="310" y1="${y3}" x2="340" y2="${y3}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3 4"/><text x="350" y="${y3 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">center-to-port guide</text>`,
    `<circle cx="498" cy="${y3}" r="5.8" fill="#d946ef" fill-opacity="0.22" stroke="#a21caf" stroke-width="1.5"/><line x1="494" y1="${y3 - 4}" x2="502" y2="${y3 + 4}" stroke="#a21caf" stroke-width="1.2"/><line x1="494" y1="${y3 + 4}" x2="502" y2="${y3 - 4}" stroke="#a21caf" stroke-width="1.2"/><text x="512" y="${y3 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">K</text>`,
    `<circle cx="548" cy="${y3}" r="4.7" fill="#fb923c" fill-opacity="0.22" stroke="#ea580c" stroke-width="1.4"/><line x1="544" y1="${y3}" x2="552" y2="${y3}" stroke="#ea580c" stroke-width="1.2"/><line x1="548" y1="${y3 - 4}" x2="548" y2="${y3 + 4}" stroke="#ea580c" stroke-width="1.2"/><text x="562" y="${y3 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#0f172a">M failures</text>`,
  ].join("\n");
}

function legendBand(size: number, title: string): string {
  return [
    `<rect x="0" y="0" width="${size}" height="116" rx="8" fill="#ffffff" fill-opacity="0.96" stroke="#cbd5e1"/>`,
    `<text x="14" y="18" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="#0f172a">${escapeXml(title)}</text>`,
  ].join("\n");
}

function round2(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
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
