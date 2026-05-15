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
  type BPStudioArchetype,
  type BPStudioComplexityBucket,
} from "./bp-studio-spec.ts";

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
