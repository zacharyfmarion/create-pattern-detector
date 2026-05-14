import { SeededRandom } from "./random.ts";
import {
  BP_STUDIO_ARCHETYPES,
  BP_STUDIO_COMPLEXITY_BUCKETS,
  BP_STUDIO_SPEC_SCHEMA_VERSION,
} from "./bp-studio-spec.ts";
import type {
  BPStudioAdapterSpec,
  BPStudioArchetype,
  BPStudioBodyPlacement,
  BPStudioComplexityBucket,
  BPStudioExpectedComplexity,
  BPStudioFlapClass,
  BPStudioFlapPlacement,
  BPStudioLayoutSpec,
  BPStudioOptimizerHints,
  BPStudioRiverBend,
  BPStudioRiverHint,
  BPStudioSheetSpec,
  BPStudioSide,
  BPStudioSymmetry,
  BPStudioTreeEdge,
  BPStudioTreeEdgeRole,
  BPStudioTreeNode,
  BPStudioTreeNodeKind,
} from "./bp-studio-spec.ts";

export interface BPStudioSamplerConfig {
  seed?: number;
  id?: string;
  archetype?: BPStudioArchetype;
  bucket?: BPStudioComplexityBucket;
  gridSize?: number;
  symmetry?: BPStudioSymmetry;
  variation?: number;
}

export interface BPStudioSpecMatrixConfig {
  seed?: number;
  archetypes?: readonly BPStudioArchetype[];
  buckets?: readonly BPStudioComplexityBucket[];
}

interface BucketProfile extends BPStudioExpectedComplexity {
  accessoryFlaps: [number, number];
  optimizerIterations: number;
  targetUtilization: [number, number];
}

interface Draft {
  seed: number;
  id?: string;
  archetype: BPStudioArchetype;
  bucket: BPStudioComplexityBucket;
  profile: BucketProfile;
  sheet: BPStudioSheetSpec;
  symmetry: BPStudioSymmetry;
  variation: number;
  rng: SeededRandom;
  nodes: BPStudioTreeNode[];
  edges: BPStudioTreeEdge[];
  bodies: BPStudioBodyPlacement[];
  flaps: BPStudioFlapPlacement[];
  anchorNodes: string[];
  notes: string[];
  nextNodeIndex: number;
  nextEdgeIndex: number;
}

interface NodeOptions {
  width?: number;
  height?: number;
  elevation?: number;
  tags?: string[];
}

interface EdgeOptions {
  width?: number;
  tags?: string[];
}

interface FlapOptions {
  flapClass?: BPStudioFlapClass;
  width?: number;
  height?: number;
  elevation?: number;
  terminalRadius?: number;
  priority?: number;
  mirroredWith?: string;
  tags?: string[];
}

const BUCKET_PROFILES: Record<BPStudioComplexityBucket, BucketProfile> = {
  small: {
    bucket: "small",
    targetFlaps: [6, 10],
    targetTreeEdges: [8, 16],
    targetCreases: [80, 300],
    expectedGridSize: [28, 36],
    accessoryFlaps: [0, 2],
    optimizerIterations: 350,
    targetUtilization: [0.5, 0.7],
  },
  medium: {
    bucket: "medium",
    targetFlaps: [10, 18],
    targetTreeEdges: [14, 28],
    targetCreases: [300, 900],
    expectedGridSize: [36, 52],
    accessoryFlaps: [3, 8],
    optimizerIterations: 650,
    targetUtilization: [0.58, 0.78],
  },
  dense: {
    bucket: "dense",
    targetFlaps: [18, 32],
    targetTreeEdges: [26, 48],
    targetCreases: [900, 2500],
    expectedGridSize: [52, 76],
    accessoryFlaps: [8, 18],
    optimizerIterations: 1100,
    targetUtilization: [0.64, 0.84],
  },
  superdense: {
    bucket: "superdense",
    targetFlaps: [32, 52],
    targetTreeEdges: [44, 78],
    targetCreases: [2500, 6000],
    expectedGridSize: [76, 112],
    accessoryFlaps: [18, 34],
    optimizerIterations: 1800,
    targetUtilization: [0.68, 0.88],
  },
};

const ARCHETYPE_WEIGHTS: Record<BPStudioArchetype, number> = {
  insect: 0.28,
  quadruped: 0.22,
  bird: 0.22,
  object: 0.14,
  abstract: 0.14,
};

const BUCKET_WEIGHTS: Record<BPStudioComplexityBucket, number> = {
  small: 0.2,
  medium: 0.4,
  dense: 0.3,
  superdense: 0.1,
};

const SYMMETRY_WEIGHTS: Record<BPStudioArchetype, Record<BPStudioSymmetry, number>> = {
  insect: { "bilateral-x": 0.62, "bilateral-y": 0.02, "rotational-2": 0.08, "radial-4": 0.03, asymmetric: 0.25 },
  quadruped: { "bilateral-x": 0.35, "bilateral-y": 0.08, "rotational-2": 0.05, "radial-4": 0.02, asymmetric: 0.5 },
  bird: { "bilateral-x": 0.52, "bilateral-y": 0.03, "rotational-2": 0.05, "radial-4": 0.02, asymmetric: 0.38 },
  object: { "bilateral-x": 0.22, "bilateral-y": 0.22, "rotational-2": 0.22, "radial-4": 0.1, asymmetric: 0.24 },
  abstract: { "bilateral-x": 0.18, "bilateral-y": 0.12, "rotational-2": 0.25, "radial-4": 0.18, asymmetric: 0.27 },
};

export function generateBPStudioSpec(config: BPStudioSamplerConfig = {}): BPStudioAdapterSpec {
  const seed = config.seed ?? 9170;
  const variation = config.variation ?? 0;
  const choiceRng = new SeededRandom(seed + variation * 65537);
  const archetype = config.archetype ?? choiceRng.weightedChoice(ARCHETYPE_WEIGHTS);
  const bucket = config.bucket ?? choiceRng.weightedChoice(BUCKET_WEIGHTS);
  const profile = BUCKET_PROFILES[bucket];
  const rng = new SeededRandom(seed ^ (variation * 0x9e3779b1) ^ archetypeHash(archetype) ^ bucketHash(bucket));
  const gridSize = config.gridSize ?? roundToEven(rng.int(profile.expectedGridSize[0], profile.expectedGridSize[1]));
  const sheet = makeSheet(gridSize);
  const symmetry = config.symmetry ?? rng.weightedChoice(SYMMETRY_WEIGHTS[archetype]);
  const draft = createDraft({
    seed,
    id: config.id,
    archetype,
    bucket,
    profile,
    sheet,
    symmetry,
    variation,
    rng,
  });

  if (archetype === "insect") buildInsect(draft);
  else if (archetype === "quadruped") buildQuadruped(draft);
  else if (archetype === "bird") buildBird(draft);
  else if (archetype === "object") buildObject(draft);
  else buildAbstract(draft);

  addAccessoryFlaps(draft);
  const rivers = makeRiverHints(draft);
  const optimizerHints = makeOptimizerHints(draft);
  const layout: BPStudioLayoutSpec = {
    symmetry: draft.symmetry,
    bodies: draft.bodies,
    flaps: draft.flaps,
    rivers,
    optimizerHints,
  };

  return {
    schemaVersion: BP_STUDIO_SPEC_SCHEMA_VERSION,
    id: draft.id ?? makeSpecId(draft),
    seed,
    archetype,
    expectedComplexity: {
      bucket,
      targetFlaps: profile.targetFlaps,
      targetTreeEdges: profile.targetTreeEdges,
      targetCreases: profile.targetCreases,
      expectedGridSize: profile.expectedGridSize,
    },
    sheet,
    tree: {
      rootId: "root",
      nodes: draft.nodes,
      edges: draft.edges,
    },
    layout,
    sampler: {
      samplerVersion: "bp-studio-sampler/v1",
      grammar: `${archetype}-grammar/v1`,
      seed,
      variation,
      symmetry: draft.symmetry,
      notes: draft.notes,
    },
  };
}

export function generateBPStudioSpecMatrix(config: BPStudioSpecMatrixConfig = {}): BPStudioAdapterSpec[] {
  const seed = config.seed ?? 9170;
  const archetypes = config.archetypes ?? BP_STUDIO_ARCHETYPES;
  const buckets = config.buckets ?? BP_STUDIO_COMPLEXITY_BUCKETS;
  const specs: BPStudioAdapterSpec[] = [];
  let index = 0;
  for (const archetype of archetypes) {
    for (const bucket of buckets) {
      specs.push(generateBPStudioSpec({ seed: seed + index * 104729, archetype, bucket, variation: index }));
      index += 1;
    }
  }
  return specs;
}

export function validateBPStudioSpec(spec: BPStudioAdapterSpec): string[] {
  const issues: string[] = [];
  const nodeIds = new Set<string>();
  for (const node of spec.tree.nodes) {
    if (nodeIds.has(node.id)) issues.push(`duplicate node id: ${node.id}`);
    nodeIds.add(node.id);
  }
  if (!nodeIds.has(spec.tree.rootId)) issues.push(`missing root node: ${spec.tree.rootId}`);
  for (const edge of spec.tree.edges) {
    if (!nodeIds.has(edge.from)) issues.push(`edge ${edge.id} has missing from node ${edge.from}`);
    if (!nodeIds.has(edge.to)) issues.push(`edge ${edge.id} has missing to node ${edge.to}`);
    if (edge.length <= 0) issues.push(`edge ${edge.id} has non-positive length`);
  }
  for (const body of spec.layout.bodies) {
    if (!nodeIds.has(body.nodeId)) issues.push(`body placement has missing node ${body.nodeId}`);
    if (!insideSheet(spec.sheet, body.center)) issues.push(`body ${body.nodeId} is outside sheet`);
    if (body.width <= 0 || body.height <= 0) issues.push(`body ${body.nodeId} has non-positive dimensions`);
  }
  for (const flap of spec.layout.flaps) {
    if (!nodeIds.has(flap.nodeId)) issues.push(`flap placement has missing node ${flap.nodeId}`);
    if (!insideSheet(spec.sheet, flap.terminal)) issues.push(`flap ${flap.nodeId} is outside sheet`);
    if (flap.width <= 0 || flap.height <= 0) issues.push(`flap ${flap.nodeId} has non-positive dimensions`);
    if (flap.mirroredWith && !nodeIds.has(flap.mirroredWith)) issues.push(`flap ${flap.nodeId} mirrors missing node ${flap.mirroredWith}`);
  }
  if (spec.layout.flaps.length < spec.expectedComplexity.targetFlaps[0]) {
    issues.push(`expected at least ${spec.expectedComplexity.targetFlaps[0]} flaps, got ${spec.layout.flaps.length}`);
  }
  if (spec.tree.edges.length < spec.expectedComplexity.targetTreeEdges[0]) {
    issues.push(`expected at least ${spec.expectedComplexity.targetTreeEdges[0]} tree edges, got ${spec.tree.edges.length}`);
  }
  if (spec.layout.rivers.length === 0) issues.push("spec has no river hints");
  return issues;
}

export function assertValidBPStudioSpec(spec: BPStudioAdapterSpec): void {
  const issues = validateBPStudioSpec(spec);
  if (issues.length > 0) {
    throw new Error(`Invalid BP Studio spec ${spec.id}: ${issues.join("; ")}`);
  }
}

function createDraft(args: {
  seed: number;
  id?: string;
  archetype: BPStudioArchetype;
  bucket: BPStudioComplexityBucket;
  profile: BucketProfile;
  sheet: BPStudioSheetSpec;
  symmetry: BPStudioSymmetry;
  variation: number;
  rng: SeededRandom;
}): Draft {
  const draft: Draft = {
    ...args,
    nodes: [],
    edges: [],
    bodies: [],
    flaps: [],
    anchorNodes: [],
    notes: [
      "Adapter-neutral BP Studio input spec; no FOLD graph or crease assignments are generated here.",
      "Tree edges and river hints are intended for the headless BP Studio adapter contract.",
    ],
    nextNodeIndex: 0,
    nextEdgeIndex: 0,
  };
  draft.nodes.push({ id: "root", kind: "root", label: "root" });
  return draft;
}

function buildInsect(draft: Draft): void {
  const g = draft.sheet.gridSize;
  const center = jitterPoint(draft, { x: g * 0.5, y: g * 0.52 }, g * 0.035);
  const thorax = addBodyNode(draft, "thorax", center, g * 0.18, g * 0.22, 1, ["insect", "thorax"]);
  const head = addBodyNode(draft, "head", { x: center.x, y: center.y + g * 0.18 }, g * 0.13, g * 0.12, 1, ["insect", "head"]);
  const abdomen = addBodyNode(draft, "abdomen", { x: center.x + draft.rng.int(-2, 2), y: center.y - g * 0.21 }, g * 0.17, g * 0.22, 1, [
    "insect",
    "abdomen",
  ]);
  addEdge(draft, "root", thorax, "body", sampleBodyLength(draft), { width: 3 });
  addEdge(draft, thorax, head, "river", sampleBodyLength(draft), { width: 2, tags: ["neck-river"] });
  addEdge(draft, thorax, abdomen, "river", sampleBodyLength(draft), { width: 2, tags: ["abdomen-river"] });

  const legFractions = [0.28, 0.5, 0.72];
  for (const [index, fraction] of legFractions.entries()) {
    const left = addAppendageFlap(draft, thorax, `left ${["front", "middle", "hind"][index]} leg`, "left", fraction, {
      tags: ["leg", "paired"],
      priority: 8 - index,
    });
    const right = addAppendageFlap(draft, thorax, `right ${["front", "middle", "hind"][index]} leg`, "right", fraction + draft.rng.float(-0.035, 0.035), {
      mirroredWith: left,
      tags: ["leg", "paired"],
      priority: 8 - index,
    });
    mirrorFlaps(draft, left, right);
  }

  const leftAntenna = addAppendageFlap(draft, head, "left antenna", "top", 0.42, {
    width: 1,
    height: 3,
    tags: ["antenna", "paired"],
    priority: 5,
  });
  const rightAntenna = addAppendageFlap(draft, head, "right antenna", "top", 0.58, {
    width: 1,
    height: 3,
    mirroredWith: leftAntenna,
    tags: ["antenna", "paired"],
    priority: 5,
  });
  mirrorFlaps(draft, leftAntenna, rightAntenna);

  if (draft.bucket !== "small" || draft.rng.next() < 0.55) {
    const leftWing = addInteriorFlap(draft, thorax, "left wing pad", { x: center.x - g * 0.18, y: center.y + g * 0.08 }, {
      flapClass: "elevation",
      width: Math.max(3, Math.round(g * 0.08)),
      height: Math.max(5, Math.round(g * 0.18)),
      elevation: 2,
      tags: ["wing", "wide", "paired"],
      priority: 7,
    });
    const rightWing = addInteriorFlap(draft, thorax, "right wing pad", { x: center.x + g * 0.18, y: center.y + g * 0.08 }, {
      flapClass: "elevation",
      width: Math.max(3, Math.round(g * 0.08)),
      height: Math.max(5, Math.round(g * 0.18)),
      elevation: 2,
      mirroredWith: leftWing,
      tags: ["wing", "wide", "paired"],
      priority: 7,
    });
    mirrorFlaps(draft, leftWing, rightWing);
  }

  draft.notes.push("Insect grammar uses thorax/head/abdomen bodies with six boundary-biased leg flaps and antennae.");
}

function buildQuadruped(draft: Draft): void {
  const g = draft.sheet.gridSize;
  const center = jitterPoint(draft, { x: g * 0.48, y: g * 0.51 }, g * 0.04);
  const body = addBodyNode(draft, "torso", center, g * 0.28, g * 0.16, 1, ["quadruped", "torso"]);
  const neck = addHubNode(draft, "neck hub", { width: 2, height: 2, elevation: 1, tags: ["neck"] });
  const head = addBodyNode(draft, "head", { x: center.x + g * 0.22, y: center.y + g * 0.06 }, g * 0.12, g * 0.11, 1, ["quadruped", "head"]);
  addEdge(draft, "root", body, "body", sampleBodyLength(draft), { width: 4 });
  addEdge(draft, body, neck, "river", sampleAppendageLength(draft), { width: 2, tags: ["neck"] });
  addEdge(draft, neck, head, "river", sampleAppendageLength(draft), { width: 2, tags: ["head"] });

  const legAnchors = [
    ["front left leg", "bottom", 0.62],
    ["front right leg", "bottom", 0.76],
    ["rear left leg", "bottom", 0.24],
    ["rear right leg", "bottom", 0.38],
  ] as const;
  for (const [label, side, fraction] of legAnchors) {
    addAppendageFlap(draft, body, label, side, fraction + draft.rng.float(-0.035, 0.035), {
      width: 2,
      height: 5,
      tags: ["leg"],
      priority: 8,
    });
  }

  addAppendageFlap(draft, head, "muzzle", "right", 0.58, {
    flapClass: "wide",
    width: 3,
    height: 3,
    tags: ["head-detail"],
    priority: 6,
  });
  addAppendageFlap(draft, body, "tail", "left", 0.58, {
    width: 2,
    height: 6,
    tags: ["tail"],
    priority: 6,
  });

  const leftEar = addAppendageFlap(draft, head, "left ear", "top", 0.58, { width: 1, height: 3, tags: ["ear"], priority: 5 });
  const rightEar = addAppendageFlap(draft, head, "right ear", "top", 0.72, {
    width: 1,
    height: 3,
    mirroredWith: leftEar,
    tags: ["ear"],
    priority: 5,
  });
  mirrorFlaps(draft, leftEar, rightEar);

  draft.notes.push("Quadruped grammar uses a long torso, head/neck river, four legs, tail, and small head-detail flaps.");
}

function buildBird(draft: Draft): void {
  const g = draft.sheet.gridSize;
  const center = jitterPoint(draft, { x: g * 0.5, y: g * 0.53 }, g * 0.035);
  const body = addBodyNode(draft, "body", center, g * 0.2, g * 0.18, 1, ["bird", "body"]);
  const neck = addHubNode(draft, "neck hub", { width: 2, height: 2, elevation: 1, tags: ["neck"] });
  const head = addBodyNode(draft, "head", { x: center.x + g * 0.06, y: center.y + g * 0.22 }, g * 0.1, g * 0.1, 1, ["bird", "head"]);
  addEdge(draft, "root", body, "body", sampleBodyLength(draft), { width: 3 });
  addEdge(draft, body, neck, "river", sampleAppendageLength(draft), { width: 2, tags: ["neck"] });
  addEdge(draft, neck, head, "river", sampleAppendageLength(draft), { width: 2, tags: ["head"] });

  const leftWing = addInteriorFlap(draft, body, "left wing", { x: center.x - g * 0.28, y: center.y + g * 0.02 }, {
    flapClass: "wide",
    width: Math.max(5, Math.round(g * 0.16)),
    height: Math.max(8, Math.round(g * 0.28)),
    elevation: 1,
    tags: ["wing", "paired", "wide"],
    priority: 9,
  });
  const rightWing = addInteriorFlap(draft, body, "right wing", { x: center.x + g * 0.28, y: center.y + g * 0.02 }, {
    flapClass: "wide",
    width: Math.max(5, Math.round(g * 0.16)),
    height: Math.max(8, Math.round(g * 0.28)),
    elevation: 1,
    mirroredWith: leftWing,
    tags: ["wing", "paired", "wide"],
    priority: 9,
  });
  mirrorFlaps(draft, leftWing, rightWing);

  addAppendageFlap(draft, head, "beak", "top", 0.56, { width: 1, height: 3, tags: ["beak"], priority: 6 });
  addAppendageFlap(draft, body, "left leg", "bottom", 0.46, { width: 1, height: 4, tags: ["leg"], priority: 5 });
  addAppendageFlap(draft, body, "right leg", "bottom", 0.54, { width: 1, height: 4, tags: ["leg"], priority: 5 });

  const tailCount = draft.bucket === "small" ? 3 : draft.bucket === "medium" ? 5 : draft.rng.int(5, 9);
  for (let i = 0; i < tailCount; i += 1) {
    addAppendageFlap(draft, body, `tail feather ${i + 1}`, "bottom", 0.34 + (0.32 * i) / Math.max(1, tailCount - 1), {
      width: 1,
      height: draft.rng.int(4, 7),
      tags: ["tail", "fan"],
      priority: 6,
    });
  }

  draft.notes.push("Bird grammar uses paired wide wing flaps, a head/neck chain, legs, and a nonuniform tail fan.");
}

function buildObject(draft: Draft): void {
  const g = draft.sheet.gridSize;
  const center = jitterPoint(draft, { x: g * 0.51, y: g * 0.49 }, g * 0.04);
  const core = addBodyNode(draft, "core body", center, g * 0.24, g * 0.22, 1, ["object", "core"]);
  const handleHub = addHubNode(draft, "handle hub", { width: 3, height: 2, elevation: 1, tags: ["handle"] });
  addEdge(draft, "root", core, "body", sampleBodyLength(draft), { width: 4 });
  addEdge(draft, core, handleHub, "river", sampleBodyLength(draft), { width: 3, tags: ["handle"] });

  const handleTop = addAppendageFlap(draft, handleHub, "upper handle", "top", 0.52, {
    flapClass: "wide",
    width: 4,
    height: 6,
    tags: ["handle"],
    priority: 8,
  });
  const handleBottom = addAppendageFlap(draft, handleHub, "lower handle", "bottom", 0.48, {
    flapClass: "wide",
    width: 4,
    height: 6,
    mirroredWith: handleTop,
    tags: ["handle"],
    priority: 8,
  });
  mirrorFlaps(draft, handleTop, handleBottom);

  for (const [label, side, fraction] of [
    ["left prong", "left", 0.34],
    ["right prong", "right", 0.66],
    ["top cap", "top", 0.34],
    ["bottom counterweight", "bottom", 0.66],
  ] as const) {
    addAppendageFlap(draft, core, label, side, fraction, {
      flapClass: label.includes("counterweight") ? "elevation" : "terminal",
      width: draft.rng.int(2, 5),
      height: draft.rng.int(4, 8),
      elevation: label.includes("counterweight") ? 2 : 0,
      tags: ["object-detail"],
      priority: 6,
    });
  }

  draft.notes.push("Object grammar uses a central body with handle, prongs, caps, and weighted terminal features.");
}

function buildAbstract(draft: Draft): void {
  const g = draft.sheet.gridSize;
  const center = jitterPoint(draft, { x: g * 0.5, y: g * 0.5 }, g * 0.035);
  const core = addBodyNode(draft, "central hub", center, g * 0.2, g * 0.2, 1, ["abstract", "core"]);
  const secondary = addBodyNode(draft, "offset hub", { x: center.x + g * 0.12, y: center.y - g * 0.1 }, g * 0.14, g * 0.12, 1, [
    "abstract",
    "offset",
  ]);
  addEdge(draft, "root", core, "body", sampleBodyLength(draft), { width: 4 });
  addEdge(draft, core, secondary, "river", sampleBodyLength(draft), { width: 2, tags: ["offset"] });

  const spokes = draft.bucket === "small" ? 6 : draft.bucket === "medium" ? 8 : 12;
  const sides: Array<Exclude<BPStudioSide, "interior">> = ["top", "right", "bottom", "left"];
  for (let i = 0; i < spokes; i += 1) {
    const side = sides[i % sides.length];
    const fraction = 0.2 + 0.6 * ((i * 5) % spokes) / Math.max(1, spokes - 1);
    addAppendageFlap(draft, i % 3 === 0 ? secondary : core, `abstract prong ${i + 1}`, side, fraction + draft.rng.float(-0.05, 0.05), {
      flapClass: i % 4 === 0 ? "elevation" : "terminal",
      width: draft.rng.int(1, 4),
      height: draft.rng.int(3, 8),
      elevation: i % 4 === 0 ? 2 : 0,
      tags: ["abstract-prong"],
      priority: 4 + (i % 4),
    });
  }

  draft.notes.push("Abstract grammar uses asymmetric hubs and radial prongs to stress layout optimization without a named animal form.");
}

function addAccessoryFlaps(draft: Draft): void {
  const target = draft.rng.int(draft.profile.targetFlaps[0], draft.profile.targetFlaps[1]);
  const randomExtra = draft.rng.int(draft.profile.accessoryFlaps[0], draft.profile.accessoryFlaps[1]);
  const desired = Math.max(target, draft.flaps.length + randomExtra);
  while (draft.flaps.length < desired || draft.edges.length < draft.profile.targetTreeEdges[0]) {
    if (isMirroringUseful(draft) && draft.flaps.length + 1 < desired && draft.rng.next() < 0.55) {
      addMirroredAccessoryPair(draft);
    } else {
      addSingleAccessoryFlap(draft);
    }
  }
  draft.notes.push(`Accessory pass produced ${draft.flaps.length} flap placements for ${draft.bucket} complexity.`);
}

function addMirroredAccessoryPair(draft: Draft): void {
  const parent = pickAnchor(draft);
  const side = draft.rng.choice(["left", "right"] as const);
  const mirrorSide: Exclude<BPStudioSide, "interior"> = side === "left" ? "right" : "left";
  const fraction = draft.rng.float(0.2, 0.8);
  const label = accessoryLabel(draft);
  const first = addAppendageFlap(draft, parent, `${label} left`, side, fraction, {
    width: draft.rng.int(1, 3),
    height: draft.rng.int(3, 7),
    tags: ["accessory", "paired"],
    priority: draft.rng.int(2, 6),
  });
  const second = addAppendageFlap(draft, parent, `${label} right`, mirrorSide, 1 - fraction + draft.rng.float(-0.025, 0.025), {
    width: draft.rng.int(1, 3),
    height: draft.rng.int(3, 7),
    mirroredWith: first,
    tags: ["accessory", "paired"],
    priority: draft.rng.int(2, 6),
  });
  mirrorFlaps(draft, first, second);
}

function addSingleAccessoryFlap(draft: Draft): void {
  const parent = pickAnchor(draft);
  const sides: Array<Exclude<BPStudioSide, "interior">> = ["top", "right", "bottom", "left"];
  const side = draft.rng.choice(sides);
  const label = accessoryLabel(draft);
  const elevated = draft.rng.next() < 0.18;
  addAppendageFlap(draft, parent, label, side, draft.rng.float(0.12, 0.88), {
    flapClass: elevated ? "elevation" : "terminal",
    width: draft.rng.int(1, 4),
    height: draft.rng.int(3, 9),
    elevation: elevated ? 2 : 0,
    tags: ["accessory"],
    priority: draft.rng.int(1, 5),
  });
}

function addBodyNode(
  draft: Draft,
  label: string,
  center: { x: number; y: number },
  width: number,
  height: number,
  elevation: number,
  tags: string[],
): string {
  const id = addNode(draft, "body", label, {
    width: Math.max(1, Math.round(width)),
    height: Math.max(1, Math.round(height)),
    elevation,
    tags,
  });
  draft.bodies.push({
    nodeId: id,
    label,
    center: clampPoint(draft.sheet, center),
    width: Math.max(1, Math.round(width)),
    height: Math.max(1, Math.round(height)),
    elevation,
    tags,
  });
  draft.anchorNodes.push(id);
  return id;
}

function addHubNode(draft: Draft, label: string, options: NodeOptions = {}): string {
  const id = addNode(draft, "hub", label, options);
  draft.anchorNodes.push(id);
  return id;
}

function addNode(draft: Draft, kind: BPStudioTreeNodeKind, label: string, options: NodeOptions = {}): string {
  const id = `${slug(label)}-${draft.nextNodeIndex}`;
  draft.nextNodeIndex += 1;
  draft.nodes.push({ id, kind, label, ...options });
  return id;
}

function addEdge(draft: Draft, from: string, to: string, role: BPStudioTreeEdgeRole, length: number, options: EdgeOptions = {}): string {
  const id = `edge-${draft.nextEdgeIndex}-${from}-${to}`;
  draft.nextEdgeIndex += 1;
  draft.edges.push({
    id,
    from,
    to,
    role,
    length: Math.max(1, Math.round(length)),
    ...options,
  });
  return id;
}

function addAppendageFlap(
  draft: Draft,
  parent: string,
  label: string,
  side: Exclude<BPStudioSide, "interior">,
  fraction: number,
  options: FlapOptions = {},
): string {
  const nodeId = addNode(draft, "flap", label, {
    width: options.width,
    height: options.height,
    elevation: options.elevation,
    tags: options.tags,
  });
  addEdge(draft, parent, nodeId, "appendage", sampleAppendageLength(draft), {
    width: Math.max(1, Math.round((options.width ?? 1) + (options.height ?? 3) * 0.2)),
    tags: options.tags,
  });
  const terminal = terminalPoint(draft, side, fraction, draft.rng.int(0, 2));
  draft.flaps.push(makeFlapPlacement(draft, nodeId, label, terminal, side, options));
  return nodeId;
}

function addInteriorFlap(draft: Draft, parent: string, label: string, terminal: { x: number; y: number }, options: FlapOptions = {}): string {
  const nodeId = addNode(draft, "flap", label, {
    width: options.width,
    height: options.height,
    elevation: options.elevation,
    tags: options.tags,
  });
  addEdge(draft, parent, nodeId, "appendage", sampleAppendageLength(draft), {
    width: Math.max(1, Math.round((options.width ?? 2) * 0.7)),
    tags: options.tags,
  });
  draft.flaps.push(makeFlapPlacement(draft, nodeId, label, clampPoint(draft.sheet, terminal), "interior", options));
  return nodeId;
}

function makeFlapPlacement(
  draft: Draft,
  nodeId: string,
  label: string,
  terminal: { x: number; y: number },
  side: BPStudioSide,
  options: FlapOptions,
): BPStudioFlapPlacement {
  return {
    nodeId,
    label,
    class: options.flapClass ?? "terminal",
    terminal: clampPoint(draft.sheet, terminal),
    side,
    width: Math.max(1, Math.round(options.width ?? draft.rng.int(1, 3))),
    height: Math.max(1, Math.round(options.height ?? draft.rng.int(3, 7))),
    elevation: Math.max(0, Math.round(options.elevation ?? 0)),
    terminalRadius: Math.max(1, Math.round(options.terminalRadius ?? draft.rng.int(1, 2))),
    priority: Math.max(1, Math.round(options.priority ?? 4)),
    mirroredWith: options.mirroredWith,
    tags: options.tags,
  };
}

function mirrorFlaps(draft: Draft, firstId: string, secondId: string): void {
  const first = draft.flaps.find((flap) => flap.nodeId === firstId);
  const second = draft.flaps.find((flap) => flap.nodeId === secondId);
  if (!first || !second) return;
  first.mirroredWith = second.nodeId;
  second.mirroredWith = first.nodeId;
}

function makeRiverHints(draft: Draft): BPStudioRiverHint[] {
  const flapByNode = new Map(draft.flaps.map((flap) => [flap.nodeId, flap]));
  return draft.edges
    .filter((edge) => edge.role !== "body")
    .map((edge) => {
      const terminalFlap = flapByNode.get(edge.to) ?? flapByNode.get(edge.from);
      const side = terminalFlap?.side ?? "interior";
      const preferredAxis = side === "left" || side === "right" ? "horizontal" : side === "top" || side === "bottom" ? "vertical" : "diagonal";
      const bend: BPStudioRiverBend =
        edge.tags?.includes("fan") || terminalFlap?.tags?.includes("fan")
          ? "fan"
          : edge.role === "river"
            ? draft.rng.choice(["direct", "dogleg", "staircase"] as const)
            : draft.rng.choice(["direct", "dogleg", "staircase"] as const);
      return {
        edgeId: edge.id,
        from: edge.from,
        to: edge.to,
        width: Math.max(1, Math.round(edge.width ?? 1)),
        preferredAxis,
        bend,
        clearance: Math.max(1, Math.round((edge.width ?? 1) + draft.rng.int(0, 2))),
        tags: edge.tags,
      };
    });
}

function makeOptimizerHints(draft: Draft): BPStudioOptimizerHints {
  const objective: BPStudioOptimizerHints["objective"] =
    draft.bucket === "small" ? "compact-flaps" : draft.bucket === "superdense" ? "river-clearance" : "utilization-balanced";
  return {
    objective,
    keepSymmetry: draft.symmetry !== "asymmetric",
    allowTerminalRelaxation: draft.bucket !== "small",
    maxIterations: draft.profile.optimizerIterations,
    topK: draft.bucket === "superdense" ? 12 : draft.bucket === "dense" ? 8 : 5,
  };
}

function makeSheet(gridSize: number): BPStudioSheetSpec {
  const margin = Math.max(3, Math.round(gridSize * 0.08));
  return {
    width: gridSize,
    height: gridSize,
    gridSize,
    margin,
    coordinateSystem: "integer-grid",
    unit: "bp-grid",
  };
}

function terminalPoint(draft: Draft, side: Exclude<BPStudioSide, "interior">, rawFraction: number, depth: number): { x: number; y: number } {
  const { sheet, rng } = draft;
  const fraction = clamp(rawFraction, 0.08, 0.92);
  const spanX = sheet.width - sheet.margin * 2;
  const spanY = sheet.height - sheet.margin * 2;
  const jitter = Math.max(1, Math.round(sheet.gridSize * 0.035));
  const offset = sheet.margin + depth;
  if (side === "top") {
    return clampPoint(sheet, { x: sheet.margin + spanX * fraction + rng.int(-jitter, jitter), y: sheet.height - offset });
  }
  if (side === "bottom") {
    return clampPoint(sheet, { x: sheet.margin + spanX * fraction + rng.int(-jitter, jitter), y: offset });
  }
  if (side === "left") {
    return clampPoint(sheet, { x: offset, y: sheet.margin + spanY * fraction + rng.int(-jitter, jitter) });
  }
  return clampPoint(sheet, { x: sheet.width - offset, y: sheet.margin + spanY * fraction + rng.int(-jitter, jitter) });
}

function jitterPoint(draft: Draft, point: { x: number; y: number }, amount: number): { x: number; y: number } {
  return clampPoint(draft.sheet, {
    x: point.x + draft.rng.float(-amount, amount),
    y: point.y + draft.rng.float(-amount, amount),
  });
}

function clampPoint(sheet: BPStudioSheetSpec, point: { x: number; y: number }): { x: number; y: number } {
  return {
    x: Math.round(clamp(point.x, sheet.margin, sheet.width - sheet.margin)),
    y: Math.round(clamp(point.y, sheet.margin, sheet.height - sheet.margin)),
  };
}

function insideSheet(sheet: BPStudioSheetSpec, point: { x: number; y: number }): boolean {
  return point.x >= 0 && point.y >= 0 && point.x <= sheet.width && point.y <= sheet.height;
}

function sampleBodyLength(draft: Draft): number {
  return draft.rng.int(Math.max(3, Math.round(draft.sheet.gridSize * 0.1)), Math.max(5, Math.round(draft.sheet.gridSize * 0.22)));
}

function sampleAppendageLength(draft: Draft): number {
  return draft.rng.int(Math.max(3, Math.round(draft.sheet.gridSize * 0.12)), Math.max(6, Math.round(draft.sheet.gridSize * 0.32)));
}

function pickAnchor(draft: Draft): string {
  if (draft.anchorNodes.length === 0) return "root";
  return draft.rng.choice(draft.anchorNodes);
}

function accessoryLabel(draft: Draft): string {
  const labels: Record<BPStudioArchetype, readonly string[]> = {
    insect: ["tarsus", "wing vein", "mandible", "side spur"],
    quadruped: ["toe", "horn", "cheek spur", "heel"],
    bird: ["feather barb", "crest", "talon", "secondary feather"],
    object: ["small prong", "rim tab", "handle spur", "counter tab"],
    abstract: ["satellite prong", "offset barb", "corner tab", "radial spur"],
  };
  return `${draft.rng.choice(labels[draft.archetype])} ${draft.flaps.length + 1}`;
}

function isMirroringUseful(draft: Draft): boolean {
  return draft.symmetry === "bilateral-x" || draft.symmetry === "rotational-2" || draft.symmetry === "radial-4";
}

function makeSpecId(draft: Draft): string {
  return `bpstudio-${draft.archetype}-${draft.bucket}-${draft.seed}-${draft.variation}`;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function roundToEven(value: number): number {
  return value % 2 === 0 ? value : value + 1;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function archetypeHash(archetype: BPStudioArchetype): number {
  return BP_STUDIO_ARCHETYPES.indexOf(archetype) * 0x45d9f3b;
}

function bucketHash(bucket: BPStudioComplexityBucket): number {
  return BP_STUDIO_COMPLEXITY_BUCKETS.indexOf(bucket) * 0x27d4eb2d;
}

if (import.meta.main) {
  const specs = generateBPStudioSpecMatrix();
  const issues = specs.flatMap((spec) => validateBPStudioSpec(spec).map((issue) => `${spec.id}: ${issue}`));
  if (issues.length > 0) {
    console.error(JSON.stringify({ valid: false, issues }, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify({ valid: true, count: specs.length, specs }, null, 2));
}
