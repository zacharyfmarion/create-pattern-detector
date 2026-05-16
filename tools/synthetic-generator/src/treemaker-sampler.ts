import { SeededRandom } from "./random.ts";
import type {
  GenerationConfig,
  TreeMakerArchetype,
  TreeMakerSamplerConfig,
  TreeMakerSymmetryClass,
  TreeMakerSymmetryVariant,
  TreeMakerTopology,
  TreeMetadata,
} from "./types.ts";

export interface TreeMakerNodeSpec {
  id: string;
  kind: "root" | "hub" | "terminal";
  label: string;
  x: number;
  y: number;
}

export interface TreeMakerEdgeSpec {
  id: string;
  from: string;
  to: string;
  length: number;
}

export interface TreeMakerAdapterSpec {
  schemaVersion: "treemaker-adapter-spec/v1";
  id: string;
  seed: number;
  targetCreases: number;
  archetype: TreeMakerArchetype;
  symmetryClass: TreeMakerSymmetryClass;
  symmetryVariant: TreeMakerSymmetryVariant;
  topology: TreeMakerTopology;
  nodes: TreeMakerNodeSpec[];
  edges: TreeMakerEdgeSpec[];
}

const DEFAULT_SYMMETRY_WEIGHTS: Record<TreeMakerSymmetryClass, number> = {
  diagonal: 0.425,
  "middle-axis": 0.425,
  asymmetric: 0.15,
};

const DEFAULT_MIDDLE_AXIS_WEIGHTS = {
  vertical: 1,
  horizontal: 1,
} satisfies Partial<Record<TreeMakerSymmetryVariant, number>>;

const DEFAULT_DIAGONAL_WEIGHTS = {
  "main-diagonal": 1,
  "anti-diagonal": 1,
} satisfies Partial<Record<TreeMakerSymmetryVariant, number>>;

const DEFAULT_ARCHETYPE_WEIGHTS: Record<TreeMakerArchetype, number> = {
  insect: 1.3,
  quadruped: 1.2,
  bird: 1,
  creature: 1,
  object: 0.8,
  abstract: 0.9,
};

const DEFAULT_TOPOLOGY_WEIGHTS: Record<TreeMakerTopology, number> = {
  "radial-star": 0.22,
  "hubbed-limbs": 0.34,
  "spine-chain": 0.28,
  "branched-hybrid": 0.16,
};

export function generateTreeMakerSpec(config: GenerationConfig): TreeMakerAdapterSpec {
  const rng = new SeededRandom(config.seed);
  const sampler = config.treeMakerSampler ?? {};
  const symmetryClass = chooseSymmetryClass(rng, sampler);
  const symmetryVariant = chooseSymmetryVariant(rng, symmetryClass, sampler);
  const archetype = chooseArchetype(rng, sampler);
  const topology = chooseTopology(rng, sampler, config.numCreases);
  const builder = new TreeBuilder(config.id, config.seed, config.numCreases, archetype, symmetryClass, symmetryVariant, topology);

  if (symmetryClass === "asymmetric") {
    addAsymmetricTopology(builder, rng, archetype, topology, config.numCreases);
  } else {
    addSymmetricTopology(builder, rng, archetype, topology, config.numCreases);
  }
  return builder.spec();
}

export function validateTreeMakerSpec(spec: TreeMakerAdapterSpec): string[] {
  const errors: string[] = [];
  if (spec.schemaVersion !== "treemaker-adapter-spec/v1") errors.push("schemaVersion must be treemaker-adapter-spec/v1");
  if (!spec.nodes.some((node) => node.kind === "root")) errors.push("requires root node");
  const nodeIds = new Set(spec.nodes.map((node) => node.id));
  for (const edge of spec.edges) {
    if (!nodeIds.has(edge.from)) errors.push(`edge ${edge.id} references missing from node ${edge.from}`);
    if (!nodeIds.has(edge.to)) errors.push(`edge ${edge.id} references missing to node ${edge.to}`);
    if (!Number.isFinite(edge.length) || edge.length <= 0) errors.push(`edge ${edge.id} has invalid length`);
  }
  if (spec.nodes.filter((node) => node.kind === "terminal").length < 3) errors.push("requires at least three terminal nodes");
  if (spec.symmetryClass !== "asymmetric") errors.push(...validateMirroredTopology(spec));
  return errors;
}

export function treeMetadataFromSpec(spec: TreeMakerAdapterSpec): TreeMetadata {
  return {
    generator: "treemaker-tree",
    archetype: spec.archetype,
    symmetryClass: spec.symmetryClass,
    symmetryVariant: spec.symmetryVariant,
    topology: spec.topology,
    rootId: "root",
    nodeCount: spec.nodes.length,
    terminalCount: spec.nodes.filter((node) => node.kind === "terminal").length,
    branchDepth: branchDepth(spec),
    edgeLengths: spec.edges.map((edge) => edge.length),
    nodes: spec.nodes,
    edges: spec.edges,
  };
}

export function chooseSymmetryClass(rng: SeededRandom, sampler: TreeMakerSamplerConfig = {}): TreeMakerSymmetryClass {
  return rng.weightedChoice({ ...DEFAULT_SYMMETRY_WEIGHTS, ...(sampler.symmetryWeights ?? {}) });
}

function chooseSymmetryVariant(
  rng: SeededRandom,
  symmetryClass: TreeMakerSymmetryClass,
  sampler: TreeMakerSamplerConfig,
): TreeMakerSymmetryVariant {
  if (symmetryClass === "middle-axis") {
    return rng.weightedChoice({ ...DEFAULT_MIDDLE_AXIS_WEIGHTS, ...(sampler.middleAxisWeights ?? {}) }) as TreeMakerSymmetryVariant;
  }
  if (symmetryClass === "diagonal") {
    return rng.weightedChoice({ ...DEFAULT_DIAGONAL_WEIGHTS, ...(sampler.diagonalWeights ?? {}) }) as TreeMakerSymmetryVariant;
  }
  return "none";
}

function chooseArchetype(rng: SeededRandom, sampler: TreeMakerSamplerConfig): TreeMakerArchetype {
  return rng.weightedChoice({ ...DEFAULT_ARCHETYPE_WEIGHTS, ...(sampler.archetypeWeights ?? {}) });
}

function chooseTopology(rng: SeededRandom, sampler: TreeMakerSamplerConfig, targetCreases: number): TreeMakerTopology {
  const weights = { ...DEFAULT_TOPOLOGY_WEIGHTS, ...(sampler.topologyWeights ?? {}) };
  if (targetCreases > 500 && sampler.topologyWeights === undefined) {
    weights["radial-star"] *= 0.45;
    weights["spine-chain"] *= 1.25;
    weights["branched-hybrid"] *= 1.35;
  }
  return rng.weightedChoice(weights);
}

class TreeBuilder {
  private nodes: TreeMakerNodeSpec[] = [{ id: "root", kind: "root", label: "root", x: 0.5, y: 0.5 }];
  private edges: TreeMakerEdgeSpec[] = [];
  private nextTerminal = 0;
  private nextHub = 0;

  constructor(
    private readonly id: string,
    private readonly seed: number,
    private readonly targetCreases: number,
    private readonly archetype: TreeMakerArchetype,
    private readonly symmetryClass: TreeMakerSymmetryClass,
    readonly symmetryVariant: TreeMakerSymmetryVariant,
    private readonly topology: TreeMakerTopology,
  ) {}

  spec(): TreeMakerAdapterSpec {
    return {
      schemaVersion: "treemaker-adapter-spec/v1",
      id: this.id,
      seed: this.seed,
      targetCreases: this.targetCreases,
      archetype: this.archetype,
      symmetryClass: this.symmetryClass,
      symmetryVariant: this.symmetryVariant,
      topology: this.topology,
      nodes: this.nodes,
      edges: this.edges,
    };
  }

  addTerminalPair(label: string, point: [number, number], length: number): void {
    const left = this.addTerminal(`${label}-a`, point);
    const mirrored = mirrorPoint(point, this.symmetryVariant);
    const right = this.addTerminal(`${label}-b`, mirrored);
    this.addEdge("root", left, length);
    this.addEdge("root", right, length);
  }

  addHubPair(label: string, point: [number, number], length: number): [string, string] {
    const left = this.addHub(`${label}-a`, point, "root", length);
    const mirrored = mirrorPoint(point, this.symmetryVariant);
    const right = this.addHub(`${label}-b`, mirrored, "root", length);
    return [left, right];
  }

  addMirroredTerminalPair(label: string, parentA: string, parentB: string, point: [number, number], length: number): [string, string] {
    const left = this.addTerminal(`${label}-a`, point);
    const mirrored = mirrorPoint(point, this.symmetryVariant);
    const right = this.addTerminal(`${label}-b`, mirrored);
    this.addEdge(parentA, left, length);
    this.addEdge(parentB, right, length);
    return [left, right];
  }

  addAxisHub(label: string, position: number, parent: string, length: number): string {
    return this.addHub(label, pointOnSymmetryAxis(this.symmetryVariant, position), parent, length);
  }

  addTerminal(label: string, point: [number, number]): string {
    const id = `t${this.nextTerminal++}`;
    this.nodes.push({ id, kind: "terminal", label, x: round(point[0]), y: round(point[1]) });
    return id;
  }

  addHub(label: string, point: [number, number], parent = "root", length = 0.22): string {
    const id = `h${this.nextHub++}`;
    this.nodes.push({ id, kind: "hub", label, x: round(point[0]), y: round(point[1]) });
    this.addEdge(parent, id, length);
    return id;
  }

  addEdge(from: string, to: string, length: number): void {
    this.edges.push({ id: `e${this.edges.length}`, from, to, length: round(length) });
  }

  node(id: string): TreeMakerNodeSpec {
    const node = this.nodes.find((item) => item.id === id);
    if (!node) throw new Error(`unknown TreeMaker node: ${id}`);
    return node;
  }
}

function addSymmetricTopology(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  topology: TreeMakerTopology,
  targetCreases: number,
): void {
  if (topology === "hubbed-limbs") {
    addSymmetricHubbedLimbs(builder, rng, archetype, targetCreases);
  } else if (topology === "spine-chain") {
    addSymmetricSpineChain(builder, rng, archetype, targetCreases);
  } else if (topology === "branched-hybrid") {
    addSymmetricBranchedHybrid(builder, rng, archetype, targetCreases);
  } else {
    addSymmetricRadialStar(builder, rng, archetype, targetCreases);
  }
}

function addSymmetricRadialStar(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  const pairs = pairCountFor(archetype, targetCreases);
  for (let index = 0; index < pairs; index++) {
    const point = mirroredSidePoint(builder.symmetryVariant, rng, index, pairs, rng.float(0.3, 0.47));
    builder.addTerminalPair(`${archetype}-flap-${index}`, point, lengthFor(archetype, index, rng));
  }
}

function addSymmetricHubbedLimbs(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  const pairs = pairCountFor(archetype, targetCreases) + (targetCreases > 500 ? 1 : 0);
  const hubPairs = Math.max(1, Math.min(4, Math.ceil(pairs / 2)));
  let terminalIndex = 0;
  for (let hubIndex = 0; hubIndex < hubPairs; hubIndex++) {
    const hubPoint = mirroredSidePoint(builder.symmetryVariant, rng, hubIndex, hubPairs, rng.float(0.16, 0.31));
    const [hubA, hubB] = builder.addHubPair(`${archetype}-hub-${hubIndex}`, hubPoint, rng.float(0.13, 0.23));
    const children = 1 + (hubIndex < pairs - hubPairs ? 1 : 0) + (targetCreases > 500 && rng.next() < 0.45 ? 1 : 0);
    for (let child = 0; child < children; child++) {
      const point = offsetPoint(hubPoint, rng.float(-Math.PI, Math.PI), rng.float(0.12, 0.24));
      builder.addMirroredTerminalPair(`${archetype}-limb-${terminalIndex}`, hubA, hubB, point, lengthFor(archetype, terminalIndex, rng) * rng.float(0.72, 1.05));
      terminalIndex++;
    }
  }
}

function addSymmetricSpineChain(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  const spineHubs = targetCreases > 500 ? rng.int(3, 5) : rng.int(2, 4);
  const positions = sortedAxisPositions(spineHubs, rng);
  let parent = "root";
  let terminalIndex = 0;
  for (let index = 0; index < spineHubs; index++) {
    const hub = builder.addAxisHub(`${archetype}-spine-${index}`, positions[index], parent, index === 0 ? rng.float(0.12, 0.22) : rng.float(0.1, 0.2));
    const pairCount = index === 0 || index === spineHubs - 1 ? 1 : rng.int(1, 2);
    for (let pair = 0; pair < pairCount; pair++) {
      const point = offsetFromSymmetryAxis(builder.symmetryVariant, positions[index], rng.float(0.16, 0.33), rng.float(-0.06, 0.06));
      builder.addMirroredTerminalPair(`${archetype}-spine-flap-${terminalIndex}`, hub, hub, point, lengthFor(archetype, terminalIndex, rng) * rng.float(0.75, 1.1));
      terminalIndex++;
    }
    parent = hub;
  }
}

function addSymmetricBranchedHybrid(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  addSymmetricSpineChain(builder, rng, archetype, Math.max(180, targetCreases));
  const extraPairs = targetCreases > 500 ? 2 : 1;
  for (let index = 0; index < extraPairs; index++) {
    const hubPoint = mirroredSidePoint(builder.symmetryVariant, rng, index, extraPairs + 1, rng.float(0.2, 0.34));
    const [hubA, hubB] = builder.addHubPair(`${archetype}-side-hub-${index}`, hubPoint, rng.float(0.12, 0.2));
    const terminal = offsetPoint(hubPoint, rng.float(-Math.PI, Math.PI), rng.float(0.16, 0.26));
    builder.addMirroredTerminalPair(`${archetype}-side-terminal-${index}`, hubA, hubB, terminal, lengthFor(archetype, index + 5, rng) * 0.9);
  }
}

function addAsymmetricTopology(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  topology: TreeMakerTopology,
  targetCreases: number,
): void {
  if (topology === "radial-star") {
    addAsymmetricRadialStar(builder, rng, archetype, targetCreases);
  } else {
    addAsymmetricBranchedTree(builder, rng, archetype, targetCreases, topology);
  }
}

function addAsymmetricRadialStar(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  const terminals = pairCountFor(archetype, targetCreases) * 2 + rng.int(0, 2);
  for (let index = 0; index < terminals; index++) {
    const angle = rng.float(0, Math.PI * 2);
    const radius = rng.float(0.22, 0.47);
    const id = builder.addTerminal(`${archetype}-flap-${index}`, [
      clamp(0.5 + Math.cos(angle) * radius, 0.06, 0.94),
      clamp(0.5 + Math.sin(angle) * radius, 0.06, 0.94),
    ]);
    builder.addEdge("root", id, lengthFor(archetype, index, rng));
  }
}

function addAsymmetricBranchedTree(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
  topology: TreeMakerTopology,
): void {
  const hubCount = topology === "spine-chain" ? rng.int(2, 4) : rng.int(2, targetCreases > 500 ? 5 : 3);
  const hubs: string[] = [];
  let parent = "root";
  for (let index = 0; index < hubCount; index++) {
    const point: [number, number] = [rng.float(0.24, 0.76), rng.float(0.24, 0.76)];
    const hub = builder.addHub(`${archetype}-hub-${index}`, point, topology === "spine-chain" ? parent : "root", rng.float(0.12, 0.24));
    hubs.push(hub);
    if (topology === "spine-chain") parent = hub;
  }
  const terminals = pairCountFor(archetype, targetCreases) * 2 + (targetCreases > 500 ? rng.int(2, 4) : rng.int(0, 2));
  for (let index = 0; index < terminals; index++) {
    const parentHub = rng.choice(hubs);
    const parentNode = builder.node(parentHub);
    const point = offsetPoint([parentNode.x, parentNode.y], rng.float(-Math.PI, Math.PI), rng.float(0.13, 0.3));
    const id = builder.addTerminal(`${archetype}-flap-${index}`, point);
    builder.addEdge(parentHub, id, lengthFor(archetype, index, rng) * rng.float(0.68, 1.12));
  }
}

function pairCountFor(archetype: TreeMakerArchetype, targetCreases: number): number {
  const densityBump = targetCreases > 500 ? 2 : targetCreases > 180 ? 1 : 0;
  if (archetype === "insect") return 4 + densityBump;
  if (archetype === "quadruped") return 3 + densityBump;
  if (archetype === "bird") return 3 + densityBump;
  if (archetype === "object") return 2 + densityBump;
  return 3 + densityBump;
}

function lengthFor(archetype: TreeMakerArchetype, index: number, rng: SeededRandom): number {
  const base = archetype === "insect" ? 0.28 : archetype === "bird" ? 0.34 : archetype === "object" ? 0.22 : 0.3;
  return base + (index % 3) * 0.035 + rng.float(-0.025, 0.035);
}

function validateMirroredTopology(spec: TreeMakerAdapterSpec): string[] {
  const errors: string[] = [];
  const terminals = spec.nodes.filter((node) => node.kind === "terminal");
  const unmatched = new Set(terminals.map((node) => node.id));
  for (const node of terminals) {
    if (!unmatched.has(node.id)) continue;
    const mirrored = mirrorPoint([node.x, node.y], spec.symmetryVariant);
    const partner = terminals.find((candidate) =>
      candidate.id !== node.id &&
      unmatched.has(candidate.id) &&
      Math.abs(candidate.x - mirrored[0]) < 1e-6 &&
      Math.abs(candidate.y - mirrored[1]) < 1e-6
    );
    if (!partner) {
      errors.push(`terminal ${node.id} has no mirrored partner`);
      continue;
    }
    const edge = spec.edges.find((item) => item.to === node.id);
    const partnerEdge = spec.edges.find((item) => item.to === partner.id);
    if (!edge || !partnerEdge || Math.abs(edge.length - partnerEdge.length) > 1e-6) {
      errors.push(`terminal ${node.id} and ${partner.id} have mismatched mirrored edge lengths`);
    }
    unmatched.delete(node.id);
    unmatched.delete(partner.id);
  }
  return errors;
}

function branchDepth(spec: TreeMakerAdapterSpec): number {
  const adjacency = new Map<string, string[]>();
  for (const edge of spec.edges) {
    adjacency.set(edge.from, [...(adjacency.get(edge.from) ?? []), edge.to]);
    adjacency.set(edge.to, [...(adjacency.get(edge.to) ?? []), edge.from]);
  }
  let maxDepth = 0;
  const seen = new Set(["root"]);
  const queue: Array<{ id: string; depth: number }> = [{ id: "root", depth: 0 }];
  while (queue.length) {
    const current = queue.shift()!;
    maxDepth = Math.max(maxDepth, current.depth);
    for (const next of adjacency.get(current.id) ?? []) {
      if (seen.has(next)) continue;
      seen.add(next);
      queue.push({ id: next, depth: current.depth + 1 });
    }
  }
  return maxDepth;
}

function mirrorPoint(point: [number, number], symmetry: TreeMakerSymmetryVariant): [number, number] {
  const [x, y] = point;
  if (symmetry === "vertical") return [1 - x, y];
  if (symmetry === "horizontal") return [x, 1 - y];
  if (symmetry === "main-diagonal") return [y, x];
  if (symmetry === "anti-diagonal") return [1 - y, 1 - x];
  return point;
}

function pointOnSymmetryAxis(symmetry: TreeMakerSymmetryVariant, position: number): [number, number] {
  const t = clamp(position, 0.16, 0.84);
  if (symmetry === "vertical") return [0.5, t];
  if (symmetry === "horizontal") return [t, 0.5];
  if (symmetry === "main-diagonal") return [t, t];
  if (symmetry === "anti-diagonal") return [t, 1 - t];
  return [t, 0.5];
}

function sortedAxisPositions(count: number, rng: SeededRandom): number[] {
  const positions: number[] = [];
  for (let index = 0; index < count; index++) {
    const base = 0.22 + (index / Math.max(1, count - 1)) * 0.56;
    positions.push(clamp(base + rng.float(-0.035, 0.035), 0.16, 0.84));
  }
  return positions.sort((a, b) => a - b);
}

function mirroredSidePoint(
  symmetry: TreeMakerSymmetryVariant,
  rng: SeededRandom,
  index: number,
  total: number,
  radius: number,
): [number, number] {
  const t = clamp(0.18 + ((index + 0.55) / Math.max(1, total)) * 0.64 + rng.float(-0.035, 0.035), 0.12, 0.88);
  const d = clamp(radius * rng.float(0.45, 0.9), 0.08, 0.34);
  if (symmetry === "vertical") return [clamp(0.5 - d, 0.06, 0.44), t];
  if (symmetry === "horizontal") return [t, clamp(0.5 - d, 0.06, 0.44)];
  if (symmetry === "main-diagonal") return [clamp(t - d * 0.65, 0.06, 0.94), clamp(t + d * 0.65, 0.06, 0.94)];
  if (symmetry === "anti-diagonal") return [clamp(t + d * 0.65, 0.06, 0.94), clamp(1 - t + d * 0.65, 0.06, 0.94)];
  return [rng.float(0.08, 0.92), rng.float(0.08, 0.92)];
}

function offsetFromSymmetryAxis(
  symmetry: TreeMakerSymmetryVariant,
  position: number,
  distance: number,
  alongJitter: number,
): [number, number] {
  const t = clamp(position + alongJitter, 0.12, 0.88);
  const d = clamp(distance, 0.08, 0.36);
  if (symmetry === "vertical") return [clamp(0.5 - d, 0.06, 0.44), t];
  if (symmetry === "horizontal") return [t, clamp(0.5 - d, 0.06, 0.44)];
  if (symmetry === "main-diagonal") return [clamp(t - d * 0.7, 0.06, 0.94), clamp(t + d * 0.7, 0.06, 0.94)];
  if (symmetry === "anti-diagonal") return [clamp(t + d * 0.7, 0.06, 0.94), clamp(1 - t + d * 0.7, 0.06, 0.94)];
  return [clamp(0.5 + d, 0.06, 0.94), t];
}

function offsetPoint(point: [number, number], angle: number, radius: number): [number, number] {
  return [
    clamp(point[0] + Math.cos(angle) * radius, 0.06, 0.94),
    clamp(point[1] + Math.sin(angle) * radius, 0.06, 0.94),
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
