import { SeededRandom } from "./random.ts";
import type {
  GenerationConfig,
  TreeMakerArchetype,
  TreeMakerSamplerConfig,
  TreeMakerSymmetryClass,
  TreeMakerSymmetryVariant,
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

export function generateTreeMakerSpec(config: GenerationConfig): TreeMakerAdapterSpec {
  const rng = new SeededRandom(config.seed);
  const sampler = config.treeMakerSampler ?? {};
  const symmetryClass = chooseSymmetryClass(rng, sampler);
  const symmetryVariant = chooseSymmetryVariant(rng, symmetryClass, sampler);
  const archetype = chooseArchetype(rng, sampler);
  const builder = new TreeBuilder(config.id, config.seed, config.numCreases, archetype, symmetryClass, symmetryVariant);

  if (symmetryClass === "asymmetric") {
    addAsymmetricTerminals(builder, rng, archetype, config.numCreases);
  } else {
    addSymmetricTerminals(builder, rng, archetype, config.numCreases);
  }
  builder.addOptionalSpine(rng);
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
    private readonly symmetryVariant: TreeMakerSymmetryVariant,
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

  addTerminal(label: string, point: [number, number]): string {
    const id = `t${this.nextTerminal++}`;
    this.nodes.push({ id, kind: "terminal", label, x: round(point[0]), y: round(point[1]) });
    return id;
  }

  addHub(label: string, point: [number, number]): string {
    const id = `h${this.nextHub++}`;
    this.nodes.push({ id, kind: "hub", label, x: round(point[0]), y: round(point[1]) });
    this.addEdge("root", id, 0.22);
    return id;
  }

  addEdge(from: string, to: string, length: number): void {
    this.edges.push({ id: `e${this.edges.length}`, from, to, length: round(length) });
  }

  addOptionalSpine(rng: SeededRandom): void {
    if (this.archetype === "abstract" || this.edges.length < 8 || rng.next() < 0.45) return;
    const hubPoint: [number, number] = this.symmetryClass === "asymmetric"
      ? [rng.float(0.35, 0.65), rng.float(0.35, 0.65)]
      : this.symmetryVariant === "vertical"
        ? [0.5, rng.float(0.32, 0.68)]
        : this.symmetryVariant === "horizontal"
          ? [rng.float(0.32, 0.68), 0.5]
          : [0.5, 0.5];
    this.addHub("secondary-hub", hubPoint);
  }
}

function addSymmetricTerminals(
  builder: TreeBuilder,
  rng: SeededRandom,
  archetype: TreeMakerArchetype,
  targetCreases: number,
): void {
  const pairs = pairCountFor(archetype, targetCreases);
  for (let index = 0; index < pairs; index++) {
    const angle = (index + 0.6) / pairs * Math.PI * 0.92;
    const radius = rng.float(0.28, 0.46);
    const point: [number, number] = [
      clamp(0.5 - Math.cos(angle) * radius, 0.08, 0.92),
      clamp(0.5 + Math.sin(angle) * radius * rng.float(0.75, 1.15), 0.08, 0.92),
    ];
    builder.addTerminalPair(`${archetype}-flap-${index}`, point, lengthFor(archetype, index, rng));
  }
}

function addAsymmetricTerminals(
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
