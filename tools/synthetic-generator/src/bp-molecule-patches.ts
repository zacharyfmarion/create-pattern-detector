import type {
  CompletionPoint,
  CompletionSegment,
  MoleculeInstance,
  MoleculeKind,
  MoleculePatch,
  MoleculeTemplate,
  MoleculeTransform,
  Port,
} from "./bp-completion-contracts.ts";
import type { BPRole } from "./types.ts";

type Point = [number, number];

const LIBRARY_VERSION = "strict-bp-molecules/v0.1.0";

export const MOLECULE_ADDITION_ORDER: MoleculeKind[] = [
  "corner-fan",
  "river-corridor",
  "diagonal-staircase",
  "body-panel",
  "diamond-connector",
  "stretch-gadget",
];

export function moleculePatchLibrary(): MoleculePatch[] {
  return [
    patch("terminal-corner-fan", "corner-fan", [
      vertex("c", 0, 0),
      vertex("e", 2, 0),
      vertex("n", 0, 2),
      vertex("s", 0, -2),
      vertex("ne", 2, 2),
      vertex("se", 2, -2),
    ], [
      segment("fan-e", "c", "e", "V", "axis"),
      segment("fan-n", "c", "n", "V", "hinge"),
      segment("fan-s", "c", "s", "V", "hinge"),
      segment("fan-ne", "c", "ne", "M", "ridge"),
      segment("fan-se", "c", "se", "M", "ridge"),
    ], [
      port("corridor", "terminal-corner-fan", "horizontal", "right", 2, "axis", { x: 2, y: 0 }),
    ]),
    patch("straight-pleat-corridor", "river-corridor", [
      vertex("w", -2, 0),
      vertex("c", 0, 0),
      vertex("e", 2, 0),
      vertex("n", 0, 1),
      vertex("s", 0, -1),
    ], [
      segment("corridor-w", "w", "c", "V", "axis"),
      segment("corridor-e", "c", "e", "V", "axis"),
      segment("corridor-n", "c", "n", "M", "hinge"),
      segment("corridor-s", "c", "s", "V", "hinge"),
    ], [
      port("west", "straight-pleat-corridor", "horizontal", "left", 2, "axis", { x: -2, y: 0 }),
      port("east", "straight-pleat-corridor", "horizontal", "right", 2, "axis", { x: 2, y: 0 }),
    ]),
    patch("diagonal-staircase-turn", "diagonal-staircase", [
      vertex("a", -2, 0),
      vertex("c", 0, 0),
      vertex("b", 2, 2),
      vertex("e", 2, 0),
      vertex("n", 0, 2),
    ], [
      segment("turn-in", "a", "c", "V", "axis"),
      segment("turn-diagonal", "c", "b", "M", "ridge"),
      segment("turn-e", "c", "e", "V", "hinge"),
      segment("turn-n", "c", "n", "V", "axis"),
    ], [
      port("west", "diagonal-staircase-turn", "horizontal", "left", 2, "axis", { x: -2, y: 0 }),
      port("northeast", "diagonal-staircase-turn", "diagonal-positive", "interior", 2, "ridge", { x: 2, y: 2 }),
    ]),
    patch("body-hub-panel", "body-panel", [
      vertex("c", 0, 0),
      vertex("e", 2, 0),
      vertex("w", -2, 0),
      vertex("n", 0, 2),
      vertex("s", 0, -2),
      vertex("ne", 2, 2),
      vertex("nw", -2, 2),
      vertex("se", 2, -2),
      vertex("sw", -2, -2),
    ], [
      segment("hub-e", "c", "e", "V", "hinge"),
      segment("hub-w", "c", "w", "V", "hinge"),
      segment("hub-n", "c", "n", "V", "axis"),
      segment("hub-s", "c", "s", "V", "axis"),
      segment("hub-ne", "c", "ne", "M", "ridge"),
      segment("hub-nw", "c", "nw", "M", "ridge"),
      segment("hub-se", "c", "se", "M", "ridge"),
      segment("hub-sw", "c", "sw", "M", "ridge"),
    ], [
      port("west", "body-hub-panel", "horizontal", "left", 2, "hinge", { x: -2, y: 0 }),
      port("east", "body-hub-panel", "horizontal", "right", 2, "hinge", { x: 2, y: 0 }),
      port("north", "body-hub-panel", "vertical", "top", 2, "axis", { x: 0, y: 2 }),
      port("south", "body-hub-panel", "vertical", "bottom", 2, "axis", { x: 0, y: -2 }),
    ]),
    patch("diamond-chevron-connector", "diamond-connector", [
      vertex("n", 0, 2),
      vertex("e", 2, 0),
      vertex("s", 0, -2),
      vertex("w", -2, 0),
    ], [
      segment("diamond-ne", "n", "e", "M", "ridge"),
      segment("diamond-es", "e", "s", "M", "ridge"),
      segment("diamond-sw", "s", "w", "M", "ridge"),
      segment("diamond-wn", "w", "n", "M", "ridge"),
    ], [
      port("west", "diamond-chevron-connector", "diagonal-negative", "left", 2, "ridge", { x: -2, y: 0 }),
      port("east", "diamond-chevron-connector", "diagonal-positive", "right", 2, "ridge", { x: 2, y: 0 }),
    ]),
    patch("conservative-stretch-gadget", "stretch-gadget", [
      vertex("w", -2, 0),
      vertex("c", 0, 0),
      vertex("e", 2, 0),
      vertex("n", 0, 2),
      vertex("s", 0, -2),
      vertex("ne", 2, 2),
      vertex("sw", -2, -2),
    ], [
      segment("stretch-axis-w", "w", "c", "V", "stretch"),
      segment("stretch-axis-e", "c", "e", "V", "stretch"),
      segment("stretch-cross-n", "c", "n", "M", "axis"),
      segment("stretch-cross-s", "c", "s", "V", "axis"),
      segment("stretch-ridge-ne", "c", "ne", "M", "ridge"),
      segment("stretch-ridge-sw", "c", "sw", "M", "ridge"),
    ], [
      port("west", "conservative-stretch-gadget", "horizontal", "left", 2, "stretch", { x: -2, y: 0 }),
      port("east", "conservative-stretch-gadget", "horizontal", "right", 2, "stretch", { x: 2, y: 0 }),
    ]),
  ];
}

export function moleculeTemplateFromPatch(patch: MoleculePatch, id = patch.id): MoleculeTemplate {
  return {
    id,
    kind: patch.kind,
    version: patch.version,
    ports: patch.ports.map((item) => ({ ...item, moleculeId: id })),
    localProof: {
      kawasaki: true,
      maekawa: true,
      fixture: `${patch.id}-certification-fixture`,
    },
  };
}

export function createMoleculeInstance(
  id: string,
  patch: MoleculePatch,
  transform: MoleculeTransform,
): MoleculeInstance {
  return {
    id,
    templateId: patch.id,
    kind: patch.kind,
    transform,
    patch,
    ports: patch.ports.map((item) => {
      const position = item.position ? transformPoint([item.position.x, item.position.y], transform) : undefined;
      return {
        ...item,
        id: `${id}:${item.id}`,
        moleculeId: id,
        position: position ? { x: position[0], y: position[1] } : undefined,
      };
    }),
  };
}

export function instantiatePatchSegments(instance: MoleculeInstance): CompletionSegment[] {
  if (!instance.patch) return [];
  const vertices = new Map(instance.patch.vertices.map((item) => [item.id, transformPoint([item.x, item.y], instance.transform)]));
  return instance.patch.segments.map((segmentItem): CompletionSegment => {
    const p1 = vertices.get(segmentItem.from);
    const p2 = vertices.get(segmentItem.to);
    if (!p1 || !p2) throw new Error(`molecule patch ${instance.patch?.id} has missing segment vertex`);
    return {
      id: `${instance.id}:${segmentItem.id}`,
      moleculeId: instance.id,
      moleculeKind: instance.kind,
      p1,
      p2,
      assignment: segmentItem.assignment,
      role: segmentItem.role,
    };
  });
}

export function compassRayToSheet(center: Point, direction: Point): Point {
  const candidates: number[] = [];
  if (direction[0] > 0) candidates.push((1 - center[0]) / direction[0]);
  if (direction[0] < 0) candidates.push((0 - center[0]) / direction[0]);
  if (direction[1] > 0) candidates.push((1 - center[1]) / direction[1]);
  if (direction[1] < 0) candidates.push((0 - center[1]) / direction[1]);
  const t = Math.min(...candidates.filter((candidate) => Number.isFinite(candidate) && candidate > 1e-9));
  return roundPoint([center[0] + direction[0] * t, center[1] + direction[1] * t]);
}

export function roleForDirection(direction: Point): Exclude<BPRole, "border"> {
  if (Math.abs(direction[0]) > 0 && Math.abs(direction[1]) > 0) return "ridge";
  if (Math.abs(direction[0]) > 0) return "hinge";
  return "axis";
}

function patch(
  id: string,
  kind: MoleculeKind,
  vertices: MoleculePatch["vertices"],
  segments: MoleculePatch["segments"],
  ports: Port[],
): MoleculePatch {
  const xs = vertices.map((item) => item.x);
  const ys = vertices.map((item) => item.y);
  return {
    id,
    kind,
    version: LIBRARY_VERSION,
    vertices,
    segments,
    ports,
    bounds: {
      x1: Math.min(...xs),
      y1: Math.min(...ys),
      x2: Math.max(...xs),
      y2: Math.max(...ys),
    },
  };
}

function vertex(id: string, x: number, y: number): MoleculePatch["vertices"][number] {
  return { id, x, y };
}

function segment(
  id: string,
  from: string,
  to: string,
  assignment: "M" | "V",
  role: Exclude<BPRole, "border">,
): MoleculePatch["segments"][number] {
  return { id, from, to, assignment, role };
}

function port(
  id: string,
  moleculeId: string,
  orientation: Port["orientation"],
  side: Port["side"],
  width: number,
  role: BPRole,
  position: CompletionPoint,
): Port {
  return {
    id,
    moleculeId,
    orientation,
    side,
    coordinate: orientation === "vertical" ? position.x : position.y,
    width,
    parity: Number.isInteger(position.x) && Number.isInteger(position.y) ? "integer" : "half",
    role,
    expectedPartnerRole: role,
    position,
  };
}

function transformPoint(point: Point, transform: MoleculeTransform): Point {
  const scale = transform.scale ?? 1;
  let x = point[0] * scale;
  let y = point[1] * scale;
  if (transform.mirrorX) x = -x;
  if (transform.mirrorY) y = -y;
  const turns = transform.rotateQuarterTurns ?? 0;
  for (let i = 0; i < turns; i++) {
    const nextX = -y;
    y = x;
    x = nextX;
  }
  return roundPoint([x + transform.translate.x, y + transform.translate.y]);
}

function roundPoint(point: Point): Point {
  return [round(point[0]), round(point[1])];
}

function round(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}
