import { access } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveBpStudioRoot } from "./bp-studio-root.ts";

export interface BpOptimizerFlapRequest {
  id: number;
  width: number;
  height: number;
}

export interface BpOptimizerHierarchy {
  leaves: number[];
  distMap: Array<[number, number, number]>;
  parents: Array<{
    id: number;
    radius: number;
    children: number[];
  }>;
}

export interface BpOptimizerRequest {
  type: "rect";
  flaps: BpOptimizerFlapRequest[];
  hierarchies: BpOptimizerHierarchy[];
  layout: "view" | "random";
  useBH: boolean;
  random: number;
  vec: Array<{ x: number; y: number }> | null;
}

export interface BpOptimizerFlapResult {
  id: number;
  x: number;
  y: number;
}

export interface BpOptimizerResult {
  width: number;
  height: number;
  flaps: BpOptimizerFlapResult[];
}

export interface BpOptimizerOptions {
  bpStudioRoot?: string;
  seed?: number;
}

interface RawOptimizerInstance {
  init(): void;
  solve(data: object, seed: number): Promise<RawVectorInt>;
}

interface RawVectorInt {
  size(): number;
  get(index: number): number;
  delete(): void;
}

interface RawOptimizerModule {
  default(options: { print?: (message: string) => void; checkInterrupt?: () => boolean }): Promise<RawOptimizerInstance>;
}

const UINT_MAX = 4294967295;

export async function solveWithBpStudioOptimizer(
  request: BpOptimizerRequest,
  options: BpOptimizerOptions = {},
): Promise<BpOptimizerResult> {
  const root = resolveBpStudioRoot(options.bpStudioRoot);
  const modulePath = await resolveOptimizerModule(root);
  const module = await import(pathToFileURL(modulePath).href) as RawOptimizerModule;
  const instance = await module.default({
    print: () => undefined,
    checkInterrupt: () => false,
  });
  instance.init();

  const raw = await instance.solve(toRawRequest(request), normalizeSeed(options.seed));
  try {
    const size = raw.size();
    if (size === 0) throw new Error("BP Studio optimizer did not find a solution");
    const sheet = raw.get(size - 1);
    return {
      width: sheet,
      height: sheet,
      flaps: request.flaps.map((flap, index) => ({
        id: flap.id,
        x: raw.get(index * 2),
        y: raw.get(index * 2 + 1),
      })),
    };
  } finally {
    raw.delete();
  }
}

export async function canLoadBpStudioOptimizer(bpStudioRoot?: string): Promise<boolean> {
  try {
    await resolveOptimizerModule(resolveBpStudioRoot(bpStudioRoot));
    return true;
  } catch {
    return false;
  }
}

async function resolveOptimizerModule(root: string): Promise<string> {
  const candidates = [
    join(root, "lib", "optimizer", "debug", "optimizer.js"),
    join(root, "lib", "optimizer", "dist", "optimizer.js"),
  ];
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error(
    `Cannot find BP Studio optimizer under ${root}. Clone BP Studio and set BP_STUDIO_ROOT, or pass --bp-studio-root.`,
  );
}

function toRawRequest(request: BpOptimizerRequest): object {
  return {
    type: request.type === "rect" ? 1 : 2,
    flaps: request.flaps,
    hierarchies: request.hierarchies,
    useView: request.layout === "view",
    useBH: request.useBH,
    random: request.random,
    vec: request.vec,
  };
}

function normalizeSeed(seed: number | undefined): number {
  if (seed === undefined || !Number.isInteger(seed) || seed < 0 || seed > UINT_MAX) {
    return Math.floor(Math.random() * UINT_MAX);
  }
  return seed;
}
