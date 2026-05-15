import type { BPRole, EdgeAssignment } from "./types.ts";

export type PortAssignment = Extract<EdgeAssignment, "M" | "V">;
export type SolverPortOrientation = "horizontal" | "vertical" | "diagonal-positive" | "diagonal-negative";
export type SolverPortParity = "integer" | "half";

export interface SolverPort {
  id: string;
  orientation: SolverPortOrientation;
  side: "top" | "right" | "bottom" | "left" | "interior";
  width: number;
  parity: SolverPortParity;
  sequence: PortAssignment[];
  role?: BPRole;
}

export interface RegionState {
  id: string;
  regionId: string;
  label?: string;
  templateId?: string;
  phase?: number;
  cost?: number;
  ports: SolverPort[];
}

export interface RegionVariable {
  id: string;
  rank?: number;
  states: RegionState[];
}

export interface ConnectorState {
  id: string;
  label?: string;
  cost?: number;
  from: SolverPort;
  to: SolverPort;
}

export interface PortConstraint {
  id: string;
  aRegion: string;
  aPort: string;
  bRegion: string;
  bPort: string;
  sequenceOrder?: "same" | "reversed";
  connectorStates?: ConnectorState[];
}

export interface PortSolverProblem {
  id: string;
  regions: RegionVariable[];
  constraints: PortConstraint[];
}

export interface PortCompatibilityResult {
  ok: boolean;
  reason?: string;
}

export interface PortJoinResolution {
  constraintId: string;
  aRegion: string;
  aState: string;
  aPort: string;
  bRegion: string;
  bState: string;
  bPort: string;
  connectorId?: string;
  mode: "direct" | "connector";
}

export interface PortSolverTraceEvent {
  step: number;
  event: "select-region" | "try-state" | "accept-state" | "reject-state" | "backtrack" | "solution";
  regionId?: string;
  stateId?: string;
  detail?: string;
}

export interface PortSolverOptions {
  maxSteps?: number;
}

export interface PortSolverResult {
  ok: boolean;
  problemId: string;
  steps: number;
  assignments: Record<string, string>;
  states: RegionState[];
  joins: PortJoinResolution[];
  errors: string[];
  trace: PortSolverTraceEvent[];
}

const DEFAULT_MAX_STEPS = 25_000;
const EPSILON = 1e-9;

export function solvePortAssignmentProblem(
  problem: PortSolverProblem,
  options: PortSolverOptions = {},
): PortSolverResult {
  const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
  const errors = validateProblem(problem);
  if (errors.length) return emptyResult(problem, 0, errors);

  const variables = stableVariables(problem.regions);
  const variableById = new Map(variables.map((variable) => [variable.id, variable]));
  const degree = regionDegrees(problem);
  const assigned = new Map<string, RegionState>();
  const trace: PortSolverTraceEvent[] = [];
  let steps = 0;
  let stepLimitHit = false;

  const emit = (event: Omit<PortSolverTraceEvent, "step">): void => {
    trace.push({ step: steps, ...event });
  };

  const viableStates = (variable: RegionVariable): RegionState[] =>
    stableStates(variable.states).filter((state) => stateViable(problem, state, assigned, variableById));

  const chooseVariable = (): RegionVariable | undefined => {
    const candidates = variables.filter((variable) => !assigned.has(variable.id));
    if (!candidates.length) return undefined;
    const ranked = candidates.map((variable) => ({
      variable,
      viableCount: viableStates(variable).length,
      degree: degree.get(variable.id) ?? 0,
    })).sort((a, b) =>
      a.viableCount - b.viableCount ||
      (a.variable.rank ?? Number.MAX_SAFE_INTEGER) - (b.variable.rank ?? Number.MAX_SAFE_INTEGER) ||
      b.degree - a.degree ||
      a.variable.id.localeCompare(b.variable.id)
    );
    return ranked[0]?.variable;
  };

  const search = (): boolean => {
    steps += 1;
    if (steps > maxSteps) {
      stepLimitHit = true;
      return false;
    }

    const variable = chooseVariable();
    if (!variable) {
      emit({ event: "solution", detail: "all regions assigned" });
      return true;
    }

    const candidates = viableStates(variable);
    emit({
      event: "select-region",
      regionId: variable.id,
      detail: `${candidates.length}/${variable.states.length} viable states`,
    });
    for (const state of candidates) {
      emit({ event: "try-state", regionId: variable.id, stateId: state.id });
      assigned.set(variable.id, state);
      const failure = firstAssignedConstraintFailure(problem, assigned);
      if (failure) {
        emit({
          event: "reject-state",
          regionId: variable.id,
          stateId: state.id,
          detail: failure,
        });
        assigned.delete(variable.id);
        continue;
      }
      const emptyDomain = firstEmptyForwardDomain(problem, assigned, variableById);
      if (emptyDomain) {
        emit({
          event: "reject-state",
          regionId: variable.id,
          stateId: state.id,
          detail: `forward-domain-empty:${emptyDomain}`,
        });
        assigned.delete(variable.id);
        continue;
      }
      emit({ event: "accept-state", regionId: variable.id, stateId: state.id });
      if (search()) return true;
      emit({ event: "backtrack", regionId: variable.id, stateId: state.id });
      assigned.delete(variable.id);
    }

    return false;
  };

  const ok = search();
  if (!ok) {
    return {
      ok: false,
      problemId: problem.id,
      steps,
      assignments: {},
      states: [],
      joins: [],
      errors: [stepLimitHit ? `port-solver-step-limit:${maxSteps}` : "port-solver-unsat"],
      trace,
    };
  }

  const states = variables.map((variable) => assigned.get(variable.id)).filter((state): state is RegionState => Boolean(state));
  return {
    ok: true,
    problemId: problem.id,
    steps,
    assignments: Object.fromEntries(states.map((state) => [state.regionId, state.id])),
    states,
    joins: problem.constraints.map((constraint) => resolveConstraint(constraint, assigned)!),
    errors: [],
    trace,
  };
}

export function compatiblePorts(
  a: SolverPort,
  b: SolverPort,
  sequenceOrder: "same" | "reversed" = "same",
): PortCompatibilityResult {
  if (a.orientation !== b.orientation) return { ok: false, reason: `orientation:${a.orientation}:${b.orientation}` };
  if (Math.abs(a.width - b.width) > EPSILON) return { ok: false, reason: `width:${a.width}:${b.width}` };
  if (a.parity !== b.parity) return { ok: false, reason: `parity:${a.parity}:${b.parity}` };
  const expected = sequenceOrder === "reversed" ? [...b.sequence].reverse() : b.sequence;
  if (a.sequence.length !== expected.length) return { ok: false, reason: `lane-count:${a.sequence.length}:${expected.length}` };
  for (let index = 0; index < a.sequence.length; index += 1) {
    if (a.sequence[index] !== expected[index]) {
      return {
        ok: false,
        reason: `sequence:${sequenceToString(a.sequence)}:${sequenceToString(expected)}`,
      };
    }
  }
  return { ok: true };
}

export function alternatingSequence(start: PortAssignment, laneCount: number): PortAssignment[] {
  return Array.from({ length: laneCount }, (_, index) => index % 2 === 0 ? start : flipAssignment(start));
}

export function sequenceToString(sequence: PortAssignment[]): string {
  return sequence.join("");
}

export function port(
  id: string,
  sequence: PortAssignment[],
  options: Partial<Omit<SolverPort, "id" | "sequence">> = {},
): SolverPort {
  return {
    id,
    orientation: options.orientation ?? "horizontal",
    side: options.side ?? "interior",
    width: options.width ?? sequence.length,
    parity: options.parity ?? "integer",
    sequence,
    role: options.role,
  };
}

function resolveConstraint(
  constraint: PortConstraint,
  assigned: Map<string, RegionState>,
): PortJoinResolution | undefined {
  const aState = assigned.get(constraint.aRegion);
  const bState = assigned.get(constraint.bRegion);
  if (!aState || !bState) return undefined;
  const aPort = findPort(aState, constraint.aPort);
  const bPort = findPort(bState, constraint.bPort);
  if (!aPort || !bPort) return undefined;
  if (compatiblePorts(aPort, bPort, constraint.sequenceOrder).ok) {
    return resolution(constraint, aState, bState, "direct");
  }
  for (const connector of stableConnectors(constraint.connectorStates ?? [])) {
    const aToConnector = compatiblePorts(aPort, connector.from, constraint.sequenceOrder).ok;
    const connectorToB = compatiblePorts(connector.to, bPort, constraint.sequenceOrder).ok;
    if (aToConnector && connectorToB) {
      return resolution(constraint, aState, bState, "connector", connector.id);
    }
  }
  return undefined;
}

function resolution(
  constraint: PortConstraint,
  aState: RegionState,
  bState: RegionState,
  mode: PortJoinResolution["mode"],
  connectorId?: string,
): PortJoinResolution {
  return {
    constraintId: constraint.id,
    aRegion: constraint.aRegion,
    aState: aState.id,
    aPort: constraint.aPort,
    bRegion: constraint.bRegion,
    bState: bState.id,
    bPort: constraint.bPort,
    mode,
    connectorId,
  };
}

function stateViable(
  problem: PortSolverProblem,
  state: RegionState,
  assigned: Map<string, RegionState>,
  variableById: Map<string, RegionVariable>,
): boolean {
  const nextAssigned = new Map(assigned);
  nextAssigned.set(state.regionId, state);
  return !firstAssignedConstraintFailure(problem, nextAssigned) &&
    !firstEmptyForwardDomain(problem, nextAssigned, variableById);
}

function firstAssignedConstraintFailure(problem: PortSolverProblem, assigned: Map<string, RegionState>): string | undefined {
  for (const constraint of problem.constraints) {
    if (!assigned.has(constraint.aRegion) || !assigned.has(constraint.bRegion)) continue;
    if (!resolveConstraint(constraint, assigned)) return `incompatible-port:${constraint.id}`;
  }
  return undefined;
}

function firstEmptyForwardDomain(
  problem: PortSolverProblem,
  assigned: Map<string, RegionState>,
  variableById: Map<string, RegionVariable>,
): string | undefined {
  for (const variable of variableById.values()) {
    if (assigned.has(variable.id)) continue;
    const hasViable = stableStates(variable.states).some((state) => {
      const probe = new Map(assigned);
      probe.set(variable.id, state);
      return !firstAssignedConstraintFailure(problem, probe);
    });
    if (!hasViable) return variable.id;
  }
  return undefined;
}

function validateProblem(problem: PortSolverProblem): string[] {
  const errors: string[] = [];
  const regionIds = new Set<string>();
  for (const region of problem.regions) {
    if (regionIds.has(region.id)) errors.push(`duplicate-region:${region.id}`);
    regionIds.add(region.id);
    if (!region.states.length) errors.push(`empty-region-domain:${region.id}`);
    for (const state of region.states) {
      if (state.regionId !== region.id) errors.push(`state-region-mismatch:${region.id}:${state.id}:${state.regionId}`);
      const portIds = new Set<string>();
      for (const item of state.ports) {
        if (portIds.has(item.id)) errors.push(`duplicate-state-port:${state.id}:${item.id}`);
        portIds.add(item.id);
      }
    }
  }
  for (const constraint of problem.constraints) {
    if (!regionIds.has(constraint.aRegion)) errors.push(`missing-constraint-region:${constraint.id}:${constraint.aRegion}`);
    if (!regionIds.has(constraint.bRegion)) errors.push(`missing-constraint-region:${constraint.id}:${constraint.bRegion}`);
  }
  return errors;
}

function emptyResult(problem: PortSolverProblem, steps: number, errors: string[]): PortSolverResult {
  return {
    ok: false,
    problemId: problem.id,
    steps,
    assignments: {},
    states: [],
    joins: [],
    errors,
    trace: [],
  };
}

function findPort(state: RegionState, portId: string): SolverPort | undefined {
  return state.ports.find((item) => item.id === portId);
}

function stableVariables(variables: RegionVariable[]): RegionVariable[] {
  return [...variables].sort((a, b) =>
    (a.rank ?? Number.MAX_SAFE_INTEGER) - (b.rank ?? Number.MAX_SAFE_INTEGER) ||
    a.id.localeCompare(b.id)
  );
}

function stableStates(states: RegionState[]): RegionState[] {
  return [...states].sort((a, b) =>
    (a.cost ?? 0) - (b.cost ?? 0) ||
    (a.phase ?? 0) - (b.phase ?? 0) ||
    a.id.localeCompare(b.id)
  );
}

function stableConnectors(connectors: ConnectorState[]): ConnectorState[] {
  return [...connectors].sort((a, b) =>
    (a.cost ?? 0) - (b.cost ?? 0) ||
    a.id.localeCompare(b.id)
  );
}

function regionDegrees(problem: PortSolverProblem): Map<string, number> {
  const result = new Map<string, number>();
  for (const constraint of problem.constraints) {
    result.set(constraint.aRegion, (result.get(constraint.aRegion) ?? 0) + 1);
    result.set(constraint.bRegion, (result.get(constraint.bRegion) ?? 0) + 1);
  }
  return result;
}

function flipAssignment(assignment: PortAssignment): PortAssignment {
  return assignment === "M" ? "V" : "M";
}
