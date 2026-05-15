import { completeBoxPleatLayout, fixtureCompletionLayout } from "./bp-completion.ts";
import { validateFold } from "./validate.ts";
import type { CompositionFixture, MoleculeCertificationReport, MoleculeKind } from "./bp-completion-contracts.ts";
import type { FOLDFormat, ValidationConfig } from "./types.ts";

export const COMPLETION_FIXTURE_LADDER: CompositionFixture[] = [
  {
    id: "two-flap-stretch",
    description: "two terminal fans joined through a central body/stretch connector",
    requiredMoleculeKinds: ["corner-fan", "river-corridor", "flap-contour", "body-panel", "diamond-connector", "stretch-gadget"],
  },
  {
    id: "three-flap-relay",
    description: "three terminal fans routed into one body hub",
    requiredMoleculeKinds: ["corner-fan", "river-corridor", "flap-contour", "body-panel", "diamond-connector", "stretch-gadget"],
  },
  {
    id: "five-flap-uniaxial",
    description: "multi-terminal uniaxial chain around one body hub",
    requiredMoleculeKinds: ["corner-fan", "river-corridor", "flap-contour", "body-panel", "diamond-connector", "stretch-gadget"],
  },
  {
    id: "insect-lite",
    description: "paired appendage layout with antenna/tail-like terminal",
    requiredMoleculeKinds: ["corner-fan", "river-corridor", "flap-contour", "body-panel", "diamond-connector", "stretch-gadget"],
  },
];

const STRICT_CERTIFICATION_VALIDATION: ValidationConfig = {
  strictGlobal: true,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-6,
  maxVertices: 4000,
  maxEdges: 4000,
  requireBoxPleat: true,
  boxPleatMode: "dense",
  requireDense: false,
  requireRealistic: false,
};

export async function certifyCompletionFixture(
  fixtureId: CompositionFixture["id"],
  validationConfig: ValidationConfig = STRICT_CERTIFICATION_VALIDATION,
): Promise<MoleculeCertificationReport> {
  const fixture = COMPLETION_FIXTURE_LADDER.find((item) => item.id === fixtureId);
  if (!fixture) {
    return {
      fixtureId,
      ok: false,
      moleculeKinds: [],
      checkedPortJoins: 0,
      rejectedPortJoins: 0,
      danglingEndpointCount: 0,
      validationPassed: [],
      validationFailed: ["fixture-lookup"],
      errors: [`unknown fixture: ${fixtureId}`],
    };
  }

  const completion = completeBoxPleatLayout(fixtureCompletionLayout(fixture.id as Parameters<typeof fixtureCompletionLayout>[0]));
  if (!completion.ok || !completion.fold) {
    return {
      fixtureId,
      ok: false,
      moleculeKinds: completion.moleculeInstances?.map((instance) => instance.kind) ?? [],
      checkedPortJoins: completion.portJoins.length,
      rejectedPortJoins: completion.portJoins.filter((join) => !join.accepted).length,
      danglingEndpointCount: 0,
      validationPassed: [],
      validationFailed: ["completion"],
      errors: completion.rejected.map((item) => `${item.code}:${item.message}`),
    };
  }

  const validation = await validateFold(completion.fold, validationConfig);
  const danglingEndpointCount = countInteriorDanglingActiveEndpoints(completion.fold);
  const moleculeKinds = Object.keys(completion.fold.molecule_metadata?.molecules ?? {}) as MoleculeKind[];
  const missingKinds = fixture.requiredMoleculeKinds.filter((kind) => !moleculeKinds.includes(kind));
  const errors = [
    ...validation.errors,
    ...missingKinds.map((kind) => `missing-required-molecule-kind:${kind}`),
  ];
  if (danglingEndpointCount > 0) errors.push(`dangling-active-endpoints:${danglingEndpointCount}`);

  return {
    fixtureId,
    ok: validation.valid && danglingEndpointCount === 0 && missingKinds.length === 0,
    moleculeKinds,
    checkedPortJoins: completion.portJoins.length,
    rejectedPortJoins: completion.portJoins.filter((join) => !join.accepted).length,
    danglingEndpointCount,
    validationPassed: validation.passed,
    validationFailed: validation.failed,
    errors,
  };
}

export function countInteriorDanglingActiveEndpoints(fold: FOLDFormat): number {
  const borderVertices = new Set<number>();
  const activeDegrees = Array.from({ length: fold.vertices_coords.length }, () => 0);
  for (const [edgeIndex, [a, b]] of fold.edges_vertices.entries()) {
    const assignment = fold.edges_assignment[edgeIndex];
    if (assignment === "B") {
      borderVertices.add(a);
      borderVertices.add(b);
      continue;
    }
    if (assignment === "M" || assignment === "V") {
      activeDegrees[a] += 1;
      activeDegrees[b] += 1;
    }
  }
  return activeDegrees.filter((degree, vertex) => degree === 1 && !borderVertices.has(vertex)).length;
}
