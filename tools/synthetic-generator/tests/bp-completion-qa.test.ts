import { expect, test } from "bun:test";
import { completeBoxPleatLayout, fixtureCompletionLayout } from "../src/bp-completion.ts";
import { runBPCompletionQA } from "../src/bp-completion-qa.ts";

test("completion QA separates strict label readiness from production distribution readiness", () => {
  const result = completeBoxPleatLayout(fixtureCompletionLayout("insect-lite"));
  expect(result.ok).toBe(true);

  const report = runBPCompletionQA(result.fold!);
  expect(report.strictLabelReady).toBe(true);
  expect(report.productionDistributionReady).toBe(false);
  expect(report.errors).toEqual([]);
  expect(report.warnings).toContain("baseline-compiler-version-not-production");
});

test("completion QA rejects raw or ambiguous label provenance", () => {
  const result = completeBoxPleatLayout(fixtureCompletionLayout("two-flap-stretch"));
  const fold = {
    ...result.fold!,
    label_policy: {
      labelSource: "bp-studio-raw" as const,
      geometrySource: "bp-studio-raw" as const,
      assignmentSource: "bp-studio-raw" as const,
      trainingEligible: false,
      notes: ["test"],
    },
  };

  const report = runBPCompletionQA(fold);
  expect(report.strictLabelReady).toBe(false);
  expect(report.errors).toContain("labels-not-compiler-training-eligible");
});
