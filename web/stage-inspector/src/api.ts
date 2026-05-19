import type { Stage4Diagnostic, Stage4ExamplesResponse, StageInfo } from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const body = await response.text();
  const payload = body ? JSON.parse(body) : {};
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function fetchStages(): Promise<{ stages: StageInfo[] }> {
  return requestJson("/api/stages");
}

export function fetchStage4Examples(): Promise<Stage4ExamplesResponse> {
  return requestJson("/api/stage4/examples");
}

export function fetchStage4Diagnostic(exampleKey: string): Promise<Stage4Diagnostic> {
  return requestJson(`/api/stage4/examples/${encodeURIComponent(exampleKey)}`);
}

export function fetchStage5Examples(): Promise<Stage4ExamplesResponse> {
  return requestJson("/api/stage5/examples");
}

export function fetchStage5Diagnostic(exampleKey: string): Promise<Stage4Diagnostic> {
  return requestJson(`/api/stage5/examples/${encodeURIComponent(exampleKey)}`);
}

export function recomputeStage4Diagnostic(input: {
  exampleKey: string;
  threshold?: number;
  inferAssignments?: boolean;
}): Promise<Stage4Diagnostic> {
  return requestJson("/api/stage4/recompute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}
