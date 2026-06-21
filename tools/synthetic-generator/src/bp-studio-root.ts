import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// The vendored Box Pleating Studio subset (optimizer + layout core) lives next
// to this file under vendor/box-pleating-studio. Resolve it relative to the
// module so it works regardless of the process CWD. Override with the
// BP_STUDIO_ROOT env var or a --bp-studio-root flag to use a full clone.
export const VENDORED_BP_STUDIO_ROOT = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "vendor",
  "box-pleating-studio",
);

export function resolveBpStudioRoot(explicit?: string): string {
  return explicit ?? process.env.BP_STUDIO_ROOT ?? VENDORED_BP_STUDIO_ROOT;
}
