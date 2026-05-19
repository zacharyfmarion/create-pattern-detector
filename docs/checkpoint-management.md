# Checkpoint Management

This repo does not commit model weights. Checkpoints are large, machine-local
artifacts under the ignored `checkpoints/` directory. The repo commits small
checkpoint manifests under `artifacts/checkpoints/` so future agents can tell
which local file is the current best checkpoint, how it was trained, and how to
verify it.

## What Gets Committed

Commit:

- `artifacts/checkpoints/*.json`: small manifests for blessed or important
  checkpoints.
- Docs that explain how a checkpoint should be used.
- Config examples and scripts needed to reproduce or evaluate a checkpoint.

Do not commit:

- `.pt`, `.pth`, `.onnx`, `.safetensors`, or other weight files.
- `checkpoints/`, prediction caches, TensorBoard logs, or large visualization
  outputs.
- RunPod API keys or machine-specific secrets.

The repo currently ignores `checkpoints/` and `visualizations/` for this reason.

## Directory Convention

Use stable, descriptive run directories:

```text
checkpoints/
  phase3_local_smoke/
  phase3_1024_hrnet_preflight/
  runpod_phase3_curriculum/
    stage-base/
    stage-balanced/
```

Each real training run should write:

- `latest.pt`: best or final loadable checkpoint for downstream use.
- `post_train.pt`: immediate post-training state before validation, when useful.
- `run_config.json`: exact training configuration.
- `summary.json`: validation summary and checkpoint path.
- `train_history.jsonl`: step-level training metrics.
- `train.log`: stdout/stderr log for RunPod runs.

Short smoke runs can omit logs, but should still write `run_config.json` and
`summary.json`.

## Blessed Checkpoints

A blessed checkpoint is the one downstream phases should load by default. It
must have a manifest in `artifacts/checkpoints/`.

Current blessed checkpoint:

```text
artifacts/checkpoints/phase3-v1-cpline.json
```

The manifest records:

- Local relative path under `checkpoints/`.
- Original RunPod path, if applicable.
- SHA-256 checksum and size.
- Model architecture and image size.
- Dataset, training profile, seed, and key hyperparameters.
- Required inference/eval settings, especially BatchNorm mode.
- Posthoc graph-eval metrics and known caveats.
- Links to related docs.

The manifest does not make the weight file portable by itself. It is a registry
entry for the local/shared artifact.

## Registering A New Checkpoint

Before replacing a blessed checkpoint:

1. Run the relevant validation/eval suite.
2. Confirm the new checkpoint improves the intended metric without regressing
   the supported input envelope.
3. Compute its checksum:

   ```bash
   shasum -a 256 checkpoints/path/to/latest.pt
   stat -f '%z bytes' checkpoints/path/to/latest.pt
   ```

4. Create or update a manifest under `artifacts/checkpoints/`.
5. Include enough eval numbers that another agent can understand why it is
   blessed.
6. Link the manifest from the relevant phase-status doc or roadmap section.

Prefer creating a new manifest id when a checkpoint changes the supported input
envelope or model architecture. Updating an existing manifest is okay for small
metadata corrections.

## Loading A Checkpoint

For Phase 3 V1 CPLineNet, load:

```text
checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
```

Use the config stored inside the checkpoint to construct the model:

- `CPLineNet`
- `backbone=hrnet_w18`
- `hidden_channels=128`
- `image_size=1024`

Use `batch-stats` BatchNorm behavior for validation/vectorization. Mixed
light/dark/photo-style training made plain eval-mode BatchNorm misleading for
batch-size-1 graph evaluation.

## If A Checkpoint Is Missing

Do not retrain by default. First:

1. Check `artifacts/checkpoints/*.json` for the intended path and checksum.
2. Check the shared workspace or previous RunPod copy-back directory.
3. Ask the user whether there is an external artifact store or backup.
4. Only retrain after confirming the artifact cannot be recovered.

If a future artifact store is added, record its URI in the manifest alongside
the local relative path and checksum.

## RunPod Notes

RunPod pods should copy checkpoints back before stopping or deleting the pod.
Network volumes or object storage are better long-term homes than stopped pod
volumes. The manifest should point to the canonical local path after copy-back,
and optionally to the original RunPod path for traceability.

After every RunPod training run, stop the pod once artifacts are copied back.
RunPod can continue billing while a pod is running even if training is done.
