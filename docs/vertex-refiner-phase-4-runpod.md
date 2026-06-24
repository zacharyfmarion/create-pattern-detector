# Vertex Refiner Phase 4 RunPod Handoff

This is the source-only GPU probe path for frame-aware `VertexRefinerV2`.

## Goal

Measure whether a high-resolution crop model can recover vertex locations and
local topology directly from crease-pattern source pixels, without waiting for
or depending on the full-image CPLineNet/HRNet dense pass.

The Phase 4 probe must keep this runtime shape:

```text
source image -> source-image proposals -> VertexRefiner crops -> graph
```

Do not train the first real V2 checkpoint with CPLineNet dense junction
channels. That would force a product waterfall:

```text
source image -> CPLineNet full-image pass -> VertexRefiner crops -> graph
```

## Current Result

The latest source-only V2 run is the cached 1024px/base-48 RTX 4090 run copied
back under:

```text
checkpoints/runpod_vertex_refiner_v2_cached_20260624_4090/
checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/
```

The current best checkpoint is:

```text
checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/full/latest.pt
```

Key findings:

- Cached crop refs fixed the paid CPU-side setup bottleneck; the 4090 run wrote
  `run_config.json` promptly and reached the first logged loss.
- The first full run completed `2000` steps at `LR=0.0003`, then a warm
  continuation completed another `2000` steps at `LR=0.0001`.
- The final sampled validation point was still the best point: precision
  `0.9683`, recall `0.9854`, F1 `0.9768`, boundary-contact F1 `0.9565`, and
  close-pair F1 `0.9323`.
- Local MPS eval from the copied checkpoint over the full cached validation crop
  set (`7,096` crops) measured precision `0.9692`, recall `0.9651`, F1
  `0.9671`, boundary-contact F1 `0.9495`, close-pair F1 `0.9029`, and corner
  F1 `0.9498`.
- These are crop-level V2 validation results, not product promotion. The next
  gate is corrected native/full-pattern CPOOGLE eval and failure sheets.

Earlier 2026-06-23 warm 512px runs remain useful historical context but are no
longer the current checkpoint candidate.

## Required Invariants

Every Phase 4 source-only run must record these values in `run_config.json`:

```text
auxiliary_mode=zero
model_version=v2
include_gt_training_anchors=true
include_val_gt_anchors=false
```

`auxiliary_mode=zero` is still correct for V2. It means the legacy CPLineNet
auxiliary channels are not used. The source/frame channels come from
`model_version=v2` in the trainer and `input_version=v2` in the crop-ref cache.

`include_gt_training_anchors=true` is allowed only for the training split, where
it ensures positive crop coverage. Validation and standalone eval must leave GT
anchors off so metrics reflect source-image proposal quality.

Rendered junction labels are test-only. Real CPLineNet dense caches are reserved
for a later ablation after source-only quality is known.

## Budget

Keep the first probe under `$10` of RunPod compute.

Preferred GPUs:

```text
NVIDIA RTX A5000
NVIDIA GeForce RTX 4090
NVIDIA RTX 3090
NVIDIA L4
NVIDIA A40
```

Avoid H100/H200/B200/A100-class GPUs for this probe. The launcher refuses those
GPU names unless `ALLOW_EXPENSIVE_VERTEX_REFINER_GPU=1` is set.

Use `--stop-after` when creating the pod. Eight hours is a conservative cap for
an RTX 4090-class probe, and much more than the initial probe should need.

## Pod Setup

Follow `docs/runpod-quickstart.md` for API key setup, SSH, repo bootstrap, Torch
CUDA verification, and dereferenced `cp_training_mix_v1` upload. For this probe,
create an inexpensive 24GB+ pod, for example:

```bash
STOP_AFTER="$(date -u -v+8H +"%Y-%m-%dT%H:%M:%SZ")"
set -a; source configs/runpod.env; set +a
runpodctl pod create \
  --template-id runpod-torch-v220 \
  --gpu-id "NVIDIA RTX A5000" \
  --name vertex-refiner-source-only-probe \
  --container-disk-in-gb 40 \
  --volume-in-gb 80 \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --stop-after "$STOP_AFTER"
```

If A5000 capacity is unavailable, retry with `NVIDIA GeForce RTX 4090`.

## Preflight

On the pod:

```bash
cd /workspace/create-pattern-detector
scripts/setup_python_env.sh
export CP_SHARED_DATA_ROOT=/workspace/datasets/create-pattern-detector
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1

PREFLIGHT_ONLY=1 \
scripts/training/run_vertex_refiner_runpod_source_only_probe.sh
```

The preflight runs CUDA for two steps and verifies:

- `device=cuda`
- `auxiliary_mode=zero`
- `model_version=v2`
- `include_gt_training_anchors=true`
- `include_val_gt_anchors=false`
- configured `image_size`, `max_edges`, and `base_channels`

Do not continue if this verification fails.

## Known-Good 2026-06-23 Recipe

The successful warm continuation used:

```text
template: runpod-torch-v220
gpu: NVIDIA GeForce RTX 4090
container disk: 50GB
workspace volume: 40GB local pod volume
network volume: none
dataset: dereferenced cp_training_mix_v1 linked at data/generated/synthetic/cp_training_mix_v1
```

`scripts/setup_python_env.sh` initially installed a too-new Torch wheel for the
pod driver. Pinning back to the CUDA 12.1 stack in `docs/runpod-quickstart.md`
restored CUDA. Verify `torch.cuda.is_available() == True` before running the
refiner launcher.

Blackwell GPUs are different. On the 2026-06-24 RTX PRO 4000 Blackwell attempt,
`torch==2.2.2+cu121` imported and reported CUDA availability, but a real CUDA
matrix multiply failed with `no kernel image is available for execution on the
device` because the wheel did not support `sm_120`. For RTX PRO 4000/4500/5000
Blackwell-class pods, use a CUDA 12.8+ PyTorch wheel and verify an actual CUDA
operation before training, for example:

```bash
.venv/bin/python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" \
  "torchvision==0.23.0+cu128"

.venv/bin/python -m pip install --force-reinstall \
  "numpy<2" \
  "opencv-python==4.10.0.84" \
  "opencv-python-headless==4.10.0.84"

.venv/bin/python -c \
  'import torch; x=torch.randn(16,16,device="cuda"); y=(x@x).sum(); torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), float(y.cpu()))'
```

Use this shape for a cheap warm-start continuation:

```bash
tmux new -s vertex-refiner-warm

OUTPUT_ROOT=checkpoints/runpod_vertex_refiner_warm_continue_$(date +%Y%m%d) \
INIT_CHECKPOINT=checkpoints/runpod_vertex_refiner_source_only_probe_20260622_fixed_selector/full/latest.pt \
IMAGE_SIZE=512 \
MAX_EDGES=1200 \
TRAIN_COUNT=64 \
VAL_COUNT=8 \
PROPOSALS_PER_SAMPLE=128 \
BATCH_SIZE=64 \
NUM_WORKERS=0 \
MAX_STEPS=2000 \
LR=0.00005 \
BASE_CHANNELS=24 \
CHECKPOINT_EVERY=250 \
LOG_EVERY=25 \
HEATMAP_THRESHOLD=0.15 \
EVAL_MAX_BATCHES=16 \
SEED=61 \
scripts/training/run_vertex_refiner_runpod_source_only_probe.sh
```

Why these values:

- `INIT_CHECKPOINT` warm-starts from the corrected fixed-selector source-only
  checkpoint.
- `NUM_WORKERS=0` avoids the Linux DataLoader failure
  `RuntimeError: received 0 items of ancdata`, seen with worker
  multiprocessing on this pod.
- `TRAIN_COUNT=64` kept the run bounded. A `TRAIN_COUNT=256` attempt spent
  several minutes in CPU-side rendering/proposal generation before the first
  step, so larger runs need persistent proposal/crop caches or precomputed crop
  refs before spending more GPU time.
- `BASE_CHANNELS=24` and `IMAGE_SIZE=512` are still budget-probe settings, not
  the final 1024px/base-48 target.

## 2026-06-24 Cached 4090 Run

The cached retry ran on pod `2mfc9y7nxeoca1`:

```text
gpu: NVIDIA GeForce RTX 4090
cost: about $0.69/hr
network volume: none
volumeInGb: 0
container disk: 80GB
```

Run setup:

```text
MODEL_VERSION=v2
AUXILIARY_MODE=zero
IMAGE_SIZE=1024
MAX_EDGES=1200
TRAIN_COUNT=512
VAL_COUNT=64
PROPOSALS_PER_SAMPLE=128
BATCH_SIZE=32
NUM_WORKERS=0
BASE_CHANNELS=48
TRAIN_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json
VAL_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json
SHUFFLE_TRAIN_CROPS=0
RENDERED_SAMPLE_CACHE_SIZE=4
```

The first full run used `LR=0.0003` for `2000` steps and reached validation F1
`0.9297`. The warm continuation used
`INIT_CHECKPOINT=checkpoints/runpod_vertex_refiner_v2_cached_20260624_4090/full/latest.pt`
with `LR=0.0001` for another `2000` steps.

Warm-continuation sampled validation curve:

| Step | Precision | Recall | F1 | Boundary F1 | Close-Pair F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | `0.9255` | `0.9554` | `0.9402` | `0.8782` | `0.7764` |
| 500 | `0.9427` | `0.9616` | `0.9521` | `0.9079` | `0.8285` |
| 750 | `0.9442` | `0.9762` | `0.9599` | `0.9201` | `0.8594` |
| 1000 | `0.9245` | `0.9792` | `0.9511` | `0.8954` | `0.8583` |
| 1250 | `0.9667` | `0.9808` | `0.9737` | `0.9511` | `0.9069` |
| 1500 | `0.9469` | `0.9869` | `0.9665` | `0.9354` | `0.9141` |
| 1750 | `0.9551` | `0.9808` | `0.9678` | `0.9253` | `0.8696` |
| 2000 | `0.9683` | `0.9854` | `0.9768` | `0.9565` | `0.9323` |

Because the final point was the best point, a lower-LR continuation
(`LR=0.00003`) was briefly started. It was stopped immediately when the user
asked to clean up the pod near the rate limit; it only wrote `run_config.json`
and an empty `train.log`.

The standalone eval used the corrected `VAL_SEED=1041`, but
`--full-pattern-diagnostics` stayed CPU-bound for about 12 minutes after the
training summary had already been written and copied back. It wrote no
`eval.json` before it was stopped. For continuation runs, use
`RUN_STANDALONE_EVAL=0` or `EVAL_FULL_PATTERN_DIAGNOSTICS=0` when the training
summary validation curve is sufficient.

Copied local artifacts:

```text
checkpoints/runpod_vertex_refiner_v2_cached_20260624_4090/
checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/
```

The current best checkpoint and named safety copy are:

```text
checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/full/latest.pt
checkpoints/runpod_vertex_refiner_v2_cached_continue_lr1e4_20260624_4090/full/checkpoint_step2000.pt
```

Local MPS eval from the copied best checkpoint over all cached validation crops
(`7,096` crop samples), without full-pattern diagnostics:

```text
precision: 0.9692
recall:    0.9651
F1:        0.9671
boundary-contact F1: 0.9495
close-pair F1:       0.9029
corner F1:           0.9498
```

Cleanup state after copy-back:

```text
runpodctl pod list            -> []
runpodctl network-volume list -> []
runpodctl user currentSpendPerHr -> 0
```

## 2026-06-24 V2 1024px Attempt

Do not retry the full 1024px/base-48 V2 run unchanged. On pod
`v18e53zyit0hrv` (RTX PRO 4000 Blackwell, no network volume, about
`$0.584/hr`), the trainer stayed CPU-only before writing `run_config.json`:

```text
TRAIN_COUNT=512 VAL_COUNT=64: stopped after >5 minutes pre-config
TRAIN_COUNT=128 VAL_COUNT=16: stopped after >5 minutes pre-config
TRAIN_COUNT=32  VAL_COUNT=8 : stopped after ~4 minutes pre-config
```

The process was active, not crashed, but it never reached the first GPU step in
the budget window. The likely bottleneck is eager 1024px render/proposal/crop
construction in `VertexRefinerCropDataset._build_crop_refs` plus the initial
validation preview batch, both of which happen before `run_config.json` is
written. Before another paid full run, add progress logging and precompute or
cache proposal/crop refs so the GPU job reaches the training loop predictably.

Abort notes from that attempt were copied back locally under:

```text
checkpoints/runpod_vertex_refiner_v2_source_only_full_20260624_blackwell/
checkpoints/runpod_vertex_refiner_v2_source_only_bounded_20260624_blackwell/
checkpoints/runpod_vertex_refiner_v2_source_only_signal_20260624_blackwell/
```

The pod was stopped and deleted after artifact copy-back. `runpodctl pod list`
and `runpodctl network-volume list` returned empty, and `currentSpendPerHr=0`.

## Local Crop-Ref Cache For Retry

The next 1024px V2 run should use locally precomputed crop refs. This moves the
expensive render/proposal/crop-ref selection phase off the paid pod and makes a
cache mismatch fail before training.

Current local cache artifacts:

```text
checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json
checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json
```

The train cache was built from 512 selected records and contains 95,728 crop
refs. The val cache was built from 64 selected records and contains 7,096 crop
refs. Together they are about 36 MB, so they should be uploaded with the repo
snapshot before the next RunPod launch.

To regenerate the cache locally:

```bash
.venv/bin/python scripts/data/precompute_vertex_refiner_crop_refs.py \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --split train \
  --limit 512 \
  --max-edges 1200 \
  --image-size 1024 \
  --padding 32 \
  --line-width 2 \
  --augment-profile clean \
  --seed 41 \
  --proposals-per-sample 128 \
  --include-gt-training-anchors \
  --boundary-gt-anchor-repeats 3 \
  --boundary-gt-anchor-jitter-px 6.0 \
  --auxiliary-mode zero \
  --input-version v2 \
  --crop-ref-progress-every 16 \
  --output checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json

.venv/bin/python scripts/data/precompute_vertex_refiner_crop_refs.py \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --split val \
  --limit 64 \
  --max-edges 1200 \
  --image-size 1024 \
  --padding 32 \
  --line-width 2 \
  --augment-profile clean \
  --seed 1041 \
  --proposals-per-sample 128 \
  --no-include-gt-training-anchors \
  --auxiliary-mode zero \
  --input-version v2 \
  --crop-ref-progress-every 16 \
  --output checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json
```

Use record-ordered training and a small rendered-sample cache with these refs:

```text
TRAIN_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json
VAL_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json
SHUFFLE_TRAIN_CROPS=0
RENDERED_SAMPLE_CACHE_SIZE=4
ABORT_LOSS_THRESHOLD=100000
```

The trainer builds the validation split with `SEED+1000`. For the current
`SEED=41` cached run, the validation cache was built with seed `1041`, so the
launcher sets `VAL_SEED=1041` by default via `VAL_SEED=SEED+1000` for
standalone eval. If standalone eval uses `--seed 41` with `val-refs.json`, cache
verification should fail; that is a harness configuration bug, not a model
failure.

Do not use shuffled full-crop training with an unbounded rendered-sample cache on
the 1024px run. It can grow toward all 512 rendered samples in memory and waste
the budget before the model work starts.

## Probe Launch

Start in `tmux`:

```bash
tmux new -s vertex-refiner
```

Then launch the cached source-only V2 run:

```bash
OUTPUT_ROOT=checkpoints/runpod_vertex_refiner_source_only_probe_$(date +%Y%m%d) \
MODEL_VERSION=v2 \
RUN_PREFLIGHT=0 \
IMAGE_SIZE=1024 \
MAX_EDGES=1200 \
TRAIN_COUNT=512 \
VAL_COUNT=64 \
PROPOSALS_PER_SAMPLE=128 \
BATCH_SIZE=32 \
NUM_WORKERS=0 \
MAX_STEPS=2000 \
BASE_CHANNELS=48 \
CHECKPOINT_EVERY=250 \
LOG_EVERY=25 \
EVAL_MAX_BATCHES=16 \
EARLY_EVAL_EVERY=250 \
EARLY_STOP_AFTER_STEP=500 \
BOUNDARY_GT_ANCHOR_REPEATS=3 \
BOUNDARY_GT_ANCHOR_JITTER_PX=6.0 \
TRAIN_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/train-refs.json \
VAL_CROP_REFS=checkpoints/vertex_refiner_crop_refs_v2_source_only_20260624/val-refs.json \
SHUFFLE_TRAIN_CROPS=0 \
RENDERED_SAMPLE_CACHE_SIZE=4 \
ABORT_LOSS_THRESHOLD=100000 \
VAL_SEED=1041 \
RUN_STANDALONE_EVAL=0 \
scripts/training/run_vertex_refiner_runpod_source_only_probe.sh
```

Use `RUN_STANDALONE_EVAL=0` for continuation runs when the training summary's
sampled validation curve is enough. If standalone eval is needed but
full-pattern diagnostics are not, use `EVAL_FULL_PATTERN_DIAGNOSTICS=0` so the
run still writes `eval.json` without the slow CPU-heavy diagnostics tail.

Use `PROPOSALS_PER_SAMPLE=128` or higher for the corrected Phase 4 run unless
you are intentionally running a tiny smoke. Earlier probes with very low caps
mixed up two questions: whether the proposal generator found the right areas
and whether the refiner fired inside those areas. The launcher default is now
`128`, and standalone eval writes `proposal_coverage` plus
`full_pattern_metrics` so those failures stay separated.

The launcher writes:

```text
checkpoints/runpod_vertex_refiner_source_only_probe_<date>/
  full/
    latest.pt
    latest_train.pt
    run_config.json
    summary.json
    train_history.jsonl
    train.log
    eval.json
    eval.log
    qualitative_before_after.png
```

## Monitor

From the pod:

```bash
tail -f checkpoints/runpod_vertex_refiner_source_only_probe_*/full/train.log
tail -f checkpoints/runpod_vertex_refiner_source_only_probe_*/full/train_history.jsonl
watch -n 5 nvidia-smi
```

The first decision point is not final promotion. It is whether source-only
training is learning the hard slices:

- validation loss decreases
- close-pair slice recall improves
- boundary-contact slice recall improves
- false positives per crop decrease
- `proposal_coverage.coverage` is high enough that recall failures are mostly
  covered-but-missed rather than never-cropped vertices
- qualitative overlay shows peaks tightening around source vertices

## Post-Run Eval

For the warm-continuation checkpoint, the current seed-17 product-style eval was
generated with:

```bash
.venv/bin/python scripts/evals/eval_vertex_refiner.py \
  --checkpoint checkpoints/runpod_vertex_refiner_warm_continue_20260623/full/latest.pt \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --split val \
  --limit 8 \
  --max-edges 1200 \
  --image-size 512 \
  --proposals-per-sample 128 \
  --batch-size 64 \
  --base-channels 24 \
  --device cuda \
  --auxiliary-mode zero \
  --no-include-gt-training-anchors \
  --heatmap-threshold 0.075 \
  --match-tolerance-px 2.0 \
  --full-pattern-diagnostics \
  --global-merge \
  --merge-radius-px 2.0 \
  --merge-min-score 0.25 \
  --merge-min-support 1 \
  --merge-min-support-fraction 0.45 \
  --out checkpoints/runpod_vertex_refiner_warm_continue_20260623/full/eval-global-merge-corners-best-seed17.json
```

Run the same command with `--seed 61` and a separate `--out` path for the
second sanity slice. The copied artifacts also include focused sweeps:

```text
checkpoints/runpod_vertex_refiner_warm_continue_20260623/full/eval-global-merge-corners-focused-sweep-seed17.json
checkpoints/runpod_vertex_refiner_warm_continue_20260623/full/eval-global-merge-corners-focused-sweep-seed61.json
```

Do not rerun a broad threshold sweep with the old naive global merge. It is
slow at low heatmap thresholds because decoded candidates explode. Use the
spatial-grid merge implementation in `src/evaluation/vertex_refiner_global_merge.py`
or cache decoded outputs before sweeping merge parameters.

## Copy Back And Stop

From the laptop, after the run finishes:

```bash
ssh -i ~/.ssh/id_ed25519 -p "$POD_PORT" root@"$POD_HOST" \
  'cd /workspace/create-pattern-detector &&
   tar -czf - checkpoints/runpod_vertex_refiner_source_only_probe_*' |
tar -xzf -

set -a; source configs/runpod.env; set +a
runpodctl pod stop "$RUNPOD_POD_ID"
runpodctl pod delete "$RUNPOD_POD_ID"
runpodctl pod list
runpodctl network-volume list
runpodctl user
```

RunPod bills while the pod is running. Stop the pod immediately after artifacts
are copied back. For agent-created experiment pods, delete the pod too unless
there is a specific reason to preserve its local volume. The expected final
state is empty pod list, empty network-volume list, and `currentSpendPerHr=0`.

## After The Probe

If the source-only probe is promising, keep the next iteration source-only and
tune proposal recall, hard-slice sampling, loss weights, and decode thresholds.

Only run source-plus-dense ablations after source-only has a clear baseline.
Those ablations should be named and registered separately so they cannot be
mistaken for the clean V1 architecture.

For the next run, do not spend more GPU time on broad proposal overgeneration
alone. The copied warm-continuation evals show that the measured false
negatives are already inside selected crops; improve the refiner and decoder
for those covered misses first.

For V2 runs, keep `auxiliary_mode=zero` unless intentionally running a dense
ablation. Do not write `source_frame` as an auxiliary mode; the source/frame
channels are selected by `MODEL_VERSION=v2`.
