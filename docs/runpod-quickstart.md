# RunPod Quickstart

This is the fast path for Phase 3 CPLine training on a fresh RunPod GPU pod.
It captures the setup details that are easy to miss when moving from the local
Mac worktree to Linux/CUDA.

## Local Setup

Use a restricted RunPod API key:

- `api.runpod.io/graphql`: `Read / Write`
- `api.runpod.ai`: `None`

Store it only in the ignored local env file:

```bash
cp configs/runpod.env.example configs/runpod.env
# Paste RUNPOD_API_KEY into configs/runpod.env.
set -a; source configs/runpod.env; set +a
runpodctl user
```

For agent runs, prefer this env-file path over global RunPod config:

```bash
set -a; source configs/runpod.env; set +a
runpodctl user
```

Do not assume `~/.runpod/config.toml` contains a usable key. During the
2026-06-23 vertex-refiner run, the global config existed but had an empty API
key; the working credentials were in the ignored `configs/runpod.env` file.
Never print the key itself in logs or handoff notes.

## Create A Pod

Prefer a 24GB+ GPU. A secure-cloud L4 works for the conservative first run; a
4090/A5000/5090 is faster when available.

```bash
STOP_AFTER="$(date -u -v+10H +"%Y-%m-%dT%H:%M:%SZ")"
set -a; source configs/runpod.env; set +a
runpodctl pod create \
  --template-id runpod-torch-v220 \
  --gpu-id "NVIDIA L4" \
  --name cpline-phase3-balanced \
  --container-disk-in-gb 40 \
  --volume-in-gb 50 \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --stop-after "$STOP_AFTER"
```

Set `RUNPOD_POD_ID` from the create output:

```bash
export RUNPOD_POD_ID="<pod-id>"
```

If that GPU is out of stock, retry with one of:

```text
NVIDIA GeForce RTX 4090
NVIDIA RTX A5000
NVIDIA GeForce RTX 5090
NVIDIA L40S
```

Get SSH info and keep the host/port handy for plain `ssh` and `rsync`:

```bash
runpodctl ssh info "$RUNPOD_POD_ID"
```

## Bootstrap The Repo

On the pod:

```bash
cd /workspace
git clone -b codex/phase3-real-folds https://github.com/zacharyfmarion/create-pattern-detector.git
cd create-pattern-detector
scripts/setup_python_env.sh
```

If you are uploading an uncommitted local worktree snapshot instead of cloning
from GitHub, make sure the archive includes the repo setup files as well as the
source tree. At minimum, include:

```text
.python-version
requirements.txt
pyproject.toml
src/
scripts/
configs/
docs/
implementation-plan/
tests/
```

Missing `.python-version` or `requirements.txt` will make
`scripts/setup_python_env.sh` fail on the pod. When patching a live pod with
`scp`, copy files to their exact repo paths:

```bash
scp -P "$POD_PORT" src/foo.py root@"$POD_HOST":/workspace/create-pattern-detector/src/foo.py
```

Do not copy multiple files to `/workspace/create-pattern-detector/` unless that
is intentionally the destination; `scp` will otherwise drop patched files at
the repo root instead of preserving subdirectories.

Install `tmux` if the template does not include it:

```bash
apt-get update && apt-get install -y tmux
```

The RunPod PyTorch 2.2 image may still get a newer `torch>=2` wheel from pip
during project setup. If CUDA is unavailable or Torch reports CUDA 13 on a CUDA
12 host, pin back to a compatible CUDA 12.1 stack:

```bash
.venv/bin/python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.2.2+cu121" \
  "torchvision==0.17.2+cu121"

.venv/bin/python -m pip install --force-reinstall \
  "numpy<2" \
  "opencv-python==4.10.0.84" \
  "opencv-python-headless==4.10.0.84"
```

Check CUDA:

```bash
.venv/bin/python - <<'PY'
import cv2, numpy as np, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("numpy", np.__version__, "cv2", cv2.__version__)
PY
```

Known-good package state from the 2026-06-23 RTX 4090 run:

```text
torch 2.2.2+cu121
torchvision 0.17.2+cu121
numpy 1.26.x
opencv-python 4.10.0.84
opencv-python-headless 4.10.0.84
torch.cuda.is_available() == True
```

If setup pulls a much newer Torch wheel and CUDA becomes unavailable, pin back
to the CUDA 12.1 stack above before running preflight or training.

## Upload Synthetic Data

The local `cp_training_mix_v1` root contains symlinks into the TreeMaker and
Rabbit Ear source dataset roots. Do not preserve those symlinks when copying to
RunPod; they point at local Mac paths and will break on the pod. The trainer
only needs `raw-manifest.jsonl` and `folds/`, so the quick copy skips the large
`metadata/` directory.

From the laptop, dereference symlinks with `tar -h`:

```bash
# Fill these from: runpodctl ssh info "$RUNPOD_POD_ID"
POD_HOST="<pod-ip>"
POD_PORT="<pod-port>"

ssh -i ~/.ssh/id_ed25519 -p "$POD_PORT" root@"$POD_HOST" \
  'rm -rf /workspace/datasets/create-pattern-detector/synthetic/cp_training_mix_v1 &&
   mkdir -p /workspace/datasets/create-pattern-detector/synthetic'

COPYFILE_DISABLE=1 tar -chzf - \
  --no-xattrs \
  --no-mac-metadata \
  --exclude='._*' \
  --exclude='.DS_Store' \
  -C /Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic \
  cp_training_mix_v1/raw-manifest.jsonl \
  cp_training_mix_v1/qa.json \
  cp_training_mix_v1/qa \
  cp_training_mix_v1/folds |
ssh -i ~/.ssh/id_ed25519 -p "$POD_PORT" root@"$POD_HOST" \
  'tar --warning=no-unknown-keyword --no-same-owner \
     -C /workspace/datasets/create-pattern-detector/synthetic -xzf -'
```

On the pod, link and verify:

```bash
cd /workspace/create-pattern-detector
export CP_SHARED_DATA_ROOT=/workspace/datasets/create-pattern-detector
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1

.venv/bin/python - <<'PY'
from pathlib import Path
root = Path("data/generated/synthetic/cp_training_mix_v1")
folds = list((root / "folds").glob("*.fold"))
print("manifest", (root / "raw-manifest.jsonl").exists())
print("fold_count", len(folds))
print("symlink_count", sum(1 for p in folds if p.is_symlink()))
print("broken_symlinks", sum(1 for p in folds if p.is_symlink() and not p.exists()))
PY
```

Expected: manifest exists, `fold_count` is `14000`, and broken symlinks is `0`.
If `symlink_count` is still non-zero after upload, remove the remote dataset and
repeat the `tar -h` transfer; preserving the local symlinks will leave the pod
with paths that point back to the Mac.

### Avoid Repeating Uploads

The fastest future path is to keep this dataset on persistent storage instead of
reuploading it for every pod:

- Use a RunPod network volume as the canonical GPU-side dataset location, then
  attach it to future pods at `/workspace/datasets`.
- Or build a single dereferenced artifact such as
  `cp_training_mix_v1_train.tar.zst`, upload it to object storage, and download
  it from the pod with `curl`, `rclone`, or `aws s3 cp`.
- Keep the packed artifact limited to `raw-manifest.jsonl`, `qa/`, `qa.json`,
  and `folds/` unless a training/eval script explicitly needs `metadata/`.

Do not copy the local mixed dataset with plain `scp -r` or `rsync -a` unless
`--copy-links` is set; otherwise the symlinked `.fold` files will be broken on
Linux.

## Smoke Check

Run a tiny CUDA batch before starting the paid curriculum:

```bash
TQDM_DISABLE=1 .venv/bin/python scripts/training/train_cpline_smoke.py \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --output-dir checkpoints/runpod_phase3_preflight \
  --device cuda \
  --backbone tiny \
  --hidden-channels 32 \
  --image-size 128 \
  --train-count 2 \
  --val-count 1 \
  --max-edges 100 \
  --max-steps 1 \
  --batch-size 1 \
  --num-workers 0 \
  --augment-profile stage-base \
  --skip-graph-eval \
  --log-every 1
```

## Launch Phase 3 Curriculum

Use `tmux` so training survives SSH disconnects:

```bash
tmux new -s cpline

OUTPUT_ROOT=checkpoints/runpod_phase3_curriculum \
MANIFEST=data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
TRAIN_COUNT=512 \
VAL_COUNT=64 \
TRAIN_FAMILY_SAMPLING=balanced \
BATCH_SIZE=1 \
NUM_WORKERS=4 \
IMAGE_SIZE=1024 \
BACKBONE=hrnet_w18 \
BATCHNORM_MODE=batch-stats \
RUN_TARGETED=0 \
LOG_EVERY=50 \
SKIP_GRAPH_EVAL=1 \
scripts/training/run_cpline_runpod_curriculum.sh
```

The script runs `stage-base` first, then initializes `stage-balanced` from
`stage-base/latest.pt`.

`TRAIN_FAMILY_SAMPLING=balanced` is the default for this RunPod curriculum. The
mixed manifest is intentionally not family-balanced on disk: TreeMaker is the
primary source and Rabbit Ear is a smaller supplemental source. Balanced train
sampling keeps Rabbit Ear geometry from being under-trained while clean
validation and posthoc graph eval still report natural and per-family metrics.

`BATCHNORM_MODE=batch-stats` is important for 1024px HRNet runs with batch size
1 and strong light/dark/photo style mixing. It keeps the model in eval mode for
validation/vectorization, but makes BatchNorm use per-image batch statistics
without updating running stats. Plain eval-mode BatchNorm can report bogus
validation failures after mixed-style training because running stats drift
between light and dark distributions.

## Monitor And Stop

From the pod:

```bash
tail -f checkpoints/runpod_phase3_curriculum/stage-base/train.log
tail -f checkpoints/runpod_phase3_curriculum/stage-base/train_history.jsonl
watch -n 5 nvidia-smi
```

From the laptop:

```bash
set -a; source configs/runpod.env; set +a
runpodctl pod get "$RUNPOD_POD_ID" --include-machine
```

When finished, copy checkpoints/results back, then stop the pod. The base
RunPod image may not have `rsync`, so this tar stream is the most portable
copy-back path:

```bash
ssh -i ~/.ssh/id_ed25519 -p "$POD_PORT" root@"$POD_HOST" \
  'cd /workspace/create-pattern-detector &&
   tar -czf - checkpoints/runpod_phase3_curriculum visualizations/runpod_phase3_graph_eval' |
tar -xzf -

runpodctl pod stop "$RUNPOD_POD_ID"
```

If CUDA disappears while the pod is still `RUNNING` and `nvidia-smi` reports an
NVML initialization error, restart the pod once. The `/workspace` volume should
survive the restart, but confirm CUDA and dataset links again before continuing.

RunPod bills while the pod is running, even if training has already finished.
Stopping the pod stops GPU compute billing. Delete the pod later if you do not
need its volume.

For short-lived agent-created pods, copy artifacts back, then stop and delete
the pod immediately:

```bash
set -a; source configs/runpod.env; set +a
runpodctl pod stop "$RUNPOD_POD_ID"
runpodctl pod delete "$RUNPOD_POD_ID"
runpodctl pod list
runpodctl network-volume list
runpodctl user
```

Expected cleanup state is an empty pod list, an empty network-volume list, and
`currentSpendPerHr=0`. The 2026-06-23 vertex-refiner run did not require a
network volume; do not create or keep one unless the current training code
explicitly uses it.
