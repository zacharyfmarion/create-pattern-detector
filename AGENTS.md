# AGENTS.md

Guidance for coding agents working in this repository.

## Project Goal

This project is an unfinished crease-pattern detector for origami. The intended pipeline is:

1. Read or generate `.fold` crease-pattern graphs.
2. Render or load crease-pattern images.
3. Train a pixel model to predict crease segmentation, line orientation, and junction heatmaps.
4. Convert pixel predictions into an over-complete candidate graph.
5. Train a graph head to keep/drop candidate edges, assign M/V/B/U labels, refine vertices, and eventually export a valid `.fold` graph.

The codebase is partly aspirational. Treat docs and comments as useful design notes, but verify behavior in code before relying on them.

## Current Architecture

- `src/models/cp_detector.py`: main pixel detector wrapper. It combines an HRNet backbone with `PixelHead`.
- `src/models/backbone/hrnet.py`: timm HRNet feature extractor, concatenating multiscale features at stride 4.
- `src/models/heads/pixel_head.py`: segmentation, orientation, junction heatmap, and junction offset branches.
- `src/data/`: FOLD parser, ground-truth map generation, datasets, transforms, and manifest-based synthetic data tooling.
- `tools/synthetic-generator/`: tracked Bun/Rabbit Ear validation plus BP Studio-backed and TreeMaker-backed generation package. It writes canonical FOLD files plus raw manifests.
- `recipes/synthetic/`: synthetic generation recipes. `bp_studio_realistic_v1.yaml` is BP Studio diagnostic/calibration work; `treemaker_tree_v1.yaml` is the current real external non-BP TreeMaker family. Older hand-written/fake generator recipes have been removed.
- `src/postprocessing/`: pixel-to-graph extraction. `GraphExtractor` skeletonizes segmentation, finds vertices from heatmaps/skeleton/boundaries, traces candidate edges, assigns labels, and can export FOLD-like dictionaries.
- `src/models/graph/`: graph head feature extraction, message passing layers, and edge/vertex prediction heads.
- `scripts/training/train_pixel_head.py`: primary pixel-head training entrypoint.
- `scripts/training/train_graph_head.py`: most complete graph-head training script.
- `src/training/graph_trainer.py`: stale duplicate graph trainer. It expects dataset keys and model outputs that are not provided as written.
- `data/ts-generation/`: older ignored/partial TypeScript generator work. Prefer the tracked package under `tools/synthetic-generator/`.

## Environment

Use the shared setup script instead of building a fresh dependency environment in each worktree:

```bash
scripts/setup_python_env.sh
```

The script creates or reuses a shared virtualenv under:

```text
~/.cache/create-pattern-detector/venvs
```

Then it links the current worktree to that environment:

```text
.venv -> ~/.cache/create-pattern-detector/venvs/py<version>-dev
```

If a worktree already has a local `.venv` directory from manual setup, run:

```bash
scripts/setup_python_env.sh --adopt-local
```

or, to discard that local environment and use the shared one:

```bash
scripts/setup_python_env.sh --replace-local
```

After setup, use `.venv/bin/python` and `.venv/bin/pytest` for local commands. Set `CP_PYTHON`, `CP_PYTHON_ENV_ROOT`, or `CP_PYTHON_VENV` only when a machine needs a custom interpreter or environment location.

The repo currently has `.python-version` set to `3.12.6`, but `pyproject.toml` only requires `>=3.10`. On a machine without that pyenv version, bare `python` will fail. Either install Python 3.12.6, update the local pyenv version intentionally, or run explicit commands with an available Python such as `python3.10`.

Python dependencies are not vendored. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,graph]"
```

The maintained TypeScript generator is a Bun package under `tools/synthetic-generator/`. It targets `rabbit-ear@0.9.32`; do not port new work back into `data/ts-generation` unless explicitly asked.

## Shared Scraped Dataset

Do not copy the scraped real-world crease pattern dataset into each git worktree.

The shared dataset lives outside the repo at:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/scraped
```

Each worktree should access it through this ignored symlink:

```text
data/output/scraped -> /Users/zacharymarion/Documents/datasets/create-pattern-detector/scraped
```

For a new worktree, run:

```bash
scripts/data/link_shared_scraped_data.sh
```

If the dataset lives somewhere else on a machine, set one of:

```bash
export CP_SHARED_DATA_ROOT=/path/to/create-pattern-detector-datasets
export CP_SCRAPED_DATASET=/path/to/create-pattern-detector-datasets/scraped
```

Then rerun:

```bash
scripts/data/link_shared_scraped_data.sh
```

The script refuses to replace a non-empty `data/output/scraped` directory unless it only contains metadata such as `.DS_Store`.

Keep raw dataset files, crops, manifests, and generated reports out of git. Commit small code, docs, config examples, tests, and deterministic fixture manifests instead.

BP Studio is pinned as a git submodule under `third_party/bp-studio`, with a headless adapter package in `tools/bp-studio-adapter/`. After cloning this worktree, initialize the submodule before running BP Studio-backed generation:

```bash
git submodule update --init --recursive
bun install --cwd tools/bp-studio-adapter
```

## BP Compiler Context Reset

Before touching BP Studio-backed generation or the box-pleat completion compiler, read these files in order, especially after context compaction:

1. `implementations-plans/realistic-box-pleat-production-data.md`
2. `implementations-plans/bp-studio-raw-export-rca.md`
3. `implementations-plans/bp-crease-assignment-solver.md`

The crease-assignment solver doc is the source of truth for the current M/V assignment algorithm. If the algorithm, solver contracts, or rejection policy changes, update that doc in the same change.

## Common Commands

Light syntax check:

```bash
python3.10 -m compileall -q src scripts
```

Pixel training:

```bash
python scripts/training/train_pixel_head.py \
  --fold-dir data/output/synthetic/raw/tier-a \
  --epochs 50 \
  --batch-size 4 \
  --image-size 1024
```

Graph-head training, after a pixel checkpoint exists:

```bash
python scripts/training/train_graph_head.py \
  --pixel-checkpoint checkpoints/checkpoint_epoch_8.pt \
  --fold-dir data/output/synthetic/raw/tier-a \
  --epochs 30 \
  --batch-size 4 \
  --image-size 512
```

Pipeline validation:

```bash
python scripts/validation/validate_pipeline.py --fold-dir data/output/synthetic/raw/tier-a
python scripts/validation/validate_pipeline_with_gt.py --fold-dir data/output/synthetic/raw/tier-a --image-size 512
python scripts/validation/validate_graph_head.py
```

BP Studio-backed realistic smoke:

```bash
bun install --cwd tools/synthetic-generator
bun run generate -- --recipe recipes/synthetic/bp_studio_realistic_v1.yaml --count 4 --out /tmp/bp_studio_realistic_v1 --max-attempts 120
python scripts/data/render_synthetic_dataset.py --root /tmp/bp_studio_realistic_v1
bun run folded-preview -- --root /tmp/bp_studio_realistic_v1 --limit 4
python scripts/data/bp_realism_report.py --root /tmp/bp_studio_realistic_v1
```

Generator checks:

```bash
cd tools/synthetic-generator
bun run typecheck
bun test
```

BP Studio adapter smoke:

```bash
bun --cwd tools/bp-studio-adapter test
bun run bpstudio-adapter -- --spec tools/bp-studio-adapter/fixtures/two-flap.json --out /tmp/bps.fold --metadata /tmp/bps.meta.json
```

TreeMaker external setup and smoke:

```bash
python3.10 tools/treemaker-adapter/scripts/setup_external_treemaker_cli.py
export TREEMAKER_CLI=~/.cache/cp-detector/treemaker-legacy/build/treemaker-json-cli
export TREEMAKER_CLI_ARGS=--triangulate
export TREEMAKER_TIMEOUT_MS=5000
bun run generate -- --recipe recipes/synthetic/treemaker_tree_v1.yaml --count 16 --out /tmp/treemaker_tree_v1
python3.10 scripts/data/synthetic_fold_report.py --root /tmp/treemaker_tree_v1
python3.10 scripts/data/merge_synthetic_fold_shards.py --out /tmp/treemaker_tree_v1_merged /tmp/treemaker_tree_v1
```

Code quality, once dependencies are installed:

```bash
black src/ scripts/
ruff check src/ scripts/
mypy src/
```

Do not rely on the console scripts in `pyproject.toml` (`cp-train`, `cp-evaluate`, `cp-predict`) until their target modules exist.

## Important Gotchas

- Assignment ids differ by layer:
  - FOLD/parser assignments: `M=0, V=1, B=2, U=3`.
  - Pixel segmentation classes: `BG=0, M=1, V=2, B=3, U=4`.
  - Graph head code sometimes says 4 classes and sometimes passes 5 pixel classes. Check the expected label space before changing losses or feature extractors.
- `CreasePatternDetector.forward(..., return_features=True)` is required for graph-head training. Without it, outputs do not include `features`.
- `src/training/graph_trainer.py` currently calls the pixel model without `return_features=True` and expects a `crease_pattern` batch key. Prefer `scripts/training/train_graph_head.py` or fix the stale trainer first.
- The postprocessing extractor is intentionally over-complete. Low precision is expected before the graph head; missing true edges are much more damaging than extra candidate edges.
- There is no finished inference CLI that takes an arbitrary image and writes a cleaned `.fold` file. `ExtractedGraph.to_fold_format()` exists, but full image rectification, graph-head filtering, coordinate denormalization, and validation are not wired into a productized path.
- Synthetic generation must not silently fall back to fake data. Do not reintroduce Rabbit Ear axiom/classic/single-vertex, hand-written box pleat, dense lattice, non-BP dense, or strict-completion fallback outputs as production data.
- `bp-studio-realistic` samples BP Studio-style trees/layouts, runs the pinned BP Studio adapter, normalizes raw exports, and then strict validation decides acceptance. If raw exports fail local/global constraints, fix the BP Studio adapter/sampler/normalizer path or stop with an RCA.
- `treemaker-tree` requires a real external `TREEMAKER_CLI`. The repo provides only a thin wrapper/build script; GPL TreeMaker source is cloned and built outside the repo, similar to external scraped data.
- TreeMaker graph generation comes before Phase 3 image augmentation. Keep scale checks focused on accepted `.fold` graphs, shard merges, topology/symmetry/archetype diversity, and fold distribution reports.
- `strictGlobal: true` with `globalBackend: rabbit-ear-solver` means the generator checks local Kawasaki/Maekawa, asks Rabbit Ear for globally consistent layer ordering, and computes finite flat-folded vertex coordinates. Use `folded-preview` when you need visual QA of the folded state, not only the CP drawing.
- In lightweight Python environments without OpenCV/SciPy, `src/data/annotations.py` uses optimized Pillow/NumPy fallbacks. Keep dense manifest loading smoke-tested; a naive per-edge supersampled fallback is too slow for 1,000+ edge samples.
- Several older docs mention `scripts/render_dataset.py`; the current synthetic renderer is `scripts/data/render_synthetic_dataset.py`.
- The repository currently contains tracked `.DS_Store` and some tracked `.pyc` files. Do not churn them unless the user asks for repository cleanup.

## Engineering Direction

For meaningful progress, prioritize in this order:

1. Stabilize data generation and validation, including reproducible synthetic `.fold` output.
2. Build a small, checked-in or locally generated smoke dataset for fast tests.
3. Make pixel-head training and validation reproducible on that dataset.
4. Validate graph extraction on ground-truth annotations before training the graph head.
5. Unify graph-label class conventions and remove/repair stale duplicate trainers.
6. Add a real inference path: image -> pixel outputs -> candidate graph -> graph head -> cleaned `.fold`.

When making changes, keep them narrow and verify with the smallest relevant script first. This repo has enough half-finished paths that broad refactors can easily make the working subset harder to identify.
