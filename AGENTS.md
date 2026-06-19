# Agent Notes

## Python Environment

Use the shared setup script instead of building a fresh dependency environment in
each worktree:

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

After setup, use `.venv/bin/python` and `.venv/bin/pytest` for local commands.
Set `CP_PYTHON`, `CP_PYTHON_ENV_ROOT`, or `CP_PYTHON_VENV` only when a machine
needs a custom interpreter or environment location.

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

## Box-Pleat Native Eval

For the box-pleat/grid-line-suppression diagnostic, use
`docs/evals/box-pleat-native-v1.md` and
`eval_specs/box_pleat_native_v1.json`. The eval set is intentionally
re-derived from the native converted-FOLD corpus and verified with
path-independent canonical FOLD fingerprints; do not commit local path lists or
generated contact sheets as the source of truth.

## Shared Synthetic Datasets

The maintained synthetic generator lives under `tools/synthetic-generator/`.
This PR keeps it focused on fold-only families:

- `treemaker-tree`: primary external TreeMaker-derived CP generation.
- `rabbit-ear-fold-program`: strict supplemental Rabbit Ear fold-operation generation.
- `tessellation-fold-program`: deterministic BP/tessellation-style crease evidence, currently focused on dense Rabbit Ear-validated orthogonal grids with vertical-heavy coverage.

Large generated releases live outside the repo:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/tessellation_orthogonal_bp_grid_v2_15pct
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/tessellation_miura_ori_v2_15pct
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v3_tessellation_15pct
```

Future worktrees should link the current mixed root when training:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py --root data/generated/synthetic/cp_training_mix_v1
```

Use `recipes/synthetic/tessellation_fold_program_v1.yaml` and
`scripts/data/visualize_synthetic_folds.py` for the tessellation release. Do not
mutate `cp_training_mix_v1`; build a new mixed release when tessellation samples
are added to training. The current 15% tessellation experiment mix is
`cp_training_mix_v3_tessellation_15pct`.

When training on `cp_training_mix_v3_tessellation_15pct`, do not use natural or
plain balanced family sampling. Use `TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct`
or the dedicated tessellation launcher. The preset preserves the old dense-edge
base exposure while adding tessellations:

- `42.5%` `treemaker-tree`
- `42.5%` `rabbit-ear-fold-program`
- `12%` `tessellation_orthogonal_bp_grid_v2_15pct`
- `3%` `tessellation_miura_ori_v2_15pct`

See `docs/synthetic-fold-datasets.md` for generation, shard merge, folded-preview,
and mix-building commands.

## Checkpoint Registry

Do not commit model weights. Keep `.pt` and related checkpoint files under the
ignored `checkpoints/` tree, and register blessed or important runs with small
JSON manifests under `artifacts/checkpoints/`.

Do not leave a promoted model only inside a temporary Codex worktree. After
promotion, mirror the ignored checkpoint run directory into the canonical local
checkout under `/Users/zacharymarion/Documents/code/create-pattern-detector`
using the same `checkpoints/...` relative path recorded in the manifest, then
verify the SHA-256. Downstream ONNX exports should likewise live in the
canonical `tree-maker-rust/apps/web/public/models/` tree, not only in a
throwaway worktree.

Before replacing, exporting, or using a checkpoint, read
`docs/model-training-history.md` first, then `docs/checkpoint-management.md`.
The current downstream/browser model is the corrected V3 no-guide-grid
close-pair dense-edge 15% tessellation run registered at
`artifacts/checkpoints/runpod-v3-no-guide-grid-close-pair-dense-edges-tess15-weighted-4090.json`.
It is exported in `tree-maker-rust` under the stable `cp-detector-v3` product
path and the versioned `cp-detector-v3-tess15-weighted-20260619` path.

Do not confuse the previous V3 close-pair R1 checkpoint, the no-guide-grid R1,
max700 dense-edge, or max1200 dense-edge checkpoints that this run superseded,
or the later R3 from-scratch run with the promoted model. The previous
close-pair R1 is retained at
`artifacts/checkpoints/runpod-v3-close-pair-warmstart-4090.json`; R3 is
registered as an ablation at
`artifacts/checkpoints/runpod-v3-close-pair-scratch-r3-4090.json` because it
landed statistically identical to R1 and was not promoted.

The older blessed Phase 3 V1 Python/CLI baseline remains registered at
`artifacts/checkpoints/phase3-v1-cpline.json`.

## RunPod Phase 3

For GPU training setup, follow `docs/runpod-quickstart.md` first. It includes
the CUDA-compatible Torch pin, the dereferenced synthetic dataset upload, smoke
checks, curriculum launch, monitoring, and teardown commands. Use
`docs/runpod-phase-3.md` for the longer rationale and review gates.

Before launching more Phase 3 GPU work, read `docs/phase-3-v1-status.md`.
Phase 3 V1 is complete for readable 1024px crease patterns; the remaining dense
Rabbit Ear/tiny-fold tail is tracked as V2 and should not block Phase 4 work.

## No-Guide-Grid Training Safety

For any no-guide-grid training intended to stay compatible with the current V3
close-pair product decoder, use only:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_full.sh
```

or the short probe variant:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_probe.sh
```

These canonical launchers set and verify the required R1 close-pair recipe:
`junction_sigma_px=1.5`, `junction_offset_radius_px=3.0`,
`junction_offset_weight=0.5`, `junction_focal_alpha=2.0`, and
`junction_focal_beta=4.0`. The older
`run_cpline_runpod_v3_no_guide_grid_{probe,full}.sh` names are intentionally
retired and fail with an "Are you sure?" message because they previously
launched a dense-head diagnostic with `junction_offset_radius_px=0.0`.

For dense BP follow-up probes from the current promoted tess15 weighted model,
use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_probe.sh
```

It defaults to the promoted `MAX_EDGES=1200` checkpoint and verifies the same
radius-3 close-pair recipe. It refuses to write into an existing `OUTPUT_ROOT`
unless `ALLOW_EXISTING_OUTPUT_ROOT=1` is set, so use a fresh explicit
`OUTPUT_ROOT` for new probes.

For the 15% tessellation BP-data experiment, use:

```bash
scripts/training/run_cpline_runpod_v3_no_guide_grid_close_pair_dense_edges_tess15_probe.sh
```

That launcher defaults to
`data/generated/synthetic/cp_training_mix_v3_tessellation_15pct/raw-manifest.jsonl`
and `TRAIN_FAMILY_SAMPLING=v3-tessellation-15pct`. The generic dense-edge
launcher also auto-selects this sampler for that manifest and fails with an
"Are you sure?" message if a non-default tessellation sampler is used without
`ALLOW_NONDEFAULT_TESS15_SAMPLING=1`.
