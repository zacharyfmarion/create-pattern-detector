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

## Shared Synthetic Datasets

The maintained synthetic generator lives under `tools/synthetic-generator/`.
This PR keeps it focused on two fold-only families:

- `treemaker-tree`: primary external TreeMaker-derived CP generation.
- `rabbit-ear-fold-program`: strict supplemental Rabbit Ear fold-operation generation.

Large generated releases live outside the repo:

```text
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1
/Users/zacharymarion/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v1
```

Future worktrees should link the mixed root when training:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
PYTHONPATH=. python3.10 scripts/data/smoke_shared_synthetic_data.py --root data/generated/synthetic/cp_training_mix_v1
```

See `docs/synthetic-fold-datasets.md` for generation, shard merge, folded-preview,
and mix-building commands.
