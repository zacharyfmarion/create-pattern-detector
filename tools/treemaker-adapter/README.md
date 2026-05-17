# TreeMaker Adapter

This directory does **not** vendor TreeMaker. TreeMaker source is GPL, so this
repo keeps only a thin wrapper and setup script. The actual TreeMaker checkout
and compiled CLI live outside the repo, like scraped or generated data.

## External Setup

Build the real TreeMaker-backed CLI into a local cache:

```bash
python tools/treemaker-adapter/scripts/setup_external_treemaker_cli.py
```

The script clones:

- `https://github.com/AndrewKvalheim/treemaker.git`
- branch: `legacy-environment`

and builds `tools/treemaker-adapter/headless/treemaker-json-cli.cpp` against the
external checkout. It prints the environment variables to use:

```bash
export TREEMAKER_CLI=~/.cache/cp-detector/treemaker-legacy/build/treemaker-json-cli
export TREEMAKER_CLI_ARGS=--triangulate
export TREEMAKER_TIMEOUT_MS=5000
```

`--triangulate` is currently required for sampled specs; without it, TreeMaker
often emits partial polygon content with `POLYS_NOT_VALID`.

`TREEMAKER_TIMEOUT_MS` prevents one pathological tree from hanging a large
generation shard. The synthetic generator treats timeout as a rejected attempt.

## CLI Contract

The executable must accept:

```bash
$TREEMAKER_CLI $TREEMAKER_CLI_ARGS --spec spec.json --out out.json
```

`spec.json` uses `treemaker-adapter-spec/v1` from
`tools/synthetic-generator/src/treemaker-sampler.ts`.

The output JSON contains:

- `ok`: true only when TreeMaker produced `HAS_FULL_CP`;
- `optimization.success`;
- `foldedForm.success`;
- `stats.vertices`, `stats.creases`, `stats.facets`;
- `creases[]`, with `p1`, `p2`, `assignment`, `foldAngle`, and TreeMaker `kind`.

The synthetic generator rejects `ok=false`. Partial TreeMaker exports are useful
for debugging but are not training labels.

## Why This Uses The Legacy Branch

The modern `vishvish/treemaker` fork builds on macOS, but in the current spike
the same headless model calls produced empty or invalid CPs. The
`AndrewKvalheim/treemaker` `legacy-environment` branch produced real full CPs:

- `@optimized`: 104 creases, `HAS_FULL_CP`;
- `@gusset`: 24 creases, `HAS_FULL_CP`;
- `tmModelTester_5.tmd5`: 434 creases, `HAS_FULL_CP`;
- sampled spec with `--triangulate`: 299 creases, `HAS_FULL_CP`.

Until that behavioral difference is understood, production synthetic generation
should use the legacy branch through this adapter.
