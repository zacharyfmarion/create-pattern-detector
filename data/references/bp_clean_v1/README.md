# Clean BP Reference Manifest

This directory is reserved for calibration-only clean digital box-pleat crease-pattern references.
Reference rows are not training labels. They are used by
`scripts/data/bp_realism_report.py` to compare generated samples against
real/BPStudio-like graph or raster distributions.

Do not add copyrighted CP images here unless the source and allowed use are clear.

## Manifest Path

The default reference manifest path is:

```text
data/references/bp_clean_v1/manifest.jsonl
```

Rows are JSON Lines. Paths are resolved relative to this directory. The report
script ignores rows with `"ignore": true`. Rows with `"placeholder": true` are
ignored unless at least one referenced `fold_path` or `image_path` exists.

## Required Fields

- `id`: Stable unique reference id.
- `source_url`: Original public source URL or local source note.
- `license`: License or permission status. Use values such as `MIT`,
  `CC-BY-4.0`, `public-domain`, `author-permission`, or
  `calibration-only-unverified`.

## Asset Fields

At least one of these must be present for a non-placeholder row:

- `fold_path`: Path to a clean FOLD graph with `vertices_coords`,
  `edges_vertices`, and optional `edges_assignment`/`edges_bpRole`.
- `image_path`: Path to a clean reference image. Image-only rows require Pillow
  for raster metrics; graph rows do not.

CamelCase aliases `foldPath`, `imagePath`, and `sourceUrl` are accepted for
compatibility, but new rows should use snake_case.

## Recommended Metadata

- `archetype`: One of `insect`, `quadruped`, `bird`, `object`, `abstract`, or a
  short project-specific label.
- `designer`: Designer or tool name, when known and allowed.
- `reference_type`: `real_cp`, `bp_studio_export`, `licensed_synthetic`, or
  `placeholder`.
- `review_status`: `candidate`, `accepted`, `rejected`, or `needs_review`.
- `tags`: Array of short labels such as `tree`, `fan`, `staircase`, `ridge`,
  `river`, or `dense`.
- `notes`: Short calibration note.

## Example Row

```json
{"id":"placeholder-insect-001","placeholder":true,"fold_path":"folds/placeholder-insect-001.fold","image_path":"images/placeholder-insect-001.png","source_url":"placeholder-only","license":"placeholder-no-assets","archetype":"insect","reference_type":"placeholder","review_status":"candidate","tags":["placeholder"],"notes":"Ignored until a permitted FOLD or image file exists at one of the referenced paths."}
```

## Report Command

```bash
python scripts/data/bp_realism_report.py \
  --root data/generated/synthetic/box_pleat_realistic_v3 \
  --reference-manifest data/references/bp_clean_v1/manifest.jsonl
```

The generated report is written to `qa/bp-realism-report.json` under the
dataset root unless `--out` is provided.
