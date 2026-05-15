# Real CP Scraping

This pipeline collects internal-use real origami crease-pattern examples while
keeping provenance beside every downloaded asset.

## Output Layout

Scrapes write under `data/output/scraped/`:

- `source_snapshots/`: fetched CPOogle JSON, OBB HTML, and run metadata.
- `raw_assets/`: downloaded images, PDFs, and native files. CPOogle image
  smoke/screening runs can store Drive thumbnails here; manifests mark those as
  `download_variant=drive_thumbnail`.
- `crops/`: normalized CP-only image crops split into `review/` and rejected manifest rows.
  Heuristics do not auto-accept crops; Gemini promotes review crops later.
- `native/`: preserved native assets plus converted geometry-only FOLD files.
- `manifests/`: JSONL asset, crop, native, dedupe, and summary manifests.
- `review/`: contact sheets for quick manual QA.

Raw images are assumed internal-only unless a later license review says
otherwise.

Scrape commands resume by default. Before downloading, they reuse matching
successful asset manifest rows and deterministic files already present under
`raw_assets/`. This covers both completed runs and interrupted runs that wrote
raw files before a final manifest was produced. Use `--force-download` when you
intentionally want to fetch everything again.

## Commands

One-time local setup from this worktree:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pillow numpy opencv-python pypdfium2 pytest
```

Check local scraping dependencies and, once you have a key, Drive API access:

```bash
source .venv/bin/activate
GOOGLE_DRIVE_API_KEY=... python scripts/data/check_scraping_setup.py
```

Dry-run CPOogle candidate discovery:

```bash
python scripts/data/scrape_cpoogle.py --dry-run
```

Smoke scrape a small CPOogle sample:

```bash
GOOGLE_DRIVE_API_KEY=... python scripts/data/scrape_cpoogle.py --limit-assets 25
```

The setup checker uses `GOOGLE_DRIVE_API_KEY` or `GOOGLE_API_KEY` to verify
Drive API metadata access. API-key media downloads can be blocked by Google, so
the scraper defaults to public `drive.usercontent.google.com` downloads. Use
`--drive-media-api` only if you have confirmed media downloads work for your
credentials.

For faster runs, use workers:

```bash
GOOGLE_DRIVE_API_KEY=... python scripts/data/scrape_cpoogle.py \
  --limit-assets 500 \
  --workers 32 \
  --request-delay 0
```

CPOogle image downloads default to Drive thumbnails at 1024px wide because CP
detection is usually line-structure bound and the public original downloads are
slower. Use a larger thumbnail for safer screening, or set
`--image-download-size 0` to fetch original files:

```bash
python scripts/data/scrape_cpoogle.py \
  --limit-assets 500 \
  --workers 32 \
  --image-download-size 2048
```

For final high-quality training crops, prefer re-fetching originals
(`--image-download-size 0`) for assets that passed thumbnail screening.

Scrape the OBB gallery:

```bash
python scripts/data/scrape_obb.py
```

Estimate Gemini classifier cost for generated crops:

```bash
python scripts/data/estimate_gemini_cost.py data/output/scraped/crops --model gemini-2.5-flash-lite
```

Run Gemini only on heuristic-review crops, but estimate cost without API calls:

```bash
python scripts/data/scrape_cpoogle.py \
  --limit-assets 100 \
  --gemini-mode review \
  --gemini-cost-only
```

Actually call Gemini after reviewing the estimate:

```bash
GEMINI_API_KEY=... python scripts/data/scrape_cpoogle.py \
  --limit-assets 100 \
  --gemini-mode review \
  --gemini-model gemini-2.5-flash-lite
```

The scraper defaults to `--gemini-mode off`, so normal scraping has no model
API cost.

Use a restricted API key for real runs: limit it to the Gemini API and set a
budget/alert in Google Cloud before scraping thousands of images.

## Full Extraction Workflow

Run the whole staged workflow with conservative rate-limit defaults:

```bash
set -a; source configs/scraping.env; set +a
python scripts/data/run_full_real_cp_extraction.py
```

For a small end-to-end smoke run:

```bash
python scripts/data/run_full_real_cp_extraction.py \
  --limit-assets 25 \
  --gemini-cost-only
```

The staged commands are available separately when you want more control or need
to resume:

```bash
python scripts/data/scrape_cpoogle.py \
  --no-images --no-pdfs \
  --workers 16 \
  --request-delay 0.05 \
  --timeout 30 \
  --retries 2
```

```bash
python scripts/data/scrape_cpoogle.py \
  --no-native \
  --image-download-size 2048 \
  --workers 16 \
  --request-delay 0.05 \
  --timeout 30 \
  --retries 2 \
  --gemini-mode off
```

```bash
python scripts/data/refetch_originals_from_manifest.py \
  --screening-run CPOOGLE_SCREENING_RUN_ID \
  --statuses review \
  --image-download-size 0 \
  --workers 12 \
  --request-delay 0.10
```

```bash
python scripts/data/classify_existing_crops.py \
  --crop-manifest data/output/scraped/manifests/CPOOGLE_ORIGINAL_CROPS.jsonl \
  --status review \
  --model gemini-2.5-flash-lite \
  --confidence-threshold 0.70
```

```bash
python scripts/data/build_final_real_cp_dataset.py \
  --native-run NATIVE_RUN_ID \
  --cpoogle-screening-run CPOOGLE_SCREENING_RUN_ID \
  --cpoogle-original-run CPOOGLE_ORIGINAL_RUN_ID \
  --cpoogle-original-crop-manifest GEMINI_MERGED_CROPS.jsonl \
  --obb-run OBB_RUN_ID \
  --output-root data/output/scraped/final
```

Final outputs are written under `data/output/scraped/final/`:

- `native_manifest.jsonl`: all preserved native files plus conversion status.
- `final_usable_images.jsonl`: deduped Gemini-accepted and unresolved-review crops.
- `final_rejects.jsonl`: rejected crops and duplicate records.
- `dedupe_groups.jsonl`: duplicate provenance groups.
- `summary.json`: counts for acceptance, review, rejects, duplicates, and native rows.

## Gemini Cost Notes

The built-in estimator uses current Gemini API pricing assumptions stored in
`src/data/scraping/gemini_classifier.py`, and prints the official pricing and
token documentation URLs. Recheck those links before a large run:

- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Token docs: https://ai.google.dev/gemini-api/docs/tokens

Use Gemini as a second pass on `review` crops first. That usually gives most of
the quality lift with far fewer calls than classifying every raw image.

With the current built-in `gemini-2.5-flash-lite` defaults, rough standard-tier
estimates are:

- 10k already-cropped 512px images: about `$0.70`.
- 10k already-cropped 1024px images: about `$1.47`.
- 10k raw 4096px images: about `$9.73`.

Those are estimates, not billing guarantees; run `estimate_gemini_cost.py` on
the actual crop directory before enabling model calls.
