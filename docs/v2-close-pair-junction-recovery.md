# V2 Close-Pair Junction Recovery (label + loss + decoder plan)

Status: Proposed implementation plan, 2026-06-09.
Owner branch: `codex/v2-roadmap-border-repair` (V2 heads live here only).
Downstream consumer: `tree-maker-rust` `junction-first-v1` candidate strategy
(PR #55), which is now the production default.

## Why this plan exists

Strict graph-isomorphism benchmarking in `tree-maker-rust` (clean-1024-s15,
4px vertex matching, reports `2026-06-09-strict-*`) shows the decoder stack is
at its ceiling and the residual error is one model-side phenomenon:

- `junction-first-v1` + parity repair reaches strict edge F1 **0.942**, exact
  topology on 4/15 samples (legacy baseline: 0.902, 1/15).
- **172/175 remaining missing edges and ~120/150 extra edges trace to GT vertex
  pairs closer than 8px** (median 4.9px apart, tiny connecting creases median
  6.1px). Affects 11/15 samples; worst in rabbit-ear.
- Measured on the dense cache: the junction heatmap covers these pairs with a
  **single fused ~0.99 plateau** — one local maximum in 49/53 sampled pairs —
  and `junction_offset` outputs are ±0.1px and unimodal across the blob, so no
  decoder can split them. Vertex probability at the true locations is ≥0.99;
  the information is *present but not separable*.

If close pairs become separable, the projected strict numbers are eF1
~0.97–0.99 and exact topology on ~10+/15 samples.

## Root causes in the training targets (verified, file:line)

All citations are the **actual CPLine training path**
(`CplineFoldDataset.__getitem__` → `render_cpline_sample` →
`render_augmented_cpline_sample` in `src/data/cpline_augmentations.py`). Note
`src/data/annotations.py:_generate_junction_offsets` is a different (legacy)
pipeline and is NOT what CPLine training consumes.

1. **Junction Gaussian is too wide at 1024px.**
   `src/data/cpline_augmentations.py:217`:
   `junction_sigma = max(1.0, 2.5 * image_size / 768)` → **σ ≈ 3.33px at
   image_size=1024**. Two vertices 5px apart leave a label-level midpoint dip
   of only ~0.75 (max-blended, `src/vectorization/evidence.py:124`
   `_add_gaussian` + `_add_impulse`). The label itself is nearly fused.
2. **BCE + pos_weight=50 rewards filling the dip.**
   `src/models/losses/cpline_loss.py:20` `junction_pos_weight: 50.0`.
   Under-predicting any positive pixel costs 50×; over-predicting the shallow
   dip between two peaks costs almost nothing. The model learns plateaus.
3. **Offsets are subpixel-only, supervised at one anchor pixel per vertex.**
   `src/data/cpline_augmentations.py:667` `_junction_offsets`: target
   `(x−round(x), y−round(y)) ∈ [−0.5, 0.5]` written at the single rounded
   pixel; `junction_mask` true only there. Blob pixels carry no information
   about *which* vertex they belong to. (Decode side confirms: measured raw
   offsets across fused blobs are ±0.1 and unimodal.)

## Changes

### A. CenterNet-style offset targets with radius (the splitting signal)

Rewrite `_junction_offsets` (`src/data/cpline_augmentations.py:667`):

- For every pixel within `offset_radius_px` (default **3.0**) of any degree≥1
  vertex, write `offset = (vx − px, vy − py)` toward the **nearest** vertex,
  and set `junction_mask` true.
- Range becomes [−r, r]; keep the head linear (it already is) and switch the
  decode-side clamp accordingly (see Decoder section).
- Nearest-vertex assignment is what creates the **bimodal offset field over a
  fused blob**: pixels on each side point to their own vertex.
- Implementation: KD-tree or simple grid bucketing over vertices; O(V·r²).
- Loss: existing masked L1 (`junction_offset_weight: 0.25` →raise to **0.5**
  since the task is now load-bearing, not cosmetic). Normalize targets by
  `offset_radius_px` so the loss scale is comparable to before (predict in
  [−1, 1] × radius).

### B. Sharper, dip-preserving heatmap

1. `src/data/cpline_augmentations.py:217`: junction sigma → **fixed 1.5px at
   1024** (`max(1.0, 1.5 * image_size / 1024)`). At σ=1.5 a 5px pair has a
   label dip of ~0.25 — a real valley. Keep `_add_impulse`.
2. **Penalty-reduced focal loss for the junction head** (CornerNet/CenterNet
   formulation) replacing BCE+pos_weight=50 in
   `src/models/losses/cpline_loss.py`:
   `loss = −(1−p̂)^α·log(p̂)` at peak pixels (label==1), and
   `−(1−y)^β·p̂^α·log(1−p̂)` elsewhere (β≈4 down-weights near-peak negatives but
   still penalizes confident over-prediction in the dip). α=2, β=4 standard.
   There is already focal machinery in this file for vertex_type
   (`cpline_loss.py:317-330`) to model the implementation on.
3. Optional v1.1 (only if A+B leave residue): up-weight loss pixels within 8px
   of a GT close pair (the renderer knows GT geometry; a per-sample weight map
   is cheap).

### C. Decoder rewrite in tree-maker-rust (offset-cluster vertex extraction)

`crates/oristudio-cp-detect/src/evidence_extract.rs` — gated by a new
`EvidenceExtractionConfig` flag (e.g. `offset_cluster_decode: bool`) so old
caches/models keep the current local-maxima path:

1. Collect all pixels with junction probability ≥ threshold.
2. For each, compute `target = pixel + offset` (offsets now range ±r; replace
   the `finite_offset` ±0.5 clamp at `evidence_extract.rs:715` with ±r).
3. Mean-shift / greedy-cluster the targets with bandwidth ~1.5px; each cluster
   (with summed support) becomes a `JunctionPrimitive`.
4. A fused blob over a close pair yields two target clusters → two vertices.
5. Keep NMS as a final dedupe at ~1px.

`junction-first-v1` needs no changes: with both vertices present, the tiny
edge is proposed by the adjacent-pair scorer (min span is already 3px) and the
adjacency corridor blocks the pass-through span.

## Retraining strategy: warm-start, not from scratch

Warm-start from the deployed V2 checkpoint
(`artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json`) via
`--init-checkpoint`, **re-initializing only the `junction_offset` head**
(its output semantics flip from subpixel to radius-vectors; everything else —
backbone, line/assignment/style/boundary heads — is untouched by A+B and
should not relearn). The junction head keeps its weights (its task merely
sharpens). Add a small `--reinit-heads junction_offset` switch to the trainer
for this.

Why not from scratch: identical data, 10/12 heads unchanged → warm-start
converges in a fraction of the steps with far less risk of regressing the
boundary/vertex-type heads that V2 worked hard to land. From-scratch is kept
as a **single comparison ablation** because at ~$1/run it is nearly free; if
warm-start shows any sign of being trapped in the old offset regime, the
scratch run is the fallback. Decide on strict-harness numbers, not loss curves.

## RunPod execution plan (~$25 budget, speed-balanced)

Tooling already in repo: `docs/runpod-quickstart.md`, `configs/runpod.env`
(restricted API key), `runpodctl` installed locally, curriculum script
`scripts/training/run_cpline_runpod_curriculum.sh`. Reference timing: the V2
full-dataset run (5202 train / 638 val, 800 steps, 1024px, batch 1, hrnet_w18)
took **~68 min on an RTX 4000 Ada** (~$0.26–0.39/hr secure cloud).

Pod: one **RTX 4090** (~$0.34–0.69/hr, ~2× the 4000 Ada) or 4000 Ada/L4 if
out of stock. Always create with `--stop-after` (max 10h) as the hard budget
fuse; one pod at a time.

| Run | Steps | Est. wall | Est. cost |
| --- | ---: | ---: | ---: |
| R0 smoke (256px, tiny backbone, new targets/loss) — local Mac, free | 120 | 10 min | $0 |
| R1 warm-start fine-tune, full dataset, σ=1.5 + focal + radius offsets | 1500 | ~1.2h (4090) | ~$0.8 |
| R2 ablation: warm-start, σ=1.5, focal, **no** radius offsets (isolates A) | 1500 | ~1.2h | ~$0.8 |
| R3 ablation: from-scratch, full recipe | 4000–6000 | ~3–4h | ~$2.5 |
| R4 contingency (tuning σ/α/β, offset radius 2–4, longer R1) | ≤3 runs | ~4h | ~$3 |
| Eval/export overhead (GPU) | — | ~1h | ~$0.7 |

Projected spend **≤ $10**, leaving >50% headroom of the $25 cap. Hard rules:
ledger every pod-hour in the run log; stop and reassess if cumulative spend
crosses **$18**; never leave a pod running unattended past its stop-after.

## Evaluation gates (in order; each gate must pass before spending more)

1. **R0 smoke (local)**: targets render as expected — assert in a unit test
   that a synthetic 5px pair produces (a) two label peaks with a dip < 0.5 at
   σ=1.5 and (b) a bimodal offset field (left pixels point left, right point
   right). Trainer runs 120 steps without NaN.
2. **Detector-side close-pair eval** (new small script,
   `scripts/evals/eval_close_pairs.py`): on val samples, for every GT vertex
   pair < 8px apart, decode with offset-clustering and count pairs resolved
   into two vertices ≤1.5px from GT. Gate: **≥80% resolved** (currently ~8%,
   i.e. 4/53), with overall junction recall/precision within 1% of the V2
   baseline and line/boundary losses not regressed.
3. **End-to-end strict topology** (the real gate):
   - export ONNX: `tree-maker-rust/scripts/cp-detect/export-cpline-onnx.py`
     (opset 17, 12 output heads, browser manifest);
   - regenerate the dense cache:
     `node scripts/cp-detect/run-browser-dense-cache.mjs --pack
     .../packs/clean-1024-s15/manifest.json --out
     .../dense-cache/clean-1024-s15-browser-onnx-v3` (web app running);
   - implement decoder change C, then run
     `compare_exact_solve_benchmark --candidate-source junction-first-v1
     --skip-exact-solve --skip-flat-folder --strict-vertex-tolerance-px 4`.
   Gates: `merged_edges` 78 → **≤15**; strict edge F1 **≥0.97**; exact
   topology **≥8/15**; extras not above 150; lenient selected recall not
   below 0.958. Also rerun the V2 issue-profile packs (watermark/dashed) to
   confirm no robustness regression before promoting the checkpoint.

## Stop conditions

- R0 target-rendering assertions fail → fix labels before any GPU spend.
- R1 close-pair resolution < 40% → offsets aren't learning the field; try
  raising `junction_offset_weight`, radius 4, or R3 from-scratch before any
  further tuning.
- Strict harness shows merged_edges fixed but extras ballooning → the decode
  bandwidth/NMS needs tuning, not more training.
- Cumulative spend ≥ $18 without passing gate 3 → stop, write up findings.

## Deliverables

1. Label/loss changes on `codex/v2-roadmap-border-repair` with unit tests
   (close-pair label assertions, focal-loss shape tests).
2. `--reinit-heads` trainer switch + run configs for R1–R3.
3. `scripts/evals/eval_close_pairs.py`.
4. Trained checkpoint + registry JSON under `artifacts/checkpoints/`, run logs
   with the cost ledger.
5. ONNX export + new dense cache + decoder change (C) in tree-maker-rust, with
   the strict-harness comparison report archived under
   `artifacts/cp-detect-correctness/reports/`.

## Outcome (2026-06-10)

Executed R0/R1/R3 plus the decoder work. Total GPU spend ~$7.10 of the $25
budget. Checkpoint registry: `runpod-v3-close-pair-warmstart-4090` (R1).

| config | exact | strict eF1 | miss | extra | merged | pair res. |
| --- | --- | --- | --- | --- | --- | --- |
| shipped V2 baseline | 4/15 | 0.942 | 176 | 130 | 74 | 5.4% |
| R1 warm-start + cluster decode | 4/15 | **0.953** | 145 | 101 | 65–67 | 26.5% |
| R3 from-scratch (5000 steps) | 4/15 | 0.952 | 144 | 109 | 71 | 21.5% |

What held up:

- The recipe is a real, strict-metric improvement on every axis with no
  regressions (junction recall 0.959 → 0.973 at precision 0.99; the gain is
  mostly the sigma-1.5 + focal heatmap).
- The offset field works as designed where it works: votes land median 0.35px
  from GT, and 60% of close pairs have bimodal fields on val samples.
- Seed-anchored offset-vote clustering (eval + the mirrored Rust decoder)
  converts those into split vertices without hurting overall junction PR.

What did not:

- Pair resolution plateaued at ~21–27% against the 80% gate, and **R3
  from-scratch landed identically to R1 warm-start**, refuting the
  "warm-start is the ceiling" hypothesis. Two independent inits converge to
  the same plateau: the limit is the recipe/representation. The likely root
  cause is the input itself — at 1024px with ~2px strokes, two junctions 5px
  apart with several incident creases are nearly a single ink blob; the
  remaining pairs give one-sided or midpoint-collapsed offset fields.
- Consequently the strict gates (eF1 ≥ 0.97, exact ≥ 8/15, merged ≤ 15) were
  not met; merged edges only improved 74 → ~67.

Recommendation: bank R1 (registry entry above), keep the decoder flag-gated,
and do NOT continue tuning this recipe (radius/weight ablations face two
identical plateaus). If close pairs are revisited, the levers are different in
kind: higher input resolution for the dense heads (2048px or tiled crops), a
dedicated close-pair head, or pipeline-side fused-vertex splitting from
incident-line geometry in tree-maker-rust.

Operational gotchas recorded for the next round: these checkpoints REQUIRE
batch-stats BatchNorm at inference (eval-mode running stats collapse junction
recall to ~0.35), and the fresh-checkout traps for browser dense-cache
regeneration are documented in tree-maker-rust `scripts/cp-detect/README.md`.
