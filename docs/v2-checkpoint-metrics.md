# V2 Checkpoint Metrics

This page records V2 checkpoint comparisons that should be kept in git. The
large plots, contact sheets, and per-sample visualizations remain under ignored
`visualizations/` directories.

Machine-readable snapshot:

```text
artifacts/evaluations/v2-checkpoint-comparison-20260522.json
```

Current V2 candidate checkpoint:

```text
artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json
```

## 2026-05-22 Corrective Replay Comparison

Evaluation command family:

```text
scripts/evals/eval_stage4_checkpoint.py
```

Settings:

- `image_size=1024`
- `max_edges=300`
- `split=val`
- `samples_per_profile=16`
- profiles: `clean`, `line-style`, `v2-all-issue-mix`, `v2-dark-issue-mix`
- `family_sampling=balanced`
- `batchnorm_mode=batch-stats`
- `threshold=0.65`
- deterministic square-border reconstruction enabled
- no oracle border chain

The comparison is intentionally a compact checkpoint screen, not the final V2
promotion suite. It is useful for deciding whether another run improved over
the prior checkpoints without rerunning those prior checkpoints.

| checkpoint | profile | edge P/R | border P/R/F1 | assignment | structural |
|---|---:|---:|---:|---:|---:|
| Phase 3 V1 | aggregate | 0.724/0.781 | 0.715/0.761/0.737 | 0.978 | 1.000 |
| Phase 3 V1 | clean | 0.943/0.907 | 0.862/0.860/0.861 | 0.987 | 1.000 |
| Phase 3 V1 | line-style | 0.904/0.874 | 0.832/0.836/0.834 | 0.984 | 1.000 |
| Phase 3 V1 | v2-all-issue-mix | 0.590/0.604 | 0.492/0.597/0.539 | 0.989 | 1.000 |
| Phase 3 V1 | v2-dark-issue-mix | 0.541/0.739 | 0.721/0.754/0.737 | 0.953 | 1.000 |
| V2 issue-only | aggregate | 0.763/0.835 | 0.796/0.848/0.821 | 0.990 | 0.984 |
| V2 issue-only | clean | 0.925/0.905 | 0.862/0.894/0.878 | 0.990 | 1.000 |
| V2 issue-only | line-style | 0.597/0.756 | 0.684/0.763/0.721 | 0.987 | 0.938 |
| V2 issue-only | v2-all-issue-mix | 0.791/0.839 | 0.792/0.862/0.825 | 0.990 | 1.000 |
| V2 issue-only | v2-dark-issue-mix | 0.785/0.840 | 0.856/0.874/0.865 | 0.994 | 1.000 |
| V2 replay partial | aggregate | 0.855/0.865 | 0.848/0.881/0.864 | 0.987 | 1.000 |
| V2 replay partial | clean | 0.916/0.905 | 0.855/0.896/0.875 | 0.984 | 1.000 |
| V2 replay partial | line-style | 0.775/0.817 | 0.788/0.843/0.814 | 0.985 | 1.000 |
| V2 replay partial | v2-all-issue-mix | 0.860/0.861 | 0.861/0.884/0.872 | 0.989 | 1.000 |
| V2 replay partial | v2-dark-issue-mix | 0.872/0.878 | 0.890/0.901/0.896 | 0.989 | 1.000 |
| V2 replay full | aggregate | 0.877/0.877 | 0.860/0.892/0.876 | 0.987 | 1.000 |
| V2 replay full | clean | 0.932/0.905 | 0.896/0.915/0.906 | 0.988 | 1.000 |
| V2 replay full | line-style | 0.849/0.860 | 0.836/0.889/0.862 | 0.984 | 1.000 |
| V2 replay full | v2-all-issue-mix | 0.858/0.868 | 0.847/0.867/0.857 | 0.988 | 1.000 |
| V2 replay full | v2-dark-issue-mix | 0.871/0.875 | 0.861/0.896/0.878 | 0.986 | 1.000 |

Interpretation:

- `V2 replay full` is the current best balanced checkpoint.
- It restores most of the old-profile `line-style` regression introduced by the
  issue-only V2 curriculum.
- It improves over `V2 issue-only` by `+0.114` edge precision, `+0.042` edge
  recall, and `+0.054` border F1 on this screen.
- It improves over `V2 replay partial` by `+0.022` edge precision, `+0.012`
  edge recall, and `+0.011` border F1.
- Assignment accuracy is effectively flat to slightly lower than issue-only,
  which is acceptable for this checkpoint because the current bottleneck is
  geometry/topology rather than M/V classification.

## Boundary-Contact Head Check

Evaluation command family:

```text
scripts/v2/eval_boundary_contacts.py
```

Settings:

- `image_size=1024`
- `max_edges=300`
- `count=24`
- thresholds swept over `0.25,0.35,0.45,0.55,0.65,0.75,0.85`

| checkpoint | profile | threshold | contact P/R/F1 | corner FP rate | contact vertex acc |
|---|---:|---:|---:|---:|---:|
| V2 issue-only | clean | 0.750 | 0.963/0.919/0.941 | 0.068 | 1.000 |
| V2 issue-only | line-style | 0.550 | 0.952/0.900/0.925 | 0.074 | 1.000 |
| V2 issue-only | v2-all-issue-mix | 0.650 | 0.961/0.919/0.939 | 0.068 | 1.000 |
| V2 issue-only | v2-dark-issue-mix | 0.750 | 0.963/0.919/0.941 | 0.068 | 1.000 |
| V2 replay partial | clean | 0.650 | 0.963/0.919/0.941 | 0.068 | 1.000 |
| V2 replay partial | line-style | 0.650 | 0.963/0.919/0.941 | 0.062 | 1.000 |
| V2 replay partial | v2-all-issue-mix | 0.550 | 0.961/0.922/0.941 | 0.067 | 0.997 |
| V2 replay partial | v2-dark-issue-mix | 0.250 | 0.961/0.922/0.941 | 0.067 | 1.000 |
| V2 replay full | clean | 0.750 | 0.963/0.919/0.941 | 0.068 | 1.000 |
| V2 replay full | line-style | 0.650 | 0.961/0.919/0.939 | 0.068 | 1.000 |
| V2 replay full | v2-all-issue-mix | 0.750 | 0.969/0.916/0.942 | 0.057 | 0.997 |
| V2 replay full | v2-dark-issue-mix | 0.250 | 0.961/0.919/0.939 | 0.068 | 1.000 |

Interpretation:

- Boundary-contact localization is already around `0.94` F1 on this synthetic
  contact eval.
- The remaining gap to structurally perfect FOLD output is therefore not mainly
  a missing boundary-contact head. It is the downstream square-aware topology
  problem: carrier selection, analytic intersections, border-chain splitting,
  artifact rejection, dashed support, and compile/flat-foldability gates.

## 2026-05-22 Square Topology Inspector Screen

Evaluation command family:

```text
scripts/evals/eval_stage4_checkpoint.py --decoder square
```

Settings:

- checkpoint: `artifacts/checkpoints/runpod-v2-replay-correction-full-4000ada.json`
- `image_size=1024`
- `max_edges=300`
- `split=val`
- `samples_per_profile=6`
- profiles: `clean`, `line-style`, `v2-text`, `v2-watermark`,
  `v2-guide-grid`, `v2-dashed`, `v2-faint`, `v2-ambiguous-mv`,
  `v2-combined`, `v2-dark-combined`, `v2-replay-corrective`
- `family_sampling=balanced`
- `batchnorm_mode=batch-stats`
- `threshold=0.65`
- deterministic square topology decoder enabled

Generated visual inspection bundle:

```text
visualizations/v2_square_topology_inspector/eval
```

| profile | edge P/R | border P/R/F1 | structural |
|---|---:|---:|---:|
| aggregate | 0.868/0.795 | 0.925/0.893/0.909 | 1.000 |
| clean | 0.933/0.851 | 0.961/0.942/0.951 | 1.000 |
| line-style | 0.901/0.789 | 0.929/0.917/0.923 | 1.000 |
| v2-text | 0.861/0.810 | 0.961/0.923/0.942 | 1.000 |
| v2-watermark | 0.874/0.820 | 0.961/0.923/0.942 | 1.000 |
| v2-guide-grid | 0.913/0.833 | 0.929/0.923/0.926 | 1.000 |
| v2-dashed | 0.865/0.805 | 0.921/0.891/0.906 | 1.000 |
| v2-faint | 0.920/0.829 | 0.940/0.910/0.925 | 1.000 |
| v2-ambiguous-mv | 0.918/0.831 | 0.921/0.897/0.909 | 1.000 |
| v2-combined | 0.759/0.686 | 0.891/0.840/0.865 | 1.000 |
| v2-dark-combined | 0.765/0.734 | 0.851/0.878/0.864 | 1.000 |
| v2-replay-corrective | 0.848/0.755 | 0.924/0.784/0.838 | 1.000 |

Interpretation:

- The square topology decoder is no longer over-generating every analytic
  carrier intersection; outputs are compact enough for visual inspection.
- Structural validity is perfect on this synthetic screen, so the next review
  should focus on missed/extra topology and border-contact placement rather
  than parseability failures.
- Combined, dark-combined, and replay-corrective profiles remain the weakest
  slices and need visual inspection before another training or decoder pass.

## Caveat

These numbers are Stage 4 structural metrics, not proof that exported FOLD files
fold cleanly in Orieta. Many outputs are parseable and complete-border, but the
quality report still often flags `ambiguous` or `outside_v1_envelope` because of
short edges, crowded junctions, Kawasaki/Maekawa warnings, and assignment
uncertainty.
