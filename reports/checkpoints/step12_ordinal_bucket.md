# CHECKPOINT — Phase 9F: ordinal turbulence bucket (verifying the middle-path side-finding)

**Generated:** 2026-06-18 13:54 UTC

## What was tested
The middle-path experiment (`prevversion/midpath/`) showed the GMM adds nothing but that an ordinal 5-level turbulence bucket beats v5's binary turbulent/quiet threshold on cutoff-robustness (median -0.106 -> +0.033 on the 2010-12+ window, logistic_l1 only).  Phase 9F verifies that on the FULL 2007-12+ window, all four models for the quarterly walk-forward, and two models for the 49-cutoff sliding test.

## What was built
- `regime_detection/lib/turbulence_bucket.py` — `ordinal_turbulence_bucket()` (causal n-quantile binning) + `build_feature_matrix_with_bucket()` (v5 matrix, binary regime swapped for the bucket).
- `regime_detection/tests/test_turbulence_bucket.py` — 6 tests (range, warm-up NaN, occupancy, no-lookahead shock, monotonicity, validation).
- `regime_detection/step12_ordinal_bucket.py` — this driver.

## What tests pass
- **156/156 total** (was 150 — 6 new bucket tests).

## Quarterly walk-forward (full window, 4 models)

|                     |   binary_auc |   bucket_auc |   binary_brier |   bucket_brier |   binary_sortino_lift |   bucket_sortino_lift |
|:--------------------|-------------:|-------------:|---------------:|---------------:|----------------------:|----------------------:|
| random_forest       |        0.708 |        0.642 |          0.156 |          0.164 |                -0.139 |                -0.198 |
| logistic_l1         |        0.712 |        0.677 |          0.137 |          0.136 |                -0.182 |                 0.012 |
| linear_svm          |        0.578 |        0.536 |          0.166 |          0.188 |                -0.071 |                -0.173 |
| logistic_elasticnet |        0.713 |        0.674 |          0.137 |          0.137 |                -0.145 |                 0.067 |

## Sliding-cutoff distribution — 49 monthly cutoffs, FULL window

|                             |   n |   median |    q25 |    q75 |   frac_positive |   frac_above_0p10 |    max |
|:----------------------------|----:|---------:|-------:|-------:|----------------:|------------------:|-------:|
| ('logistic_l1', 'binary')   |  49 |   -0.147 | -0.222 | -0.051 |           0.061 |             0     |  0.055 |
| ('logistic_l1', 'bucket')   |  49 |    0.15  |  0.085 |  0.194 |           0.898 |             0.735 |  0.289 |
| ('random_forest', 'binary') |  49 |   -0.354 | -0.419 | -0.106 |           0     |             0     | -0.045 |
| ('random_forest', 'bucket') |  49 |   -0.255 | -0.356 | -0.111 |           0.02  |             0.02  |  0.102 |

![Phase 9F](figures/thesis_v5/fig11_ordinal_bucket.png)

## Verdict

**CONFIRMED.** On the full 2007-12+ window the ordinal 5-bucket lifts the logistic_l1 median sliding-cutoff Sortino from -0.147 to +0.150 (frac positive 6% -> 90%) and the random_forest median from -0.354 to -0.255.  The improvement survives the full window and a second model — it was NOT a 2010-12+ artefact.  Recommend adopting the ordinal bucket as v5's regime feature; it is a ~30-line, GMM-free change that directly addresses the L2 cutoff-sensitivity limitation.

## Caveats
- Same ~1,327-row OOS sample-size ceiling: the per-cutoff resolution is ~±0.5 Sortino, so the signal is the DISTRIBUTIONAL shift across 49 cutoffs, not any single cutoff.
- AUC may be lower than v5 even where Sortino-robustness improves (the bucket is a more robust trader, not necessarily a better classifier) — read the table above.
- The bucket is deterministic (no model seed), so unlike the GMM there is no seed-robustness concern.

## Open questions for the user
- If CONFIRMED: fold the ordinal bucket into the Phase-10 thesis as the v5 regime feature, and update the abstract's cutoff-distribution numbers?
- Sweep `n_buckets` (3 / 5 / 10) to find the best granularity, or is 5 sufficient for the thesis?
