# CHECKPOINT — decomposed-turbulence features vs scalar baseline

**Generated:** 2026-06-18 13:56 UTC

## Hypothesis
Does *which factor* drives turbulence (composition) predict imminent stress better than the scalar turbulence level alone?

## What was built
- `regime_detection/lib/turbulence_decomposition.py` — exact additive per-factor decomposition of `d_t` (causal; reconstructs `d_t` to machine precision).
- `regime_detection/lib/decomposition_features.py` — `decshare_<factor>_<w>` + `decconc_<w>` (22 added columns, windows 21/63, all `.shift(1)`-ed).
- `regime_detection/step15_composition_features.py` — this driver.

## Quarterly walk-forward (full window, 4 models)

|                     |   base_auc |   dec_auc |   base_brier |   dec_brier |   base_sortino_lift |   dec_sortino_lift |
|:--------------------|-----------:|----------:|-------------:|------------:|--------------------:|-------------------:|
| random_forest       |      0.708 |     0.661 |        0.156 |       0.145 |              -0.139 |             -0.458 |
| logistic_l1         |      0.712 |     0.704 |        0.137 |       0.148 |              -0.182 |             -0.322 |
| linear_svm          |      0.578 |     0.559 |        0.166 |       0.179 |              -0.071 |             -0.321 |
| logistic_elasticnet |      0.713 |     0.71  |        0.137 |       0.143 |              -0.145 |             -0.313 |

## Sliding-cutoff distribution — 49 monthly cutoffs

|                               |   n |   median |    q25 |    q75 |   frac_positive |   frac_above_0p10 |    max |
|:------------------------------|----:|---------:|-------:|-------:|----------------:|------------------:|-------:|
| ('logistic_l1', 'baseline')   |  49 |   -0.147 | -0.222 | -0.051 |           0.061 |             0     |  0.055 |
| ('logistic_l1', 'decomp')     |  49 |   -0.176 | -0.357 | -0.023 |           0.204 |             0.061 |  0.185 |
| ('random_forest', 'baseline') |  49 |   -0.354 | -0.419 | -0.106 |           0     |             0     | -0.045 |
| ('random_forest', 'decomp')   |  49 |   -0.335 | -0.455 | -0.15  |           0     |             0     | -0.107 |

![decomp vs baseline](figures/decomposition/decomp_vs_baseline.png)

## Verdict

**NO IMPROVEMENT.** logistic_l1 median moves -0.029 (-0.147->-0.176); mean AUC change -0.020. Composition features do not beat the scalar on this OOS sample (sample-size ceiling applies). Keep them as interpretation, not prediction.

## Caveats
- Same ~1,327-row OOS ceiling: per-cutoff resolution ~±0.5 Sortino, so the signal is the distributional shift across 49 cutoffs.
- The decomposition's primary value is interpretability (which factor drives each regime); predictive lift is the secondary question tested here.