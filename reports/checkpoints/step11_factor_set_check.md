# CHECKPOINT — Phase 9D: ETF-inception splicing + 14-factor turbulence

**Generated:** 2026-06-18 14:04 UTC

## What was implemented
- **`regime_detection/lib/splicing.py`** — ports the v4 FACTOR_DEFINITIONS and FACTOR_FALLBACKS tables verbatim (see file:line citation in the module docstring).  Exposes `splice_factor()` (return-level stitch) and `build_spliced_factor_panel()` (full 14-factor reconstruction).  Documents the constant `UNSPLICEABLE_FACTORS` for the three factors with no v4 fallback (`em_credit`, `short_vol`, `low_risk`).
- **`regime_detection/tests/test_splicing.py`** — 14 tests, all green: continuity at splice date, primary-precedence on overlap, pre/post-splice match source, panel anchor = 100, missing-primary returns NaN, FACTOR_DEFINITIONS = 14 entries, UNSPLICEABLE constants match the cache's no-fallback set.
- **`regime_detection/step11_factor_set_check.py`** — driver: computes 14-factor turbulence, runs the Phase-2 crisis sanity check on it, compares against 10-factor turbulence in the common range, optionally reruns Phase 6 on the new target.

## What tests pass
- **150/150 total** (was 136 before Phase 9D — 14 new splicing tests).

## Factor coverage — what splicing achieves

The cached parquet already has v4 splicing applied.  The three unspliceable factors (em_credit, short_vol, low_risk) have no fallback in v4's table and Phase 9D does not invent new ones.  The achievable 14-factor coverage starts at the latest of these primary inception dates:

| factor      | first_valid         |   n_finite | splice_status                 |
|:------------|:--------------------|-----------:|:------------------------------|
| equity      | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| rates       | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| credit      | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| commodities | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| em_equity   | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| em_credit   | 2007-12-20 00:00:00 |       4610 | primary only (no v4 fallback) |
| fx_usd      | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| short_vol   | 2011-10-05 00:00:00 |       3655 | primary only (no v4 fallback) |
| inflation   | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| momentum    | 2007-03-02 00:00:00 |       4814 | v4 splicing applied           |
| value       | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |
| quality     | 2005-12-07 00:00:00 |       5125 | v4 splicing applied           |
| low_risk    | 2011-10-21 00:00:00 |       3643 | primary only (no v4 fallback) |
| vix         | 2005-01-04 00:00:00 |       5362 | v4 splicing applied           |

## 14-factor turbulence — crisis sanity check

| crisis            |   n_days_in_window | max_d     |   top_decile_days | verdict        |
|:------------------|-------------------:|:----------|------------------:|:---------------|
| 2008 GFC          |                  0 | —         |                 0 | SKIP (no data) |
| 2011 Euro / US dg |                  0 | —         |                 0 | SKIP (no data) |
| 2015 oil / China  |                146 | 154.460   |                43 | PASS           |
| 2018 Vol-mageddon |                 19 | 11746.103 |                 6 | PASS           |
| 2018 Q4 sell-off  |                 63 | 49.395    |                 6 | PASS           |
| 2020 COVID        |                 72 | 330.568   |                43 | PASS           |
| 2022 inflation    |                251 | 122.202   |                51 | PASS           |

## 10-factor (Phase 2) reference — crisis sanity check

| crisis            |   n_days_in_window |   max_d |   top_decile_days | verdict   |
|:------------------|-------------------:|--------:|------------------:|:----------|
| 2008 GFC          |                146 | 329.448 |                85 | PASS      |
| 2011 Euro / US dg |                 65 |  49.108 |                12 | PASS      |
| 2015 oil / China  |                146 |  56.994 |                11 | PASS      |
| 2018 Vol-mageddon |                 19 | 165.408 |                 5 | PASS      |
| 2018 Q4 sell-off  |                 63 |  59.87  |                10 | PASS      |
| 2020 COVID        |                 72 | 365.604 |                50 | PASS      |
| 2022 inflation    |                251 | 139.168 |                91 | PASS      |

## 14-factor vs 10-factor turbulence comparison

|                        | value               |
|:-----------------------|:--------------------|
| n_common               | 3138                |
| pearson_r              | 0.08524381158964801 |
| spearman_r             | 0.7689414443035536  |
| top_decile_overlap_pct | 0.45707656612529    |
| first_common           | 2013-10-25          |
| last_common            | 2026-04-14          |

### Persistence test (Skulls Table 1) on the 14-factor series

|   horizon |   E[d|spike] |   E[d] |   ratio |
|----------:|-------------:|-------:|--------:|
|         1 |       62.902 | 12.905 |   4.874 |
|         5 |       46.264 | 12.905 |   3.585 |
|        10 |       37.065 | 12.905 |   2.872 |
|        20 |       26.499 | 12.905 |   2.053 |

### Persistence test on the 10-factor reference

|   horizon |   E[d|spike] |   E[d] |   ratio |
|----------:|-------------:|-------:|--------:|
|         1 |       29.829 |  8.884 |   3.358 |
|         5 |       27.964 |  8.884 |   3.148 |
|        10 |       26.803 |  8.884 |   3.017 |
|        20 |       24.186 |  8.884 |   2.723 |

![Phase 9D factor coverage and turbulence overlay](figures/thesis_v5/fig10_factor_coverage.png)

## Phase 6 quarterly walk-forward — 14-factor target

Binary target rebuilt from 14-factor turbulence (threshold = 90th-pct of 14-factor dev turbulence).  Feature matrix uses the existing 10-factor columns + the new 14-factor turbulence-derived features.  Quarterly walk-forward refits with the same four challenger models as Phase 6.

|                     |   n_oos |   pos_rate |   auc |   brier |   sortino_strategy |   sortino_bh |   sortino_lift |   max_dd_strategy |   max_dd_bh |   ann_ret |
|:--------------------|--------:|-----------:|------:|--------:|-------------------:|-------------:|---------------:|------------------:|------------:|----------:|
| logistic_l1         |    1580 |       0.58 | 0.637 |   0.261 |              0.665 |         0.79 |         -0.126 |            -0.31  |      -0.411 |     0.1   |
| random_forest       |    1580 |       0.58 | 0.669 |   0.237 |              0.524 |         0.79 |         -0.266 |            -0.445 |      -0.411 |     0.091 |
| linear_svm          |    1580 |       0.58 | 0.638 |   0.236 |              0.593 |         0.79 |         -0.198 |            -0.38  |      -0.411 |     0.092 |
| logistic_elasticnet |    1580 |       0.58 | 0.631 |   0.252 |              0.824 |         0.79 |          0.034 |            -0.373 |      -0.411 |     0.114 |

### Comparison to Phase 6 10-factor headline

| model               |   auc |   brier |   turb_sortino_lift |   max_dd_strategy |
|:--------------------|------:|--------:|--------------------:|------------------:|
| random_forest       | 0.708 |   0.156 |              -0.139 |            -0.515 |
| logistic_l1         | 0.712 |   0.137 |              -0.182 |            -0.344 |
| linear_svm          | 0.578 |   0.166 |              -0.071 |            -0.411 |
| logistic_elasticnet | 0.713 |   0.137 |              -0.145 |            -0.344 |

Median Sortino lift across the four challengers: 14-factor = -0.162 vs 10-factor = -0.142 (Δ = -0.020).  Within noise — no thesis-narrative change required.

## Verdict

**Mixed verdict.**  14-factor crisis sanity (5/7 pass, 2 skipped, 0 fail) and correlation (0.085) sit short of the 'viable primary' bar.  Recommend keeping 10-factor primary and citing 14-factor as a sensitivity check.

## Deviations from spec
1. **The brief assumed v4 splicing extends all 14 factors to 2005-12.  It does not.**  v4's FACTOR_FALLBACKS table covers `rates, credit, commodities, fx_usd, quality, momentum` (6 of 8 needing extension) but is silent on the three factors in `UNSPLICEABLE_FACTORS` (em_credit, short_vol, low_risk).  Phase 9D ports the v4 tables verbatim and documents these limits in the module docstring; we do not invent new proxies (a future revision could).
2. **No yfinance download.** The cached `data/cache/factor_prices.parquet` was already built by the v4 loader with splicing applied; we read it directly rather than re-download.  The splicing module is ready for offline use against a fresh raw-ticker panel when needed.

## Open questions for the user
- Invent new pre-inception proxies for `em_credit`, `short_vol`, `low_risk` (e.g. ELD pre-2007 for EM debt; 1/VIX or VXX-inverse pre-2011 for short_vol; a Ken-French min-vol portfolio for low_risk)?  Adds methodology surface but would close the 2005–2011 gap on these three factors.
- For the thesis: state that 10-factor remains primary because of 2008/2011 coverage, and present 14-factor results as a robustness check on the post-2013 window?
- Per the Phase 9 brief, Phase 9E (narrative reframing in light of 9A/9B/9C/9D results) is the natural next step.
