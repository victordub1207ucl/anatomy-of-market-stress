# Scripts — the project in 18 numbered steps

Every analysis is one script, numbered in execution order. Run any of them
from the project root (`latestversion/`) as:

```bash
/opt/miniconda3/bin/python3 -m regime_detection.step01_turbulence
```

or run the whole pipeline with `python3 run_all.py [--full]` from the root.
Each step writes its figures to `figures/<group>/` and a plain-English
results writeup ("checkpoint") to `regime_detection/reports/checkpoints/`.

## A — Build the measurements (steps 01–03)

| step | script | what it does |
|---|---|---|
| 01 | `step01_turbulence.py` | Computes the walk-forward Mahalanobis turbulence index d_t and the binary regime label; writes `outputs/turbulence/turbulence_series.parquet` (the input to everything downstream) + diagnostics notebook 02. |
| 02 | `step02_turbulence_sanity_check.py` | Validates d_t: all 7 named crises must register top-decile turbulence. Fails loudly if not. |
| 03 | `step03_event_time.py` | Event-time conversion: re-paces time by cumulative turbulence; the 94%-kurtosis-reduction result; writes `outputs/event_time/` + notebook 03. |

## B — Predict and evaluate (steps 04–06)

| step | script | what it does |
|---|---|---|
| 04 | `step04_supervised_baseline.py` | Single-shot supervised baseline: 4 frozen models predict imminent stress (purged CV on dev, one OOS pass). |
| 05 | `step05_rbi_explainability.py` | Relevance-Based Importance explainer vs SHAP on the supervised model. |
| 06 | `step06_oos_evaluation.py` | The headline protocol: quarterly walk-forward (22 refits/model, 2021–2026); AUC/Brier/Sortino tables. |

## C — Stress-test every claim (steps 07–13)

| step | script | what it does |
|---|---|---|
| 07 | `step07_ablations.py` | One-knob-at-a-time ablations: which design choices matter. |
| 08 | `step08_rbi_stability.py` | RBI cross-window stability test → the "more stable than SHAP" claim is retired. |
| 09 | `step09_gmm_benchmark.py` | Fair walk-forward GMM benchmark (Hungarian label alignment) — turbulence wins on every model. |
| 10 | `step10_cutoff_robustness.py` | **The key honesty test:** 49 sliding train/test boundaries → the +0.06 headline was the max; median −0.15. |
| 11 | `step11_factor_set_check.py` | 14-factor (spliced) vs 10-factor turbulence — 10-factor confirmed primary. |
| 12 | `step12_ordinal_bucket.py` | The fix: ordinal 5-level severity bucket flips linear-model median lift −0.15 → +0.15. |
| 13 | `step13_bucket_granularity.py` | Bucket robustness: 3/5/10 levels all work — the gain is binary→ordinal, not the count. |

## D — Interpretation & causality (steps 14–18)

| step | script | what it does |
|---|---|---|
| 14 | `step14_turbulence_decomposition.py` | Exact per-factor split of d_t — each crisis's "fingerprint" (Volmageddon = 79% VIX). |
| 15 | `step15_composition_features.py` | Do composition features improve *timing* prediction? (Honest negative.) |
| 16 | `step16_causality.py` | Regime-conditional Granger networks: VIX leads escalation; the causal wiring reorients toward USD in stress. |
| 17 | `step17_crisis_typology.py` | Clusters 38 stress episodes into 3 types; shows the *composition* of the next spike is predictable (p<1e-4). |
| 18 | `step18_signed_turbulence.py` | "Downside turbulence" (signed split of d_t) — clean negative: direction adds nothing. |

`tools/` holds optional utilities (notebook regeneration) that are not part
of the numbered pipeline.


## E — Research extensions (overlay research thread)

| step | script | what it does |
|---|---|---|
| 19 | `step19_threshold_sensitivity.py` | Sweeps the regime-label quantile (q=0.60…0.90) through the 49-cutoff test — shows the binary threshold is a noisy feature and the ordinal bucket dominates it. |
| 20 | `step20_correction_overlay.py` | Calibrated probabilistic correction-timing overlay on the equity factor (isotonic calibration, dev-selected threshold, transaction costs, 49-cutoff robustness). |
| 21 | `step21_event_time_inference.py` | Reruns the correction overlay with the event-time drawdown target vs the calendar target — first test of downstream inference on the event clock. |
| 22 | `step22_msci_benchmark.py` | External-validity: reruns the correction overlay on MSCI World (ACWI) vs the US equity factor — the edge is stronger on the global benchmark. |
| 23 | `step23_absorption_ratio.py` | Absorption ratio (KLPR systemic-fragility signal, `lib/absorption.py`); validates it and tests whether fragility adds predictive value beyond turbulence (marginal). |
| 24 | `step24_cross_market_pooling.py` | Pools one global turbulence signal across 7 regional equity benchmarks (panel + market fixed effects, date-block bootstrap) — the strongest predictive result: 7/7 markets positive, basket Sortino lift +0.36 (92% bootstrap-confident). |
| 25 | `step25_eventtime_pooling.py` | Combines the two leads — event-time target × cross-market pooling. Honest negative: they CONFLICT (a global event clock mismatches heterogeneous local markets); calendar+pooling stays best. |
| 26 | `step26_typology_hedge.py` | **Additive** better-posed target: the regime says *when* to de-risk, the *signed* decomposition says *where* to park. Shows (a) magnitude crisis-typology is direction-blind for hedge choice but the signed signal is not, and (b) bonds win 2020 / gold wins 2022, so a routed/diversified hedge beats any single blind hedge (composition-aware +56% vs blind-bonds, 93% bootstrap-confident; ≈ a static 50/50 split). |
| 27 | `step27_event_clock_interpretation.py` | **Fusion** of the strongest technical result (event-time re-clocking) with the strongest interpretive ones: re-runs composition predictability (step17) and causal drivers (step16) on the **event clock** vs calendar. Composition predictability is a *wash* (0.715→0.714 — a persistence property robust to the clock); the Granger lead structure is **clock-robust** (VIX still leads p 2e-20→3e-10; full ranking preserved, marginal leaders wash out) — a clock-robustness check no prior event-time work performs. |
| 28 | `step28_eventtime_features.py` | First use of **event-time quantities as predictive *features*** (not target/clock) for correction forecasting, vs turbulence-only, same honest harness. Point AUC unchanged (~0.48) but the full event-time block flips the 49-cutoff overlay from −0.087/20%-positive to **+0.089/78%**, drawdown well below B&H. Controls: calendar momentum *hurts* (−0.095) and no subset reproduces it (overfitting flag), **but** it transfers cross-asset to MSCI World (+0.041→+0.100). Promising robustness lead, not a 95%-significant alpha. |
| 29 | `step29_eventtime_feature_robustness.py` | Stress-tests step28: 7-market panel + leave-one-feature-out + COVID-period split. Panel (6/7): event-time features improve the overlay on **5/6 markets** (materially on US/UK/Canada) but mean Δ only +0.066 and not on Japan/Pacific → **directionally real, modest, not uniform**, inside the noise band. |
| 30 | `step30_per_market_event_clocks.py` | Tests the step25 fix: pace **each market by its own** univariate-turbulence event clock instead of one global clock, re-running the pooled overlay. Per-market clocks **recover** the global clock's damage (−0.146→−0.066, 0/7→1/7) → confirms step25's global-vs-local-mismatch diagnosis — **but only to calendar parity** (calendar −0.056); no positive pooled overlay on any clock under the 2020 OOS. Event clock's genuine benefit stays confined to the matched single-asset case (step21). |
| 31 | `step31_cross_universe_typology.py` | **Defends the flagship.** Re-runs the entire composition-predictability pipeline (step17) on a different universe — 9 SPDR US sectors. It **replicates**: 47 sector episodes, k=4 archetypes, run-up→spike cosine **0.777 vs null 0.615, p<0.0001** (vs factors 0.715/0.463). Composition predictability is structural, not a Factor-Lens artefact — kills the n=38 in-sample objection. |
| 32 | `step32_persistence_baseline.py` | **Anticipation vs stickiness.** Settles the sharpest viva objection: is the run-up *anticipating* the spike composition, or just reflecting a *persistent* slow-moving composition? Compares the run-up against carry-forward predictors from earlier windows (paired sign-flip test). On **factors: ANTICIPATION** — run-up beats 1-month-back carry-forward +0.107 (p=0.003) and composition converges monotonically toward the spike (0.54→0.58→0.61→0.72). On **sectors: persistence** dominates (incremental +0.052, p=0.08, marginal). Honest bound: anticipation is clearest on the macro-factor universe; distinctiveness/persistence is what replicates out-of-universe. |
| 33 | `step33_oos_composition.py` | **Composition predictability is OOS.** The run-up→spike test is causal per episode, so restricting it to held-out-era episodes (spike ≥2020-01-01, same window as the timing nulls) gives a genuine OOS forecast. **Factors hold OOS**: 24 episodes, cosine 0.698 vs null 0.475, p<1e-4, and run-up beats carry-forward +0.102 (paired p=0.039). Makes the predictability boundary a like-for-like (both sides OOS) comparison. Sectors: distinctiveness holds OOS, anticipation persistence-marginal. |
| 34 | `step34_examiner_robustness.py` | **Three examiner objections, answered.** (A) **Circularity**: VIX still Granger-leads turbulence rebuilt WITHOUT VIX (9-factor) at p=7e-14, rank 1/10 → genuine lead, not arithmetic. (B) **Persistence floor**: naive 'today's turbulence' scores OOS AUC 0.632 vs supervised 0.712 → harness adds +0.08, the 0.71 anchor read against a 0.63 floor. (C) **Cosine-on-simplex**: predictability holds under Spearman (0.493/0.138) and 1−TV (0.640/0.485), both p<1e-4 → not a metric artefact. |

> **OOS / test-set contract (fixed 2026-06):** the held-out window
> begins **1 January 2020** (`lib/splits.py`: `DEV_END=2019-12-31`,
> `OOS_START=2020-01-01`), deliberately bringing the COVID crash into the test
> set. OOS-dependent results from steps that predate this change should be
> regenerated under the new split.

## Historical note (for readers of the white paper / phase logs)

The project was developed in phases; older documents refer to scripts by
phase number. Mapping: Phase 4→step04, Phase 5→step05, Phase 6→step06,
Phase 7→step07, Phase 9A→step08, 9B→step09, 9C→step10, 9D→step11,
9F→step12 (+13). Robustness artifacts live under `artifacts/robustness/`; the hidden
`.cache/phase9*/` folders retain historical names so expensive cached
results stay reusable.
