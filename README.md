# The Anatomy of Market Stress

**Decomposing, Explaining, and Acting on Mahalanobis Financial Turbulence**

MSc thesis research (UCL Computer Science).
Research report: [`docs/Anatomy_of_Market_Stress_Report.pdf`](docs/Anatomy_of_Market_Stress_Report.pdf).

> **The anatomy of a market crisis is predictable, but its trigger is not.**
> That stress is forming, what kind of stress it is, and which factors lead it are
> forecastable; the timing of the precipitating shock — and any return edge from acting
> on it — is not.

## Headline results

| Finding | Status |
|---|---|
| Event time cuts equity excess kurtosis 94% (8.81 → 0.57) | Rock-solid; replicates CKT (2023) |
| Every crisis has a distinct factor fingerprint (2018 "Volmageddon" = 79% VIX, unprompted) | Strong |
| Stress *composition* is distinct & persistent (cosine 0.715 vs 0.463 null, p < 10⁻⁴) | Strong; novel |
| The run-up *anticipates* the spike (cluster-robust p = 0.022; placebo p = 0.006) | Strong in-sample; directional OOS |
| VIX Granger-leads escalation — survives a VIX-free rebuild of the index (p = 7×10⁻¹⁴) | Strong (descriptive) |
| Timing equity corrections | Near chance (AUC ≈ 0.48) — an informative null |
| Signed "downside" turbulence beats total | Rejected (loss/gain symmetric) |
| Pre-registered de-risking overlay | Cuts max drawdown 33.7% → 26.5%; no timing edge over a matched static hedge |

The full graded findings ledger, with every negative result, is in the paper.

## Reproducing everything

```bash
pip install -r requirements.txt
python run_all.py --core     # the nine core pipeline scripts (~10 min)
python run_all.py --full     # + every robustness and side experiment (~30 min)
python thesis_numbers.py     # figures quoted in the thesis that no single step produces
python thesis_figures.py     # the thesis figures, into figures/thesis/
```

`run_all.py` produces the pipeline's own results. Two further scripts cover the rest of
what the write-up quotes: `thesis_numbers.py` re-derives the values assembled across steps
(the matched-exposure blend of Table 6.4, the Visibility bootstrap interval, the
date-block bootstrap on the routing correlation, the silhouette profile, the meta-crisis
grouping and the gate counts), and `thesis_figures.py` renders the figures as they appear
in the thesis. Pass `--slow` to `thesis_numbers.py` to re-derive the sector-universe
archetypes, and `--out DIR` to `thesis_figures.py` to write elsewhere.

## The nine core scripts

Everything in the report comes from nine scripts in `regime_detection/`, listed here in
the report's narrative order. Script numbers record the order experiments were originally
built — the project's lab notebook — so they are stable identifiers, kept unchanged; the
gaps in the sequence are the robustness and side experiments, all in `extras/`.

| № | Script | What it produces |
|--:|---|---|
| 1 | `step02_turbulence_sanity_check` | the walk-forward index and its validation |
| 2 | `step14_turbulence_decomposition` | the exact per-factor split and crisis fingerprints |
| 3 | `step16_causality` | what leads the build-up (lead–lag and spillovers) |
| 4 | `step17_crisis_typology` | stress episodes, the three crisis types, and the visibility test |
| 5 | `step32_persistence_baseline` | the run-up vs. its own past (the sharpening ladder) |
| 6 | `step36_cluster_robust_trend` | the sharpening tested honestly, at the crisis level |
| 7 | `step37_confound_and_placebo` | the placebo and fixed-covariance controls |
| 8 | `step06_oos_evaluation` | the supervised timing test (the null) |
| 9 | `step39_derisk_timing` | the pre-registered de-risking experiment |

Each writes a plain-English report to `reports/checkpoints/` under the same name. Shared machinery lives in
`regime_detection/lib/`, covered by the test suite in `regime_detection/tests/`.

## Repository map

| Path | Contents |
|---|---|
| `run_all.py` | single entry point: tests → analyses → results index |
| `regime_detection/` | the nine core scripts + `lib/` + `tests/` |
| `regime_detection/extras/` | every robustness check and side experiment (runs with `--full`) |
| `data/` | processed price panels (public ETF data; the pipeline runs offline) |
| `reports/checkpoints/` | one plain-English writeup per experiment |
| `figures/` | all generated figures |
| `docs/` | the research report (`Anatomy_of_Market_Stress_Report.pdf`) |

## The no-look-ahead contract

All statistics are strictly walk-forward (trailing windows through *t−1* only), and the
contract is **enforced by code, not assertion**: an automated shock-test suite (200+
checks) injects an extreme value at date *t* and requires every output at *t′ < t* to be
byte-identical. All out-of-sample evaluation begins 2020-01-01 — the COVID crash is
deliberately inside the test set. This discipline follows an audit of a predecessor
pipeline that found 48 look-ahead defects; that version was retracted and rebuilt.

## Method in one paragraph

Turbulence is the Mahalanobis distance of the day's ten-factor return vector
(Kritzman–Li 2010), computed on a trailing 10-year window with Ledoit–Wolf shrinkage. It
admits an exact per-factor decomposition *cᵢ = δᵢ(Ω⁻¹δ)ᵢ* (summing to the index to ~10⁻¹⁶),
which yields per-crisis fingerprints, a crisis typology, and the composition-predictability
tests; regime-conditional Granger causality plus a Diebold–Yılmaz connectedness
decomposition describe what leads stress and how the factor network rewires; a frozen
four-model supervised harness (purged time-series CV, isotonic calibration) tests what any
of it can predict out of sample.

## Author

Victor Dublin — UCL Department of Computer Science · `ucabvdu@ucl.ac.uk`
Academic supervisor: Prof. Philip Treleaven (UCL)
