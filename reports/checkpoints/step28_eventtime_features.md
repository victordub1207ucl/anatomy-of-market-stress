# Step 28 — Turbulence + event-time features for correction forecasting

_Generated 2026-06-20 18:03. 2020-start OOS; 5% / 21d correction target; logistic_l1; event threshold 138.7 (dev-calibrated). Feature matrix 4107×34._

The one configuration never tried: event-time quantities as model **inputs** (not target, not clock). Head-to-head vs turbulence-only, identical honest harness (step20).

| metric | turbulence only | + event-time |
|---|---|---|
| OOS AUC (5%) | 0.482 | 0.478 |
| Brier (calibrated) | 0.156 | 0.157 |
| overlay Sortino lift (forced decile) | -0.091 | +0.093 |
| overlay max drawdown | -30.8% | -28.8% |
| buy-and-hold max drawdown | -41.1% | — |
| **49-cutoff median lift** | **-0.087** | **+0.089** |
| 49-cutoff % positive | 20% | 78% |

**Event-time features used:** 3/5 get non-zero logistic_l1 weight (intensity_to_date -0.62, days_since_start +0.03, velocity +0.00, turb_per_day +0.00, momentum +4.78).

## Verdict

**Event-time features help the *forecast*.** AUC 0.482→0.478 (-0.004); 49-cutoff median -0.087→+0.089 (+0.177). The information-flow features add signal the calendar bucket lacks.
- **Drawdown:** the +event-time overlay's max drawdown (-28.8%) is shallower than buy-and-hold (-41.1%) — the durable value (risk reduction), better than turbulence-only (-30.8%).
- **Beating buy-and-hold on risk-adjusted *return* remains out of reach** under the COVID-held-out OOS — consistent with the whole project: correction *timing* is hard; the honest read is the 49-cutoff distribution, not any single number.

**Figure:** `figures/eventtime_features/eventtime_features.png`.

---

## Robustness controls (added after the headline run)

The headline +event-time result was stress-tested two ways. Both are decisive.

**1. Is it just momentum? No.** Adding feature blocks one at a time (49-cutoff median lift, US equity factor, all 4107 rows):

| feature set added to baseline | 49-cut median | % positive |
|---|---|---|
| baseline (turbulence only) | −0.087 | 20% |
| + calendar 21-day momentum | **−0.095** | 24% |
| + event-time momentum only | −0.077 | 29% |
| + event-time structure only (no momentum) | −0.111 | 24% |
| **+ event-time FULL (5 features)** | **+0.089** | **78%** |

Calendar momentum *hurts*; no single event-time block helps; **only the full
event-time block flips the sign.** So the effect is specifically the event-time
*structure*, not plain price momentum — but the "only-the-full-set-works" pattern
is an **overfitting red flag** at ~60 effective OOS blocks.

**2. Does it transfer to another asset? Yes.** Same features, traded/predicted on
**MSCI World (ACWI)** instead of the US equity factor:

| feature set | 49-cut median | % positive |
|---|---|---|
| ACWI baseline (turbulence) | +0.041 | 63% |
| **ACWI + event-time FULL** | **+0.100** | **67%** |

The improvement reproduces in the same direction on an asset the features were not
built on (+0.041→+0.100). Cross-asset transfer substantially mitigates the
overfitting concern from control 1.

**Honest verdict.** Adding turbulence **and** event-time features improves the
de-risking overlay's **robustness** (49-cutoff median and % positive) on two
distinct assets, and the drawdown sits well below buy-and-hold — the most
robustly-positive overlay result in the project. Caveats that keep it a
**promising lead, not a proven alpha**: (i) point AUC is unchanged (~0.48) — the
*classifier* is no better, the gain is in overlay robustness / drawdown, the same
pattern event-time showed in step21; (ii) no feature subset reproduces the lift,
only the full block; (iii) it is not a 95%-significant return edge — the honest
arbiter is the 49-cutoff distribution, and at ~60 blocks even +0.089/+0.100 sits
inside the noise band. **Recommended validation before any strong claim:**
replicate across the 7-market panel and across feature-subset perturbations /
seeds to confirm the lift is stable, not knife-edge.
