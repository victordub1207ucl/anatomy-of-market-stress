# Step 39 — Calibrated de-risking: does timing beat a static hedge?

_Generated 2026-06-29 00:30._

Pre-registered, FL-only, walk-forward, leak-free. Hedge sleeve = cash (BIL).

## Test period (2015–2026, COVID + 2022 held in): dynamic vs B&H vs static hedge

| thr | τ | time-in-mkt | CAGR dyn/BH/stat | Sortino dyn/BH/stat | maxDD dyn/BH/stat | Calmar dyn/BH/stat |
|--:|--:|--:|--|--|--|--|
| 3% | 0.68 | 98% | 10.9%/13.3%/13.2% | 0.90/0.97/0.97 | 31.0%/33.7%/33.2% | 0.35/0.40/0.40 |
| 5% | 0.47 | 97% | 13.9%/13.3%/13.0% | 1.23/0.97/0.97 | 26.5%/33.7%/32.9% | 0.53/0.40/0.40 |
| 8% | 0.43 | 98% | 11.8%/13.3%/13.2% | 0.97/0.97/0.97 | 24.5%/33.7%/33.2% | 0.48/0.40/0.40 |

## The decisive test — does TIMING beat a static hedge of equal average exposure?

| thr | Δ Calmar (dyn−static) 95% CI | Δ Sortino (dyn−static) 95% CI | verdict |
|--:|--|--|--|
| 3% | [-0.30, +0.05] | [-0.35, +0.10] | tie — no timing edge over static |
| 5% | [-0.21, +0.66] | [-0.21, +0.69] | tie — no timing edge over static |
| 8% | [-0.26, +0.20] | [-0.29, +0.22] | tie — no timing edge over static |

## Verdict

**Honest null (pre-registered).** The calibrated signal cuts drawdown relative to buy-and-hold, but does **not** beat a static hedge of equal average exposure on risk-adjusted terms — i.e. the value is lower exposure, not timing. This is the pre-registered kill outcome and is reported as the headline.
