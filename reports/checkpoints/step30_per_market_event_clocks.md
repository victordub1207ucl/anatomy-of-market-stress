# Step 30 — Per-market event clocks: fixing the step25 negative?

_Generated 2026-06-21 10:11 UTC. 7-market pooled overlay, 5% correction, 10bps, market fixed effects, date-block bootstrap. Global clock thr 139 (purge 50d); per-market clocks calibrated to ~21d each (pooled purge 111d)._

Step25 showed the event clock *fails* when pooled because one **global** clock is imposed on heterogeneous local markets. This tests the diagnosed fix: pace **each market by its own** univariate-turbulence clock.

| arm | basket lift | 95% CI | P(>0) | markets+ |
|---|---|---|---|---|
| calendar (step24) | -0.056 | [-0.509, +0.293] | 34% | 1/7 |
| global clock (step25) | -0.146 | [-0.633, +0.186] | 19% | 0/7 |
| **per-market clocks** | -0.066 | [-0.662, +0.379] | 32% | 1/7 |

Per-market overlay lift (calendar / global / per-market):
- US: -0.034 / -0.083 / -0.110
- Europe: -0.077 / -0.166 / -0.098
- Japan: -0.268 / -0.316 / -0.208
- UK: -0.015 / -0.027 / +0.048
- Pacific: -0.089 / -0.198 / -0.092
- Canada: +0.130 / -0.043 / -0.011
- Korea: -0.099 / -0.201 / -0.151

## Verdict

**PARTIAL — DIAGNOSIS CONFIRMED, NO NET EDGE.** Per-market clocks recover the global clock's damage — basket lift -0.146 (0/7, P>0=19%) → **-0.066** (1/7, P>0=32%), back to **calendar parity** (-0.056). This **confirms step25's diagnosis**: the pooled event-clock failure was the global-vs-local mismatch — pacing each market by its own clock removes it. **But** the event clock buys *nothing over a plain calendar horizon* once you go local, and all three arms are negative (≤1/7 markets positive): under the 2020 OOS there is **no positive pooled overlay on any clock** (consistent with step24's COVID-driven collapse). So local clocks *stop the bleeding* but don't create an edge — the event clock's genuine benefit stays confined to the matched single-asset case (step21).

**Figure:** `figures/per_market_event_clocks/per_market_event_clocks.png`. Factor Lens intact (one global signal for features; clocks are per-market only for the target horizon).
