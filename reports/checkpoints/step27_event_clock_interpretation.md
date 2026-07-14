# Step 27 — Interpretation on the event clock

_Generated 2026-06-20 17:46. Event threshold 138.7 (dev-calibrated, mean ~21 trading days); 38 episodes; 260 event-grid observations._

Fuses the strongest technical result (event-time re-clocking) with the strongest interpretive results (composition predictability, causal drivers) by measuring both on the event clock vs the calendar.

## A — Composition predictability: does the event clock sharpen it?

| clock | run-up | mean cosine | null | p |
|---|---|---|---|---|
| calendar (21 days) | 21d | 0.715 | 0.463 | 0.0000 |
| event (1 event-unit) | 20.7d | 0.714 | 0.466 | 0.0000 |

Run-up length sweep (mean run-up days → mean cosine):
- calendar: 10d→0.713, 21d→0.715, 42d→0.694, 63d→0.689
- event:    9.9d→0.712, 20.7d→0.714, 40.9d→0.717, 60.4d→0.712

**Verdict A — a wash.** The event-clock run-up predicts spike composition about as well as the calendar run-up (Δcosine -0.001); both remain highly significant (p≈0.000). The predictability is a property of composition *persistence*, not of the clock — which is itself worth stating: the result is robust to how time is measured.

## B — Causal drivers of turbulence: calendar vs event clock

| factor | calendar p(adj) | sig | event p(adj) | sig |
|---|---|---|---|---|
| vix | 2.4e-20 | ✓ | 3.1e-10 | ✓ |
| quality | 5.9e-10 | ✓ | 5.3e-04 | ✓ |
| equity | 9.8e-10 | ✓ | 6.6e-04 | ✓ |
| credit | 5.5e-09 | ✓ | 4.6e-05 | ✓ |
| value | 5.5e-09 | ✓ | 5.1e-04 | ✓ |
| em_equity | 8.9e-07 | ✓ | 5.3e-04 | ✓ |
| rates | 1.2e-02 | ✓ | 7.1e-01 | · |
| commodities | 1.7e-02 | ✓ | 8.1e-01 | · |
| inflation | 8.4e-01 | · | 2.9e-01 | · |
| fx_usd | 8.7e-01 | · | 1.9e-01 | · |

- **VIX still leads** on the event clock (p 2.4e-20 → 3.1e-10).
- Driver ranking is preserved across clocks (Spearman of −log p = 0.73); 6/10 factors are significant on **both** clocks.

**Verdict B — the causal lead structure is clock-robust.** The headline finding (volatility leads turbulence escalation; the usual suspects follow; fx/inflation do not lead) survives re-estimation on the event grid, where observations carry uniform information rather than uniform calendar spacing. This *strengthens* the causal claim: it is not an artefact of the calendar.

**Figures:** `figures/event_clock_interpretation/predictability_clocks.png`, `drivers_clocks.png`.

> Thesis use: this is the chapter that points the two strongest tools at each other. Whether the event clock *sharpens* or merely *preserves* the interpretation, the result is a genuine fusion — and a clock-robustness check that no prior event-time work has performed (CKT 2023 stop at the return distribution; they never re-run inference on the clock).
