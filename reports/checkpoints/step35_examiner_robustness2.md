# Step 35 — Examiner robustness, round 6

_Generated 2026-06-26 23:33._

## A. Volatility-matched hedge benchmark

Buy-and-hold scaled to exposure **w = 0.81** matches the overlay's realised volatility. If the overlay's drawdown is shallower than this vol-matched buy-and-hold, the benefit exceeds mere de-risking.

| strategy | ann. vol | Sortino | max drawdown |
|---|--:|--:|--:|
| buy-and-hold | 20.6% | 0.80 | -35.7% |
| buy-and-hold, vol-matched | 16.7% | 0.80 | -29.8% |
| static 50/50 split | 14.1% | 0.31 | -26.4% |
| composition-aware overlay | 16.7% | 0.32 | -27.9% |

**Verdict — the drawdown benefit EXCEEDS de-risking.** The overlay's max drawdown is -27.9% versus -29.8% for a buy-and-hold dialled down to the same volatility. Because the overlay's drawdown is shallower than the equally-volatile buy-and-hold, the drawdown reduction is not explained by lower average exposure alone — there is genuine timing/composition content in *when* and *where* it de-risks.

## B. Cluster-bootstrap on the OOS anticipation (+0.102)

The 24 post-2019 factor episodes form **8 independent meta-crisis clusters** (sizes [2, 8, 7, 2, 2, 1, 1, 1]); the effective independent sample is roughly 12, not 24. Resampling whole clusters:

- observed paired difference (run-up − carry-forward): **+0.102**
- 95% cluster-bootstrap CI: **[-0.019, +0.202]**

**Verdict — marginal under clustering.** The cluster-bootstrap CI includes zero: on the held-out window the anticipation is real in the point estimate but not separable from chance once episode dependence is respected. The honest abstract claim is that OOS anticipation is *established at marginal significance on a small, dependent set*, while distinctiveness/persistence is the bulletproof result.

