# Step 40 — Correlation surprise (K&T 2014): forward-looking content

_Generated 2026-07-06 22:25. n=4579 days, walk-forward, no look-ahead._

## A. K&T conditional sort — high-magnitude days, split by correlation surprise

High-magnitude days (top quintile): 916; of which correlation-surprise ratio>1: 193, ratio<=1: 723.

| forward 21d vol (ann.) | ratio>1 | ratio<=1 | diff | block-perm p |
|---|--:|--:|--:|--:|
| equity | 26.4% | 24.1% | +2.3% | 0.3064 |
| factor basket | 10.5% | 9.1% | +1.4% | 0.1714 |
| forward mean turbulence | 19.1 | 15.8 | +3.3 | 0.2724 |

## B. Incremental $R^2$ — beyond persistence and magnitude

- trailing vol only: $R^2$ = 0.408
- + magnitude surprise: 0.444  (+3.64 pp)
- + correlation surprise: 0.444  (**+0.00 pp incremental**)

## Robustness — split rule and horizon (equity forward vol)

| horizon | split | ratio-high | ratio-low | diff | p |
|--:|---|--:|--:|--:|--:|
| 5d | ratio>1 | 27.7% | 24.5% | +3.2% | 0.171 |
| 5d | median | 25.2% | 25.1% | +0.1% | 0.474 |
| 21d | ratio>1 | 26.6% | 24.1% | +2.6% | 0.287 |
| 21d | median | 24.3% | 24.9% | -0.6% | 0.555 |

## Verdict

**The K&T claim does NOT replicate on this panel.** Among equally large moves, unusually-correlated days are followed by only directionally higher forward vol (equity 26.4% vs 24.1%), never significant across horizons (5d/21d) or split rules (ratio>1 / median), and the correlation component adds zero incremental $R^2$ beyond trailing vol and magnitude surprise. This is a boundary of the claim, not a refutation of K&T (2014): their panel (asset-class indices, an earlier era) differs from this 10-factor ETF panel over 2007--2026, so the honest statement is that on *this* universe and period the forward-looking content of turbulence sits in the magnitude component and in persistence, with the correlation component contributing nothing incremental. Consistent with the thesis's recurring shape: the informative axis of the decomposition is *which factors* (composition), not *magnitude-versus-correlation*.
