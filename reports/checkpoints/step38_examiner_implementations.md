# Step 38 — Connectedness, partial-R^2 Granger, and CoDA distinctiveness

_Generated 2026-06-28 23:14._

## A. Diebold-Yilmaz connectedness (implemented network claim)

**Calm regime** (n=4158): total connectedness 61.1%. Net transmitters (NET>0) vs receivers (NET<0):
  - biggest net *receiver* (sink): commodities -19.5%, fx_usd -16.6%, credit -10.8%
  - biggest net *transmitter*: equity +25.2%, value +21.5%, quality +11.9%
  - USD net directional: **-16.6%** (net receiver / sink)

**Turbulent regime** (n=462): total connectedness 64.8%. Net transmitters (NET>0) vs receivers (NET<0):
  - biggest net *receiver* (sink): fx_usd -25.4%, inflation -23.1%, commodities -19.0%
  - biggest net *transmitter*: equity +31.4%, value +28.9%, quality +22.7%
  - USD net directional: **-25.4%** (net receiver / sink)

**Verdict — CONFIRMS the dollar-sink claim.** Under stress the dollar moves toward (or further into) net-receiver status (NET -16.6% calm -> -25.4% turbulent), so the qualitative 'central sink' reading is backed by a cardinal connectedness decomposition, not only by a rank-Granger picture.


![connectedness](figures/causality/connectedness_regimes.png)

## B. Partial-$R^2$ Granger effect measure (sample-size-invariant)

Incremental $R^2$ for one-step turbulence escalation (caused variable $d\log d_t$, 5 lags, matching the Granger setup of Table 4.3), factor lags over a turbulence-own-lag baseline (partial $R^2$; does not scale with sample size):

| rank | factor | partial $R^2$ |
|---|---|--:|
| 1 | vix | 1.492% |
| 2 | quality | 0.782% |
| 3 | equity | 0.756% |
| 4 | credit | 0.694% |
| 5 | value | 0.689% |
| 6 | em_equity | 0.530% |
| 7 | commodities | 0.204% |
| 8 | rates | 0.185% |
| 9 | inflation | 0.007% |
| 10 | fx_usd | 0.007% |

**Verdict — confirms the ordinal ranking.** The cardinal partial-$R^2$ order is vix, quality, equity, credit, value, ... — VIX first, then the quality/equity/credit/value block, then EM, then the macro tail (rates, commodities, inflation, USD) near zero. This reproduces the Table 4.3 F-ranking almost exactly (only the negligible rates/commodities pair, both $\approx$0.2\%, swaps), so the ranking the thesis relies on now carries a sample-size-invariant cardinal complement that agrees with the F-ordering rather than resting on F alone.

## C. Aitchison (CoDA) log-ratio distinctiveness test

On the centred-log-ratio (Aitchison) coordinates, run-up vs spike:
- Aitchison similarity (neg. mean distance): observed -12.842 vs null -15.811, p = <0.0001
- cosine on clr coordinates: observed 0.445 vs null 0.150, p = <0.0001

**Verdict — distinctiveness holds in the principled simplex geometry.** Re-run in Aitchison (log-ratio) geometry — the metric a CoDA-literate examiner asks for — the run-up still resembles its own spike far more than a permuted one, so the distinctiveness is not an artefact of cosine on near-simplex vectors; CoDA is now a primary check, not a footnote.

