# Step 34 — Examiner-grade robustness

_Generated 2026-06-23 18:47._

## A. VIX-in-turbulence circularity

Driver Granger ranking (BH-FDR adjusted $p$) on turbulence built WITH vs WITHOUT VIX. If VIX still leads the nine-factor (VIX-free) index, the lead is economic, not arithmetic.

| factor | with VIX (10f) | WITHOUT VIX (9f) |
|---|--:|--:|
| vix | 2.4e-20 | 7.0e-14 |
| quality | 5.9e-10 | 3.4e-09 |
| equity | 9.8e-10 | 3.4e-09 |
| credit | 5.5e-09 | 4.0e-08 |
| value | 5.5e-09 | 3.1e-08 |
| em_equity | 8.9e-07 | 1.4e-06 |
| rates | 1.2e-02 | 1.4e-02 |
| commodities | 1.7e-02 | 1.4e-02 |
| inflation | 8.4e-01 | 6.3e-01 |
| fx_usd | 8.7e-01 | 8.0e-01 |

**Verdict — GENUINE.** On the VIX-free nine-factor index, VIX returns Granger-lead escalation at adjusted p = 7.0e-14 (rank 1 of 10). Because VIX leads an index it is no longer part of, the lead is genuine economic precedence, not mechanical co-determination — the circularity objection is answered.

## B. Persistence baseline behind entry AUC ≈ 0.71

OOS positive base rate = 86% (dev 56%). AUC of trivial predictors vs the supervised harness:

| predictor | dev AUC | OOS AUC |
|---|--:|--:|
| naive: turbulence level | 0.617 | 0.632 |
| naive: turbulence percentile | 0.551 | 0.607 |
| **supervised (turbulence features)** | — | **0.712** |

**Verdict.** A trivial 'today's turbulence level' predictor scores OOS AUC 0.632; the supervised harness scores 0.712 (Δ = +0.080). The harness adds a non-trivial margin over naive persistence, so the entry signal is not merely autocorrelation.

## C. Cosine-on-simplex robustness

Composition vectors carry a small negative mass (fraction of entries < 0: 13.4%; mean negative magnitude 0.0081), so they are near-, not strictly-, non-negative. The permutation null already absorbs the structurally high baseline; the result also holds under two non-cosine metrics:

| similarity | observed | null | p |
|---|--:|--:|--:|
| cosine | 0.715 | 0.463 | <0.0001 |
| Spearman | 0.493 | 0.138 | <0.0001 |
| 1 - total variation | 0.640 | 0.485 | <0.0001 |

**Verdict — ROBUST.** The run-up→spike predictability clears the permutation null under all three metrics, so it is not an artefact of cosine on a near-simplex.

