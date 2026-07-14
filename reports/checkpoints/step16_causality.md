# CHECKPOINT — regime-conditional causality (clean v5 port)

**Generated:** 2026-06-18 13:51 UTC

## What this is
A descriptive characterisation of the factor causal structure, conditioned on v5's **causal** turbulence regime — ports the strong idea from v4's `causal.py` (regime-conditional Granger with contiguous-episode segmentation + Fisher combination) onto leak-free foundations. Interpretation/causality contribution, not an OOS predictor.

## Method note
At q<0.05 almost every directed edge is 'significant' (the sample is large), so significance is uninformative as a network filter. We therefore rank edges by **Granger strength** (-log10 BH-p) and report the strongest links per regime, plus a **rewiring** view: edges that rise most in *within-regime rank* from calm to turbulent (rank-based to neutralise the ~5x calm/turbulent sample-size asymmetry).

## 1. Factor causal network by regime

- **Calm — top causal hubs (out-degree):** equity (2), fx_usd (2), value (2); strongest links: rates→inflation, inflation→rates, value→em_equity, value→inflation, fx_usd→rates.
- **Turbulent — top hubs:** inflation (3), equity (2), value (2); strongest links: credit→fx_usd, inflation→fx_usd, value→fx_usd, vix→em_equity, equity→fx_usd.
- **Stress rewiring (links strengthening most in turbulence):** inflation→credit, credit→fx_usd, value→fx_usd, equity→rates, vix→quality, credit→rates.

![networks](../../../figures/causality/regime_causal_networks.png)

## 2. What drives turbulence escalation

Factor returns Granger-causing the change in log-turbulence (full sample, BH-FDR q<0.05), ranked by strength:

| factor      |   p_value_raw |   p_value_adj |
|:------------|--------------:|--------------:|
| vix         |   4.99242e-21 |   4.99242e-20 |
| quality     |   1.62968e-10 |   8.14842e-10 |
| equity      |   5.06067e-10 |   1.68689e-09 |
| credit      |   3.03814e-09 |   7.59536e-09 |
| value       |   4.10967e-09 |   8.21933e-09 |
| em_equity   |   8.55269e-07 |   1.42545e-06 |
| rates       |   0.00958688  |   0.0136955   |
| commodities |   0.0213749   |   0.0267187   |

![drivers](../../../figures/causality/turbulence_drivers.png)

## Caveats
- Granger causality is predictive lead-lag, not structural intervention causality. The PC-algorithm DAG (v4's `estimate_dag`) was **not** ported: `causal-learn` is not installed — offer to add it as a structural complement.
- Regime labels are causal, but the Granger estimation is full-sample within each regime — a stylised-fact characterisation, deliberately not a walk-forward predictor.