# Step 26 — Typology-conditioned hedge selection (composition-aware de-risking)

_Generated 2026-06-23 18:26. OOS = 2020-01-02 … 2026-04-14 (1578 trading days), fixed 2020-01-01 constraint._

## The question (better-posed than 'time the drawdown')

The turbulence *regime* already says **when** to de-risk. This step asks **where to park** — and shows the right hedge depends on the *kind* of stress. The 2020-start OOS hands us both archetypes as held-out events:

| crisis | type | bonds (TLT) | gold (GLD) | cash | equity (SPY) |
|---|---|---|---|---|---|
| 2020 COVID (Feb-Apr) | equity-led | +15% | +5% | +0% | -13% |
| 2022 inflation (full yr) | rates-led | -31% | -1% | +1% | -18% |

→ **The hedge that works flips.** Long Treasuries are the perfect COVID hedge and a disaster in 2022; gold is the reverse. A single blind hedge is therefore fragile across regimes.

## Finding 1 — magnitude typology is direction-blind

Does a routing signal predict *bonds − gold* forward return on de-risk days? (Spearman ρ, high-turbulence subset)

| signal | dev ρ (p) | oos ρ (p) |
|---|---|---|
| magnitude rates-share (step17 typology) | -0.025 (0.498) | -0.008 (0.826) |
| **signed** duration trend (step18 spirit) | -0.142 (0.000) | -0.117 (0.002) |

The **magnitude** crisis-typology fingerprint carries ~zero information for hedge choice — both 2020 and 2022 load heavily on rates/inflation, so |contribution| cannot tell a flight-to-safety rally from a rates crash. The **signed** decomposition can. This is a concrete reason the signed-turbulence layer (step18) earns its keep.

## Finding 2 — out-of-sample hedge performance (2020-01-01 +)

Equity (SPY) protected; de-risk gate = turbulence > dev 75th pct; 5 bps per switch.

| strategy | total | CAGR | vol | Sharpe | Sortino | max DD |
|---|---|---|---|---|---|---|
| buy_and_hold | +136.1% | +14.7% | 20.5% | 0.77 | 0.95 | -33.7% |
| blind_cash | +7.7% | +1.2% | 9.7% | 0.17 | 0.19 | -17.7% |
| blind_bonds | -18.5% | -3.2% | 16.6% | -0.11 | -0.16 | -38.4% |
| blind_gold | +79.4% | +9.8% | 16.2% | 0.66 | 0.94 | -20.1% |
| split_bond_gold | +20.9% | +3.1% | 14.1% | 0.29 | 0.41 | -25.1% |
| composition_aware | +27.4% | +3.9% | 16.7% | 0.31 | 0.44 | -25.1% |

## Finding 3 — routing vs the natural comparators

- **composition_aware − blind_bonds**: cumulative +56.2%; date-block bootstrap P(daily edge>0) = **93%**, 90% CI [-0.29, +5.45] bps/day.
- **composition_aware − split_bond_gold**: cumulative +5.3%; date-block bootstrap P(daily edge>0) = **56%**, 90% CI [-2.01, +2.23] bps/day.

## Verdict

- **Magnitude typology cannot route hedges; the signed decomposition can** — a clean, falsifiable result that motivates step18 directly.
- But the edge over a naive bonds hedge is **not 95%-significant** (P≈93%); with only ~2 archetype crises in the OOS the honest reading is that the value comes mostly from **not concentrating in the single hedge that fails** (diversification / routing), not from precise crisis typing — composition_aware is **statistically indistinguishable from a static 50/50 bond-gold split** (P≈56%).
- Always-gold won this particular OOS outright, but that is **hindsight hedge selection** (gold's 2020-2026 secular run); ex ante you cannot know which single sleeve will dominate, which is exactly why the routed / diversified hedge is the defensible choice.
- **Additive, not a replacement:** the turbulence detector, decomposition and supervised models are unchanged — this is a hedge-selection layer bolted on top of the existing de-risk gate.

**Figures:** `figures/typology_hedge/oos_equity_curves.png`, `crisis_archetype_hedges.png`, `routing_timeline.png`.
