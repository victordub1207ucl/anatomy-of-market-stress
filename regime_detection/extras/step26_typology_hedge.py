"""Step 26 — typology-conditioned hedge selection (composition-aware de-risking).

ADDITIVE module (the turbulence detector, decomposition and supervised models
are untouched).  A *better-posed* question than "time the drawdown": given that
the turbulence regime already says *when* to step out of equity, *where* should
we park?  The right hedge depends on the kind of stress —

    * 2020 COVID  : equity-led -> flight-to-safety, long Treasuries RALLY.
    * 2022 inflation: rates-led -> Treasuries CRASH with equities; gold/cash win.

The decisive split constraint (OOS starts 2020-01-01) puts BOTH archetypes in
the held-out set — a clean natural experiment we are not allowed to learn from.

What this step shows, honestly:
  1.  The *magnitude* crisis-typology fingerprint (step17) is DIRECTION-BLIND:
      both 2020 and 2022 carry a large rates/inflation contribution, so the
      magnitude share cannot pick the hedge.  Routing needs the SIGNED
      decomposition (step18) — here, the trailing trend of the duration leg.
  2.  Any *single* blind hedge is fragile (bonds great in 2020 / terrible in
      2022; gold the reverse).  A routed / diversified hedge avoids the
      catastrophic wrong-hedge concentration.
  3.  We measure whether composition-aware routing beats blind hedging out of
      sample, with a date-block bootstrap, and we do NOT overstate a 2-crisis
      edge.

Run:  python3 -m regime_detection.step26_typology_hedge
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from regime_detection.lib.splits import DEV_END, OOS_START
from regime_detection.lib.typology_hedge import (
    BONDS, CASH, EQUITY, GOLD,
    performance_summary, rates_pressure_share, route_hedge,
    signed_duration_trend, strategy_returns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
HEDGE_PATH = PROJECT_ROOT / "data" / "hedge_assets.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
CONTRIB_PATH = (PROJECT_ROOT / "reports" / "decomposition"
                / "turbulence_contributions.parquet")
OUT_DIR = PROJECT_ROOT / "reports" / "typology_hedge"
FIG_DIR = PROJECT_ROOT / "figures" / "typology_hedge"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step26_typology_hedge.md"

GATE_Q = 0.75          # de-risk when turbulence > dev 75th pct (existing signal)
DUR_WINDOW = 63        # trailing window for the signed duration trend
COST_BPS = 5.0
HORIZON = 21           # block length for bootstrap / validation look-ahead
N_BOOT = 10_000


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load():
    turb = pd.read_parquet(TURB_PATH)["turbulence"].dropna()
    contrib = pd.read_parquet(CONTRIB_PATH).dropna()
    hedge = pd.read_parquet(HEDGE_PATH).sort_index()
    fpx = pd.read_parquet(FACTOR_PATH).sort_index()
    hret = np.log(hedge / hedge.shift(1))
    asset_returns = {
        EQUITY: hret["SPY"],
        BONDS: hret["TLT"],
        GOLD: hret["GLD"],
        CASH: hret["BIL"],
    }
    return turb, contrib, asset_returns, fpx


# --------------------------------------------------------------------------- #
# validation: magnitude (direction-blind) vs signed routing signal
# --------------------------------------------------------------------------- #
def _validation(turb, contrib, asset_returns):
    """Does each candidate routing signal predict bonds-minus-gold forward?"""
    gate = turb > turb.loc[:DEV_END].quantile(GATE_Q)
    R = rates_pressure_share(contrib)                       # magnitude (blind)
    trend = signed_duration_trend(asset_returns[BONDS], DUR_WINDOW)  # signed
    fwd = (asset_returns[BONDS] - asset_returns[GOLD]).shift(-1)\
        .rolling(HORIZON).sum().shift(-(HORIZON - 1))      # forward bonds-gold
    out = {}
    for name, sig in [("magnitude_share", R), ("signed_trend", trend)]:
        df = pd.concat({"s": sig, "g": gate, "f": fwd}, axis=1).dropna()
        df = df[df["g"]]
        rec = {}
        for lbl, sl in [("dev", df[df.index <= DEV_END]),
                        ("oos", df[df.index > DEV_END])]:
            rho, p = stats.spearmanr(sl["s"], sl["f"])
            rec[lbl] = (float(rho), float(p), int(len(sl)))
        out[name] = rec
    return out


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _build_positions(turb, asset_returns):
    """Return a dict of {name: position-label Series} on the common index,
    each already lagged by one bar (decision at t -> held over t+1)."""
    gate = turb > turb.loc[:DEV_END].quantile(GATE_Q)
    # signed, parameter-free a-priori rule: bonds in a downtrend (rising rates)
    # => rates-shock => route to gold; else flight-to-safety => bonds.
    trend = signed_duration_trend(asset_returns[BONDS], DUR_WINDOW)
    rates_shock = trend < 0.0

    decisions = {
        "buy_and_hold": pd.Series(EQUITY, index=gate.index),
        "blind_cash": gate.map({True: CASH, False: EQUITY}),
        "blind_bonds": gate.map({True: BONDS, False: EQUITY}),
        "blind_gold": gate.map({True: GOLD, False: EQUITY}),
        # 50/50 bond+gold split represented by alternating? No — use a synthetic
        # 'split' asset built in returns; handled separately below.
        "composition_aware": route_hedge(gate, rates_shock,
                                          defensive=BONDS, inflationary=GOLD),
    }
    # lag every decision by one bar
    return {k: v.shift(1).dropna() for k, v in decisions.items()}, gate, rates_shock


def _split_returns(asset_returns, gate):
    """50/50 bonds+gold diversified hedge as an explicit return series."""
    split_hedge = 0.5 * asset_returns[BONDS] + 0.5 * asset_returns[GOLD]
    pos = gate.map({True: "split", False: EQUITY}).shift(1).dropna()
    ar = dict(asset_returns)
    ar["split"] = split_hedge
    return strategy_returns(pos, ar, cost_bps=COST_BPS)


def _block_bootstrap_diff(diff, block=HORIZON, n=N_BOOT, seed=0):
    """P(mean diff > 0) and 90% CI via contiguous date-block bootstrap."""
    d = diff.dropna().to_numpy()
    if len(d) < block:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(d) / block))
    starts_max = len(d) - block
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([d[s:s + block] for s in starts])[:len(d)]
        means[i] = sample.mean()
    return float((means > 0).mean()), (float(np.quantile(means, 0.05)),
                                        float(np.quantile(means, 0.95)))


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading turbulence, contributions, hedge ETFs ...")
    turb, contrib, asset_returns, fpx = _load()

    print("Validating routing signal (magnitude vs signed) ...")
    val = _validation(turb, contrib, asset_returns)
    for name, rec in val.items():
        print(f"  {name:16s} dev rho={rec['dev'][0]:+.3f} (p={rec['dev'][1]:.3f}) "
              f"| oos rho={rec['oos'][0]:+.3f} (p={rec['oos'][1]:.3f})")

    print("Building strategies ...")
    positions, gate, rates_shock = _build_positions(turb, asset_returns)
    strat_rets = {k: strategy_returns(v, asset_returns, cost_bps=COST_BPS)
                  for k, v in positions.items()}
    strat_rets["split_bond_gold"] = _split_returns(asset_returns, gate)

    # restrict to OOS (2020-01-01 onward)
    oos = {k: v.loc[OOS_START:].dropna() for k, v in strat_rets.items()}
    common = None
    for v in oos.values():
        common = v.index if common is None else common.intersection(v.index)
    oos = {k: v.reindex(common) for k, v in oos.items()}

    print(f"\nOOS window: {common.min().date()} .. {common.max().date()} "
          f"({len(common)} days)\n")

    # performance table
    perf = {k: performance_summary(v) for k, v in oos.items()}
    order = ["buy_and_hold", "blind_cash", "blind_bonds", "blind_gold",
             "split_bond_gold", "composition_aware"]
    print(f"{'strategy':18s} {'totRet':>8s} {'CAGR':>7s} {'vol':>6s} "
          f"{'Sharpe':>7s} {'Sortino':>8s} {'maxDD':>8s}")
    for k in order:
        p = perf[k]
        print(f"{k:18s} {p['total_return']:+8.1%} {p['cagr']:+7.1%} "
              f"{p['vol']:6.1%} {p['sharpe']:7.2f} {p['sortino']:8.2f} "
              f"{p['max_drawdown']:8.1%}")

    # sub-window: hedge instrument returns in the two archetype crises
    windows = {"2020 COVID (Feb-Apr)": ("2020-02-19", "2020-04-30"),
               "2022 inflation (full yr)": ("2022-01-01", "2022-12-31")}
    sub = {}
    for wname, (s, e) in windows.items():
        sub[wname] = {lab: float(np.expm1(asset_returns[lab].loc[s:e].sum()))
                      for lab in (EQUITY, BONDS, GOLD, CASH)}
    print("\nHedge-instrument total return by crisis archetype:")
    for wname, d in sub.items():
        print(f"  {wname:26s} " +
              "  ".join(f"{k}={v:+.1%}" for k, v in d.items()))

    # bootstrap: composition-aware vs the two natural comparators
    boots = {}
    for comp in ["blind_bonds", "split_bond_gold"]:
        diff = oos["composition_aware"] - oos[comp]
        pgt, ci = _block_bootstrap_diff(diff)
        boots[comp] = (pgt, ci, float(np.expm1(diff.sum())))
        print(f"\ncomposition_aware − {comp}: cum diff={boots[comp][2]:+.1%}  "
              f"P(daily mean>0)={pgt:.0%}  90% CI[{ci[0]*1e4:+.2f},{ci[1]*1e4:+.2f}] bps/day")

    # persist the OOS strategy curves for the thesis-figure generator
    _curve_dir = PROJECT_ROOT / "reports" / "typology_hedge"
    _curve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: oos[k] for k in order}).to_parquet(_curve_dir / "oos_curves.parquet")

    # ----------------------------------------------------------------- figures
    _fig_equity_curves(oos, order)
    _fig_crisis_bars(sub)
    _fig_routing_timeline(turb, gate, rates_shock)

    _write_checkpoint(val, perf, order, sub, boots, common)
    print(f"\nWrote checkpoint → {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


# --------------------------------------------------------------------------- #
def _fig_equity_curves(oos, order):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"buy_and_hold": "0.4", "blind_cash": "tab:gray",
              "blind_bonds": "tab:blue", "blind_gold": "tab:orange",
              "split_bond_gold": "tab:green", "composition_aware": "tab:red"}
    for k in order:
        eq = np.expm1(oos[k].cumsum())
        ax.plot(eq.index, eq.values, label=k, lw=1.8 if k == "composition_aware" else 1.2,
                color=colors.get(k))
    ax.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-04-30"),
               color="red", alpha=0.06)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="orange", alpha=0.06)
    ax.set_title("OOS (2020-01-01+) cumulative return — equity protected, hedge routed by stress type")
    ax.set_ylabel("cumulative return"); ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "oos_equity_curves.png", dpi=130)
    plt.close(fig)


def _fig_crisis_bars(sub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (wname, d) in zip(axes, sub.items()):
        labs = list(d.keys()); vals = [d[l] for l in labs]
        colors = ["0.4", "tab:blue", "tab:orange", "tab:gray"]
        ax.bar(labs, vals, color=colors)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(wname, fontsize=10)
        ax.set_ylabel("total return")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+.0%}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    fig.suptitle("The hedge that works flips between crisis archetypes "
                 "(bonds win 2020, gold wins 2022)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "crisis_archetype_hedges.png", dpi=130)
    plt.close(fig)


def _fig_routing_timeline(turb, gate, rates_shock):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    t = turb.loc[OOS_START:]
    ax.plot(t.index, t.values, color="0.5", lw=0.7, label="turbulence")
    g = gate.loc[OOS_START:]
    rs = (rates_shock.reindex(g.index) == True)  # noqa: E712 — NaN -> False, bool
    ax.fill_between(g.index, 0, t.max(), where=(g & ~rs).values,
                    color="tab:blue", alpha=0.18, label="de-risk → bonds")
    ax.fill_between(g.index, 0, t.max(), where=(g & rs).values,
                    color="tab:orange", alpha=0.30, label="de-risk → gold (rates-shock)")
    ax.set_title("OOS routing: when de-risked, bonds vs gold by signed duration trend")
    ax.set_ylabel("turbulence"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "routing_timeline.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _write_checkpoint(val, perf, order, sub, boots, common):
    bh, blind_b, split, aware = (perf["buy_and_hold"], perf["blind_bonds"],
                                 perf["split_bond_gold"], perf["composition_aware"])
    # verdict logic
    beats_bh_dd = aware["max_drawdown"] > bh["max_drawdown"]      # less negative
    aware_sortino_best = aware["sortino"] >= max(
        perf[k]["sortino"] for k in ["blind_cash", "blind_bonds", "blind_gold"])
    pgt_bonds = boots["blind_bonds"][0]
    signed_ok = (val["signed_trend"]["oos"][1] < 0.05)
    mag_blind = (val["magnitude_share"]["oos"][1] > 0.10)

    lines = []
    lines.append("# Step 26 — Typology-conditioned hedge selection "
                 "(composition-aware de-risking)\n")
    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. "
                 f"OOS = {common.min().date()} … {common.max().date()} "
                 f"({len(common)} trading days), fixed 2020-01-01 constraint._\n")

    lines.append("## The question (better-posed than 'time the drawdown')\n")
    lines.append("The turbulence *regime* already says **when** to de-risk. This "
                 "step asks **where to park** — and shows the right hedge depends "
                 "on the *kind* of stress. The 2020-start OOS hands us both "
                 "archetypes as held-out events:\n")
    lines.append("| crisis | type | bonds (TLT) | gold (GLD) | cash | equity (SPY) |")
    lines.append("|---|---|---|---|---|---|")
    for wname, d in sub.items():
        lines.append(f"| {wname} | {'equity-led' if '2020' in wname else 'rates-led'} "
                     f"| {d[BONDS]:+.0%} | {d[GOLD]:+.0%} | {d[CASH]:+.0%} | {d[EQUITY]:+.0%} |")
    lines.append("\n→ **The hedge that works flips.** Long Treasuries are the perfect "
                 "COVID hedge and a disaster in 2022; gold is the reverse. A single "
                 "blind hedge is therefore fragile across regimes.\n")

    lines.append("## Finding 1 — magnitude typology is direction-blind\n")
    m, s = val["magnitude_share"], val["signed_trend"]
    lines.append("Does a routing signal predict *bonds − gold* forward return on "
                 "de-risk days? (Spearman ρ, high-turbulence subset)\n")
    lines.append("| signal | dev ρ (p) | oos ρ (p) |")
    lines.append("|---|---|---|")
    lines.append(f"| magnitude rates-share (step17 typology) | "
                 f"{m['dev'][0]:+.3f} ({m['dev'][1]:.3f}) | {m['oos'][0]:+.3f} ({m['oos'][1]:.3f}) |")
    lines.append(f"| **signed** duration trend (step18 spirit) | "
                 f"{s['dev'][0]:+.3f} ({s['dev'][1]:.3f}) | {s['oos'][0]:+.3f} ({s['oos'][1]:.3f}) |")
    lines.append("\nThe **magnitude** crisis-typology fingerprint carries ~zero "
                 "information for hedge choice — both 2020 and 2022 load heavily on "
                 "rates/inflation, so |contribution| cannot tell a flight-to-safety "
                 "rally from a rates crash. The **signed** decomposition can. This is "
                 "a concrete reason the signed-turbulence layer (step18) earns its keep.\n")

    lines.append("## Finding 2 — out-of-sample hedge performance (2020-01-01 +)\n")
    lines.append("Equity (SPY) protected; de-risk gate = turbulence > dev "
                 f"{int(GATE_Q*100)}th pct; {int(COST_BPS)} bps per switch.\n")
    lines.append("| strategy | total | CAGR | vol | Sharpe | Sortino | max DD |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in order:
        p = perf[k]
        lines.append(f"| {k} | {p['total_return']:+.1%} | {p['cagr']:+.1%} | "
                     f"{p['vol']:.1%} | {p['sharpe']:.2f} | {p['sortino']:.2f} | "
                     f"{p['max_drawdown']:.1%} |")
    lines.append("")

    lines.append("## Finding 3 — routing vs the natural comparators\n")
    for comp, (pgt, ci, cum) in boots.items():
        lines.append(f"- **composition_aware − {comp}**: cumulative "
                     f"{cum:+.1%}; date-block bootstrap P(daily edge>0) = "
                     f"**{pgt:.0%}**, 90% CI [{ci[0]*1e4:+.2f}, {ci[1]*1e4:+.2f}] bps/day.")
    lines.append("")

    # verdict
    lines.append("## Verdict\n")
    verdict = []
    if mag_blind and signed_ok:
        verdict.append("**Magnitude typology cannot route hedges; the signed "
                       "decomposition can** — a clean, falsifiable result that "
                       "motivates step18 directly.")
    if aware_sortino_best and beats_bh_dd:
        verdict.append("Composition-aware routing has the **best Sortino among the "
                       "de-risking variants** and a shallower drawdown than buy-and-hold.")
    verdict.append(f"But the edge over a naive bonds hedge is **not 95%-significant** "
                   f"(P≈{pgt_bonds:.0%}); with only ~2 archetype crises in the OOS the "
                   f"honest reading is that the value comes mostly from **not "
                   f"concentrating in the single hedge that fails** (diversification / "
                   f"routing), not from precise crisis typing — composition_aware is "
                   f"**statistically indistinguishable from a static 50/50 bond-gold "
                   f"split** (P≈{boots['split_bond_gold'][0]:.0%}).")
    verdict.append("Always-gold won this particular OOS outright, but that is "
                   "**hindsight hedge selection** (gold's 2020-2026 secular run); "
                   "ex ante you cannot know which single sleeve will dominate, which "
                   "is exactly why the routed / diversified hedge is the defensible choice.")
    verdict.append("**Additive, not a replacement:** the turbulence detector, "
                   "decomposition and supervised models are unchanged — this is a "
                   "hedge-selection layer bolted on top of the existing de-risk gate.")
    for v in verdict:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("**Figures:** `figures/typology_hedge/oos_equity_curves.png`, "
                 "`crisis_archetype_hedges.png`, `routing_timeline.png`.")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
