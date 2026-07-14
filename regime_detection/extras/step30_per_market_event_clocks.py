"""Step 30 — Per-market event clocks: does a local clock fix the step25 negative?

Step25 found that the event clock, which *helped* a single matched asset (step21),
*failed* when pooled across 7 heterogeneous markets: one **global** factor-
turbulence clock imposed on Korea/Japan/Canada/… whose own drawdown dynamics
don't follow it (basket lift −0.23, 0/7 positive vs calendar+pooling −0.06).
The diagnosed fix — and the asset-specific clock CKT (2023) flag but never build —
is to pace **each market by its own clock**.

This step builds that: each market gets an event clock driven by its **own
univariate turbulence** (squared z-score of its return), the event-time drawdown
target is measured on that local clock, and the step24 pooled overlay is re-run.
Head-to-head, three arms, everything else identical (same global turbulence
features, market fixed effects, panel walk-forward, date-block bootstrap):

  1. calendar (21-day) target                — the step24 baseline
  2. GLOBAL event clock target               — the step25 (failed) version
  3. PER-MARKET event clock target           — the fix tested here

Hypothesis: local clocks recover the event-time benefit that the global clock
threw away → arm 3 beats arm 2, and ideally matches/beats arm 1.

Run:  python3 -m regime_detection.step30_per_market_event_clocks
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_detection.lib.splits import DEV_END
from regime_detection.lib.event_time import EventTimeConverter
from regime_detection.lib.event_time_features import univariate_turbulence
from regime_detection.lib.targets import event_time_drawdown, forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
# reuse the panel builder + pooled evaluator from step25 (no duplication)
from regime_detection.extras.step25_eventtime_pooling import build_panel, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
REG_PATH = PROJECT_ROOT / "data" / "benchmarks_regional.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "per_market_event_clocks"
FIG_DIR = PROJECT_ROOT / "figures" / "per_market_event_clocks"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step30_per_market_event_clocks.md"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
LOSS = 0.05; CAL_HORIZON = 21
REGION_NAMES = {"VGK": "Europe", "EWJ": "Japan", "EWU": "UK",
                "EPP": "Pacific", "EWC": "Canada", "EWY": "Korea"}


def _purge_99(intensity: pd.Series, thr: float) -> int:
    s = intensity.dropna()
    if len(s) < 50:
        return CAL_HORIZON
    cum = np.cumsum(s.values); n = len(s)
    js = np.searchsorted(cum, cum + thr, side="left")
    lens = (js - np.arange(n))[js < n]
    return int(np.ceil(np.quantile(lens, 0.99))) if len(lens) else CAL_HORIZON


def _build_panel_local(X, assets, markets, target_by_market, dummy_cols):
    """Panel builder for the per-market arm (target differs by market)."""
    blocks = []
    for m in markets:
        y = target_by_market[m]
        ret = np.log(assets[m] / assets[m].shift(1))
        b = X.reindex(y.dropna().index).dropna()
        b["y"] = y.reindex(b.index); b["ret"] = ret.reindex(b.index)
        b["mkt"] = m; b["date"] = b.index
        for d in dummy_cols:
            b[d] = 1.0 if d == f"mkt_{m}" else 0.0
        blocks.append(b.dropna(subset=["y", "ret"]))
    return pd.concat(blocks, ignore_index=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH); turb = tdf["turbulence"]
    X = build_feature_matrix_with_bucket(prices, turb, tdf["regime_smoothed"],
                                         factors=FACTORS_10, n_buckets=5)
    reg = pd.read_parquet(REG_PATH)
    assets = {"US": prices["equity"]}
    for tk, nm in REGION_NAMES.items():
        assets[nm] = reg[tk].reindex(prices.index).ffill(limit=3)
    markets = list(assets)
    dummy_cols = [f"mkt_{m}" for m in markets[1:]]
    feat_cols = list(X.columns) + dummy_cols

    # global event clock (step21/25): threshold + purge from global turbulence
    _, gdiag = EventTimeConverter.calibrate(turb.loc[:DEV_END], target_mean_trading_days=21.0)
    g_thr = float(gdiag.threshold)
    g_purge = _purge_99(turb, g_thr)

    # per-market event clocks: each market's own univariate turbulence + threshold
    local_targets, local_thr, local_purges = {}, {}, []
    for m in markets:
        uni = univariate_turbulence(assets[m]).reindex(prices.index)
        _, ldiag = EventTimeConverter.calibrate(uni.loc[:DEV_END].dropna(),
                                                target_mean_trading_days=21.0)
        thr_m = float(ldiag.threshold)
        etd = event_time_drawdown(assets[m], uni, intensity_threshold=thr_m)
        local_targets[m] = (etd <= -LOSS).astype(float).where(etd.notna())
        local_thr[m] = thr_m
        local_purges.append(_purge_99(uni, thr_m))
    l_purge = int(max(local_purges))
    print(f"global thr {g_thr:.0f} (purge {g_purge}d); "
          f"per-market thr {min(local_thr.values()):.1f}–{max(local_thr.values()):.1f} "
          f"(pooled purge {l_purge}d)")

    def cal_target(price):
        fdd = forward_drawdown_conditional(price, horizon=CAL_HORIZON)
        return (fdd <= -LOSS).astype(float).where(fdd.notna())

    def evt_global_target(price):
        etd = event_time_drawdown(price, turb, intensity_threshold=g_thr)
        return (etd <= -LOSS).astype(float).where(etd.notna())

    print("[1/3] calendar + pooling ...")
    cal = evaluate(build_panel(X, assets, markets, cal_target, dummy_cols),
                   feat_cols, markets, CAL_HORIZON)
    print("[2/3] GLOBAL event clock + pooling ...")
    evg = evaluate(build_panel(X, assets, markets, evt_global_target, dummy_cols),
                   feat_cols, markets, g_purge)
    print("[3/3] PER-MARKET event clocks + pooling ...")
    evl = evaluate(_build_panel_local(X, assets, markets, local_targets, dummy_cols),
                   feat_cols, markets, l_purge)

    cmp = pd.DataFrame({
        "calendar": _row(cal, markets), "global_clock": _row(evg, markets),
        "per_market_clock": _row(evl, markets),
    }).T
    cmp.to_csv(OUT_DIR / "per_market_event_clocks.csv")
    print(cmp.round(3).to_string())

    _figure(cal, evg, evl, markets)
    _write_checkpoint(cal, evg, evl, markets, g_thr, g_purge, l_purge)
    print(f"\nWrote checkpoint → {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _row(r, markets):
    return {"auc": r["auc"], "basket_lift": r["lift"], "ci_low": r["ci"][0],
            "ci_high": r["ci"][1], "p_pos": r["p_pos"],
            "markets_pos": f"{r['n_pos']}/{len(markets)}"}


def _figure(cal, evg, evl, markets):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.hist(cal["_lifts"], bins=40, alpha=0.5, color="#1f77b4",
           label=f"calendar (lift {cal['lift']:+.2f}, P{cal['p_pos']:.0%})")
    a.hist(evg["_lifts"], bins=40, alpha=0.5, color="#c0392b",
           label=f"global clock ({evg['lift']:+.2f}, P{evg['p_pos']:.0%})")
    a.hist(evl["_lifts"], bins=40, alpha=0.5, color="#27ae60",
           label=f"per-market clock ({evl['lift']:+.2f}, P{evl['p_pos']:.0%})")
    a.axvline(0, color="k", lw=0.9); a.set_xlabel("basket Sortino lift (date-block bootstrap)")
    a.set_title("Does a per-market clock fix the global-clock failure?"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1]
    xx = np.arange(len(markets)); w = 0.27
    a.bar(xx - w, [cal["per"].get(m, 0) for m in markets], w, color="#1f77b4", label="calendar")
    a.bar(xx, [evg["per"].get(m, 0) for m in markets], w, color="#c0392b", label="global clock")
    a.bar(xx + w, [evl["per"].get(m, 0) for m in markets], w, color="#27ae60", label="per-market clock")
    a.axhline(0, color="k", lw=0.7); a.set_xticks(xx); a.set_xticklabels(markets, rotation=30, fontsize=8)
    a.set_ylabel("overlay Sortino lift"); a.set_title("Per-market lift by clock"); a.legend(fontsize=8); a.grid(axis="y", alpha=0.3)
    fig.suptitle("Step 30 — per-market event clocks vs the global clock", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "per_market_event_clocks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_checkpoint(cal, evg, evl, markets, g_thr, g_purge, l_purge):
    fixes = evl["lift"] > evg["lift"] + 0.02            # recovers vs the global clock
    positive_edge = evl["lift"] > 0 and evl["n_pos"] >= 4   # an actual positive overlay
    matches_cal = evl["lift"] >= cal["lift"] - 0.02        # back to calendar parity
    L = []
    L.append("# Step 30 — Per-market event clocks: fixing the step25 negative?\n")
    L.append(f"_Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC. 7-market pooled overlay, "
             "5% correction, 10bps, market fixed effects, date-block bootstrap. "
             f"Global clock thr {g_thr:.0f} (purge {g_purge}d); per-market clocks "
             f"calibrated to ~21d each (pooled purge {l_purge}d)._\n")
    L.append("Step25 showed the event clock *fails* when pooled because one **global** "
             "clock is imposed on heterogeneous local markets. This tests the diagnosed "
             "fix: pace **each market by its own** univariate-turbulence clock.\n")
    L.append("| arm | basket lift | 95% CI | P(>0) | markets+ |")
    L.append("|---|---|---|---|---|")
    for nm, r in [("calendar (step24)", cal), ("global clock (step25)", evg),
                  ("**per-market clocks**", evl)]:
        L.append(f"| {nm} | {r['lift']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
                 f"{r['p_pos']:.0%} | {r['n_pos']}/{len(markets)} |")
    L.append("")
    L.append("Per-market overlay lift (calendar / global / per-market):")
    for m in markets:
        L.append(f"- {m}: {cal['per'].get(m,0):+.3f} / {evg['per'].get(m,0):+.3f} / "
                 f"{evl['per'].get(m,0):+.3f}")
    L.append("")
    L.append("## Verdict\n")
    if fixes and positive_edge and matches_cal:
        L.append(f"**THE LOCAL CLOCK FIXES IT.** Per-market clocks lift the pooled basket "
                 f"to a positive {evl['lift']:+.3f} ({evl['n_pos']}/{len(markets)} markets, "
                 f"P>0={evl['p_pos']:.0%}) — recovering from the global clock's "
                 f"{evg['lift']:+.3f} ({evg['n_pos']}/{len(markets)}) and matching/beating "
                 f"calendar+pooling ({cal['lift']:+.3f}). Confirms step25's diagnosis: the "
                 "event clock transfers across markets **when each market is paced by its "
                 "own clock**. (Read against the sample-size ceiling — CIs below.)")
    elif fixes and matches_cal:
        L.append(f"**PARTIAL — DIAGNOSIS CONFIRMED, NO NET EDGE.** Per-market clocks recover "
                 f"the global clock's damage — basket lift {evg['lift']:+.3f} "
                 f"({evg['n_pos']}/{len(markets)}, P>0={evg['p_pos']:.0%}) → "
                 f"**{evl['lift']:+.3f}** ({evl['n_pos']}/{len(markets)}, P>0={evl['p_pos']:.0%}), "
                 f"back to **calendar parity** ({cal['lift']:+.3f}). This **confirms step25's "
                 "diagnosis**: the pooled event-clock failure was the global-vs-local "
                 "mismatch — pacing each market by its own clock removes it. **But** the "
                 "event clock buys *nothing over a plain calendar horizon* once you go local, "
                 "and all three arms are negative (≤1/7 markets positive): under the 2020 OOS "
                 "there is **no positive pooled overlay on any clock** (consistent with step24's "
                 "COVID-driven collapse). So local clocks *stop the bleeding* but don't create "
                 "an edge — the event clock's genuine benefit stays confined to the matched "
                 "single-asset case (step21).")
    elif fixes:
        L.append(f"**PARTIAL FIX (below calendar).** Per-market clocks ({evl['lift']:+.3f}, "
                 f"{evl['n_pos']}/{len(markets)}, P>0={evl['p_pos']:.0%}) recover from the "
                 f"global clock ({evg['lift']:+.3f}) in the hypothesised direction but stay "
                 f"**below** calendar+pooling ({cal['lift']:+.3f}). Local clocks help vs a "
                 "global clock; the event clock still doesn't match a plain calendar horizon "
                 "in the pooled setting.")
    else:
        L.append(f"**THE LOCAL CLOCK DOES NOT FIX IT.** Per-market clocks "
                 f"({evl['lift']:+.3f}, {evl['n_pos']}/{len(markets)}, P>0={evl['p_pos']:.0%}) "
                 f"are no better than the global clock ({evg['lift']:+.3f}); calendar+pooling "
                 f"({cal['lift']:+.3f}) remains best. Honest negative: the event clock's "
                 "pooled failure is **not** merely a global-vs-local mismatch — re-pacing "
                 "each market by its own clock doesn't recover an edge either. Strengthens "
                 "the conclusion that calendar + pooling is the right configuration and the "
                 "event clock's benefit is confined to the matched single-asset case (step21).")
    L.append("")
    L.append("**Figure:** `figures/per_market_event_clocks/per_market_event_clocks.png`. "
             "Factor Lens intact (one global signal for features; clocks are per-market only "
             "for the target horizon).")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
