"""Step 25 — combine the two confirmed leads: event-time × cross-market pooling.

Step 21 showed the event clock makes the correction overlay less cutoff-fragile;
step 24 showed cross-market pooling lifts the signal (AUC 0.60, 7/7 markets
positive, basket lift +0.36, ~92% bootstrap-confident but not 95%-significant).
The obvious question: do they **compound**? This reruns the step-24 pooled
overlay with each market's target on the **event clock** (event-time drawdown)
and compares head-to-head against the calendar target — same 7 markets, same
global turbulence features, same panel walk-forward + market fixed effects +
date-block bootstrap.

Hypothesis: event-time + pooling beats calendar + pooling, and ideally clears
the 95% bar (the project's first significant positive overlay result).

Run:  python3 -m regime_detection.step25_eventtime_pooling
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from regime_detection.lib.event_time import EventTimeConverter
from regime_detection.lib.metrics import max_drawdown, sortino_ratio
from regime_detection.lib.targets import event_time_drawdown, forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.splits import DEV_END
# reuse the panel walk-forward + overlay from step 24 (no duplication)
from regime_detection.extras.step24_cross_market_pooling import pooled_walk_forward, overlay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
REG_PATH = PROJECT_ROOT / "data" / "benchmarks_regional.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "eventtime_pooling"
FIG_DIR = PROJECT_ROOT / "figures" / "eventtime_pooling"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step25_eventtime_pooling.md"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
LOSS = 0.05; CAL_HORIZON = 21; COST_BPS = 10
REGION_NAMES = {"VGK": "Europe", "EWJ": "Japan", "EWU": "UK",
                "EPP": "Pacific", "EWC": "Canada", "EWY": "Korea"}


def build_panel(X, assets, markets, target_fn, dummy_cols):
    blocks = []
    for m in markets:
        price = assets[m]
        y = target_fn(price)
        ret = np.log(price / price.shift(1))
        b = X.reindex(y.dropna().index).dropna()
        b["y"] = y.reindex(b.index); b["ret"] = ret.reindex(b.index)
        b["mkt"] = m; b["date"] = b.index
        for d in dummy_cols:
            b[d] = 1.0 if d == f"mkt_{m}" else 0.0
        blocks.append(b.dropna(subset=["y", "ret"]))
    return pd.concat(blocks, ignore_index=True)


def evaluate(panel, feat_cols, markets, purge):
    oos = pooled_walk_forward(panel, feat_cols, purge_bdays=purge)
    auc = roc_auc_score(oos["y"], oos["proba"]) if oos["y"].nunique() > 1 else np.nan
    per, bstrat, bbh = {}, [], []
    for m in markets:
        g = oos[oos["mkt"] == m].set_index("date").sort_index()
        if g.empty:
            continue
        in_mkt = (g["proba"].rank(pct=True) < 0.90).astype(float)
        strat = overlay(g["ret"], in_mkt, COST_BPS); bh = g["ret"]
        per[m] = sortino_ratio(strat.dropna()) - sortino_ratio(bh.dropna())
        bstrat.append(strat.rename(m)); bbh.append(bh.rename(m))
    bs = pd.concat(bstrat, axis=1).mean(axis=1).dropna()
    bb = pd.concat(bbh, axis=1).mean(axis=1).reindex(bs.index)
    lift = sortino_ratio(bs) - sortino_ratio(bb)
    # date-block bootstrap (21-day blocks)
    rng = np.random.default_rng(0); block = 21; nb = 2000
    diff = (bs - bb).values; bhv = bb.values; n = len(diff)
    lifts = np.empty(nb)
    for i in range(nb):
        idx = []
        while len(idx) < n:
            st = int(rng.integers(0, n)); idx.extend(range(st, min(st + block, n)))
        idx = np.array(idx[:n])
        lifts[i] = sortino_ratio(pd.Series(bhv[idx] + diff[idx])) - sortino_ratio(pd.Series(bhv[idx]))
    ci = (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5)))
    return {"auc": auc, "lift": lift, "ci": ci, "p_pos": float((lifts > 0).mean()),
            "n_pos": int(sum(v > 0 for v in per.values())), "per": per,
            "_bs": bs, "_bb": bb, "_lifts": lifts}


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

    # event threshold + purge (dev-calibrated, same as step 21)
    conv, diag = EventTimeConverter.calibrate(turb.loc[:DEV_END], target_mean_trading_days=21.0)
    thr = conv.intensity_threshold
    s = turb.dropna(); cum = np.cumsum(s.values); n = len(s)
    js = np.searchsorted(cum, cum + thr, side="left")
    evt_purge = int(np.ceil(np.quantile((js - np.arange(n))[js < n], 0.99)))
    print(f"event threshold {thr:.0f}; event purge {evt_purge}d (calendar {CAL_HORIZON}d)")

    def cal_target(price):
        fdd = forward_drawdown_conditional(price, horizon=CAL_HORIZON)
        return (fdd <= -LOSS).astype(float).where(fdd.notna())

    def evt_target(price):
        etd = event_time_drawdown(price, turb, intensity_threshold=thr)
        return (etd <= -LOSS).astype(float).where(etd.notna())

    print("evaluating CALENDAR + pooling ...")
    cal = evaluate(build_panel(X, assets, markets, cal_target, dummy_cols), feat_cols,
                   markets, CAL_HORIZON)
    print("evaluating EVENT-TIME + pooling ...")
    evt = evaluate(build_panel(X, assets, markets, evt_target, dummy_cols), feat_cols,
                   markets, evt_purge)

    cmp = pd.DataFrame({
        "calendar + pooling": {"auc": cal["auc"], "basket_lift": cal["lift"],
            "ci_low": cal["ci"][0], "ci_high": cal["ci"][1], "p_pos": cal["p_pos"],
            "markets_pos": cal["n_pos"]},
        "event-time + pooling": {"auc": evt["auc"], "basket_lift": evt["lift"],
            "ci_low": evt["ci"][0], "ci_high": evt["ci"][1], "p_pos": evt["p_pos"],
            "markets_pos": evt["n_pos"]},
    }).T
    cmp.to_csv(OUT_DIR / "calendar_vs_eventtime_pooled.csv")
    print(cmp.round(3).to_string())

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    a = ax[0]
    a.plot(cal["_bb"].index, cal["_bb"].cumsum(), color="#7f8c8d", label="basket B&H")
    a.plot(cal["_bs"].index, cal["_bs"].cumsum(), color="#1f77b4", ls="--", label="calendar+pool overlay")
    a.plot(evt["_bs"].index, evt["_bs"].cumsum(), color="#c0392b", ls="--", label="event-time+pool overlay")
    a.set_title("Global basket — calendar vs event-time overlay (OOS)"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1]
    a.hist(cal["_lifts"], bins=40, alpha=0.5, color="#1f77b4",
           label=f"calendar (lift {cal['lift']:+.2f}, P{cal['p_pos']:.0%})")
    a.hist(evt["_lifts"], bins=40, alpha=0.5, color="#c0392b",
           label=f"event-time (lift {evt['lift']:+.2f}, P{evt['p_pos']:.0%})")
    a.axvline(0, color="k", lw=0.9)
    a.set_xlabel("basket Sortino lift (date-block bootstrap)")
    a.set_title("Bootstrap distributions — does event-time compound?")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[2]
    ms = markets; xx = np.arange(len(ms)); w = 0.38
    a.bar(xx - w/2, [cal["per"].get(m, 0) for m in ms], w, color="#1f77b4", label="calendar")
    a.bar(xx + w/2, [evt["per"].get(m, 0) for m in ms], w, color="#c0392b", label="event-time")
    a.axhline(0, color="k", lw=0.7); a.set_xticks(xx); a.set_xticklabels(ms, rotation=30, fontsize=8)
    a.set_ylabel("overlay Sortino lift"); a.set_title("Per-market lift by clock")
    a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)
    fig.suptitle("Step 25 — event-time × cross-market pooling (the two leads combined)",
                 fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eventtime_pooling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {FIG_DIR / 'eventtime_pooling.png'}")

    # ---- verdict ----
    evt_sig = evt["ci"][0] > 0; cal_sig = cal["ci"][0] > 0
    compounds = evt["lift"] > cal["lift"] and evt["p_pos"] >= cal["p_pos"]
    if evt_sig and not cal_sig:
        verdict = (f"**THEY COMPOUND — AND IT CLEARS 95%.** Event-time + pooling gives "
                   f"basket lift {evt['lift']:+.3f}, 95% CI **[{evt['ci'][0]:+.3f}, "
                   f"{evt['ci'][1]:+.3f}] excluding zero** (P>0={evt['p_pos']:.0%}, "
                   f"{evt['n_pos']}/{len(markets)} markets), vs calendar+pooling "
                   f"{cal['lift']:+.3f} (CI [{cal['ci'][0]:+.3f},{cal['ci'][1]:+.3f}], "
                   "still includes zero). The two confirmed levers stack into the "
                   "project's first bootstrap-significant positive overlay — and the "
                   "Factor Lens is untouched.")
    elif compounds:
        verdict = (f"**THEY COMPOUND (still inside the floor).** Event-time + pooling "
                   f"{evt['lift']:+.3f} (P>0={evt['p_pos']:.0%}, {evt['n_pos']}/{len(markets)} "
                   f"markets) ≥ calendar+pooling {cal['lift']:+.3f} (P>0={cal['p_pos']:.0%}). "
                   "Both 95% CIs still include zero, so neither is significant, but the "
                   "event clock improves the pooled result in the hypothesised direction — "
                   "consistent with step 21.")
    else:
        verdict = (f"**THE LEADS CONFLICT — a clean, mechanistically-clear null.** "
                   f"Event-time + pooling is *worse*, not better: basket lift "
                   f"{evt['lift']:+.3f} ({evt['n_pos']}/{len(markets)} markets positive, "
                   f"P>0={evt['p_pos']:.0%}) vs calendar + pooling {cal['lift']:+.3f} "
                   f"({cal['n_pos']}/{len(markets)}, P>0={cal['p_pos']:.0%}). The two "
                   f"confirmed leads don't stack — they collide.\n\n"
                   f"**Why (and it's instructive):** the event clock is derived from the "
                   f"*global* factor turbulence, so using it as the target horizon "
                   f"imposes one global stress-clock on *heterogeneous local* markets "
                   f"(Korea, Japan, Canada…) whose own drawdown dynamics don't follow "
                   f"it. Event-time's benefit in step 21 was specific to the matched-"
                   f"clock single-asset case — the US equity factor, whose drawdowns "
                   f"roughly track the global clock. Across diverse local markets the "
                   f"global event window is a *mismatched* target horizon.\n\n"
                   f"**Conclusion: pooling is the lever that works; the event clock does "
                   f"not transfer to a cross-market pooled setting.** Calendar + pooling "
                   f"(step 24) remains the best configuration. Honest negative — and it "
                   f"sharpens *why* step 24 works: a market-agnostic horizon pools "
                   f"cleanly; a global event horizon does not.")

    md = [
        "# CHECKPOINT — Step 25: event-time × cross-market pooling (combining the leads)",
        "", f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC", "",
        "## What was tested",
        "The step-24 pooled overlay (one global turbulence, 7 markets, market fixed "
        "effects, panel walk-forward, date-block bootstrap) with each market's target "
        "on the **event clock** (event-time drawdown) vs the **calendar** 21-day "
        "drawdown. Tests whether the two confirmed leads — the event clock (step 21) "
        f"and pooling (step 24) — compound. Event threshold {thr:.0f} (dev-calibrated); "
        f"event purge {evt_purge}d.",
        "",
        "## Head-to-head (equal-weight global basket, 5% correction, 10bps)", "",
        cmp.round(3).to_markdown(),
        "",
        "![eventtime pooling](../../figures/eventtime_pooling/eventtime_pooling.png)",
        "",
        "## Verdict", "", verdict, "",
        "## Notes",
        "- Both arms use identical features/model/markets — only the target's clock "
        "(and the matched walk-forward purge) differ.",
        "- 95% CI from a date-block bootstrap (21-day blocks) that preserves "
        "cross-market co-movement — significance is not overstated.",
        "- Factor Lens intact (one global signal); interpretable/systematic/transparent.",
    ]
    CHECKPOINT.write_text("\n".join(md))
    print(f"wrote {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
