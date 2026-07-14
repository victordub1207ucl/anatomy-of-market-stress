"""Step 24 — cross-market pooling: one Factor-Lens turbulence, many benchmarks.

The honest way to raise the sample-size ceiling: keep the SAME global
turbulence signal (unchanged — no per-market turbulence, so the Factor Lens
stays intact) and apply the correction overlay to a panel of regional equity
benchmarks (US, Europe, Japan, UK, Pacific, Canada, Korea). Pooling the
(market, date) observations multiplies the events available to estimate the
common turbulence → correction relationship.

Stays interpretable / systematic / transparent:
  * One global signal → one logistic_l1 model, applied uniformly to every market.
  * **Market fixed effects** (dummies) absorb per-market base rates, so a real
    common signal isn't blurred by heterogeneity.
  * **Honest inference:** markets co-move (esp. in crises), so N markets ≠ N×
    independent data. The headline is an equal-weight global basket overlay vs
    buy-and-hold, with a **date-block bootstrap** (resamples whole dates, keeping
    cross-market dependence) — never overstating the power gain.

Run:  python3 -m regime_detection.step24_cross_market_pooling
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
from sklearn.preprocessing import StandardScaler

from regime_detection.lib.train import MODEL_BUILDERS, _predict_proba
from regime_detection.lib.metrics import sortino_ratio, max_drawdown
from regime_detection.lib.targets import forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.splits import DEV_END, OOS_START

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
REG_PATH = PROJECT_ROOT / "data" / "benchmarks_regional.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "cross_market_pooling"
FIG_DIR = PROJECT_ROOT / "figures" / "cross_market_pooling"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step24_cross_market_pooling.md"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"; LOSS = 0.05; HORIZON = 21; COST_BPS = 10
REGION_NAMES = {"VGK": "Europe", "EWJ": "Japan", "EWU": "UK",
                "EPP": "Pacific", "EWC": "Canada", "EWY": "Korea"}


def overlay(eq_ret, in_market, cost_bps):
    pos = in_market.shift(1).fillna(1.0)
    cost = (cost_bps / 1e4) * pos.diff().abs().fillna(0.0)
    return eq_ret * pos - cost


def pooled_walk_forward(panel, feat_cols, purge_bdays=HORIZON):
    """Quarterly expanding-window walk-forward on a (market,date) panel.
    Returns the panel restricted to OOS rows with a 'proba' column."""
    qs = pd.date_range(DEV_END, panel["date"].max(), freq="QE")
    out = []
    for i in range(len(qs) - 1):
        q0, q1 = qs[i], qs[i + 1]
        train_end = q0 - pd.tseries.offsets.BDay(purge_bdays)
        tr = panel[panel["date"] <= train_end]
        te = panel[(panel["date"] > q0) & (panel["date"] <= q1)]
        if len(tr) < 200 or te.empty or tr["y"].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feat_cols].values)
        est = MODEL_BUILDERS[MODEL](random_state=0)
        est.fit(sc.transform(tr[feat_cols].values), tr["y"].values)
        p = _predict_proba(est, sc.transform(te[feat_cols].values))
        te = te.copy(); te["proba"] = p
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH); turb = tdf["turbulence"]
    X = build_feature_matrix_with_bucket(prices, turb, tdf["regime_smoothed"],
                                         factors=FACTORS_10, n_buckets=5)
    reg = pd.read_parquet(REG_PATH)

    # assemble the asset price panel: US (equity factor) + 6 regions
    assets = {"US": prices["equity"]}
    for tk, nm in REGION_NAMES.items():
        assets[nm] = reg[tk].reindex(prices.index).ffill(limit=3)
    markets = list(assets)
    print(f"markets ({len(markets)}): {markets}")

    # build the long panel: one block per market, shared global features + market dummy
    feat_base = list(X.columns)
    dummy_cols = [f"mkt_{m}" for m in markets[1:]]   # drop first market = reference
    blocks = []
    for m in markets:
        price = assets[m]
        y = (forward_drawdown_conditional(price, horizon=HORIZON) <= -LOSS).astype(float)
        y = y.where(forward_drawdown_conditional(price, horizon=HORIZON).notna())
        ret = np.log(price / price.shift(1))
        b = X.copy()
        b = b.reindex(y.dropna().index).dropna()
        b["y"] = y.reindex(b.index); b["ret"] = ret.reindex(b.index)
        b["mkt"] = m; b["date"] = b.index
        for d in dummy_cols:
            b[d] = 1.0 if d == f"mkt_{m}" else 0.0
        blocks.append(b.dropna(subset=["y", "ret"]))
    panel = pd.concat(blocks, ignore_index=True)
    feat_cols = feat_base + dummy_cols
    print(f"pooled panel: {len(panel):,} (market,date) rows, base rate {panel['y'].mean():.3f}")

    # --- pooled walk-forward ---
    oos = pooled_walk_forward(panel, feat_cols)
    auc_pool = roc_auc_score(oos["y"], oos["proba"]) if oos["y"].nunique() > 1 else np.nan

    # --- per-market overlay (top-decile within market) + equal-weight basket ---
    per_mkt, basket_strat, basket_bh = {}, [], []
    for m in markets:
        g = oos[oos["mkt"] == m].set_index("date").sort_index()
        if g.empty: continue
        in_mkt = (g["proba"].rank(pct=True) < 0.90).astype(float)
        strat = overlay(g["ret"], in_mkt, COST_BPS)
        bh = g["ret"]
        per_mkt[m] = {"auc": roc_auc_score(g["y"], g["proba"]) if g["y"].nunique() > 1 else np.nan,
                      "lift": sortino_ratio(strat.dropna()) - sortino_ratio(bh.dropna()),
                      "maxdd_strat": max_drawdown(strat.dropna()), "maxdd_bh": max_drawdown(bh.dropna())}
        basket_strat.append(strat.rename(m)); basket_bh.append(bh.rename(m))
    pm = pd.DataFrame(per_mkt).T
    # equal-weight global basket (rebalanced daily)
    bs = pd.concat(basket_strat, axis=1).mean(axis=1).dropna()
    bb = pd.concat(basket_bh, axis=1).mean(axis=1).reindex(bs.index)
    basket_lift = sortino_ratio(bs) - sortino_ratio(bb)
    n_pos = int((pm["lift"] > 0).sum())

    # --- date-block bootstrap of the basket Sortino lift (respects co-movement) ---
    rng = np.random.default_rng(0); block = 21; n_boot = 2000
    diff = (bs - bb).values; n = len(diff); s_bh = sortino_ratio(bb)
    lifts = np.empty(n_boot)
    for i in range(n_boot):
        idx = []
        while len(idx) < n:
            st = int(rng.integers(0, n)); idx.extend(range(st, min(st + block, n)))
        idx = np.array(idx[:n])
        strat_bs = pd.Series(bb.values[idx] + diff[idx])
        lifts[i] = sortino_ratio(strat_bs) - sortino_ratio(pd.Series(bb.values[idx]))
    ci = (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5)))
    p_pos = float((lifts > 0).mean())
    print(f"pooled AUC {auc_pool:.3f}; basket lift {basket_lift:+.3f} "
          f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] P(>0)={p_pos:.2f}; {n_pos}/{len(pm)} markets positive")
    print(pm.round(3).to_string())
    pm.to_csv(OUT_DIR / "per_market.csv")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    a = ax[0]
    a.plot(bb.index, bb.cumsum(), color="#7f8c8d", label="equal-weight basket B&H")
    a.plot(bs.index, bs.cumsum(), color="#c0392b", ls="--", label=f"basket + overlay ({COST_BPS}bps)")
    a.set_title("Global equity basket — overlay vs buy-and-hold (OOS)"); a.legend(fontsize=9); a.grid(alpha=0.3)
    a = ax[1]
    order = pm["lift"].sort_values()
    a.barh(order.index, order.values, color=["#2ca02c" if v > 0 else "#c0392b" for v in order.values])
    a.axvline(0, color="k", lw=0.8); a.set_xlabel("overlay Sortino lift vs B&H")
    a.set_title(f"Per-market overlay lift ({n_pos}/{len(pm)} positive)"); a.grid(axis="x", alpha=0.3)
    a = ax[2]
    a.hist(lifts, bins=40, color="#8e44ad", alpha=0.7)
    a.axvline(0, color="k", lw=0.9)
    a.axvline(basket_lift, color="#c0392b", ls="--", lw=1.5, label=f"observed {basket_lift:+.3f}")
    a.axvspan(ci[0], ci[1], color="#8e44ad", alpha=0.12, label=f"95% CI [{ci[0]:+.2f},{ci[1]:+.2f}]")
    a.set_xlabel("basket Sortino lift"); a.set_title(f"Date-block bootstrap  ·  P(lift>0)={p_pos:.0%}")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Step 24 — cross-market pooling (one global turbulence, 7 equity benchmarks)",
                 fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cross_market_pooling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {FIG_DIR / 'cross_market_pooling.png'}")

    # ---- verdict ----
    sig = ci[0] > 0
    if sig:
        verdict = (f"**POSITIVE AND BOOTSTRAP-SIGNIFICANT.** Pooling a single global "
                   f"turbulence signal across {len(pm)} equity benchmarks gives an "
                   f"equal-weight basket overlay Sortino lift of {basket_lift:+.3f}, "
                   f"with a date-block-bootstrap 95% CI of [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                   f"that **excludes zero** (P(lift>0)={p_pos:.0%}); {n_pos}/{len(pm)} "
                   "markets positive. The cross-market panel raised the effective "
                   "sample enough to resolve a positive edge the single US series "
                   "could not — and the Factor Lens is untouched (one global signal).")
    elif basket_lift > 0 and p_pos > 0.8:
        verdict = (f"**ENCOURAGING, NOT YET SIGNIFICANT.** Basket overlay lift "
                   f"{basket_lift:+.3f} (P(lift>0)={p_pos:.0%}, {n_pos}/{len(pm)} markets "
                   f"positive), but the 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] still "
                   "includes zero — cross-market co-movement caps the effective sample "
                   "gain. A real directional improvement over the US-only result "
                   "(which was clearly negative); pooling helps but doesn't fully "
                   "break the ceiling with 7 markets.")
    else:
        verdict = (f"**NO POOLED EDGE.** Basket lift {basket_lift:+.3f}, CI "
                   f"[{ci[0]:+.3f}, {ci[1]:+.3f}], {n_pos}/{len(pm)} markets positive. "
                   "Pooling does not surface a robust correction-timing edge; the "
                   "weak signal is consistent across markets, not a US artefact. The "
                   "value is the honest multi-market test and the validated panel "
                   "framework (Factor Lens intact).")

    md = [
        "# CHECKPOINT — Step 24: cross-market pooling",
        "", f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC", "",
        "## What was built",
        f"One **global** Factor-Lens turbulence signal (unchanged) applied to a "
        f"panel of **{len(markets)} regional equity benchmarks** ({', '.join(markets)}), "
        "pooled into a single logistic_l1 with **market fixed effects**, quarterly "
        "walk-forward. Honest inference via a **date-block bootstrap** (resamples "
        "whole dates → preserves cross-market co-movement). The Factor Lens is not "
        "diverged from — there is no per-market turbulence; the global signal is "
        "tested across markets.",
        "",
        "## Result (OOS 2020+, 5% correction, equal-weight basket, 10bps)",
        "",
        f"- **Pooled model AUC:** {auc_pool:.3f} (across {len(panel):,} market-date rows).",
        f"- **Basket overlay Sortino lift:** {basket_lift:+.3f}; date-block-bootstrap "
        f"95% CI **[{ci[0]:+.3f}, {ci[1]:+.3f}]**, P(lift>0) = {p_pos:.0%}.",
        f"- **{n_pos}/{len(pm)} markets** have a positive overlay lift.",
        "",
        pm.round(3).to_markdown(),
        "",
        "![pooling](../../figures/cross_market_pooling/cross_market_pooling.png)",
        "",
        "## Verdict", "", verdict, "",
        "## Why this respects the Factor-Lens requirements",
        "- **One global signal**, not per-market turbulence — the lens is intact and "
        "its universality is *tested* across geographies.",
        "- **Interpretable/systematic:** a single logistic_l1 + market dummies (a "
        "panel regression), applied uniformly; every prediction is traceable.",
        "- **Honest power accounting:** the date-block bootstrap respects that markets "
        "co-move, so the effective sample gain (and any significance) is not overstated.",
        "",
        "## Notes",
        "- Regional ETFs (VGK/EWJ/EWU/EPP/EWC/EWY, 2007+) cached in "
        "`data/benchmarks_regional.parquet`. US = the equity factor.",
        "- Next lever if more power is needed: add sectors / more regions, or combine "
        "with the event clock (step 21).",
    ]
    CHECKPOINT.write_text("\n".join(md))
    print(f"wrote {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
