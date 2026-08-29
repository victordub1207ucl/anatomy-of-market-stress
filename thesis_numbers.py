"""Figures quoted in the thesis that are not produced by a numbered pipeline step.

Each block prints one or more numbers that appear in the write-up, so every quoted
value has code behind it. Run:  python3 thesis_numbers.py [--slow]

--slow additionally re-derives the sector-universe archetypes, which recomputes a
walk-forward decomposition over the nine SPDR sectors and takes a few minutes.
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
from regime_detection.lib.splits import DEV_END                      # noqa: E402
from regime_detection.lib.crisis_typology import (                    # noqa: E402
    extract_episodes, episode_composition)
from regime_detection.lib.typology_hedge import signed_duration_trend  # noqa: E402

SEED, N_BOOT, HORIZON = 0, 10_000, 21


def _rd(p):
    p = ROOT / p
    return pd.read_parquet(p) if str(p).endswith("parquet") else pd.read_csv(p)


def matched_exposure_blend():
    """Table 6.4: the static blend at the routed overlay's own average equity weight."""
    cur = _rd("reports/typology_hedge/oos_curves.parquet")
    h = _rd("data/hedge_assets.parquet").sort_index()
    r = np.log(h / h.shift(1)).reindex(cur.index)
    turb = _rd("reports/turbulence/turbulence_series.parquet")["turbulence"].dropna()
    gate = (turb > turb.loc[:DEV_END].quantile(0.75)).shift(1).reindex(cur.index)
    w = 1 - gate.mean()

    # use the pipeline's own statistics so the rows match Table 6.4 exactly
    from regime_detection.lib.typology_hedge import performance_summary

    def perf(x):
        d = performance_summary(x.dropna())
        return d["total_return"], d["sortino"], d["max_drawdown"]

    print("== Table 6.4: matched-exposure blend ==")
    print(f"  gate fires on {int(gate.sum())} of {len(cur)} evaluation days "
          f"({gate.mean():.1%}); overlay equity weight {w:.1%}")
    # a daily-rebalanced blend averages SIMPLE returns, then is carried back to log space
    simple = np.expm1(r)
    blend = np.log1p(w * simple["SPY"] + (1 - w) * simple["BIL"])
    blend0 = np.log1p(w * simple["SPY"])          # same weight against a zero-yield cash proxy
    for name, s in [("buy-and-hold", cur["buy_and_hold"]),
                    ("composition-aware (routed)", cur["composition_aware"]),
                    (f"static, matched ({w:.1%} SPY + BIL)", blend),
                    ("matched, zero-yield cash", blend0)]:
        tot, srt, dd = perf(s)
        print(f"  {name:36s} total {tot:+7.1%}  Sortino {srt:5.2f}  maxDD {dd:7.1%}")
    print(f"  BIL annualised return {r['BIL'].mean()*252:.2%}  "
          "(why the matched blend's Sortino exceeds buy-and-hold's)")


def visibility_gap_interval():
    """Section 5.3: episode-bootstrap 95% interval on the observed-minus-null gap."""
    c = _rd("reports/decomposition/turbulence_contributions.parquet").dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep)
    idx = c.index
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    pre, real = [], []
    for s0 in comp.index:
        loc = idx.searchsorted(s0)
        seg = c.iloc[max(0, loc - HORIZON):loc].dropna(how="all").mean(axis=0)
        if seg.sum() <= 0:
            continue
        pre.append((seg / seg.sum()).values)
        real.append(comp.loc[s0].values)
    pre, real = np.array(pre), np.array(real)
    n = len(pre)
    rng = np.random.default_rng(SEED)
    obs = np.mean([cos(pre[i], real[i]) for i in range(n)])
    null = np.mean([np.mean([cos(pre[i], real[p[i]]) for i in range(n)])
                    for p in (rng.permutation(n) for _ in range(2000))])
    boot = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        o = np.mean([cos(pre[i], real[i]) for i in ix])
        q = rng.permutation(len(ix))
        nl = np.mean([cos(pre[ix[k]], real[ix[q[k]]]) for k in range(len(ix))])
        boot.append(o - nl)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n== Section 5.3: Visibility ==")
    print(f"  n={n}  observed {obs:.4f}  null {null:.4f}  gap {obs-null:+.4f}")
    print(f"  episode-bootstrap 95% CI on the gap [{lo:+.3f}, {hi:+.3f}]")


def duration_trend_block_bootstrap():
    """Section 6.3.4: the routing correlation under date-block resampling."""
    turb = _rd("reports/turbulence/turbulence_series.parquet")["turbulence"].dropna()
    h = _rd("data/hedge_assets.parquet").sort_index()
    hret = np.log(h / h.shift(1))
    gate = turb > turb.loc[:DEV_END].quantile(0.75)
    trend = signed_duration_trend(hret["TLT"], 63)
    fwd = (hret["TLT"] - hret["GLD"]).shift(-1).rolling(HORIZON).sum().shift(-(HORIZON - 1))
    df = pd.concat({"s": trend, "g": gate, "f": fwd}, axis=1).dropna()
    df = df[df["g"]]
    oos = df[df.index > DEV_END]
    rho, p = stats.spearmanr(oos["s"], oos["f"])
    n_gate = int(gate[gate.index > DEV_END].sum())
    x, y, n = oos["s"].values, oos["f"].values, len(oos)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / HORIZON))
    boot = []
    for _ in range(N_BOOT):
        st = rng.integers(0, n - HORIZON, nb)
        ix = np.concatenate([np.arange(s0, s0 + HORIZON) for s0 in st])[:n]
        boot.append(stats.spearmanr(x[ix], y[ix])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n== Section 6.3.4: signed duration trend vs forward bonds-minus-gold ==")
    print(f"  gated held-out days {n_gate}; with a complete {HORIZON}-day forward window {n} "
          f"({n_gate - n} lost to the label)")
    print(f"  Spearman rho {rho:+.3f}  naive p {p:.3f}")
    print(f"  date-block bootstrap (block={HORIZON}) 95% CI [{lo:+.2f}, {hi:+.2f}]")


def silhouette_and_meta_clusters():
    """Section 5.3: silhouette profile, and the 90-day meta-crisis grouping."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    c = _rd("reports/decomposition/turbulence_contributions.parquet").dropna()
    ep = extract_episodes(c.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(c, ep)
    prof = [silhouette_score(comp.values,
            AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(comp.values))
            for k in range(2, 7)]
    print("\n== Section 5.3: cluster count and independence ==")
    print("  silhouette k=2..6: " + ", ".join(f"{v:.3f}" for v in prof))
    st = pd.DatetimeIndex(comp.index)
    gaps = np.diff(st.values).astype("timedelta64[D]").astype(int)
    cl = np.concatenate([[0], np.cumsum(gaps > 90)])
    oos = st > pd.Timestamp(DEV_END)
    print(f"  90-day meta-crisis rule: {len(st)} episodes -> {len(set(cl))} clusters; "
          f"held out {int(oos.sum())} -> {len(set(cl[oos]))}")


def dollar_inbound_links():
    """Section 4.3 and Figure 4.4: inbound links to USD among the strongest eleven."""
    rw = _rd("reports/causality/rewiring_differential.csv").rename(
        columns={"causing_factor": "src", "caused_factor": "dst"})
    fac = ["equity", "rates", "credit", "commodities", "em_equity", "fx_usd",
           "inflation", "value", "quality", "vix"]
    rw = rw[rw.src.isin(fac) & rw.dst.isin(fac)]
    print("\n== Section 4.3: the dollar as a net sink ==")
    for col, lbl in [("calm_strength", "calm"), ("turb_strength", "turbulent")]:
        top = rw.nlargest(11, col)
        print(f"  {lbl:9s}: {int((top.dst=='fx_usd').sum())} of the 11 strongest links "
              "point into the dollar")


def post2019_correction_counts():
    """Section 6.3.5: why the de-risking test uses the longer window."""
    h = _rd("data/hedge_assets.parquet").sort_index()
    spy = h["SPY"]
    fwd = (spy.shift(-1).rolling(HORIZON).min().shift(-(HORIZON - 1)) / spy) - 1
    print("\n== Section 6.3.5: independent corrections by window ==")
    for lbl, start in [("2015-2026", "2015-01-01"), ("post-2019", "2020-01-01")]:
        seg = fwd.loc[start:].dropna()
        out = []
        for thr in (0.03, 0.05, 0.08):
            idx = seg[seg <= -thr].index
            n = 0 if len(idx) == 0 else 1 + int(
                (np.diff(idx.values).astype("timedelta64[D]").astype(int) > HORIZON).sum())
            out.append(f"{int(thr*100)}%: ~{n}")
        print(f"  {lbl:10s} " + "   ".join(out))


def sector_archetypes():
    """Section 5.3: the four sector-universe archetypes (slow)."""
    from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer
    from regime_detection.lib.crisis_typology import cluster_episodes
    names = {"XLF": "Financials", "XLK": "Technology", "XLE": "Energy", "XLV": "Health Care",
             "XLI": "Industrials", "XLY": "Cons. Disc.", "XLP": "Cons. Staples",
             "XLU": "Utilities", "XLB": "Materials"}
    px = _rd("data/sector_prices.parquet").rename(columns=names)
    rets = np.log(px / px.shift(1)).dropna(how="all")
    contrib = TurbulenceDecomposer(lookback_days=2520, min_periods=504,
                                   shrinkage="ledoit-wolf").fit_transform(rets).dropna()
    ep = extract_episodes(contrib.sum(axis=1).dropna(), min_periods=252)
    comp = episode_composition(contrib, ep)
    labels, k, _, _ = cluster_episodes(comp, k_range=range(2, 6))
    print(f"\n== Section 5.3: sector universe ({len(comp)} episodes, k={k}) ==")
    for ci in sorted(set(labels)):
        top = comp[labels == ci].mean().sort_values(ascending=False).head(2)
        print(f"  cluster {ci} (n={int((labels==ci).sum())}): "
              + ", ".join(f"{f} {v:.2f}" for f, v in top.items()))


def granger_f_ranking():
    """Table 4.3 — the Granger F column.

    step16 persists the min-over-lags p-value, which is what the ranking and the
    "leads escalation?" verdict come from; the F ratios printed alongside them are
    the largest ssr-F over lags 1..5 on the full turbulence window (every day with
    a turbulence value, before the regime intersection step16 applies for its
    regime-conditional half). The ordering is the same either way.
    """
    import io, contextlib
    from statsmodels.tsa.stattools import grangercausalitytests
    F10 = ["equity", "rates", "credit", "commodities", "em_equity",
           "fx_usd", "inflation", "value", "quality", "vix"]
    px = _rd("data/factor_prices.parquet").sort_index()[F10]
    rets = np.log(px / px.shift(1)).dropna(how="any")
    turb = _rd("reports/turbulence/turbulence_series.parquet")["turbulence"]
    y = np.log(turb.replace(0, np.nan)).diff().rename("dlog_turb")
    data = pd.concat([y, rets], axis=1).dropna()
    rows = []
    for f in F10:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data[["dlog_turb", f]], maxlag=5)
        Fs = [res[l][0]["ssr_ftest"][0] for l in range(1, 6)]
        ps = [res[l][0]["ssr_chi2test"][1] for l in range(1, 6)]
        rows.append({"factor": f, "F": max(Fs), "min_p": min(ps)})
    df = pd.DataFrame(rows).sort_values("min_p").reset_index(drop=True)
    print(f"\n== Table 4.3: Granger drivers of escalation (n={len(data)}) ==")
    for i, r in df.iterrows():
        print(f"  {i+1:2d}. {r.factor:12s} F {r.F:6.2f}   min-lag p {r.min_p:.2e}")


def frozen_threshold_variants():
    """Table A.6 — the three implementable rules built on a frozen dev threshold.

    step20 persists the walk-forward scores and the dev-selected policy; these three
    rows freeze the dev 90th percentile of the *model score* instead and are what the
    text means by a rule that could have been run in real time.
    """
    from regime_detection.lib.metrics import sortino_ratio
    from regime_detection.extras.step20_correction_overlay import (
        overlay_with_costs, HEADLINE_COST)
    dev = _rd("reports/correction_overlay/headline_dev_5pct.parquet")
    oos = _rd("reports/correction_overlay/headline_oos_5pct.parquet")
    r = oos["eq_ret"]
    bh = sortino_ratio(r.dropna())

    def lift(in_market):
        strat, _, _ = overlay_with_costs(
            r, pd.Series(in_market, index=r.index), HEADLINE_COST)
        return sortino_ratio(strat.dropna()) - bh

    p = oos["pred_raw"].values
    tau = float(np.percentile(dev["pred_raw"], 90))
    med = float(np.percentile(dev["pred_raw"], 50))
    fires = p >= tau

    plain = (~fires).astype(float)

    out, hysteresis = False, np.ones(len(p))          # re-enter at the dev median
    for i, x in enumerate(p):
        if not out and x >= tau:
            out = True
        elif out and x < med:
            out = False
        hysteresis[i] = 0.0 if out else 1.0

    capped, spell = np.ones(len(p)), 0                # cash spells capped at 21 days
    for i, x in enumerate(p):
        if x < tau:
            spell = 0
        elif spell < HORIZON:
            capped[i], spell = 0.0, spell + 1
        else:
            spell = 0

    print("\n== Table A.6: rules on a frozen dev 90th-percentile score threshold ==")
    print(f"  threshold {tau:.4f} is exceeded on {100*fires.mean():.0f}% of held-out days "
          f"(the dev distribution puts 10% above it)")
    print(f"  frozen threshold                 Sortino lift {lift(plain):+.3f}")
    print(f"    with re-entry at the dev median             {lift(hysteresis):+.3f}")
    print(f"    with cash spells capped at {HORIZON} days           {lift(capped):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true",
                    help="also re-derive the sector-universe archetypes")
    args = ap.parse_args()
    granger_f_ranking()
    matched_exposure_blend()
    visibility_gap_interval()
    duration_trend_block_bootstrap()
    silhouette_and_meta_clusters()
    dollar_inbound_links()
    post2019_correction_counts()
    frozen_threshold_variants()
    if args.slow:
        sector_archetypes()
    else:
        print("\n(sector archetypes: re-run with --slow)")
