"""Step 29 — Robustness of the event-time-feature lift (step28).

Step28 found that adding the full event-time feature block flips the de-risking
overlay's 49-cutoff distribution strongly positive — but with two warning signs:
(i) no feature *subset* reproduces it, and (ii) ~60 effective OOS blocks cannot
resolve a +0.09 lift on its own.  This step stress-tests that result three ways,
to tell a *real* signal (survives changes that shouldn't matter) from a *lucky*
artefact (appears only at one exact configuration):

  Part 1 — EXTERNAL VALIDITY: re-run turbulence-only vs +event-time on a
           7-market equity panel (US factor + 6 regional ETFs).  Does adding the
           event-time block improve the overlay on most markets, or only the one
           it was built on?
  Part 2 — INTERNAL STABILITY (leave-one-feature-out): drop each event-time
           feature in turn (US factor).  Does the lift survive dropping any one,
           or does it hinge on a single load-bearing feature?
  Part 3 — PERIOD CONCENTRATION: split the 49 cutoffs at the COVID boundary —
           is the positive lift spread across the window or concentrated in the
           crisis cutoffs?

Honest caveats carried throughout: markets co-move, so N positive markets are not
N independent confirmations; everything is read as the *distribution* of the
49-cutoff Sortino lift, never a single number.

Run:  python3 -m regime_detection.step29_eventtime_feature_robustness
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
from regime_detection.lib.cutoff_sensitivity import run_sliding_cutoff, summary_statistics
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.event_time import EventTimeConverter
from regime_detection.lib.event_time_features import build_event_time_features
from regime_detection.extras.step20_correction_overlay import correction_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
REG_PATH = PROJECT_ROOT / "data" / "benchmarks_regional.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "eventtime_robustness"
FIG_DIR = PROJECT_ROOT / "figures" / "eventtime_robustness"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step29_eventtime_feature_robustness.md"
CACHE = PROJECT_ROOT / ".cache" / "eventtime_robustness"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"; HORIZON = 21; LOSS = 0.05
ET_COLS = ["et_intensity_to_date", "et_days_since_start", "et_velocity",
           "et_turb_per_day", "et_momentum"]
REGION_NAMES = {"VGK": "Europe", "EWJ": "Japan", "EWU": "UK",
                "EPP": "Pacific", "EWC": "Canada", "EWY": "Korea"}
CUTOFFS = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
COVID_SPLIT = pd.Timestamp("2020-06-30")


def _median(X, y, eq, tag):
    """49-cutoff Sortino-lift summary for one (features, asset). Distinct
    cache_dir per tag — run_sliding_cutoff caches by (model, cutoff) only."""
    Xf = X.dropna()
    idx = Xf.index.intersection(y.dropna().index).intersection(eq.dropna().index)
    Xf = Xf.loc[idx]
    df = run_sliding_cutoff(Xf, y.reindex(idx), eq.reindex(idx), cutoff_dates=CUTOFFS,
                            model_name=MODEL, purge_window=HORIZON,
                            cache_dir=CACHE / tag, verbose=False)
    s = summary_statistics(df, "sortino_lift")
    return s["median"], s["frac_positive"], df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    turb = pd.read_parquet(TURB_PATH)["turbulence"]
    regime = pd.read_parquet(TURB_PATH)["regime_smoothed"]
    reg = pd.read_parquet(REG_PATH).sort_index()
    X_turb = build_feature_matrix_with_bucket(prices, turb, regime,
                                              factors=FACTORS_10, n_buckets=5)
    _, diag = EventTimeConverter.calibrate(turb.loc[:DEV_END], target_mean_trading_days=21.0)
    thr = float(diag.threshold)

    markets = {"US": prices["equity"]}
    for tk, nm in REGION_NAMES.items():
        if tk in reg.columns:
            markets[nm] = reg[tk]

    # ---------------------------------------------------------- Part 1: panel
    print("[1] 7-market panel: turbulence-only vs +event-time ...")
    panel = []
    for mkt, px in markets.items():
        eq = np.log(px / px.shift(1)); y = correction_label(px, LOSS, HORIZON)
        et = build_event_time_features(turb, px, thr)
        m_b, p_b, _ = _median(X_turb, y, eq, f"p1_{mkt}_base")
        m_e, p_e, df_e = _median(X_turb.join(et), y, eq, f"p1_{mkt}_et")
        panel.append({"market": mkt, "base_median": m_b, "base_pos": p_b,
                      "et_median": m_e, "et_pos": p_e, "delta": m_e - m_b})
        print(f"  {mkt:8s} base {m_b:+.3f}/{p_b:.0%}  +ET {m_e:+.3f}/{p_e:.0%}  Δ{m_e-m_b:+.3f}")
        if mkt == "US":
            us_et_df = df_e
    panel = pd.DataFrame(panel).set_index("market")

    # --------------------------------------------- Part 2: leave-one-out (US)
    print("\n[2] leave-one-feature-out (US factor) ...")
    eq_us = np.log(prices["equity"] / prices["equity"].shift(1))
    y_us = correction_label(prices["equity"], LOSS, HORIZON)
    et_us = build_event_time_features(turb, prices["equity"], thr)
    loo = []
    full_m = panel.loc["US", "et_median"]
    base_m = panel.loc["US", "base_median"]
    for drop in ET_COLS:
        keep = [c for c in ET_COLS if c != drop]
        m, p, _ = _median(X_turb.join(et_us[keep]), y_us, eq_us, f"p2_drop_{drop}")
        loo.append({"dropped": drop, "median": m, "pos": p})
        print(f"  drop {drop:22s} median {m:+.3f}/{p:.0%}")
    loo = pd.DataFrame(loo).set_index("dropped")

    # ------------------------------------------- Part 3: period concentration
    print("\n[3] period concentration (US +event-time) ...")
    lifts = us_et_df["sortino_lift"].dropna()
    lifts.index = pd.to_datetime(lifts.index)
    early = lifts[lifts.index <= COVID_SPLIT]; late = lifts[lifts.index > COVID_SPLIT]
    print(f"  early cutoffs (<= {COVID_SPLIT.date()}): median {early.median():+.3f} (n={len(early)})")
    print(f"  late  cutoffs (>  {COVID_SPLIT.date()}): median {late.median():+.3f} (n={len(late)})")

    _figure(panel, loo, lifts, base_m, full_m)
    _write_checkpoint(panel, loo, early, late, base_m, full_m, thr)
    print(f"\nWrote checkpoint → {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _figure(panel, loo, lifts, base_m, full_m):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # panel
    ax = axes[0]; x = np.arange(len(panel))
    ax.bar(x - 0.2, panel["base_median"], 0.4, label="turb-only", color="tab:gray")
    ax.bar(x + 0.2, panel["et_median"], 0.4, label="+event-time", color="tab:red")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x)
    ax.set_xticklabels(panel.index, rotation=40, ha="right")
    ax.set_ylabel("49-cutoff median Sortino lift"); ax.set_title("Part 1 — 7-market panel"); ax.legend(fontsize=8)
    # leave-one-out
    ax = axes[1]
    ax.axhline(full_m, color="tab:red", ls="--", lw=1, label=f"full 5-feat ({full_m:+.3f})")
    ax.axhline(base_m, color="0.5", ls=":", lw=1, label=f"baseline ({base_m:+.3f})")
    ax.bar(range(len(loo)), loo["median"], color="tab:purple")
    ax.set_xticks(range(len(loo)))
    ax.set_xticklabels([d.replace("et_", "drop ") for d in loo.index], rotation=40, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("49-cutoff median (US)")
    ax.set_title("Part 2 — leave-one-feature-out"); ax.legend(fontsize=7)
    # period
    ax = axes[2]
    ax.plot(lifts.index, lifts.values, "o-", ms=4, color="tab:red")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(COVID_SPLIT, color="0.5", ls="--", lw=1)
    ax.set_title("Part 3 — lift per cutoff (US +event-time)"); ax.set_ylabel("Sortino lift")
    fig.suptitle("Step 29 — robustness of the event-time-feature lift")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eventtime_robustness.png", dpi=130)
    plt.close(fig)


def _write_checkpoint(panel, loo, early, late, base_m, full_m, thr):
    n_improved = int((panel["delta"] > 0).sum())
    n_positive = int((panel["et_median"] > 0).sum())
    n_mkt = len(panel)
    loo_min = loo["median"].min(); loo_max = loo["median"].max()
    loo_all_pos = bool((loo["median"] > 0).all())
    crisis_concentrated = (late.median() > 0.05) and (early.median() <= 0)

    L = []
    L.append("# Step 29 — Robustness of the event-time-feature lift\n")
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. 2020-start OOS; 5%/21d "
             f"correction; logistic_l1; event threshold {thr:.1f} (dev-calibrated). "
             "Stress-tests step28's +event-time overlay lift._\n")

    L.append("## Part 1 — 7-market panel (external validity)\n")
    L.append("| market | turb-only (median / %pos) | +event-time | Δ median |")
    L.append("|---|---|---|---|")
    for m in panel.index:
        r = panel.loc[m]
        L.append(f"| {m} | {r['base_median']:+.3f} / {r['base_pos']:.0%} | "
                 f"{r['et_median']:+.3f} / {r['et_pos']:.0%} | {r['delta']:+.3f} |")
    L.append(f"\n**{n_improved}/{n_mkt} markets improve** with the event-time block; "
             f"**{n_positive}/{n_mkt}** have a positive +event-time median. "
             f"Mean Δ = {panel['delta'].mean():+.3f}. *(Markets co-move — read as "
             "directional consistency, not independent confirmations.)*\n")

    L.append("## Part 2 — leave-one-feature-out (US factor)\n")
    L.append(f"Baseline (no event-time) {base_m:+.3f}; full 5-feature {full_m:+.3f}.\n")
    L.append("| dropped feature | 49-cut median | %pos |")
    L.append("|---|---|---|")
    for d in loo.index:
        L.append(f"| {d} | {loo.loc[d,'median']:+.3f} | {loo.loc[d,'pos']:.0%} |")
    L.append(f"\nLeave-one-out range [{loo_min:+.3f}, {loo_max:+.3f}]; "
             f"all four-feature variants {'stay POSITIVE' if loo_all_pos else 'do NOT all stay positive'}.")
    if not loo_all_pos:
        worst = loo['median'].idxmin()
        L.append(f" Dropping **{worst}** collapses the lift → it is load-bearing "
                 "(the result is not distributed across features — a fragility sign).")
    L.append("")

    L.append("## Part 3 — period concentration\n")
    L.append(f"- Early cutoffs (≤ {COVID_SPLIT.date()}): median lift "
             f"**{early.median():+.3f}** (n={len(early)}).")
    L.append(f"- Late cutoffs (> {COVID_SPLIT.date()}): median lift "
             f"**{late.median():+.3f}** (n={len(late)}).")
    L.append("")

    # overall verdict
    L.append("## Verdict\n")
    strong = (n_improved >= 5 and n_positive >= 5 and loo_all_pos and not crisis_concentrated)
    partial = (n_improved >= 4 or n_positive >= 4)
    if strong:
        L.append("**ROBUST.** The event-time lift generalises across the panel "
                 f"({n_improved}/{n_mkt} markets improve, {n_positive}/{n_mkt} positive), "
                 "survives dropping any single feature, and is not confined to the "
                 "crisis cutoffs. This is the project's first genuinely robust, "
                 "directionally-positive overlay result — defensible as a real "
                 "(if not 95%-significant) edge.")
    elif partial:
        L.append(f"**PARTIALLY ROBUST — directionally real, not significant.** The "
                 f"lift improves on {n_improved}/{n_mkt} markets "
                 f"({n_positive}/{n_mkt} positive) but with wide spread"
                 + (", and depends on specific features (leave-one-out not all positive)"
                    if not loo_all_pos else "")
                 + (", and is concentrated in the crisis cutoffs"
                    if crisis_concentrated else "")
                 + ". Consistent with the project's sample-size ceiling: a real "
                 "direction, reported distributionally, not a significance claim.")
    else:
        L.append(f"**NOT ROBUST — likely an artefact.** The lift fails to "
                 f"generalise ({n_improved}/{n_mkt} markets improve, "
                 f"{n_positive}/{n_mkt} positive)"
                 + ("" if loo_all_pos else " and/or hinges on a single feature")
                 + ". Honest cautionary finding: step28's +0.089 was specific to the "
                 "equity factor and does not hold up — report as a negative.")
    L.append("")
    L.append("**Figure:** `figures/eventtime_robustness/eventtime_robustness.png`.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
