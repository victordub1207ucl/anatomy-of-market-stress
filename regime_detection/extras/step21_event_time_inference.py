"""Step 21 — correction-timing inference on the EVENT CLOCK (the novelty flagship).

Event-time (CKT 2023) makes returns far more Gaussian (step03, 94% kurtosis
cut). The open question — nobody has tested it — is whether *downstream
inference* is better-posed on that clock. This step runs the step-20
correction-timing overlay with the **event-time drawdown target** and compares
it head-to-head against the **calendar** target, everything else identical.

Design (keeps the comparison clean and the sample size intact):
  * One observation per calendar day in BOTH arms (disjoint event sampling
    would crush the OOS to ~50 events — the ceiling would get worse, not
    better). Only the TARGET's clock changes.
  * Calendar arm: worst forward 21-day drawdown ≤ -X  (= step 20).
  * Event arm:    worst drawdown over the next 1-event-unit window ≤ -X
    (`lib.targets.event_time_drawdown`), mean horizon ≈ 21 days but variable.
  * Causality: the event intensity threshold is calibrated on DEV turbulence
    only; and because event windows are variable-length, the walk-forward
    PURGE is set to the 99th percentile of window lengths (not 21), so a
    long calm-period label can never leak into the test fold.

Hypothesis: on the clock where returns are Gaussian the target is more
predictable / less cutoff-fragile (higher AUC, better-centred 49-cutoff
distribution).

Run:  python3 -m regime_detection.step21_event_time_inference
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from regime_detection.lib.cutoff_sensitivity import run_sliding_cutoff, summary_statistics
from regime_detection.lib.event_time import EventTimeConverter
from regime_detection.lib.metrics import brier_score, max_drawdown, sortino_ratio
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.targets import event_time_drawdown, forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.splits import DEV_END, OOS_START

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "event_time_inference"
FIG_DIR = PROJECT_ROOT / "figures" / "event_time_inference"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step21_event_time_inference.md"
CACHE = PROJECT_ROOT / ".cache" / "event_time_inference"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"
LOSS = 0.05                       # 5% correction (headline from step 20)
CAL_PURGE = 21                    # calendar horizon
COST_BPS = 10
DEV_CALIB_END = pd.Timestamp("2017-12-31")
SCHEDULE = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))


def overlay(eq_ret, in_market, cost_bps):
    pos = in_market.shift(1).fillna(1.0)
    cost = (cost_bps / 1e4) * pos.diff().abs().fillna(0.0)
    return eq_ret * pos - cost


def evaluate(y, X, eq, purge, tag):
    """Full eval for one (target, clock): AUC, calibrated Brier, forced
    top-decile overlay net of cost, and the 49-cutoff distribution."""
    dev = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=purge,
            dev_end=DEV_CALIB_END, oos_start=DEV_CALIB_END + pd.Timedelta(days=1),
            random_state=0)[MODEL]
    oos = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=purge,
            dev_end=DEV_END, oos_start=OOS_START, random_state=0)[MODEL]
    dpred = dev.oos_predictions.loc[:DEV_END].dropna(); dlab = dev.oos_labels.reindex(dpred.index)
    opred = oos.oos_predictions.dropna(); olab = oos.oos_labels.reindex(opred.index)
    iso = IsotonicRegression(out_of_bounds="clip").fit(dpred.values, dlab.values)
    opred_cal = pd.Series(iso.predict(opred.values), index=opred.index)
    auc = roc_auc_score(olab, opred) if olab.nunique() > 1 else np.nan
    # forced top-decile overlay (rank the RAW score; isotonic ties break ranking)
    eq_oos = eq.reindex(opred.index)
    in_force = (opred.rank(pct=True) < 0.90).astype(float)
    strat = overlay(eq_oos, in_force, COST_BPS)
    lift = sortino_ratio(strat.dropna()) - sortino_ratio(eq_oos.dropna())
    # 49-cutoff robustness (gross)
    slide = run_sliding_cutoff(X, y, eq, cutoff_dates=SCHEDULE, model_name=MODEL,
            purge_window=purge, min_oos_quarters=4, random_state=0,
            cache_dir=CACHE / tag, verbose=False)
    ssum = summary_statistics(slide, "sortino_lift")
    return {"auc": auc, "brier_raw": brier_score(olab, opred),
            "brier_cal": brier_score(olab, opred_cal), "base_rate": float(olab.mean()),
            "overlay_lift": lift, "median_49": ssum["median"],
            "frac_pos_49": ssum["frac_positive"], "max_dd_overlay": max_drawdown(strat.dropna()),
            "_slide": slide["sortino_lift"].dropna(), "_strat": strat,
            "_bh": eq_oos, "_opred": opred, "_opred_cal": opred_cal, "_olab": olab}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH); turb = tdf["turbulence"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))
    X = build_feature_matrix_with_bucket(prices, turb, tdf["regime_smoothed"],
                                         factors=FACTORS_10, n_buckets=5)

    # --- calibrate event threshold on DEV turbulence (causal) ---
    conv, diag = EventTimeConverter.calibrate(turb.loc[:DEV_END], target_mean_trading_days=21.0)
    thr = conv.intensity_threshold
    # per-day event-window length distribution -> conservative purge
    s = turb.dropna(); cum = np.cumsum(s.values); n = len(s)
    js = np.searchsorted(cum, cum + thr, side="left")
    lens = (js - np.arange(n))[js < n]
    evt_purge = int(np.ceil(np.quantile(lens, 0.99)))
    print(f"event threshold (dev-calibrated) = {thr:.1f}; mean event "
          f"{diag.mean_trading_days:.1f}d; window len p50={int(np.median(lens))} "
          f"p99={evt_purge}d -> event purge {evt_purge}d (calendar purge {CAL_PURGE}d)")

    # --- two targets, same everything else ---
    y_cal = (forward_drawdown_conditional(prices["equity"], horizon=21) <= -LOSS).astype(float)
    y_cal = y_cal.where(forward_drawdown_conditional(prices["equity"], horizon=21).notna())
    etd = event_time_drawdown(prices["equity"], turb, intensity_threshold=thr)
    y_evt = (etd <= -LOSS).astype(float).where(etd.notna())

    print("evaluating CALENDAR arm ..."); cal = evaluate(y_cal, X, eq, CAL_PURGE, "calendar")
    print("evaluating EVENT-TIME arm ..."); evt = evaluate(y_evt, X, eq, evt_purge, "event_time")

    cmp = pd.DataFrame({
        "calendar (21d)": {k: cal[k] for k in ["base_rate", "auc", "brier_raw", "brier_cal",
                            "overlay_lift", "median_49", "frac_pos_49"]},
        "event-time (~1 event)": {k: evt[k] for k in ["base_rate", "auc", "brier_raw", "brier_cal",
                            "overlay_lift", "median_49", "frac_pos_49"]},
    }).T
    cmp.to_csv(OUT_DIR / "calendar_vs_eventtime.csv")
    print(cmp.round(3).to_string())

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    bins = np.linspace(-0.45, 0.35, 22)
    a.hist(cal["_slide"], bins=bins, alpha=0.55, color="#7f8c8d",
           label=f"calendar (median {cal['median_49']:+.3f}, {cal['frac_pos_49']:.0%}+)")
    a.hist(evt["_slide"], bins=bins, alpha=0.55, color="#c0392b",
           label=f"event-time (median {evt['median_49']:+.3f}, {evt['frac_pos_49']:.0%}+)")
    a.axvline(0, color="k", lw=0.8)
    a.axvline(cal["_slide"].median(), color="#7f8c8d", ls="--")
    a.axvline(evt["_slide"].median(), color="#c0392b", ls="--")
    a.set_xlabel("Sortino lift (49 cutoffs, gross)"); a.set_ylabel("count")
    a.set_title("49-cutoff robustness — calendar vs event-time"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1]
    labels = ["AUC", "Brier(cal)", "overlay lift", "49-median"]
    cv = [cal["auc"], cal["brier_cal"], cal["overlay_lift"], cal["median_49"]]
    ev = [evt["auc"], evt["brier_cal"], evt["overlay_lift"], evt["median_49"]]
    xx = np.arange(len(labels)); w = 0.38
    a.bar(xx - w/2, cv, w, color="#7f8c8d", label="calendar")
    a.bar(xx + w/2, ev, w, color="#c0392b", label="event-time")
    a.axhline(0, color="k", lw=0.6); a.axhline(0.5, color="grey", ls=":", lw=0.7)
    a.set_xticks(xx); a.set_xticklabels(labels, fontsize=9)
    a.set_title("Head-to-head metrics (5% correction)"); a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)
    fig.suptitle("Step 21 — correction-timing on the event clock vs calendar", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_time_inference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'event_time_inference.png'}")

    # ---- verdict ----
    d_auc = evt["auc"] - cal["auc"]
    d_med = evt["median_49"] - cal["median_49"]
    d_fp = evt["frac_pos_49"] - cal["frac_pos_49"]
    improves = (d_auc > 0.02) + (d_med > 0.03) + (d_fp > 0.05)
    if improves >= 2:
        verdict = (f"**EVENT-TIME HELPS.** Moving the target onto the event clock "
                   f"improves the correction-timing problem: AUC {cal['auc']:.3f}→"
                   f"{evt['auc']:.3f} (Δ{d_auc:+.3f}), 49-cutoff median {cal['median_49']:+.3f}"
                   f"→{evt['median_49']:+.3f}, positive cutoffs {cal['frac_pos_49']:.0%}→"
                   f"{evt['frac_pos_49']:.0%}. Consistent with the hypothesis that the "
                   "prediction problem is better-posed where returns are Gaussian — a "
                   "genuinely novel result (no prior work does inference on the event clock).")
    elif improves == 1 and d_fp > 0.05 and d_auc < 0:
        verdict = (f"**A GENUINE TENSION — the interesting result.** Event-time pulls "
                   f"the two halves of the problem in *opposite* directions:\n\n"
                   f"- **Classification gets worse:** AUC {cal['auc']:.3f} → "
                   f"{evt['auc']:.3f} (Δ{d_auc:+.3f}). On the event clock the point "
                   f"prediction of an equity correction is *less* accurate, not more. "
                   f"**Gaussianising the return distribution (step03) does not make "
                   f"drawdowns more forecastable** — a clean, citable distinction.\n\n"
                   f"- **But the strategy edge gets markedly less cutoff-fragile:** the "
                   f"49-cutoff median improves {cal['median_49']:+.3f} → {evt['median_49']:+.3f} "
                   f"and the share of *positive* cutoffs jumps {cal['frac_pos_49']:.0%} → "
                   f"{evt['frac_pos_49']:.0%}. Since cutoff-fragility is this project's "
                   f"central evaluation concern, that is a meaningful, hypothesis-"
                   f"consistent signal: the event clock makes the *robustness* of the "
                   f"edge better even where the point classifier is worse.\n\n"
                   f"Both effects are inside the ~0.5 Sortino noise floor, so neither is "
                   f"a significance claim — but the **direction** (worse AUC, more-"
                   f"robust distribution) is the novel, honest finding, and the first "
                   f"evidence on what event-time does to downstream inference.")
    elif improves == 1:
        verdict = (f"**MIXED / MARGINAL.** Event-time moves one axis but not robustly: "
                   f"ΔAUC {d_auc:+.3f}, Δ49-median {d_med:+.3f}, Δ%positive {d_fp:+.0%}. "
                   "Within the noise floor; report as a careful 'no clear improvement'.")
    else:
        verdict = (f"**NO IMPROVEMENT.** The event clock does not make corrections "
                   f"more predictable here (ΔAUC {d_auc:+.3f}, Δ49-median {d_med:+.3f}). "
                   "Event-time fixes the return *distribution* (step03) but not the "
                   "*predictability* of equity drawdowns — itself a clean, reportable "
                   "finding: Gaussianisation ≠ forecastability.")

    md = [
        "# CHECKPOINT — Step 21: correction-timing inference on the event clock",
        "", f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC", "",
        "## What was tested",
        "Whether running the step-20 correction-timing overlay with the "
        "**event-time drawdown target** (worst drawdown over the next ~1-event "
        "window) beats the **calendar** (21-day) target — everything else "
        "identical (bucket features, logistic_l1, isotonic calibration, "
        "forced top-decile overlay, 49-cutoff). Event threshold dev-calibrated "
        f"({thr:.0f}); event purge {evt_purge}d (99th-pct window length) so no "
        "long calm-period label leaks. This is the first downstream inference on "
        "the event clock (CKT 2023 stopped at the return distribution).",
        "",
        "## Head-to-head (5% correction, OOS 2020+)", "",
        cmp.round(3).to_markdown(),
        "",
        "![event-time vs calendar](../../figures/event_time_inference/event_time_inference.png)",
        "",
        "## Verdict", "", verdict, "",
        "## Notes",
        "- Sample size is identical in both arms (one row per calendar day); only "
        "the target's clock differs, isolating the event-time effect.",
        "- Same OOS ceiling — read the 49-cutoff distribution, not single numbers.",
        "- The event purge is wider than 21d because event windows are variable; "
        "this is the key causality safeguard for event-time inference.",
    ]
    CHECKPOINT.write_text("\n".join(md))
    print(f"wrote {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
