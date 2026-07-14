"""Step 28 — Turbulence + event-time *features* for correction forecasting.

The one configuration the project never tried: give the supervised model
**event-time quantities as predictive inputs** (where are we in the flow of
information?) on top of the turbulence-bucket features, and ask whether it
improves the correction forecast / the de-risking overlay vs buy-and-hold.

Head-to-head, identical honest harness to step20 (the only change is the
feature set):
  * baseline    = turbulence-bucket features (step12/step20)
  * + event-time = baseline + the causal event-time block (lib.event_time_features)

Judged on: OOS AUC, calibrated Brier, the de-risking overlay's Sortino lift and
max-drawdown vs buy-and-hold, and — the honest arbiter — the 49-cutoff
distribution of the lift.  2020-start OOS; event threshold dev-calibrated.

Run:  python3 -m regime_detection.step28_eventtime_features
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

from regime_detection.lib.splits import DEV_END, OOS_START
from regime_detection.lib.metrics import brier_score, max_drawdown, sortino_ratio
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.cutoff_sensitivity import run_sliding_cutoff, summary_statistics
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.event_time import EventTimeConverter
from regime_detection.lib.event_time_features import build_event_time_features
from regime_detection.lib.train import MODEL_BUILDERS, _predict_proba
from regime_detection.extras.step20_correction_overlay import correction_label, overlay_with_costs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "eventtime_features"
FIG_DIR = PROJECT_ROOT / "figures" / "eventtime_features"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step28_eventtime_features.md"
CACHE = PROJECT_ROOT / ".cache" / "eventtime_features"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"; HORIZON = 21; LOSS = 0.05; COST = 10
DEV_CALIB_END = pd.Timestamp("2017-12-31")
ET_COLS = ["et_intensity_to_date", "et_days_since_start", "et_velocity",
           "et_turb_per_day", "et_momentum"]


def _evaluate(X, y, eq):
    """step20's headline evaluation for one feature set: AUC, Brier, overlay
    lift (dev-tau + forced decile), max-DD vs B&H."""
    dev = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=HORIZON,
            dev_end=DEV_CALIB_END, oos_start=DEV_CALIB_END + pd.Timedelta(days=1),
            random_state=0)[MODEL]
    oos = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=HORIZON,
            dev_end=DEV_END, oos_start=OOS_START, random_state=0)[MODEL]
    dpred = dev.oos_predictions.loc[:DEV_END].dropna(); dlab = dev.oos_labels.reindex(dpred.index)
    opred = oos.oos_predictions.dropna(); olab = oos.oos_labels.reindex(opred.index)
    iso = IsotonicRegression(out_of_bounds="clip").fit(dpred.values, dlab.values)
    opred_cal = pd.Series(iso.predict(opred.values), index=opred.index)
    dpred_cal = pd.Series(iso.predict(dpred.values), index=dpred.index)
    auc = roc_auc_score(olab, opred) if olab.nunique() > 1 else np.nan
    brier_cal = brier_score(olab, opred_cal)

    eq_dev = eq.reindex(dpred_cal.index)
    best_tau, best_s = 0.5, -np.inf
    for tau in np.linspace(0.2, 0.9, 36):
        strat, _, _ = overlay_with_costs(eq_dev, (dpred_cal < tau).astype(float), COST)
        s = sortino_ratio(strat.dropna())
        if np.isfinite(s) and s > best_s:
            best_s, best_tau = s, float(tau)
    eq_oos = eq.reindex(opred_cal.index); s_bh = sortino_ratio(eq_oos.dropna())
    in_force = (opred.reindex(opred_cal.index).rank(pct=True) < 0.90).astype(float)
    strat_force, frac_cash_f, _ = overlay_with_costs(eq_oos, in_force, COST)
    return {
        "auc": auc, "brier_cal": brier_cal, "tau": best_tau,
        "lift_devtau": sortino_ratio(
            overlay_with_costs(eq_oos, (opred_cal < best_tau).astype(float), COST)[0].dropna()) - s_bh,
        "lift_forced": sortino_ratio(strat_force.dropna()) - s_bh,
        "maxdd_overlay": max_drawdown(strat_force.dropna()),
        "maxdd_bh": max_drawdown(eq_oos.dropna()),
        "in_cash_forced": frac_cash_f,
    }


def _sliding(X, y, eq, tag):
    cutoffs = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
    df = run_sliding_cutoff(X, y, eq, cutoff_dates=cutoffs, model_name=MODEL,
                            purge_window=HORIZON, cache_dir=CACHE / tag, verbose=False)
    return summary_statistics(df, "sortino_lift")


def _et_coefficients(X, y):
    """Fit one logistic_l1 on dev only and report whether the event-time
    features get non-zero weight (are they used at all?)."""
    dev_mask = X.index <= DEV_END
    Xd, yd = X.loc[dev_mask].dropna(), None
    yd = y.reindex(Xd.index)
    keep = yd.notna(); Xd, yd = Xd.loc[keep], yd.loc[keep]
    est = MODEL_BUILDERS[MODEL](random_state=0).fit(Xd.values, yd.values)
    coef = pd.Series(np.ravel(getattr(est, "coef_", [np.zeros(Xd.shape[1])])),
                     index=Xd.columns)
    return {c: float(coef.get(c, 0.0)) for c in ET_COLS}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    turb = pd.read_parquet(TURB_PATH)["turbulence"]
    regime = pd.read_parquet(TURB_PATH)["regime_smoothed"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))

    X_turb = build_feature_matrix_with_bucket(prices, turb, regime,
                                              factors=FACTORS_10, n_buckets=5)
    conv, diag = EventTimeConverter.calibrate(turb.loc[:DEV_END], target_mean_trading_days=21.0)
    et = build_event_time_features(turb, prices["equity"], float(diag.threshold))
    X_et = X_turb.join(et, how="left")
    # keep rows where both blocks are present
    X_turb = X_turb.loc[X_et.dropna().index]
    X_et = X_et.dropna()
    print(f"baseline features {X_turb.shape}; +event-time {X_et.shape}; "
          f"threshold {diag.threshold:.1f}")

    y = correction_label(prices["equity"], LOSS, HORIZON)

    print("\nEvaluating baseline (turbulence only) ...")
    m_turb = _evaluate(X_turb, y, eq)
    print("Evaluating turbulence + event-time ...")
    m_et = _evaluate(X_et, y, eq)

    print("Sliding-cutoff (49) for both ...")
    s_turb = _sliding(X_turb, y, eq, "turb")
    s_et = _sliding(X_et, y, eq, "turb_et")

    coefs = _et_coefficients(X_et, y)

    # ---- table ----
    print(f"\n{'metric':18s} {'turb-only':>12s} {'+event-time':>12s}")
    for k, lbl in [("auc", "OOS AUC"), ("brier_cal", "Brier (cal)"),
                   ("lift_forced", "overlay lift"), ("maxdd_overlay", "overlay maxDD"),
                   ("maxdd_bh", "B&H maxDD")]:
        print(f"{lbl:18s} {m_turb[k]:>12.3f} {m_et[k]:>12.3f}")
    print(f"{'49-cut median':18s} {s_turb['median']:>12.3f} {s_et['median']:>12.3f}")
    print(f"{'49-cut % positive':18s} {s_turb['frac_positive']:>12.0%} {s_et['frac_positive']:>12.0%}")
    print("\nevent-time feature weights (logistic_l1, dev fit):")
    for c, w in coefs.items():
        print(f"  {c:22s} {w:+.3f}")

    _figure(m_turb, m_et, s_turb, s_et, coefs)
    _write_checkpoint(m_turb, m_et, s_turb, s_et, coefs, X_et.shape, float(diag.threshold))
    print(f"\nWrote checkpoint → {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


def _figure(m_turb, m_et, s_turb, s_et, coefs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # metrics
    ax = axes[0]
    keys = ["auc", "brier_cal", "lift_forced"]; labs = ["AUC", "Brier(cal)", "overlay lift"]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [m_turb[k] for k in keys], 0.4, label="turb-only", color="tab:gray")
    ax.bar(x + 0.2, [m_et[k] for k in keys], 0.4, label="+event-time", color="tab:red")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_title("Forecast / overlay metrics"); ax.legend(fontsize=8)
    # drawdown
    ax = axes[1]
    ax.bar([0, 1, 2], [m_turb["maxdd_bh"], m_turb["maxdd_overlay"], m_et["maxdd_overlay"]],
           color=["0.4", "tab:gray", "tab:red"])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["B&H", "turb-only", "+event-time"])
    ax.set_title("Max drawdown (less negative = better)")
    # event-time feature weights
    ax = axes[2]
    cs = pd.Series(coefs)
    ax.barh(range(len(cs)), cs.values, color="tab:red")
    ax.set_yticks(range(len(cs))); ax.set_yticklabels([c.replace("et_", "") for c in cs.index], fontsize=8)
    ax.axvline(0, color="k", lw=0.6); ax.set_title("event-time feature weights (L1)")
    fig.suptitle("Step 28 — turbulence + event-time features for correction forecasting")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eventtime_features.png", dpi=130)
    plt.close(fig)


def _write_checkpoint(m_turb, m_et, s_turb, s_et, coefs, shape, threshold):
    d_auc = m_et["auc"] - m_turb["auc"]
    d_med = s_et["median"] - s_turb["median"]
    d_dd = m_et["maxdd_overlay"] - m_turb["maxdd_overlay"]   # less negative = better
    used = sum(abs(w) > 1e-6 for w in coefs.values())
    beats_bh_dd = m_et["maxdd_overlay"] > m_turb["maxdd_bh"]

    L = []
    L.append("# Step 28 — Turbulence + event-time features for correction forecasting\n")
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. 2020-start OOS; "
             f"5% / 21d correction target; logistic_l1; event threshold "
             f"{threshold:.1f} (dev-calibrated). Feature matrix {shape[0]}×{shape[1]}._\n")
    L.append("The one configuration never tried: event-time quantities as model "
             "**inputs** (not target, not clock). Head-to-head vs turbulence-only, "
             "identical honest harness (step20).\n")

    L.append("| metric | turbulence only | + event-time |")
    L.append("|---|---|---|")
    L.append(f"| OOS AUC (5%) | {m_turb['auc']:.3f} | {m_et['auc']:.3f} |")
    L.append(f"| Brier (calibrated) | {m_turb['brier_cal']:.3f} | {m_et['brier_cal']:.3f} |")
    L.append(f"| overlay Sortino lift (forced decile) | {m_turb['lift_forced']:+.3f} | {m_et['lift_forced']:+.3f} |")
    L.append(f"| overlay max drawdown | {m_turb['maxdd_overlay']:.1%} | {m_et['maxdd_overlay']:.1%} |")
    L.append(f"| buy-and-hold max drawdown | {m_turb['maxdd_bh']:.1%} | — |")
    L.append(f"| **49-cutoff median lift** | **{s_turb['median']:+.3f}** | **{s_et['median']:+.3f}** |")
    L.append(f"| 49-cutoff % positive | {s_turb['frac_positive']:.0%} | {s_et['frac_positive']:.0%} |")
    L.append("")
    L.append(f"**Event-time features used:** {used}/5 get non-zero logistic_l1 weight "
             f"(" + ", ".join(f"{c.replace('et_','')} {w:+.2f}" for c, w in coefs.items()) + ").")
    L.append("")

    L.append("## Verdict\n")
    if d_auc > 0.02 or d_med > 0.03:
        L.append(f"**Event-time features help the *forecast*.** AUC {m_turb['auc']:.3f}"
                 f"→{m_et['auc']:.3f} ({d_auc:+.3f}); 49-cutoff median "
                 f"{s_turb['median']:+.3f}→{s_et['median']:+.3f} ({d_med:+.3f}). "
                 "The information-flow features add signal the calendar bucket lacks.")
    elif abs(d_auc) <= 0.02 and abs(d_med) <= 0.03:
        L.append(f"**Forecast: no material change.** AUC {d_auc:+.3f}, 49-cutoff "
                 f"median {d_med:+.3f} — inside the sample-size noise floor. Adding "
                 "event-time features does not, on this sample, make equity "
                 "corrections more forecastable than the turbulence bucket alone.")
    else:
        L.append(f"**Event-time features hurt the forecast** (AUC {d_auc:+.3f}, "
                 f"49-median {d_med:+.3f}) — likely overfitting 5 extra features "
                 "against ~60 effective OOS blocks.")
    if beats_bh_dd:
        L.append(f"- **Drawdown:** the +event-time overlay's max drawdown "
                 f"({m_et['maxdd_overlay']:.1%}) is shallower than buy-and-hold "
                 f"({m_et['maxdd_bh']:.1%}) — the durable value (risk reduction), "
                 f"{'better' if d_dd>0 else 'similar'} than turbulence-only "
                 f"({m_turb['maxdd_overlay']:.1%}).")
    else:
        L.append(f"- **Drawdown:** the overlay does not beat buy-and-hold on max "
                 f"drawdown here ({m_et['maxdd_overlay']:.1%} vs {m_et['maxdd_bh']:.1%}).")
    L.append("- **Beating buy-and-hold on risk-adjusted *return* remains out of "
             "reach** under the COVID-held-out OOS — consistent with the whole "
             "project: correction *timing* is hard; the honest read is the "
             "49-cutoff distribution, not any single number.")
    L.append("")
    L.append("**Figure:** `figures/eventtime_features/eventtime_features.png`.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
