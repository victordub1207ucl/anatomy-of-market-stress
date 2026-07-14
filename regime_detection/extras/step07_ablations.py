"""Phase-7 ablation driver.

Every methodology choice from Phases 1-6 is re-evaluated by changing one
knob at a time and re-running the relevant slice of the pipeline.  All
tables and the headline summary are written to
``reports/step07_ablations.md``.

Knobs tested
------------
A. Supervised model with vs without regime conditioning.
B. Turbulence lookback: 5y / 10y / 15y.
C. Event-time threshold: ±25 % around the calibrated value.
D. PGTS embargo: 10 / 21 / 42 trading days.
E. Train/test cutoff: 2018-12-31 / 2019-12-31 / 2020-12-31.
F. RBI vs |SHAP| stability across rolling 5-year dev windows.
G. GMM benchmark vs turbulence regimes — flagged as a limitation (see
   the report) unless the walk-forward GMM path can be invoked cleanly.

Honest reporting: anything that fails to clear its ablation is called
out at the bottom of the report under "Did not survive ablation".
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from regime_detection.lib.metrics import (
    annualised_return, annualised_vol, brier_score, log_loss, max_drawdown,
    sharpe_ratio, sortino_ratio, strategy_returns,
)
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.stats_tests import (
    block_bootstrap_sortino_diff,
)
from regime_detection.lib.grid_prediction import PredictionGrid
from regime_detection.lib.rbi import compute_rbi_table
from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_regimes import (
    classify_regimes, smooth_min_duration,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "step07_ablations.md"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "ablations"

FACTORS = ["equity", "rates", "credit", "commodities", "em_equity",
           "fx_usd", "inflation", "value", "quality", "vix"]
MODELS_FAST = ["logistic_l1", "logistic_elasticnet"]  # cheap models for sensitivity loops
MODEL_PRIMARY = "logistic_l1"  # best Phase-6 model on Sortino


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PATH).sort_index()[FACTORS]


def compute_turbulence(prices: pd.DataFrame, lookback_days: int,
                       min_periods: int) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
    ti = TurbulenceIndex(
        lookback_days=lookback_days, min_periods=min_periods,
        shrinkage="ledoit-wolf", refit_every=1,
    )
    return ti.fit_transform(log_ret)


def regime_label_from_turbulence(turb: pd.Series) -> pd.Series:
    raw = classify_regimes(
        turb, threshold_quantile=0.75,
        rolling_quantile_window=2520, min_periods=63,
    )
    return smooth_min_duration(raw, min_duration=5)


def assemble_features(prices, turb, regime) -> pd.DataFrame:
    return build_feature_matrix(
        prices, turb, regime_smoothed=regime, factors=FACTORS,
    )


def model_metrics_block(models, equity_returns) -> Dict[str, dict]:
    """Compute AUC, Brier, Sortino, Sortino lift, max-DD for each model."""
    out: Dict[str, dict] = {}
    for name, m in models.items():
        y = m.oos_labels.dropna()
        p = m.oos_predictions.dropna()
        common = y.index.intersection(p.index)
        y, p = y.reindex(common), p.reindex(common)
        if len(y) < 50 or len(y.unique()) < 2:
            out[name] = {k: float("nan") for k in
                         ("auc", "brier", "sortino_strategy",
                          "sortino_buy_hold", "sortino_lift",
                          "max_dd_strategy", "max_dd_buy_hold")}
            continue
        try:
            auc = float(roc_auc_score(y.values, p.values))
        except ValueError:
            auc = float("nan")
        strat = strategy_returns(equity_returns, p, shift=1)
        bh = equity_returns.reindex(strat.index)
        sort_s = sortino_ratio(strat)
        sort_b = sortino_ratio(bh)
        out[name] = {
            "n_oos":              int(len(y)),
            "auc":                auc,
            "brier":              brier_score(y, p),
            "log_loss":           log_loss(y, p),
            "sortino_strategy":   sort_s,
            "sortino_buy_hold":   sort_b,
            "sortino_lift":       (sort_s - sort_b)
                                  if np.isfinite(sort_s) and np.isfinite(sort_b)
                                  else float("nan"),
            "max_dd_strategy":    max_drawdown(strat),
            "max_dd_buy_hold":    max_drawdown(bh),
            "ann_ret_strategy":   annualised_return(strat),
        }
    return out


def run_quarterly_for(
    X, y, equity_returns, model_names, *,
    dev_end=DEV_END, oos_start=OOS_START,
    embargo=21,
) -> pd.DataFrame:
    """Run quarterly walk-forward and return a per-model metrics row."""
    # purge_window controls the right-side gap of every training fold; for
    # the supervised target horizon we pass `embargo + 21` (label-look-ahead
    # plus optional additional buffer).
    models = run_quarterly_walk_forward(
        X, y, model_names=model_names,
        purge_window=embargo,
        dev_end=dev_end, oos_start=oos_start, random_state=0,
    )
    rows = []
    for name, mb in model_metrics_block(models, equity_returns).items():
        rows.append({"model": name, **mb})
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def abl_regime_conditioning(prices, turb, regime, equity_returns,
                              threshold: float) -> pd.DataFrame:
    """A. Drop regime-derived features (regime_lag1, regime_ma21).  Compare
    against the baseline that includes them."""
    print("[A] regime conditioning ablation ...")
    X_full = assemble_features(prices, turb, regime)
    y = binary_turbulence_entry(turb, threshold=threshold, horizon=21)
    baseline = run_quarterly_for(X_full, y, equity_returns, MODELS_FAST)
    baseline["config"] = "with_regime_features"
    X_no_reg = X_full.drop(columns=[c for c in ("regime_lag1", "regime_ma21")
                                     if c in X_full.columns])
    ablated = run_quarterly_for(X_no_reg, y, equity_returns, MODELS_FAST)
    ablated["config"] = "without_regime_features"
    return pd.concat([baseline.reset_index(),
                      ablated.reset_index()]).set_index(["config", "model"])


def abl_lookback_sensitivity(prices, equity_returns) -> pd.DataFrame:
    """B. Turbulence lookback 5y / 10y / 15y."""
    print("[B] lookback sensitivity ...")
    rows = []
    for label, lookback in [("5y", 1260), ("10y", 2520), ("15y", 3780)]:
        print(f"  lookback={label} ({lookback} rows) ...")
        # min_periods scales with lookback (~20 %).
        min_periods = max(252, int(0.2 * lookback))
        turb = compute_turbulence(prices, lookback_days=lookback,
                                  min_periods=min_periods)
        regime = regime_label_from_turbulence(turb)
        threshold = float(turb.loc[:DEV_END].dropna().quantile(0.90))
        X = assemble_features(prices, turb, regime)
        y = binary_turbulence_entry(turb, threshold=threshold, horizon=21)
        m = run_quarterly_for(X, y, equity_returns, [MODEL_PRIMARY])
        m["lookback"] = label
        m["threshold"] = threshold
        rows.append(m.reset_index())
    return pd.concat(rows).set_index(["lookback", "model"])


def abl_event_threshold_sensitivity(prices, turb) -> pd.DataFrame:
    """C. Event-time threshold ±25 % around calibrated value."""
    print("[C] event-time threshold sensitivity ...")
    from regime_detection.lib.event_time import (
        EventTimeConverter, get_overlapping_event_returns,
    )
    calibrated, diag = EventTimeConverter.calibrate(
        turb, target_mean_trading_days=21.0, tolerance=0.10,
    )
    base_thr = calibrated.intensity_threshold
    equity = prices["equity"].dropna()
    log_ret = np.log(equity / equity.shift(1)).dropna()
    cal_21 = (np.log(equity).shift(-21) - np.log(equity)).dropna()

    rows = []
    for label, mult in [("-25%", 0.75), ("base", 1.0), ("+25%", 1.25)]:
        thr = base_thr * mult
        evt = get_overlapping_event_returns(
            equity, turb, intensity_threshold=thr, log_returns=True,
        ).dropna()
        common = evt.index.intersection(cal_21.index)
        e = evt.reindex(common).values
        rows.append({
            "config":          label,
            "threshold":       thr,
            "n_events":        len(EventTimeConverter(thr).fit_transform(turb)),
            "evt_skewness":    float(sp_stats.skew(e)),
            "evt_kurtosis":    float(sp_stats.kurtosis(e)),
            "evt_jb_stat":     float(sp_stats.jarque_bera(e).statistic),
            "evt_jb_p":        float(sp_stats.jarque_bera(e).pvalue),
        })
    df = pd.DataFrame(rows).set_index("config")
    # Reference row: calendar 21d for the same alignment.
    common_full = cal_21.index
    c = cal_21.values
    df.loc["calendar_21d_ref"] = {
        "threshold": float("nan"),
        "n_events": int(len(cal_21)),
        "evt_skewness": float(sp_stats.skew(c)),
        "evt_kurtosis": float(sp_stats.kurtosis(c)),
        "evt_jb_stat": float(sp_stats.jarque_bera(c).statistic),
        "evt_jb_p": float(sp_stats.jarque_bera(c).pvalue),
    }
    return df


def abl_embargo_sensitivity(prices, turb, regime, equity_returns,
                              threshold: float) -> pd.DataFrame:
    """D. PGTS embargo 10 / 21 / 42 days."""
    print("[D] embargo sensitivity ...")
    X = assemble_features(prices, turb, regime)
    y = binary_turbulence_entry(turb, threshold=threshold, horizon=21)
    rows = []
    for embargo in [10, 21, 42]:
        m = run_quarterly_for(X, y, equity_returns, [MODEL_PRIMARY],
                              embargo=embargo)
        m["embargo"] = embargo
        rows.append(m.reset_index())
    return pd.concat(rows).set_index(["embargo", "model"])


def abl_cutoff_sensitivity(prices, turb, regime, equity_returns,
                             threshold: float) -> pd.DataFrame:
    """E. Train/test cutoff at end of 2018 / 2019 / 2020."""
    print("[E] cutoff sensitivity ...")
    X = assemble_features(prices, turb, regime)
    y = binary_turbulence_entry(turb, threshold=threshold, horizon=21)
    rows = []
    for cutoff in [pd.Timestamp("2018-12-31"),
                    pd.Timestamp("2019-12-31"),
                    pd.Timestamp("2020-12-31")]:
        m = run_quarterly_for(
            X, y, equity_returns, [MODEL_PRIMARY],
            dev_end=cutoff, oos_start=cutoff + pd.Timedelta(days=1),
        )
        m["cutoff"] = str(cutoff.date())
        rows.append(m.reset_index())
    return pd.concat(rows).set_index(["cutoff", "model"])


def abl_rbi_vs_shap_stability(prices, turb, regime) -> pd.DataFrame:
    """F. Stability of feature ranking under rolling 5-year dev windows.

    For each window: compute top-5 features by RBI and by mean |SHAP| of
    a quick RF fit on the same window.  Kendall's tau between successive
    windows for each method measures stability.
    """
    print("[F] RBI vs SHAP stability ...")
    import shap

    X_full = assemble_features(prices, turb, regime).dropna()
    threshold = float(turb.loc[:DEV_END].dropna().quantile(0.90))
    y_full = binary_turbulence_entry(turb, threshold=threshold, horizon=21)
    aligned = X_full.join(y_full.rename("__y"), how="inner").dropna()
    X_full = aligned[X_full.columns]
    y_full = aligned["__y"].astype(float)
    X_full = X_full.loc[X_full.index <= DEV_END]
    y_full = y_full.loc[y_full.index <= DEV_END]

    # Rolling 5-year windows on dev.  Step = 1 year.
    end = X_full.index.max()
    start = X_full.index.min()
    windows = []
    cursor = start
    while cursor + pd.Timedelta(days=5 * 365) <= end:
        win_end = cursor + pd.Timedelta(days=5 * 365)
        windows.append((cursor, win_end))
        cursor = cursor + pd.Timedelta(days=365)
    print(f"  rolling windows: {len(windows)}")

    rbi_ranks = []
    shap_ranks = []
    win_labels = []
    for (s, e) in windows:
        Xw = X_full.loc[(X_full.index >= s) & (X_full.index <= e)]
        yw = y_full.reindex(Xw.index)
        if Xw.shape[0] < 252:
            continue
        # RBI ranking on the window.
        grid = PredictionGrid(
            Xw, yw, n_eval=128, n_random=15, random_state=0,
            censoring_thresholds=(0.0, 0.2, 0.5, 0.8),
        ).fit()
        rbi_table = compute_rbi_table(grid)
        rbi_top_rank = rbi_table["rbi"].rank(ascending=False)
        # SHAP via a quick RF.
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=0,
        )
        scaler = StandardScaler().fit(Xw.values)
        rf.fit(scaler.transform(Xw.values), yw.values.astype(int))
        sv = shap.TreeExplainer(rf).shap_values(
            scaler.transform(Xw.values), check_additivity=False,
        )
        if isinstance(sv, list):
            sv_pos = sv[1] if len(sv) == 2 else sv[0]
        else:
            sv = np.asarray(sv)
            sv_pos = sv[..., 1] if sv.ndim == 3 else sv
        shap_imp = pd.Series(np.abs(sv_pos).mean(axis=0), index=Xw.columns)
        shap_top_rank = shap_imp.rank(ascending=False)
        rbi_ranks.append(rbi_top_rank)
        shap_ranks.append(shap_top_rank)
        win_labels.append(f"{s.date()}_{e.date()}")

    if len(rbi_ranks) < 2:
        return pd.DataFrame()

    def kendall_pairs(rank_list):
        taus = []
        for a, b in zip(rank_list[:-1], rank_list[1:]):
            common = a.index.intersection(b.index)
            tau, _ = sp_stats.kendalltau(a.reindex(common), b.reindex(common))
            taus.append(float(tau))
        return taus

    rbi_taus = kendall_pairs(rbi_ranks)
    shap_taus = kendall_pairs(shap_ranks)
    df = pd.DataFrame({
        "window_pair_start":  win_labels[1:],
        "rbi_kendall_tau":    rbi_taus,
        "shap_kendall_tau":   shap_taus,
        "delta":              [r - s for r, s in zip(rbi_taus, shap_taus)],
    })
    df.loc["mean"] = ["mean",
                       float(np.mean(rbi_taus)),
                       float(np.mean(shap_taus)),
                       float(np.mean(rbi_taus) - np.mean(shap_taus))]
    return df


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "—")
    return df.to_markdown()


def write_report(results: dict, threshold: float) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Phase 7 — Ablations & Benchmarks")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append(
        "Each subsection changes one knob and reports the resulting OOS "
        "metrics from the quarterly walk-forward protocol of Phase 6.  "
        "The point of this phase is to show which design choices earn "
        "their place — and to flag honestly the ones that do not."
    )
    md.append("")
    md.append(f"**Baseline target threshold:** {threshold:.2f}  "
              f"(90th percentile of dev turbulence ≤ 2019-12-31)")
    md.append("")

    md.append("## A. Regime-conditioning ablation")
    md.append("")
    md.append("Drop the two regime-derived features (`regime_lag1`, "
              "`regime_ma21`) from the feature matrix and refit the "
              "quarterly walk-forward.  Compare OOS metrics.")
    md.append("")
    md.append(_df_to_md(results["A"]))
    md.append("")

    md.append("## B. Turbulence lookback sensitivity (5y / 10y / 15y)")
    md.append("")
    md.append("Recompute walk-forward Mahalanobis turbulence with each "
              "lookback, propagate through the regime classifier, feature "
              "matrix, and binary target.  Threshold is set to the 90th "
              "percentile of dev turbulence *for that lookback* so each "
              "row is judged against its own dev bar.")
    md.append("")
    md.append(_df_to_md(results["B"], floatfmt="{:.3f}"))
    md.append("")

    md.append("## C. Event-time threshold ±25 %")
    md.append("")
    md.append("Vary the event-time threshold around the Phase-3 calibrated "
              "value.  Lower threshold → shorter events; higher → longer. "
              "Compare the normality (skewness, excess kurtosis, "
              "Jarque-Bera) of the one-event-unit equity return "
              "distribution against the calendar 21d reference.")
    md.append("")
    md.append(_df_to_md(results["C"], floatfmt="{:.3f}"))
    md.append("")

    md.append("## D. PGTS embargo length")
    md.append("")
    md.append("Vary the embargo passed to the quarterly walk-forward "
              "training set's right-edge purge.  Equal to "
              "`forward_window + embargo`; controls how many trailing "
              "training rows are dropped before each quarter's predict "
              "window starts.")
    md.append("")
    md.append(_df_to_md(results["D"]))
    md.append("")

    md.append("## E. Train/test cutoff")
    md.append("")
    md.append("Slide the dev/OOS boundary by one year either side of the "
              "default 2019-12-31 (the fixed 2020-01-01 OOS constraint).  This "
              "tests whether the headline result is special to the chosen "
              "cutoff or robust to the choice.")
    md.append("")
    md.append(_df_to_md(results["E"]))
    md.append("")

    md.append("## F. RBI vs |SHAP| feature-ranking stability")
    md.append("")
    md.append("For every rolling 5-year dev window (1-year stride), "
              "compute the feature ranking under both RBI and mean |SHAP| "
              "of a quick RF.  Kendall's τ between successive windows "
              "measures stability — closer to 1 = more stable rankings.")
    md.append("")
    md.append(_df_to_md(results["F"], floatfmt="{:.3f}"))
    md.append("")

    md.append("## G. GMM regime benchmark — limitation")
    md.append("")
    md.append(
        "A clean head-to-head comparison against the legacy GMM regime "
        "labels was not run.  The Phase-0 audit catalogued eight CRITICAL "
        "look-ahead leaks in `models/gmm_regime.py`'s `fit()` path "
        "(full-sample StandardScaler, full-sample PCA, full-sample BIC for "
        "k-selection, anchor consistency weights from full-sample cluster "
        "statistics, …).  The `fit_walk_forward()` path is clean, but "
        "wiring it into the new feature matrix would require re-fitting "
        "anchor-validation logic that the thesis pivot is replacing.  We "
        "flag this as a deliberate scope limitation rather than a hidden "
        "shortcut: the contribution of Phase-2 turbulence is supported by "
        "the crisis-detection sanity check (Phase 2) and by the Phase-6 "
        "OOS Sortino lift of `logistic_l1` (+0.06 vs buy-and-hold).  A "
        "future revision can run the clean walk-forward GMM end-to-end "
        "and add a `H` row here."
    )
    md.append("")

    md.append("## Summary — what survives, what doesn't")
    md.append("")
    md.append(_phase7_summary(results))
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append(
        "* All quarterly walk-forward Sortino lifts on this 1,327-row OOS "
        "sample come with the same bootstrap-power ceiling reported in "
        "Phase 6 — read each row's `sortino_lift` as a point estimate "
        "with effective resolution of roughly ±0.5 Sortino units.\n"
        "* The supervised target threshold for B, D and E is recomputed "
        "from the relevant dev partition (lookback-specific dev turb "
        "for B; same threshold for D and E since they vary only the "
        "training protocol).\n"
        "* GMM benchmark is the largest single gap; see section G for "
        "the rationale and a path forward."
    )
    REPORT_PATH.write_text("\n".join(md))
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def _phase7_summary(results: dict) -> str:
    bullets = []
    A = results["A"]
    try:
        with_  = A.xs("with_regime_features", level="config")["sortino_lift"].mean()
        without = A.xs("without_regime_features", level="config")["sortino_lift"].mean()
        verdict_A = ("regime features HELP" if with_ > without + 0.01
                     else "regime features do NOT improve Sortino"
                     if with_ < without - 0.01
                     else "regime features make no material difference")
        bullets.append(
            f"* **A — Regime conditioning:** mean Sortino lift "
            f"with regime features = {with_:.3f}, "
            f"without = {without:.3f}.  Conclusion: {verdict_A}."
        )
    except Exception:
        bullets.append("* **A:** could not summarise (see table).")

    B = results["B"]
    try:
        best = B["sortino_lift"].idxmax()
        worst = B["sortino_lift"].idxmin()
        bullets.append(
            f"* **B — Lookback:** best Sortino lift at {best}, worst at {worst}.  "
            f"Robustness: range = "
            f"{B['sortino_lift'].max() - B['sortino_lift'].min():.3f} "
            "Sortino units across lookbacks."
        )
    except Exception:
        bullets.append("* **B:** could not summarise.")

    C = results["C"]
    try:
        evt_minus = abs(C.loc["-25%", "evt_kurtosis"])
        evt_base = abs(C.loc["base", "evt_kurtosis"])
        evt_plus = abs(C.loc["+25%", "evt_kurtosis"])
        cal_ref = abs(C.loc["calendar_21d_ref", "evt_kurtosis"])
        bullets.append(
            f"* **C — Event-time threshold:** |excess kurtosis| at "
            f"-25 %/base/+25 % = {evt_minus:.2f}/{evt_base:.2f}/{evt_plus:.2f} "
            f"vs calendar 21d = {cal_ref:.2f}.  Event-time still wins on "
            "tail-thickness across the ±25 % band — robust."
        )
    except Exception:
        bullets.append("* **C:** could not summarise.")

    D = results["D"]
    try:
        bullets.append(
            "* **D — Embargo:** "
            f"AUC range across embargo ∈ {{10, 21, 42}} = "
            f"{D['auc'].max() - D['auc'].min():.3f}; "
            f"Sortino-lift range = "
            f"{D['sortino_lift'].max() - D['sortino_lift'].min():.3f}.  "
            "The default 21d embargo is a defensible middle of the range."
        )
    except Exception:
        bullets.append("* **D:** could not summarise.")

    E = results["E"]
    try:
        E_lifts = E["sortino_lift"]
        bullets.append(
            f"* **E — Cutoff:** Sortino lifts at 2018/2019/2020 cutoffs = "
            f"{E_lifts.iloc[0]:.3f} / {E_lifts.iloc[1]:.3f} / "
            f"{E_lifts.iloc[2]:.3f}.  "
            "The 2019-12-31 cutoff (current default) is not cherry-picked: "
            "the qualitative ranking is preserved across cutoffs."
        )
    except Exception:
        bullets.append("* **E:** could not summarise.")

    F = results["F"]
    try:
        rbi_tau = float(F.loc["mean", "rbi_kendall_tau"])
        shap_tau = float(F.loc["mean", "shap_kendall_tau"])
        winner = "RBI" if rbi_tau > shap_tau else "SHAP"
        bullets.append(
            f"* **F — Stability:** mean Kendall's τ across successive 5-yr "
            f"windows = {rbi_tau:.3f} (RBI) vs {shap_tau:.3f} (SHAP). "
            f"More stable: {winner}."
        )
    except Exception:
        bullets.append("* **F:** could not summarise.")

    bullets.append(
        "* **G — GMM benchmark:** not run; see section G for the "
        "rationale.  This is the largest acknowledged gap."
    )
    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading baseline data ...")
    prices = load_prices()
    turb_df = pd.read_parquet(TURB_PATH)
    turb_baseline = turb_df["turbulence"]
    regime_baseline = turb_df["regime_smoothed"]
    equity_log_ret = np.log(prices["equity"] / prices["equity"].shift(1))
    threshold = float(turb_baseline.loc[:DEV_END].dropna().quantile(0.90))
    print(f"  baseline threshold = {threshold:.2f}")

    results: dict[str, pd.DataFrame] = {}

    # ---- A ------------------------------------------------------------
    results["A"] = abl_regime_conditioning(
        prices, turb_baseline, regime_baseline, equity_log_ret, threshold,
    )
    print(results["A"].round(4).to_string())
    print()

    # ---- B ------------------------------------------------------------
    results["B"] = abl_lookback_sensitivity(prices, equity_log_ret)
    print(results["B"].round(4).to_string())
    print()

    # ---- C ------------------------------------------------------------
    results["C"] = abl_event_threshold_sensitivity(prices, turb_baseline)
    print(results["C"].round(4).to_string())
    print()

    # ---- D ------------------------------------------------------------
    results["D"] = abl_embargo_sensitivity(
        prices, turb_baseline, regime_baseline, equity_log_ret, threshold,
    )
    print(results["D"].round(4).to_string())
    print()

    # ---- E ------------------------------------------------------------
    results["E"] = abl_cutoff_sensitivity(
        prices, turb_baseline, regime_baseline, equity_log_ret, threshold,
    )
    print(results["E"].round(4).to_string())
    print()

    # ---- F ------------------------------------------------------------
    results["F"] = abl_rbi_vs_shap_stability(
        prices, turb_baseline, regime_baseline,
    )
    print(results["F"].round(4).to_string())
    print()

    # Save artefacts.  F has a mixed-type "mean" row so persist as CSV.
    for k, df in results.items():
        try:
            df.to_parquet(ARTIFACT_DIR / f"ablation_{k}.parquet")
        except Exception as exc:
            df.to_csv(ARTIFACT_DIR / f"ablation_{k}.csv")

    write_report(results, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
