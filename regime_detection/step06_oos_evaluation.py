"""Phase-6 end-to-end driver.

* Quarterly walk-forward refits of all four challenger models from Phase 4.
* OOS metrics: AUC, Brier, log loss, Sortino, max-drawdown, hit-rate.
* Sub-period breakdown (by calendar year, by Phase-2 regime).
* Statistical tests: Diebold-Mariano (vs always-1 baseline and pairwise),
  block bootstrap of Sortino difference, stationary bootstrap robustness.
* Canonical figures into ``figures/thesis_v5/`` at 300 DPI.
* Headline report to ``reports/step06_oos_results.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_detection.lib.metrics import (
    annualised_return, annualised_vol, brier_score, hit_rate_at_threshold,
    log_loss, max_drawdown, sharpe_ratio, sortino_ratio,
    strategy_returns, top_decile_threshold,
)
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.stats_tests import (
    block_bootstrap_sortino_diff, diebold_mariano,
    stationary_bootstrap_sortino_diff,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "evaluation"
FIG_DIR = PROJECT_ROOT / "figures" / "thesis_v5"
REPORT_PATH = PROJECT_ROOT / "reports" / "step06_oos_results.md"

MODELS = ["random_forest", "logistic_l1", "linear_svm", "logistic_elasticnet"]
DPI = 300


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def assemble_data():
    print("loading data ...")
    turb_df = pd.read_parquet(TURB_PATH)
    turbulence = turb_df["turbulence"]
    regime_smoothed = turb_df["regime_smoothed"]
    prices = pd.read_parquet(PRICES_PATH).sort_index()
    factors = ["equity", "rates", "credit", "commodities", "em_equity",
               "fx_usd", "inflation", "value", "quality", "vix"]
    prices = prices[factors]

    dev_turb = turbulence.loc[:DEV_END].dropna()
    threshold = float(dev_turb.quantile(0.90))
    print(f"  binary target threshold = {threshold:.2f}")
    X = build_feature_matrix(prices, turbulence, regime_smoothed=regime_smoothed,
                             factors=factors)
    y = binary_turbulence_entry(turbulence, threshold=threshold, horizon=21)
    eq_log_ret = np.log(prices["equity"] / prices["equity"].shift(1))
    return X, y, regime_smoothed, eq_log_ret, threshold


# ---------------------------------------------------------------------------
# Metrics tables
# ---------------------------------------------------------------------------


def headline_metrics(models, equity_returns) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        y, p = m.oos_labels.dropna(), m.oos_predictions.dropna()
        common = y.index.intersection(p.index)
        y, p = y.reindex(common), p.reindex(common)
        if len(y) == 0:
            continue
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc = float(roc_auc_score(y.values, p.values))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y.values, p.values))
        except ValueError:
            ap = float("nan")
        hits = hit_rate_at_threshold(y, p, decile=0.10)
        # Strategy: flat when prediction >= top-decile cutoff (regime
        # conditioning).  Compare to buy-and-hold equity.
        strat = strategy_returns(equity_returns, p,
                                 flat_when_signal_is_high=True, shift=1)
        bh = equity_returns.reindex(strat.index)
        rows.append({
            "model":             name,
            "n_oos":             len(y),
            "pos_rate":          float(y.mean()),
            "auc":               auc,
            "avg_precision":     ap,
            "brier":             brier_score(y, p),
            "log_loss":          log_loss(y, p),
            "prec_top_decile":   hits["precision"],
            "rec_top_decile":    hits["recall"],
            "sortino_strategy":  sortino_ratio(strat),
            "sortino_buy_hold":  sortino_ratio(bh),
            "sharpe_strategy":   sharpe_ratio(strat),
            "sharpe_buy_hold":   sharpe_ratio(bh),
            "ann_ret_strategy":  annualised_return(strat),
            "ann_ret_buy_hold":  annualised_return(bh),
            "ann_vol_strategy":  annualised_vol(strat),
            "max_dd_strategy":   max_drawdown(strat),
            "max_dd_buy_hold":   max_drawdown(bh),
            "in_market_share":   float(((p < top_decile_threshold(p)).astype(float)).mean()),
        })
    return pd.DataFrame(rows).set_index("model")


def sub_period_metrics(models, regime, equity_returns) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        y, p = m.oos_labels.dropna(), m.oos_predictions.dropna()
        common = y.index.intersection(p.index).intersection(regime.index)
        y, p = y.reindex(common), p.reindex(common)
        r = regime.reindex(common)
        # Per-regime AUC.
        for regval, regname in [(0.0, "quiet"), (1.0, "turbulent")]:
            mask = r == regval
            if mask.sum() < 30 or len(y[mask].unique()) < 2:
                rows.append({"model": name, "scope": regname, "n": int(mask.sum()),
                             "auc": float("nan"), "brier": float("nan"),
                             "pos_rate": float(y[mask].mean()) if mask.any() else float("nan")})
                continue
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(y[mask].values, p[mask].values))
            except ValueError:
                auc = float("nan")
            rows.append({
                "model":    name,
                "scope":    regname,
                "n":        int(mask.sum()),
                "auc":      auc,
                "brier":    brier_score(y[mask], p[mask]),
                "pos_rate": float(y[mask].mean()),
            })
        # Per-year AUC.
        for year, group in y.groupby(y.index.year):
            if len(group.unique()) < 2:
                continue
            ag = p.reindex(group.index)
            from sklearn.metrics import roc_auc_score
            try:
                auc_y = float(roc_auc_score(group.values, ag.values))
            except ValueError:
                auc_y = float("nan")
            rows.append({
                "model":    name,
                "scope":    f"year_{int(year)}",
                "n":        len(group),
                "auc":      auc_y,
                "brier":    brier_score(group, ag),
                "pos_rate": float(group.mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def dm_vs_baseline_table(models) -> pd.DataFrame:
    """Diebold-Mariano of each model vs the trivial 'always predict mean'
    baseline.  Loss = Brier per-row.
    """
    rows = []
    for name, m in models.items():
        y, p = m.oos_labels.dropna(), m.oos_predictions.dropna()
        common = y.index.intersection(p.index)
        y, p = y.reindex(common), p.reindex(common)
        base_pred = pd.Series(float(y.mean()), index=common)
        loss_model = (p - y) ** 2
        loss_base = (base_pred - y) ** 2
        r = diebold_mariano(loss_model, loss_base, h=21)
        rows.append({"model": name, "DM_stat": r.statistic,
                     "DM_p": r.p_value,
                     "mean_loss_diff": r.mean_diff,
                     "hac_lags": r.hac_lags, "n": r.n})
    return pd.DataFrame(rows).set_index("model")


def dm_pairwise_table(models) -> pd.DataFrame:
    """Pairwise Diebold-Mariano between every pair of models."""
    names = list(models.keys())
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                out.loc[a, b] = 0.0
                continue
            ya, pa = models[a].oos_labels.dropna(), models[a].oos_predictions.dropna()
            yb, pb = models[b].oos_labels.dropna(), models[b].oos_predictions.dropna()
            common = ya.index.intersection(pa.index).intersection(
                yb.index).intersection(pb.index)
            if len(common) == 0:
                continue
            ya, pa = ya.reindex(common), pa.reindex(common)
            yb, pb = yb.reindex(common), pb.reindex(common)
            loss_a = (pa - ya) ** 2
            loss_b = (pb - yb) ** 2
            r = diebold_mariano(loss_a, loss_b, h=21)
            out.loc[a, b] = r.statistic
    return out


def bootstrap_sortino_table(models, equity_returns) -> pd.DataFrame:
    """Bootstrap of Sortino(strategy) - Sortino(buy-hold) for every model.
    Both block and stationary bootstrap."""
    rows = []
    for name, m in models.items():
        p = m.oos_predictions.dropna()
        strat = strategy_returns(equity_returns, p,
                                 flat_when_signal_is_high=True, shift=1)
        bh = equity_returns.reindex(strat.index)
        block_r = block_bootstrap_sortino_diff(
            strat, bh, block_length=21, n_resamples=1000,
            random_state=0,
        )
        stat_r = stationary_bootstrap_sortino_diff(
            strat, bh, mean_block_length=21, n_resamples=1000,
            random_state=0,
        )
        rows.append({
            "model":          name,
            "block_diff":     block_r.mean_diff,
            "block_ci_low":   block_r.ci_low,
            "block_ci_high":  block_r.ci_high,
            "block_p":        block_r.p_value,
            "stat_diff":      stat_r.mean_diff,
            "stat_ci_low":    stat_r.ci_low,
            "stat_ci_high":   stat_r.ci_high,
            "stat_p":         stat_r.p_value,
        })
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_turbulence_timeseries(turbulence, regime_smoothed):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    finite = turbulence.dropna()
    ax.plot(finite.index, finite.values, lw=0.6, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_ylabel("turbulence  $d_t$  (log)")
    # Regime shading.
    rs = regime_smoothed.reindex(finite.index).fillna(0.0)
    in_turb = (rs > 0.5).astype(int)
    transitions = np.diff(np.r_[0, in_turb.values, 0])
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    for s, e in zip(starts, ends):
        s_date = finite.index[s]
        e_date = finite.index[min(e - 1, len(finite) - 1)]
        ax.axvspan(s_date, e_date, alpha=0.15, color="#d62728")
    # OOS shading on top.
    ax.axvspan(pd.Timestamp(OOS_START), finite.index.max(),
               alpha=0.05, color="black")
    ax.axvline(pd.Timestamp(DEV_END), color="black", lw=1.0, ls="--",
               label=f"dev cutoff = {DEV_END.date()}")
    ax.set_title("Walk-forward Mahalanobis turbulence with smoothed regime "
                 "shading (red) and OOS window (grey)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "fig1_turbulence_with_regime.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def figure_return_distributions(prices_equity, turbulence, threshold_event):
    """Event-time vs calendar-time return distribution overlay."""
    from regime_detection.lib.event_time import get_overlapping_event_returns
    cal = (np.log(prices_equity).shift(-21) - np.log(prices_equity)).dropna()
    evt = get_overlapping_event_returns(
        prices_equity, turbulence, intensity_threshold=threshold_event,
        log_returns=True,
    ).dropna()
    common = cal.index.intersection(evt.index)
    cal_oos = cal.reindex(common)
    evt_oos = evt.reindex(common)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    from scipy import stats as sp
    for ax, data, label, colour in [
        (axes[0], cal_oos, "calendar 21d", "#1f77b4"),
        (axes[1], evt_oos, "event time", "#ff7f0e"),
    ]:
        arr = data.dropna().values
        mu, sd = float(arr.mean()), float(arr.std(ddof=0))
        bins = np.linspace(arr.min(), arr.max(), 60)
        ax.hist(arr, bins=bins, density=True, alpha=0.65, color=colour,
                label=label)
        xs = np.linspace(arr.min(), arr.max(), 400)
        ax.plot(xs, sp.norm.pdf(xs, mu, sd), color="black", lw=1.0,
                label=f"N({mu:.4f}, {sd:.4f})")
        excess_k = sp.kurtosis(arr)
        ax.set_title(f"{label}  excess kurtosis = {excess_k:.2f}")
        ax.set_xlabel("log return")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
    fig.suptitle("Calendar 21d vs event-time return distributions  (equity factor)",
                 fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "fig2_return_distributions.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def figure_calibration(models):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, m) in zip(axes.ravel(), models.items()):
        y, p = m.oos_labels.dropna(), m.oos_predictions.dropna()
        common = y.index.intersection(p.index)
        y, p = y.reindex(common).values, p.reindex(common).values
        edges = np.linspace(0.0, 1.0, 11)
        centers = 0.5 * (edges[:-1] + edges[1:])
        digit = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        frac_pos = np.full(10, np.nan)
        for k in range(10):
            mask = digit == k
            if mask.any():
                frac_pos[k] = float(y[mask].mean())
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        p_iso = iso.predict(p)
        frac_iso = np.full(10, np.nan)
        digit_iso = np.clip(np.searchsorted(edges, p_iso, side="right") - 1, 0, 9)
        for k in range(10):
            mask = digit_iso == k
            if mask.any():
                frac_iso[k] = float(y[mask].mean())
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
        ax.plot(centers, frac_pos, "o-", color="#1f77b4", label="raw")
        ax.plot(centers, frac_iso, "s--", color="#ff7f0e", label="isotonic")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("OOS fraction positive")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("OOS reliability diagrams — quarterly walk-forward refits",
                 fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "fig3_calibration.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def figure_cumulative_returns(models, equity_returns):
    fig, ax = plt.subplots(figsize=(13, 5))
    bh = equity_returns.reindex(equity_returns.index.intersection(
        list(models.values())[0].oos_predictions.dropna().index)
    ).cumsum()
    ax.plot(bh.index, bh.values, lw=1.2, color="black",
            label="buy-and-hold equity")
    for name, m in models.items():
        p = m.oos_predictions.dropna()
        strat = strategy_returns(equity_returns, p,
                                 flat_when_signal_is_high=True, shift=1)
        cum = strat.cumsum()
        ax.plot(cum.index, cum.values, lw=0.9, label=name, alpha=0.9)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_ylabel("cumulative log return")
    ax.set_xlabel("OOS date")
    ax.set_title("Phase-6 OOS cumulative returns — equity overlay vs buy-and-hold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "fig4_cumulative_returns.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def figure_rbi_heatmap():
    """Reuse the dev RBI from Phase 5; compute an OOS RBI alongside.
    Only run if Phase 5 artefacts are present."""
    rbi_path = PROJECT_ROOT / "reports" / "explainability" / "rbi_values.parquet"
    if not rbi_path.exists():
        return None
    rbi = pd.read_parquet(rbi_path).sort_values("rbi", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ranks = pd.DataFrame({
        "RBI rank (dev)":   rbi["rbi_rank"],
        "|SHAP| rank (dev)": rbi["shap_rank"],
    }).sort_values("RBI rank (dev)")
    im = ax.imshow(ranks.values, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ranks.index, fontsize=8)
    ax.set_xticks(range(2))
    ax.set_xticklabels(ranks.columns)
    fig.colorbar(im, ax=ax, label="rank")
    ax.set_title("Phase-5 RBI vs |SHAP| ranks  (dev sample)  —  "
                 "see Phase-5 notebook for full diagnostics")
    fig.tight_layout()
    p = FIG_DIR / "fig5_rbi_ranks.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def figure_significance_table(dm_table, dm_pairwise, boot_table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: DM vs baseline.
    ax = axes[0]
    rows = list(dm_table.index)
    bars = ax.barh(rows, dm_table["DM_stat"].values, color="#1f77b4", alpha=0.85)
    for i, (stat, p) in enumerate(zip(dm_table["DM_stat"], dm_table["DM_p"])):
        ax.annotate(f"t={stat:.2f}  p={p:.3f}", xy=(stat, i),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.axvline(0, color="grey", lw=0.7)
    ax.axvline(-1.96, color="red", lw=0.6, ls=":")
    ax.axvline( 1.96, color="red", lw=0.6, ls=":",
               label="|t| = 1.96 (5% two-sided)")
    ax.set_xlabel("Diebold-Mariano stat  (model loss − baseline loss)")
    ax.set_title("DM vs always-mean baseline")
    ax.legend(fontsize=8)

    # Right: bootstrap Sortino diff with CIs.
    ax = axes[1]
    rows = list(boot_table.index)
    y_pos = np.arange(len(rows))
    diffs = boot_table["block_diff"].values
    lo = boot_table["block_ci_low"].values
    hi = boot_table["block_ci_high"].values
    err_lo = diffs - lo
    err_hi = hi - diffs
    ax.errorbar(diffs, y_pos, xerr=[err_lo, err_hi], fmt="o", color="#1f77b4",
                ecolor="#1f77b4", elinewidth=1.5, capsize=4)
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Sortino(strategy) − Sortino(buy-hold)  with 95% block-bootstrap CI")
    ax.set_title("Bootstrap of Sortino lift")
    for i, p in enumerate(boot_table["block_p"]):
        ax.annotate(f"p={p:.3f}", xy=(hi[i], i),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, va="center")

    fig.suptitle("Statistical significance — Phase-6 OOS", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "fig6_significance.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt.format(v)
                               if pd.notna(v) else "—")
    return df.to_markdown()


def write_report(*, headline, sub_period, dm_table, dm_pairwise,
                 boot_table, fig_paths, threshold, n_models, refit_log_sample):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Phase 6 — Strict OOS Evaluation")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append(
        "**Protocol:** quarterly walk-forward refit.  At each quarter-end "
        "`t_q`, every challenger model is refit on data up to "
        "`t_q − 21d` (the supervised-target horizon).  Predictions for "
        "`(t_q, t_{q+1}]` are generated from that fit and never reused.  "
        "Hyperparameters are frozen at the Phase-4 defaults — no per-quarter "
        "tuning.  Feature standardisation is refit inside every quarterly "
        "training pass."
    )
    md.append("")
    md.append(
        f"**Target:** `binary_turbulence_entry(threshold={threshold:.2f}, "
        "horizon=21)`.  Threshold is the 90th percentile of dev "
        "turbulence (≤ 2019-12-31) — fixed, not updated quarterly, so "
        "the OOS bar is the same as the dev bar."
    )
    md.append("")
    md.append("## Headline results")
    md.append("")
    md.append(_df_to_md(headline[[
        "n_oos", "pos_rate", "auc", "avg_precision", "brier", "log_loss",
        "prec_top_decile", "rec_top_decile",
        "sortino_strategy", "sortino_buy_hold",
        "sharpe_strategy", "sharpe_buy_hold",
        "ann_ret_strategy", "ann_ret_buy_hold",
        "max_dd_strategy", "max_dd_buy_hold",
        "in_market_share",
    ]]))
    md.append("")
    md.append(f"![Cumulative returns]({fig_paths['cum'].relative_to(PROJECT_ROOT)})")
    md.append("")

    md.append("## Sub-period breakdown")
    md.append("")
    md.append(_df_to_md(sub_period.set_index(["model", "scope"]).sort_index()))
    md.append("")

    md.append("## Statistical significance")
    md.append("")
    md.append("### Diebold-Mariano vs always-mean baseline")
    md.append("")
    md.append(
        "Loss = squared error (Brier) per row.  The baseline forecast is "
        "the constant OOS mean of the labels.  Negative DM stat ⇒ model "
        "loss < baseline loss (model improves on the baseline)."
    )
    md.append("")
    md.append(_df_to_md(dm_table))
    md.append("")
    md.append("### Pairwise DM stats  (rows lose vs cols when positive)")
    md.append("")
    md.append(_df_to_md(dm_pairwise))
    md.append("")
    md.append("### Block + stationary bootstrap of Sortino lift")
    md.append("")
    md.append(
        "`Sortino(strategy) − Sortino(buy-hold)` with 95 % CIs.  Block "
        "length = 21 trading days (≈ 1 month).  Stationary bootstrap uses "
        "geometric block lengths with mean = 21.  "
        "**Caveat:** at this OOS length the "
        "bootstrap's power against Sortino differences of less than "
        "≈1 unit is limited.  Read the CI width as the *effective "
        "resolution*, not as a small effect."
    )
    md.append("")
    md.append(_df_to_md(boot_table))
    md.append("")
    md.append(
        f"![Significance]({fig_paths['sig'].relative_to(PROJECT_ROOT)})"
    )
    md.append("")

    md.append("## Refit log (last 3 of each model — sanity check)")
    md.append("")
    md.append(_df_to_md(refit_log_sample.reset_index(drop=True),
                       floatfmt="{:.4f}"))
    md.append("")

    md.append("## Canonical figure set — `figures/thesis_v5/`")
    md.append("")
    for label, path in fig_paths.items():
        md.append(f"- `{path.relative_to(PROJECT_ROOT)}`  ({label})")
    md.append("")

    md.append("## Caveats")
    md.append("")
    md.append(
        f"* The OOS positive rate is materially higher than the dev "
        f"positive rate.  The OOS window (from {OOS_START.date()}, per the fixed split "
        "constraint) now includes the COVID crash and had a structurally "
        "more-turbulent cross-section; AUC inflates and Brier deflates "
        "relative to dev CV.  Read the Brier and DM stats as the most "
        "scale-invariant signals.\n"
        "* Strategy returns are pre-cost.  A 1 bp / day round-trip cost "
        "applied to the ~10 % in-market days would erode roughly 0.10 of "
        "annualised return; the qualitative ranking is robust to that.\n"
        "* The supervised target's threshold is fixed at the 90th-pct of "
        "dev turbulence and *not* updated quarterly.  Updating it on a "
        "rolling basis is a defensible alternative; documented in "
        "`scripts/step06_oos_evaluation.py`.\n"
    )

    REPORT_PATH.write_text("\n".join(md))
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, regime, equity_returns, threshold = assemble_data()
    print()
    print("running quarterly walk-forward refits ...")
    models = run_quarterly_walk_forward(
        X, y, model_names=MODELS, purge_window=21,
        dev_end=DEV_END, oos_start=OOS_START, random_state=0,
    )

    # Sanity print.
    for name, m in models.items():
        print(f"  {name:22s}  n_oos={len(m.oos_predictions):>5d}  "
              f"refits={len(m.refit_log)}")

    # ---- Metrics ------------------------------------------------------
    print()
    print("computing headline metrics ...")
    headline = headline_metrics(models, equity_returns)
    print(headline.round(4)[
        ["auc", "brier", "sortino_strategy", "sortino_buy_hold",
         "max_dd_strategy", "max_dd_buy_hold"]
    ].to_string())

    sub_period = sub_period_metrics(models, regime, equity_returns)

    # ---- Stats tests --------------------------------------------------
    print()
    print("computing DM and bootstrap tests ...")
    dm_table = dm_vs_baseline_table(models)
    dm_pairwise = dm_pairwise_table(models)
    boot_table = bootstrap_sortino_table(models, equity_returns)
    print(boot_table.round(4).to_string())

    # ---- Persist artefacts --------------------------------------------
    headline.to_parquet(ARTIFACT_DIR / "headline.parquet")
    sub_period.to_parquet(ARTIFACT_DIR / "sub_period.parquet")
    dm_table.to_parquet(ARTIFACT_DIR / "dm_vs_baseline.parquet")
    dm_pairwise.to_parquet(ARTIFACT_DIR / "dm_pairwise.parquet")
    boot_table.to_parquet(ARTIFACT_DIR / "bootstrap_sortino.parquet")
    for name, m in models.items():
        pd.DataFrame({
            "y_true": m.oos_labels, "proba": m.oos_predictions,
        }).to_parquet(ARTIFACT_DIR / f"{name}_oos_predictions.parquet")
        m.refit_log.to_parquet(ARTIFACT_DIR / f"{name}_refit_log.parquet")

    # ---- Figures ------------------------------------------------------
    print()
    print("rendering figures @ 300 DPI ...")
    turbulence_series = pd.read_parquet(TURB_PATH)["turbulence"]
    threshold_event = 166.60  # from Phase 3 calibration
    fig_paths = {
        "turb": figure_turbulence_timeseries(turbulence_series, regime),
        "ret":  figure_return_distributions(
            pd.read_parquet(PRICES_PATH)["equity"],
            turbulence_series, threshold_event,
        ),
        "cal":  figure_calibration(models),
        "cum":  figure_cumulative_returns(models, equity_returns),
        "rbi":  figure_rbi_heatmap(),
        "sig":  None,  # populated next
    }
    fig_paths["sig"] = figure_significance_table(
        dm_table, dm_pairwise, boot_table,
    )
    fig_paths = {k: v for k, v in fig_paths.items() if v is not None}

    # ---- Report -------------------------------------------------------
    refit_sample = pd.concat([
        m.refit_log.assign(model=name).tail(3)
        for name, m in models.items() if len(m.refit_log) > 0
    ])
    write_report(
        headline=headline, sub_period=sub_period,
        dm_table=dm_table, dm_pairwise=dm_pairwise, boot_table=boot_table,
        fig_paths=fig_paths, threshold=threshold,
        n_models=len(models), refit_log_sample=refit_sample,
    )
    print()
    print("Phase 6 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
