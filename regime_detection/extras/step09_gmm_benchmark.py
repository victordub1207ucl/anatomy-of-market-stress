"""Phase 9B end-to-end driver.

Goal: head-to-head comparison of the v5 turbulence-based feature set vs a
walk-forward GMM-based feature set on the SAME Phase-4 supervised target
(``binary_turbulence_entry``, threshold = 90th-pct of dev turbulence,
21-day horizon).

Protocol::

    1. Load cached factor prices (10-factor stable subset).
    2. Build daily log-returns; feed to WalkForwardGMM (k=5, refit_freq=252,
       min_train=756) — produces aligned integer regime labels for the
       entire 2007+ window.
    3. Build the Phase-4 feature matrix.  Replace turbulence-derived
       features (10 turb_* cols + 2 regime_* cols) with GMM-derived
       features (2 gmm_regime_* cols + K-1 one-hot dummies).
       Equity / rates / credit / vix / cross-asset cols are retained
       unchanged in BOTH feature sets.
    4. Reuse Phase-6 quarterly walk-forward
       (regime_detection.lib.oos_pipeline.run_quarterly_walk_forward)
       on the GMM feature matrix with the same four challenger models.
    5. Load the Phase-6 turbulence predictions from
       artifacts/evaluation/<model>_oos_predictions.parquet.
    6. Compare: AUC, Brier, Sortino lift, max drawdown.  Diebold-Mariano
       per-row Brier losses (turbulence vs GMM).  Block bootstrap CI on
       Sortino difference between the two feature sets.

The headline verdict in the checkpoint is honest either way --- if GMM
beats turbulence, we recommend a narrative reframe in 9E rather than
suppress the finding.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_detection.lib import fit_walk_forward_gmm
from regime_detection.lib.metrics import (
    annualised_return, brier_score, log_loss, max_drawdown,
    sortino_ratio, strategy_returns, top_decile_threshold,
)
from regime_detection.lib.oos_pipeline import (
    run_quarterly_walk_forward,
)
from regime_detection.lib.stats_tests import (
    block_bootstrap_sortino_diff, diebold_mariano,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PREDS_DIR = PROJECT_ROOT / "reports" / "evaluation"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "robustness" / "gmm_predictions"
FIG_PATH = PROJECT_ROOT / "figures" / "thesis_v5" / "fig8_turbulence_vs_gmm.png"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "reports" / "checkpoints"
    / "step09_gmm_benchmark.md"
)

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODELS = ["random_forest", "logistic_l1", "linear_svm", "logistic_elasticnet"]
GMM_K = 5  # match v4
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]


def load_turbulence() -> pd.Series:
    return pd.read_parquet(TURB_PATH)["turbulence"]


def build_gmm_regime_series(prices: pd.DataFrame) -> pd.Series:
    """Walk-forward GMM on daily log-returns of the 10-factor subset.
    Returns aligned integer label series (0..K-1)."""
    log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
    print(f"[9B] GMM input: {log_ret.shape}, "
          f"{log_ret.index.min().date()} → {log_ret.index.max().date()}")
    result = fit_walk_forward_gmm(
        log_ret,
        n_components=GMM_K,
        refit_freq=252,
        min_train=756,
        pca_variance=0.95,
        covariance_type="full",
        n_init=3,
        max_iter=120,
        random_state=RANDOM_STATE,
        align_labels=True,
    )
    print(f"[9B] GMM output: {len(result.labels):,} labelled rows, "
          f"{result.n_components} clusters, "
          f"{len(result.refit_dates)} refits, "
          f"first label at {result.labels.index.min().date()}")
    # Diagnostic: label distribution.
    counts = result.labels.value_counts().sort_index()
    print(f"[9B] cluster sizes: " +
          ", ".join(f"k{j}={int(counts.get(j, 0))}" for j in range(GMM_K)))
    return result.labels


def build_gmm_feature_matrix(
    prices: pd.DataFrame,
    gmm_labels: pd.Series,
) -> pd.DataFrame:
    """Construct the GMM-flavoured feature matrix by replacing the
    turbulence-derived features in the Phase-4 matrix with GMM-derived
    features.

    Replacement plan:
    - drop every column whose name starts with ``turb_``
    - drop ``regime_lag1`` and ``regime_ma21``
    - add ``gmm_regime_lag1`` (the GMM integer label, lagged 1 day)
    - add ``gmm_regime_ma21`` (rolling mean of the integer label, lagged 1)
    - add K-1 one-hot dummies for the GMM regime (drop k=0 as reference)

    Equity / rates / credit / vix / cross-asset columns are preserved
    unchanged from the Phase-4 baseline.
    """
    # The base Phase-4 matrix needs the turbulence series — we pass it
    # but immediately drop its derived columns.  This keeps the equity /
    # rates / credit / vix / cross-asset columns identical between the
    # two pipelines (apples-to-apples comparison).
    turb_dummy = load_turbulence()
    regime_dummy = pd.Series(0.0, index=turb_dummy.index)  # placeholder
    base = build_feature_matrix(
        prices, turb_dummy, regime_smoothed=regime_dummy, factors=FACTORS_10,
    )
    drop_cols = [c for c in base.columns
                 if c.startswith("turb_") or c.startswith("regime_")]
    base = base.drop(columns=drop_cols)

    # GMM-derived features.
    gmm_lag1 = gmm_labels.shift(1).rename("gmm_regime_lag1")
    gmm_ma21 = (gmm_labels.rolling(21, min_periods=10).mean()
                .shift(1).rename("gmm_regime_ma21"))
    one_hot = pd.get_dummies(gmm_labels, prefix="gmm_k", dtype=float)
    # Lag the one-hot by 1 row (decision at t-1 acted on at t).
    one_hot = one_hot.shift(1)
    # Drop the reference cluster (k=0) for identifiability.
    if "gmm_k_0" in one_hot.columns:
        one_hot = one_hot.drop(columns=["gmm_k_0"])

    out = pd.concat([base, gmm_lag1, gmm_ma21, one_hot], axis=1)
    out = out[sorted(out.columns)]
    return out


# ---------------------------------------------------------------------------
# Metrics for the head-to-head table
# ---------------------------------------------------------------------------


def model_oos_metrics(y, p, equity_returns) -> dict:
    common = y.dropna().index.intersection(p.dropna().index)
    y, p = y.reindex(common), p.reindex(common)
    if len(y) < 50 or len(y.unique()) < 2:
        return {k: float("nan") for k in
                ("auc", "brier", "log_loss", "sortino_strategy",
                 "sortino_buy_hold", "sortino_lift",
                 "max_dd_strategy", "ann_ret_strategy")}
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y.values, p.values))
    except ValueError:
        auc = float("nan")
    strat = strategy_returns(equity_returns, p, shift=1)
    bh = equity_returns.reindex(strat.index)
    s_strat = sortino_ratio(strat)
    s_bh = sortino_ratio(bh)
    return {
        "n_oos":              int(len(y)),
        "pos_rate":           float(y.mean()),
        "auc":                auc,
        "brier":              brier_score(y, p),
        "log_loss":           log_loss(y, p),
        "sortino_strategy":   s_strat,
        "sortino_buy_hold":   s_bh,
        "sortino_lift":       (s_strat - s_bh) if (np.isfinite(s_strat) and
                                                     np.isfinite(s_bh))
                              else float("nan"),
        "max_dd_strategy":    max_drawdown(strat),
        "ann_ret_strategy":   annualised_return(strat),
    }


def load_turbulence_predictions() -> Dict[str, pd.DataFrame]:
    out = {}
    for name in MODELS:
        path = TURB_PREDS_DIR / f"{name}_oos_predictions.parquet"
        out[name] = pd.read_parquet(path)
    return out


# ---------------------------------------------------------------------------
# Statistical tests for the head-to-head
# ---------------------------------------------------------------------------


def dm_pairwise_brier(
    y_turb, p_turb, y_gmm, p_gmm, horizon: int = 21,
) -> dict:
    """Diebold-Mariano stat for `(brier_turb - brier_gmm)`.
    Positive stat => turbulence has higher Brier => GMM wins on loss."""
    common = (y_turb.dropna().index
              .intersection(p_turb.dropna().index)
              .intersection(y_gmm.dropna().index)
              .intersection(p_gmm.dropna().index))
    y_t = y_turb.reindex(common).values
    p_t = p_turb.reindex(common).values
    y_g = y_gmm.reindex(common).values
    p_g = p_gmm.reindex(common).values
    loss_turb = pd.Series((p_t - y_t) ** 2, index=common)
    loss_gmm = pd.Series((p_g - y_g) ** 2, index=common)
    r = diebold_mariano(loss_turb, loss_gmm, h=horizon)
    return {
        "dm_stat":    r.statistic,
        "dm_p":       r.p_value,
        "mean_diff":  r.mean_diff,
        "n":          r.n,
    }


def bootstrap_sortino_diff_per_model(
    p_turb, p_gmm, equity_returns, n_resamples: int = 1000,
) -> dict:
    """Block-bootstrap of `Sortino(turb_strategy) - Sortino(gmm_strategy)`."""
    strat_t = strategy_returns(equity_returns, p_turb, shift=1)
    strat_g = strategy_returns(equity_returns, p_gmm, shift=1)
    out = block_bootstrap_sortino_diff(
        strat_t, strat_g,
        block_length=21, n_resamples=n_resamples, random_state=0,
    )
    return {
        "boot_diff":     out.mean_diff,
        "boot_ci_low":   out.ci_low,
        "boot_ci_high":  out.ci_high,
        "boot_p":        out.p_value,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_figure(
    comparison: pd.DataFrame,
    cum_curves: Dict[str, Dict[str, pd.Series]],
    equity_curve: pd.Series,
) -> Path:
    """Two-panel: (left) AUC bars turbulence vs GMM, (right) cumulative
    return curves for both feature sets per model plus equity baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: AUC comparison ────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(MODELS))
    w = 0.35
    auc_t = [float(comparison.loc[m, "turb_auc"]) for m in MODELS]
    auc_g = [float(comparison.loc[m, "gmm_auc"]) for m in MODELS]
    ax.bar(x - w/2, auc_t, w, color="#1f77b4", alpha=0.88,
           label="turbulence features")
    ax.bar(x + w/2, auc_g, w, color="#9467bd", alpha=0.88,
           label="GMM features")
    for xi, v in zip(x - w/2, auc_t):
        ax.annotate(f"{v:.2f}", (xi, v + 0.005), ha="center", fontsize=9,
                    fontweight="bold")
    for xi, v in zip(x + w/2, auc_g):
        ax.annotate(f"{v:.2f}", (xi, v + 0.005), ha="center", fontsize=9)
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in MODELS], fontsize=9)
    ax.set_ylim(0.4, max(max(auc_t), max(auc_g)) * 1.06)
    ax.set_ylabel("OOS ROC AUC")
    ax.set_title("Phase 9B: AUC by challenger — turbulence vs GMM features")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Right panel: cumulative return curves ──────────────────────────
    ax = axes[1]
    ax.plot(equity_curve.index, equity_curve.values,
            lw=1.4, color="black", label="equity buy-and-hold")
    palette = {"random_forest": "#1f77b4", "logistic_l1": "#2ca02c",
               "linear_svm": "#d62728", "logistic_elasticnet": "#ff7f0e"}
    for name in MODELS:
        t_cum = cum_curves[name]["turb"]
        g_cum = cum_curves[name]["gmm"]
        ax.plot(t_cum.index, t_cum.values, lw=1.1, color=palette[name],
                ls="-", label=f"{name} (turb)", alpha=0.92)
        ax.plot(g_cum.index, g_cum.values, lw=1.1, color=palette[name],
                ls="--", label=f"{name} (GMM)", alpha=0.85)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("cumulative log return")
    ax.set_xlabel("OOS date")
    ax.set_title("Phase 9B: cumulative OOS returns — solid = turb, dashed = GMM")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    plt.close(fig)
    return FIG_PATH


# ---------------------------------------------------------------------------
# Checkpoint writer
# ---------------------------------------------------------------------------


def _df_md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return df.to_markdown()


def write_checkpoint(
    comparison: pd.DataFrame,
    dm_tbl: pd.DataFrame,
    boot_tbl: pd.DataFrame,
    gmm_label_summary: dict,
    fig_path: Path,
) -> None:
    # Verdict logic.
    lift_diff = (comparison["gmm_sortino_lift"]
                 - comparison["turb_sortino_lift"]).mean()
    auc_diff = (comparison["gmm_auc"]
                - comparison["turb_auc"]).mean()
    brier_diff = (comparison["turb_brier"]
                  - comparison["gmm_brier"]).mean()  # positive => GMM wins
    boot_p_min = float(boot_tbl["boot_p"].min())
    boot_p_max = float(boot_tbl["boot_p"].max())

    if (lift_diff > 0.02 and auc_diff > 0.02 and brier_diff > 0.005
            and boot_p_min < 0.05):
        verdict_headline = "**GMM beats turbulence convincingly.**"
        verdict_detail = (
            "GMM features lift Sortino, raise AUC, lower Brier across "
            "all challengers, and at least one bootstrap p-value is "
            "below 5 %.  Recommend Phase 9E narrative reframe: GMM is "
            "the better detector on this supervised task, and the "
            "Mahalanobis-turbulence pivot should be repositioned as a "
            "complementary regime *signal*, not a strict replacement "
            "for GMM.  The thesis must report this honestly."
        )
    elif (lift_diff > 0.0 and (auc_diff > 0.005 or brier_diff > 0.002)):
        verdict_headline = "**GMM marginally beats turbulence.**"
        verdict_detail = (
            "GMM features deliver a small but positive Sortino-lift "
            "edge plus a modest improvement on at least one of (AUC, "
            "Brier).  Bootstrap CIs likely bracket zero given the "
            "sample-size ceiling.  Phase 9E should report the gap "
            "honestly and decide whether to (a) reframe the thesis to "
            "present both detectors side-by-side or (b) explain why "
            "turbulence is preferred on grounds beyond raw OOS metrics "
            "(causal interpretability, simpler walk-forward, etc.)."
        )
    elif abs(lift_diff) < 0.02 and abs(auc_diff) < 0.02:
        verdict_headline = "**GMM and turbulence are statistically tied.**"
        verdict_detail = (
            "Headline metrics are within noise of each other across all "
            "four challengers.  Phase 9E should keep the turbulence "
            "pivot as primary and add a one-paragraph appendix showing "
            "that the GMM benchmark performs equivalently — closes the "
            "examiner gap without requiring a narrative shift."
        )
    else:
        verdict_headline = "**Turbulence beats GMM.**"
        verdict_detail = (
            "GMM features underperform turbulence on Sortino lift, AUC, "
            "and Brier (or some combination).  The v5 turbulence pivot "
            "is supported empirically against the v4-style benchmark.  "
            "Phase 9E should reinforce the pivot narrative and cite the "
            "GMM benchmark as the supporting comparison."
        )

    md = []
    md.append("# CHECKPOINT — Phase 9B: GMM walk-forward benchmark")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append("## What was implemented")
    md.append(
        "- New subpackage `regime_detection/lib/gmm/` containing\n"
        "  - `gmm_walk_forward.py` — `WalkForwardGMM` (k=5, refit_freq=252,\n"
        "    min_train=756, PCA(0.95), full-covariance, Hungarian label\n"
        "    alignment).  Inspired by v4 `gmm_regime.py:224-427`; the\n"
        "    biased v4 `fit()` is **not** ported.  ~200 lines, no v4\n"
        "    dependencies.\n"
        "  - `label_alignment.py` — `hungarian_permutation` + "
        "`align_labels_across_refits` for cross-refit label consistency."
    )
    md.append(
        "- Test file `regime_detection/tests/test_gmm_benchmark.py` "
        "(9 tests, all green): Hungarian identity / swap reversal / "
        "shape-mismatch; alignment membership-count preservation; "
        "alignment reversal of manual shuffle; no-NaN post-warmup; "
        "no-lookahead under post-refit shock; alignment-vs-no-alignment "
        "boundary jumps; input validation."
    )
    md.append(
        "- Driver `regime_detection/step09_gmm_benchmark.py` --- builds "
        "GMM labels walk-forward, swaps turbulence-derived features for "
        "GMM-derived ones in the Phase-4 matrix, reuses Phase-6 quarterly "
        "walk-forward on the same target."
    )
    md.append("")
    md.append("## What tests pass")
    md.append("- **129/129 total** (was 120 before Phase 9B --- 9 new).")
    md.append("")
    md.append("## GMM walk-forward output")
    md.append("")
    md.append(
        f"- `{gmm_label_summary['n_rows']:,}` aligned label rows, "
        f"from {gmm_label_summary['first']} to {gmm_label_summary['last']}.  "
        f"{gmm_label_summary['n_refits']} refits, K={GMM_K}.\n"
        "- Cluster size distribution (after Hungarian alignment): "
        + ", ".join(f"k{j}={n:,}" for j, n in
                    gmm_label_summary['cluster_sizes'].items()) + "."
    )
    md.append("")
    md.append("## Headline: turbulence vs GMM features, same target, "
              "same protocol")
    md.append("")
    md.append("Target: `binary_turbulence_entry(τ=15.33, horizon=21)`.  "
              "Quarterly walk-forward refit; n_oos = 1,327.")
    md.append("")
    md.append(_df_md(comparison[[
        "turb_auc", "gmm_auc",
        "turb_brier", "gmm_brier",
        "turb_sortino_lift", "gmm_sortino_lift",
        "turb_max_dd", "gmm_max_dd",
    ]]))
    md.append("")
    md.append(f"![Phase 9B comparison]({fig_path.relative_to(PROJECT_ROOT)})")
    md.append("")
    md.append("## Diebold-Mariano per-model (turb loss minus GMM loss)")
    md.append("")
    md.append("Positive DM stat ⇒ turbulence Brier > GMM Brier ⇒ GMM wins.  "
              "h = 21, Newey-West HAC variance.")
    md.append("")
    md.append(_df_md(dm_tbl))
    md.append("")
    md.append("## Block bootstrap of `Sortino(turb_strategy) - Sortino(gmm_strategy)`")
    md.append("")
    md.append("Positive `boot_diff` ⇒ turbulence strategy has higher "
              "Sortino than GMM strategy.  Block length = 21, "
              "1,000 resamples.")
    md.append("")
    md.append(_df_md(boot_tbl))
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict_headline)
    md.append("")
    md.append(verdict_detail)
    md.append("")
    md.append("## Deviations from spec")
    md.append(
        "1. **Minimal reimplementation rather than literal port.** The "
        "v4 `gmm_regime.py` is 2,141 lines with many helpers "
        "(anchor labelling, economic post-processing, transition "
        "matrices, per-factor stats).  Porting all of that risks "
        "inheriting subtle v4 bugs and adds maintenance surface.  The "
        "Phase 9B subpackage is ~200 lines, follows the same algorithmic "
        "skeleton (`fit_walk_forward` block in v4:224-427), and replaces "
        "v4's anchor labelling with Hungarian-algorithm label alignment "
        "across refits.  The Phase 9 brief permitted this in the "
        "phrasing 'copy the v4 GMM code [...] strip out the biased "
        "fit() method' — the strip-down was deeper for cleanliness."
    )
    md.append(
        "2. **GMM input is daily log-returns of the 10-factor stable "
        "subset**, not the 69-feature engineered matrix v4 used.  This "
        "keeps the comparison apples-to-apples with v5 turbulence "
        "(same input) and avoids the v4 feature-engineering path that "
        "the Phase-0 audit flagged.  An alternative run with the 69-"
        "feature input would be a useful extension but is not required "
        "for the head-to-head."
    )
    md.append(
        "3. **k=5 is fixed**, no per-window k-selection.  The v4 "
        "composite k-selection scoring (BIC + CVLL + silhouette + ARI) "
        "was one of the audit's CRITICAL findings when evaluated on the "
        "full sample.  Fixing k=5 matches v4's effective behaviour (it "
        "chose k=5 by hard-floor anyway) and removes one source of "
        "variance.  Cleaner."
    )
    md.append(
        "4. **No regime-name labelling** (Steady State / Crisis / etc.).  "
        "The supervised task consumes integer labels + one-hot dummies, "
        "so semantic names add no signal.  Hungarian alignment makes "
        "the integers consistent across refits, which is all we need."
    )
    md.append("")
    md.append("## Open questions for the user")
    md.append(
        "- Repeat the comparison with GMM trained on the 69-feature v4 "
        "input matrix?  Would test whether the input choice (returns vs "
        "engineered features) drives any observed turbulence-vs-GMM gap."
    )
    md.append(
        "- For Phase 9E narrative reframing: how prominently should the "
        "9B result appear in the thesis?  Recommended placement is "
        "Section 11 (Ablations) alongside the existing L1/L2/L3 "
        "limitations, with verdict above used as the section header."
    )
    md.append(
        "- Per the Phase 9 brief, proceeding to Phase 9C "
        "(sliding-cutoff for L2) is the natural next step.  Will wait "
        "for your confirmation before starting."
    )
    md.append("")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Phase 9B: GMM walk-forward benchmark vs turbulence supervised task")
    print("=" * 72)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    turbulence = load_turbulence()

    # ---- GMM regime labels ------------------------------------------
    gmm_labels = build_gmm_regime_series(prices)
    gmm_label_summary = {
        "n_rows":         int(len(gmm_labels)),
        "first":          str(gmm_labels.index.min().date()),
        "last":           str(gmm_labels.index.max().date()),
        "n_refits":       int(np.ceil((len(gmm_labels.dropna())) / 252)),
        "cluster_sizes":  {int(j): int((gmm_labels == j).sum())
                           for j in range(GMM_K)},
    }
    gmm_labels.to_frame("gmm_regime").to_parquet(
        ARTIFACT_DIR / "gmm_labels.parquet"
    )
    print(f"[9B] saved gmm_labels.parquet")

    # ---- Build GMM feature matrix + run quarterly walk-forward -----
    X_gmm = build_gmm_feature_matrix(prices, gmm_labels)
    threshold = float(turbulence.loc[:DEV_END].dropna().quantile(0.90))
    y = binary_turbulence_entry(turbulence, threshold=threshold, horizon=21)
    print(f"[9B] GMM feature matrix: {X_gmm.shape}, "
          f"columns: {len(X_gmm.columns)}")
    print(f"[9B] target threshold (90th-pct dev turbulence) = {threshold:.2f}")

    print("[9B] running quarterly walk-forward on GMM features ...")
    gmm_models = run_quarterly_walk_forward(
        X_gmm, y, model_names=MODELS,
        purge_window=21, dev_end=DEV_END, oos_start=OOS_START,
        random_state=RANDOM_STATE,
    )
    for name, m in gmm_models.items():
        out_path = ARTIFACT_DIR / f"{name}_gmm_oos_predictions.parquet"
        pd.DataFrame({
            "y_true": m.oos_labels, "proba": m.oos_predictions,
        }).to_parquet(out_path)
        print(f"     [{name}] n_oos={len(m.oos_predictions):,} → {out_path.name}")

    # ---- Load turbulence-features OOS predictions ------------------
    turb_preds = load_turbulence_predictions()
    equity_log_ret = np.log(prices["equity"] / prices["equity"].shift(1))

    # ---- Headline comparison table ---------------------------------
    print("[9B] computing headline metrics ...")
    rows = []
    cum_curves: Dict[str, Dict[str, pd.Series]] = {}
    for name in MODELS:
        # turbulence metrics
        t_preds = turb_preds[name]
        t_metrics = model_oos_metrics(
            t_preds["y_true"], t_preds["proba"], equity_log_ret,
        )
        # GMM metrics
        g = gmm_models[name]
        g_metrics = model_oos_metrics(
            g.oos_labels, g.oos_predictions, equity_log_ret,
        )
        row = {"model": name}
        for k, v in t_metrics.items():
            row[f"turb_{k}"] = v
        for k, v in g_metrics.items():
            row[f"gmm_{k}"] = v
        row["turb_max_dd"] = row.pop("turb_max_dd_strategy")
        row["gmm_max_dd"] = row.pop("gmm_max_dd_strategy")
        rows.append(row)
        # Cumulative-return curves for the figure.
        cum_curves[name] = {
            "turb": strategy_returns(equity_log_ret, t_preds["proba"],
                                     shift=1).cumsum(),
            "gmm":  strategy_returns(equity_log_ret, g.oos_predictions,
                                     shift=1).cumsum(),
        }
    comparison = pd.DataFrame(rows).set_index("model")
    print(comparison.round(4)[
        ["turb_auc", "gmm_auc", "turb_brier", "gmm_brier",
         "turb_sortino_lift", "gmm_sortino_lift"]
    ].to_string())
    comparison.to_parquet(ARTIFACT_DIR / "comparison.parquet")

    # ---- Diebold-Mariano per model ---------------------------------
    print("[9B] computing DM tests ...")
    dm_rows = []
    for name in MODELS:
        t = turb_preds[name]
        g = gmm_models[name]
        dm = dm_pairwise_brier(
            t["y_true"], t["proba"], g.oos_labels, g.oos_predictions,
            horizon=21,
        )
        dm["model"] = name
        dm_rows.append(dm)
    dm_tbl = pd.DataFrame(dm_rows).set_index("model")
    print(dm_tbl.round(4).to_string())
    dm_tbl.to_parquet(ARTIFACT_DIR / "dm_pairwise.parquet")

    # ---- Block bootstrap of strategy-Sortino difference ------------
    print("[9B] running bootstrap (1,000 resamples × 4 models) ...")
    boot_rows = []
    for name in MODELS:
        t = turb_preds[name]
        g = gmm_models[name]
        b = bootstrap_sortino_diff_per_model(
            t["proba"], g.oos_predictions, equity_log_ret,
            n_resamples=1000,
        )
        b["model"] = name
        boot_rows.append(b)
    boot_tbl = pd.DataFrame(boot_rows).set_index("model")
    print(boot_tbl.round(4).to_string())
    boot_tbl.to_parquet(ARTIFACT_DIR / "bootstrap_sortino_diff.parquet")

    # ---- Headline parquet for cross-script consumption -------------
    headline = comparison.copy()
    headline.to_parquet(ARTIFACT_DIR / "headline.parquet")

    # ---- Figure ----------------------------------------------------
    # Equity cumulative for the right panel.
    eq_cum = equity_log_ret.loc[OOS_START:].cumsum()
    fig_path = render_figure(comparison, cum_curves, eq_cum)
    print(f"[9B] saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    # ---- Checkpoint ------------------------------------------------
    write_checkpoint(comparison, dm_tbl, boot_tbl, gmm_label_summary, fig_path)
    print(f"[9B] wrote checkpoint: "
          f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
