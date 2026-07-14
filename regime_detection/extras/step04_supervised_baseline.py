"""Phase-4 end-to-end driver.

Loads the Phase-2 turbulence series and the cached factor prices, builds
the feature matrix + binary turbulence-entry target, trains the four
challenger models with strict walk-forward CV inside the dev partition
(<= 2019-12-31), evaluates them once on the OOS partition (>= 2020-01-01),
saves the artefacts to ``artifacts/supervised/``, and writes the
human-readable report to ``reports/step04_supervised_results.md``.

Run from the project root::

    python3 -m regime_detection.step04_supervised_baseline
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_detection.lib.evaluate import (
    calibrate, confusion_at_top_decile, headline_table,
    in_sample_vs_oos_gap, oos_metrics, per_regime_breakdown,
    reliability_bins, sortino_overlay,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.train import (
    MODEL_BUILDERS, TrainedModel, train_all_models,
)
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "supervised"
REPORT_PATH = PROJECT_ROOT / "reports" / "step04_supervised_results.md"
PHASE4_OUT = PROJECT_ROOT / "figures" / "supervised"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def assemble_dataset():
    print("loading data ...")
    turb_df = pd.read_parquet(TURB_PATH)
    turbulence = turb_df["turbulence"]
    regime = turb_df["regime_smoothed"]

    prices = pd.read_parquet(PRICES_PATH).sort_index()
    factors = [
        "equity", "rates", "credit", "commodities", "em_equity",
        "fx_usd", "inflation", "value", "quality", "vix",
    ]
    prices = prices[factors]

    print(f"  turbulence:  {turbulence.dropna().shape[0]:,} finite rows")
    print(f"  prices:      {prices.shape}  "
          f"{prices.index.min().date()} → {prices.index.max().date()}")

    # Threshold for the binary target: the 90th percentile of dev
    # turbulence — pegging "crisis entry" to the top decile of observed
    # turbulence on the development sample.  Using the 75th percentile
    # produced a 91% positive rate (classification too easy because
    # nearly every 21-day window touches the bar); the 90th percentile
    # gives a more meaningful "imminent crisis" target.  The threshold
    # is derived ONLY from dev data so the OOS set is judged against a
    # bar it never saw.
    dev_turb = turbulence.loc[:DEV_END].dropna()
    threshold = float(dev_turb.quantile(0.90))
    print(f"  binary target threshold (90th pct of dev turbulence) "
          f"= {threshold:.2f}")

    print("building feature matrix ...")
    X = build_feature_matrix(prices, turbulence, regime_smoothed=regime,
                             factors=factors)
    print(f"  feature matrix: {X.shape}, "
          f"columns:\n    " + "\n    ".join(X.columns))

    print("building target ...")
    y = binary_turbulence_entry(turbulence, threshold=threshold, horizon=21)
    print(f"  target positive rate (full sample): {y.dropna().mean():.3f}")

    return X, y, regime, prices["equity"], threshold


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: TrainedModel, threshold: float) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = ARTIFACT_DIR / model.name
    joblib.dump(
        {"estimator": model.estimator, "scaler": model.scaler},
        out_base.with_suffix(".joblib"),
    )
    meta = {
        "model_name":        model.name,
        "fit_date_max":      str(model.fit_date_max.date()),
        "training_rows":     model.extras["dev_rows"],
        "oos_rows":          model.extras["oos_rows"],
        "feature_columns":   model.feature_columns,
        "hyperparams":       _coerce_jsonable(model.hyperparams),
        "random_state":      model.random_state,
        "target_threshold":  threshold,
        "forward_window":    model.extras["forward_window"],
        "embargo":           model.extras["embargo"],
        "dev_pos_rate":      model.extras["dev_pos_rate"],
        "oos_pos_rate":      model.extras["oos_pos_rate"],
        "saved_at_utc":      datetime.utcnow().isoformat() + "Z",
    }
    with open(out_base.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def _coerce_jsonable(obj):
    """Make hyperparams JSON-serialisable (estimator objects → str)."""
    if isinstance(obj, dict):
        return {k: _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_oos_auc(models: Dict[str, TrainedModel]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    cv_aucs = []
    oos_aucs = []
    names = []
    for name, m in models.items():
        cv_aucs.append(m.cv_scores["auc"].mean())
        oos_aucs.append(oos_metrics(m)["auc"])
        names.append(name)
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w/2, cv_aucs, w, color="#1f77b4", alpha=0.85, label="CV mean")
    ax.bar(x + w/2, oos_aucs, w, color="#ff7f0e", alpha=0.85, label="OOS")
    for xi, (cv, oos) in enumerate(zip(cv_aucs, oos_aucs)):
        ax.annotate(f"{cv:.3f}", xy=(xi - w/2, cv), ha="center", va="bottom",
                    fontsize=9)
        ax.annotate(f"{oos:.3f}", xy=(xi + w/2, oos), ha="center", va="bottom",
                    fontsize=9)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="random (AUC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Phase 4 — CV (dev) vs OOS AUC by challenger model")
    ax.set_ylim(0.4, max(max(cv_aucs), max(oos_aucs)) * 1.08)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = PHASE4_OUT / "fig1_auc_cv_vs_oos.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def figure_calibration(models: Dict[str, TrainedModel]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, m) in zip(axes.ravel(), models.items()):
        cal = calibrate(m, n_bins=10)
        if len(cal.bin_centers) == 0:
            ax.text(0.5, 0.5, "insufficient OOS variation", ha="center")
            ax.set_title(name)
            continue
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8,
                label="perfectly calibrated")
        ax.plot(cal.bin_centers, cal.raw_fraction_pos, "o-",
                color="#1f77b4", label=f"raw  Brier={cal.raw_brier:.3f}")
        ax.plot(cal.bin_centers, cal.iso_fraction_pos, "s--",
                color="#ff7f0e", label=f"isotonic  Brier={cal.iso_brier:.3f}")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("fraction positive (OOS)")
        ax.set_title(name)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("OOS reliability diagrams (10-bin) — "
                 "raw vs isotonic-recalibrated", fontsize=11)
    fig.tight_layout()
    p = PHASE4_OUT / "fig2_calibration.png"
    fig.savefig(p, dpi=120)
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


def write_report(
    models: Dict[str, TrainedModel],
    threshold: float,
    regime: pd.Series,
    equity_returns: pd.Series,
    fig_paths: Dict[str, Path],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    headline = headline_table(models)
    md_parts = []
    md_parts.append("# Phase 4 — Supervised Forecasting Results")
    md_parts.append("")
    md_parts.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC  ")
    md_parts.append(f"**Target:** `binary_turbulence_entry`  "
                    f"(turbulence crosses {threshold:.2f} within next 21 "
                    f"trading days)  ")
    md_parts.append(f"**Dev cutoff:** {DEV_END.date()}  "
                    f"**OOS:** {OOS_START.date()} onwards  ")
    md_parts.append("**Cross-validation:** PurgedGroupTimeSeriesSplit "
                    "(n_splits=5, forward_window=21, embargo=21, "
                    "strict expanding-window)")
    md_parts.append("")
    md_parts.append("## Headline OOS results (one evaluation pass per model)")
    md_parts.append("")
    md_parts.append(_df_to_md(headline[[
        "n_oos", "pos_rate", "auc", "avg_precision",
        "prec_top_decile", "rec_top_decile", "brier",
        "cv_mean_auc", "cv_mean_brier",
        "dev_pos_rate", "oos_pos_rate",
    ]]))
    md_parts.append("")
    md_parts.append(f"![CV vs OOS AUC]({fig_paths['auc'].relative_to(PROJECT_ROOT)})")
    md_parts.append("")

    # Confusion matrices
    md_parts.append("## Confusion matrices at the OOS top-decile cutoff")
    md_parts.append("")
    for name, m in models.items():
        cm, cutoff = confusion_at_top_decile(m, decile=0.10)
        md_parts.append(f"### {name}  (cutoff = {cutoff:.4f})")
        md_parts.append("")
        md_parts.append(cm.to_markdown())
        md_parts.append("")

    # Calibration
    md_parts.append("## Calibration")
    md_parts.append("")
    md_parts.append(
        "Reliability bins for the raw OOS probabilities and an "
        "isotonic-recalibrated version.  *Caveat:* the isotonic step is "
        "fit on OOS pairs as a Phase-4 v1 diagnostic — the production "
        "calibrator should be fit on out-of-fold dev predictions and "
        "applied once to OOS.  This is flagged in the README and is the "
        "one short-cut taken in Phase 4."
    )
    md_parts.append("")
    md_parts.append(f"![Calibration]({fig_paths['cal'].relative_to(PROJECT_ROOT)})")
    md_parts.append("")
    cal_rows = []
    for name, m in models.items():
        cal = calibrate(m)
        cal_rows.append({"model": name, "raw_brier": cal.raw_brier,
                         "iso_brier": cal.iso_brier,
                         "delta": cal.raw_brier - cal.iso_brier})
    cal_df = pd.DataFrame(cal_rows).set_index("model")
    md_parts.append(_df_to_md(cal_df))
    md_parts.append("")

    # Per-regime breakdown
    md_parts.append("## OOS performance per turbulence regime")
    md_parts.append("")
    md_parts.append(
        "Conditioning on the Phase-2 smoothed turbulence regime "
        "at evaluation time.  Each row is the model's OOS performance "
        "*within* that regime."
    )
    md_parts.append("")
    for name, m in models.items():
        md_parts.append(f"### {name}")
        md_parts.append("")
        breakdown = per_regime_breakdown(m, regime)
        if breakdown.empty:
            md_parts.append("_(no regime overlap on OOS)_")
        else:
            md_parts.append(_df_to_md(breakdown))
        md_parts.append("")

    # CV vs OOS gap
    md_parts.append("## In-sample (CV) vs OOS gap")
    md_parts.append("")
    md_parts.append(
        "If CV AUC is materially higher than OOS AUC the model is "
        "over-fitting to the dev window; if OOS AUC is *higher* "
        "(rare) the dev folds happened to be more difficult than OOS."
    )
    md_parts.append("")
    gap_rows = []
    for name, m in models.items():
        g = in_sample_vs_oos_gap(m)
        g["model"] = name
        gap_rows.append(g)
    gap_df = pd.DataFrame(gap_rows).set_index("model")[[
        "cv_mean_auc", "cv_std_auc", "oos_auc", "auc_gap",
        "cv_mean_brier", "oos_brier",
    ]]
    md_parts.append(_df_to_md(gap_df))
    md_parts.append("")

    # Sortino overlay
    md_parts.append("## Economic value — Sortino of equity overlay")
    md_parts.append("")
    md_parts.append(
        "Trivial overlay strategy: go flat in the equity factor whenever "
        "the model's predicted probability (next-21d turbulence-entry) "
        "exceeds its OOS top-decile cutoff; otherwise hold the equity "
        "factor long.  Signals lag by one day (decision at end of t-1 "
        "acted on at t).  Annualised Sortino, target = 0."
    )
    md_parts.append("")
    sortino_rows = []
    for name, m in models.items():
        s = sortino_overlay(m, equity_returns)
        s["model"] = name
        sortino_rows.append(s)
    sortino_df = pd.DataFrame(sortino_rows).set_index("model")[[
        "in_market_share", "ann_ret_overlay", "ann_ret_buy_hold",
        "sortino_overlay", "sortino_buy_hold", "threshold",
    ]]
    md_parts.append(_df_to_md(sortino_df, floatfmt="{:.4f}"))
    md_parts.append("")

    # Artefacts inventory
    md_parts.append("## Artefacts")
    md_parts.append("")
    md_parts.append("Saved under `artifacts/supervised/`:")
    md_parts.append("")
    for name in models:
        md_parts.append(f"- `{name}.joblib` — fitted estimator + scaler")
        md_parts.append(f"- `{name}.json` — metadata (rows, hyperparams, "
                        f"seed, threshold, features used)")
    md_parts.append("")
    md_parts.append("Figures under `outputs/supervised/`:")
    md_parts.append("")
    for label, p in fig_paths.items():
        md_parts.append(f"- `{p.relative_to(PROJECT_ROOT)}`")

    REPORT_PATH.write_text("\n".join(md_parts))
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    X, y, regime, equity_px, threshold = assemble_dataset()
    equity_returns = np.log(equity_px / equity_px.shift(1))

    print()
    print("training models ...")
    models = train_all_models(X, y, forward_window=21, embargo=21, n_splits=5,
                              random_state=0)

    print()
    print("OOS metrics:")
    print(headline_table(models).round(4).to_string())

    PHASE4_OUT.mkdir(parents=True, exist_ok=True)
    print()
    print("saving artefacts + figures ...")
    for name, m in models.items():
        save_model(m, threshold)
        # Persist the per-model OOS scores parquet too — handy for later
        # downstream analysis.
        pd.DataFrame({
            "y_true": m.oos_labels,
            "proba":  m.oos_predictions,
        }).to_parquet(ARTIFACT_DIR / f"{name}_oos_predictions.parquet")
    fig_paths = {
        "auc": figure_oos_auc(models),
        "cal": figure_calibration(models),
    }

    write_report(models, threshold, regime, equity_returns, fig_paths)
    print()
    print("Phase 4 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
