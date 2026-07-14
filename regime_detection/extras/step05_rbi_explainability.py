"""Phase-5 driver: compute RBI on the Phase-4 dev features.

Reuses the feature matrix and binary target produced by Phase 4 (the
``binary_turbulence_entry`` target, threshold = 90th-pct of dev
turbulence).  Restricts everything to the dev partition (``index <=
DEV_END``) so the RBI Mahalanobis statistics are fit purely on training
data.

Outputs
-------
* ``artifacts/explainability/rbi_values.parquet`` — RBI / tau / SHAP
  importance ranking with one row per feature.
* ``artifacts/explainability/rbi_per_regime.parquet`` — per-regime RBI
  table (quiet vs turbulent).
* ``artifacts/explainability/grid_adj_fit.parquet`` — full adjusted-fit
  matrix (rows = subset, cols = censor).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from regime_detection.lib.grid_prediction import PredictionGrid
from regime_detection.lib.rbi import compute_rbi_table
from regime_detection.lib.relevance import RelevanceCalculator
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
RF_PATH = PROJECT_ROOT / "reports" / "supervised" / "random_forest.joblib"
OUT_DIR = PROJECT_ROOT / "reports" / "explainability"


def assemble_dev_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    print("loading data ...")
    turb_df = pd.read_parquet(TURB_PATH)
    turbulence = turb_df["turbulence"]
    regime_smoothed = turb_df["regime_smoothed"]

    prices = pd.read_parquet(PRICES_PATH).sort_index()
    factors = [
        "equity", "rates", "credit", "commodities", "em_equity",
        "fx_usd", "inflation", "value", "quality", "vix",
    ]
    prices = prices[factors]

    dev_turb = turbulence.loc[:DEV_END].dropna()
    threshold = float(dev_turb.quantile(0.90))
    print(f"  binary target threshold (90th pct of dev turb) = {threshold:.2f}")

    X = build_feature_matrix(prices, turbulence, regime_smoothed=regime_smoothed,
                             factors=factors)
    y = binary_turbulence_entry(turbulence, threshold=threshold, horizon=21)

    # Restrict to dev partition with both features and labels present.
    Z = X.join(y.rename("__y"), how="inner").join(
        regime_smoothed.rename("__regime"), how="inner",
    ).dropna()
    Z = Z.loc[Z.index <= DEV_END]
    X_dev = Z[X.columns]
    y_dev = Z["__y"].astype(float)
    regime_dev = Z["__regime"]
    print(f"  dev features: {X_dev.shape}  "
          f"{X_dev.index.min().date()} → {X_dev.index.max().date()}")
    print(f"  dev positive rate: {y_dev.mean():.3f}")
    return X_dev, y_dev, regime_dev, threshold


def compute_grid_rbi(X: pd.DataFrame, y: pd.Series, *,
                     n_eval: int, n_random: int, label: str) -> Tuple[
    PredictionGrid, pd.DataFrame, pd.Series,
]:
    print(f"  building grid ({label}) ...")
    grid = PredictionGrid(
        X, y,
        n_eval=n_eval,
        n_random=n_random,
        censoring_thresholds=(0.0, 0.2, 0.5, 0.8),
        random_state=0,
    ).fit()
    # Informativeness per variable — used to scale the tau statistic to
    # paper-style units.
    calc = grid.calculator
    inf_per_var = {}
    for c in calc.columns_:
        idx = calc.columns_.index(c)
        cov_sub = calc.cov_[idx, idx]
        diff = calc.X.iloc[:, idx].values - calc.mean_[idx]
        inf_per_var[c] = float(0.5 * (diff ** 2).sum() / max(cov_sub, 1e-12) / len(diff))
    informativeness = pd.Series(inf_per_var, name="informativeness")
    # R² proxy = adjusted_fit of the full-feature subset at censor=0.0.
    full_subset = tuple(calc.columns_)
    r2_proxy = float(
        [c.adjusted_fit for c in grid.cells_
         if c.subset == full_subset and c.censor == 0.0][0]
    )
    K = grid.n_eval
    print(f"    grid label={label}  R²={r2_proxy:.3f}  K={K}  "
          f"n_subsets={len(grid.subsets_)}")
    table = compute_rbi_table(
        grid, informativeness=informativeness, K=K, r_squared=r2_proxy,
    )
    return grid, table, informativeness


def compute_shap_importance(X_dev: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature on the Phase-4 RF model."""
    import shap
    print("  computing TreeSHAP on Phase-4 random forest ...")
    bundle = joblib.load(RF_PATH)
    estimator = bundle["estimator"]
    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X_dev.values)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X_scaled, check_additivity=False)
    # For binary classification newer SHAP returns array of shape
    # (n, p, 2) — take the positive class.
    if isinstance(sv, list):
        sv_pos = sv[1] if len(sv) == 2 else sv[0]
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[-1] == 2:
            sv_pos = sv[..., 1]
        else:
            sv_pos = sv
    abs_mean = np.abs(sv_pos).mean(axis=0)
    return pd.Series(abs_mean, index=X_dev.columns, name="mean_abs_shap")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_dev, y_dev, regime_dev, threshold = assemble_dev_dataset()

    # ---- Aggregate RBI on the full dev partition ----------------------
    grid_full, table_full, inf_full = compute_grid_rbi(
        X_dev, y_dev, n_eval=200, n_random=30, label="all-dev",
    )
    grid_full.adj_fit_.to_parquet(OUT_DIR / "grid_adj_fit.parquet")

    # ---- TreeSHAP comparison ------------------------------------------
    shap_imp = compute_shap_importance(X_dev)

    # ---- Merge into one ranking table ---------------------------------
    combined = table_full.copy()
    combined["mean_abs_shap"] = shap_imp.reindex(combined.index)
    combined["shap_rank"] = combined["mean_abs_shap"].rank(ascending=False)
    combined["rbi_rank"] = combined["rbi"].rank(ascending=False)
    combined = combined.sort_values("rbi", ascending=False)
    print()
    print("RBI ranking (top 10):")
    print(combined.head(10).round(4).to_string())

    combined.to_parquet(OUT_DIR / "rbi_values.parquet")

    # ---- Per-regime RBI -----------------------------------------------
    # Strategy: rebuild the grid using only eval points whose smoothed
    # regime label is 0 (quiet) or 1 (turbulent).  The full Mahalanobis
    # fit uses ALL dev rows in both cases so the metric is comparable.
    # Implementation note: we pass a regime-restricted (X, y) to the
    # grid; this means both the Mahalanobis fit AND the eval sample
    # are restricted.  This is the natural per-regime conditional.
    print()
    print("computing per-regime RBI ...")
    per_regime: dict[int, pd.DataFrame] = {}
    for r in (0.0, 1.0):
        mask = regime_dev == r
        if mask.sum() < 200:
            print(f"  regime={int(r)}: only {mask.sum()} rows, skipping")
            continue
        X_r = X_dev.loc[mask]
        y_r = y_dev.loc[mask]
        _, table_r, _ = compute_grid_rbi(
            X_r, y_r, n_eval=128, n_random=20, label=f"regime={int(r)}",
        )
        table_r["regime"] = int(r)
        per_regime[int(r)] = table_r

    if per_regime:
        regime_df = pd.concat(per_regime.values()).reset_index()
        regime_df.to_parquet(OUT_DIR / "rbi_per_regime.parquet")
        print()
        print("Per-regime top variables (by RBI):")
        for r, df in per_regime.items():
            print(f"\n  regime={r}: top 5 by RBI")
            print(df.head(5).round(4).to_string())

    # ---- Sidecar JSON with run config ---------------------------------
    meta = {
        "dev_rows": int(len(X_dev)),
        "dev_pos_rate": float(y_dev.mean()),
        "target_threshold": float(threshold),
        "features": list(X_dev.columns),
        "n_subsets": len(grid_full.subsets_),
        "n_eval": grid_full.n_eval,
        "censors": list(grid_full.censoring_thresholds_),
        "r2_proxy_full_subset": float(
            [c.adjusted_fit for c in grid_full.cells_
             if c.subset == tuple(grid_full.columns_) and c.censor == 0.0][0]
        ),
        "shrinkage_intensity": float(grid_full.calculator.shrinkage_intensity_),
    }
    (OUT_DIR / "rbi_run_meta.json").write_text(json.dumps(meta, indent=2))
    print()
    print(f"saved RBI artefacts to {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
