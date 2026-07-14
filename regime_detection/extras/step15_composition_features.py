"""Test whether decomposed-turbulence features beat the scalar baseline.

Mirrors the Phase-9F protocol exactly, but the experimental arm adds the
per-factor turbulence-composition features
(:mod:`regime_detection.lib.decomposition_features`) on top of the
v5 baseline, rather than swapping the regime representation.

Arms:
  * **baseline** — v5 feature matrix (binary regime).  Sliding-cutoff
    numbers reused from the Phase-9C cache so the comparison is against
    the canonical thesis baseline.
  * **decomp**   — baseline + decshare_*/decconc_* features.

Outputs: quarterly walk-forward table (4 models), 49-cutoff sliding
distribution (logistic_l1 + random_forest), a comparison figure, and a
markdown summary under reports/checkpoints/.

Run::

    python3 -m regime_detection.step15_composition_features
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_detection.lib.cutoff_sensitivity import (
    run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.metrics import (
    brier_score, sortino_ratio, strategy_returns,
)
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer
from regime_detection.lib.decomposition_features import (
    build_feature_matrix_with_decomposition,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
V5_9C_PATH = PROJECT_ROOT / "reports" / "robustness" / "sliding_cutoff.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "decomp_features"
FIG_PATH = PROJECT_ROOT / "figures" / "decomposition" / "decomp_vs_baseline.png"
CHECKPOINT_PATH = (PROJECT_ROOT / "reports"
                   / "checkpoints" / "step15_composition_features.md")
CACHE_DIR = PROJECT_ROOT / ".cache" / "decomp_features"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODELS = ["random_forest", "logistic_l1", "linear_svm", "logistic_elasticnet"]
SLIDING_MODELS = ["logistic_l1", "random_forest"]
TURB_PARAMS = dict(lookback_days=2520, min_periods=504,
                   shrinkage="ledoit-wolf", refit_every=1)


def _qmetrics(qwf, eq):
    y = qwf.oos_labels.dropna(); p = qwf.oos_predictions.dropna()
    c = y.index.intersection(p.index); y, p = y.reindex(c), p.reindex(c)
    if len(y) < 50 or len(y.unique()) < 2:
        return {k: float("nan") for k in ("auc", "brier", "sortino_lift")}
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y.values, p.values))
    strat = strategy_returns(eq, p, shift=1); bh = eq.reindex(strat.index)
    ss, sb = sortino_ratio(strat), sortino_ratio(bh)
    return {"auc": auc, "brier": brier_score(y, p),
            "sortino_lift": (ss - sb) if np.isfinite(ss) and np.isfinite(sb)
                            else float("nan")}


def run_quarterly(X, y, eq, label):
    print(f"[decomp] quarterly walk-forward ({label}, X{X.shape}) ...")
    models = run_quarterly_walk_forward(
        X, y, model_names=MODELS, purge_window=21,
        dev_end=DEV_END, oos_start=OOS_START, random_state=0)
    return pd.DataFrame({n: _qmetrics(m, eq) for n, m in models.items()}).T


def run_sliding(X, y, eq, model, tag):
    sched = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
    return run_sliding_cutoff(
        X, y, eq, cutoff_dates=sched, model_name=model, purge_window=21,
        min_oos_quarters=4, random_state=0,
        cache_dir=CACHE_DIR / tag, verbose=True)


def render_figure(q_base, q_dec, slide_base, slide_dec):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    x = np.arange(len(MODELS)); w = 0.38
    ax.bar(x - w/2, [q_base.loc[m, "auc"] for m in MODELS], w,
           color="#1f77b4", label="v5 baseline (scalar)")
    ax.bar(x + w/2, [q_dec.loc[m, "auc"] for m in MODELS], w,
           color="#2ca02c", label="+ decomposed turbulence")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_", "\n") for m in MODELS],
                                         fontsize=8)
    ax.set_ylim(0.45, 0.75); ax.set_ylabel("OOS ROC AUC")
    ax.set_title("AUC — baseline vs + decomposition (full window)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bins = np.linspace(-0.45, 0.30, 26)
    sb = slide_base["logistic_l1"]["sortino_lift"].dropna()
    sd = slide_dec["logistic_l1"]["sortino_lift"].dropna()
    ax.hist(sb, bins=bins, alpha=0.6, color="#1f77b4", label="baseline")
    ax.hist(sd, bins=bins, alpha=0.6, color="#2ca02c", label="+ decomposition")
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvline(sb.median(), color="#1f77b4", ls="--", lw=1.4,
               label=f"baseline median {sb.median():+.3f}")
    ax.axvline(sd.median(), color="#2ca02c", ls="--", lw=1.4,
               label=f"decomp median {sd.median():+.3f}")
    ax.set_xlabel("Sortino lift vs buy-and-hold (49 monthly cutoffs)")
    ax.set_ylabel("count")
    ax.set_title("Cutoff-sensitivity — logistic_l1, FULL window")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Decomposed-turbulence features vs v5 scalar baseline",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FIG_PATH


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return df.to_markdown()


def write_checkpoint(q_base, q_dec, slide_summ, fig_path, n_decomp_cols):
    med_b = slide_summ.loc[("logistic_l1", "baseline"), "median"]
    med_d = slide_summ.loc[("logistic_l1", "decomp"), "median"]
    fp_b = slide_summ.loc[("logistic_l1", "baseline"), "frac_positive"]
    fp_d = slide_summ.loc[("logistic_l1", "decomp"), "frac_positive"]
    med_b_rf = slide_summ.loc[("random_forest", "baseline"), "median"]
    med_d_rf = slide_summ.loc[("random_forest", "decomp"), "median"]
    gain_l1 = med_d - med_b
    gain_rf = med_d_rf - med_b_rf
    dauc = (q_dec["auc"] - q_base["auc"])

    if gain_l1 > 0.03 and gain_rf > 0.0:
        verdict = (f"**HELPS.** Adding decomposed-turbulence features lifts the "
                   f"logistic_l1 median sliding-cutoff Sortino from {med_b:+.3f} "
                   f"to {med_d:+.3f} (frac positive {fp_b:.0%}->{fp_d:.0%}) and "
                   f"random_forest from {med_b_rf:+.3f} to {med_d_rf:+.3f}. "
                   "Composition information adds value beyond the scalar.")
    elif gain_l1 > 0.0 or dauc.mean() > 0.005:
        verdict = (f"**MARGINAL / MIXED.** logistic_l1 median moves {gain_l1:+.3f} "
                   f"({med_b:+.3f}->{med_d:+.3f}); random_forest {gain_rf:+.3f}; "
                   f"mean AUC change {dauc.mean():+.3f}. Some signal, not decisive "
                   "on this sample.")
    else:
        verdict = (f"**NO IMPROVEMENT.** logistic_l1 median moves {gain_l1:+.3f} "
                   f"({med_b:+.3f}->{med_d:+.3f}); mean AUC change {dauc.mean():+.3f}. "
                   "Composition features do not beat the scalar on this OOS sample "
                   "(sample-size ceiling applies). Keep them as interpretation, not "
                   "prediction.")

    md = [
        "# CHECKPOINT — decomposed-turbulence features vs scalar baseline",
        "",
        f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
        "",
        "## Hypothesis",
        "Does *which factor* drives turbulence (composition) predict imminent "
        "stress better than the scalar turbulence level alone?",
        "",
        "## What was built",
        "- `regime_detection/lib/turbulence_decomposition.py` — exact additive "
        "per-factor decomposition of `d_t` (causal; reconstructs `d_t` to machine "
        "precision).",
        "- `regime_detection/lib/decomposition_features.py` — "
        f"`decshare_<factor>_<w>` + `decconc_<w>` ({n_decomp_cols} added columns, "
        "windows 21/63, all `.shift(1)`-ed).",
        "- `regime_detection/step15_composition_features.py` — this driver.",
        "",
        "## Quarterly walk-forward (full window, 4 models)",
        "",
        _md(pd.DataFrame({
            "base_auc": q_base["auc"], "dec_auc": q_dec["auc"],
            "base_brier": q_base["brier"], "dec_brier": q_dec["brier"],
            "base_sortino_lift": q_base["sortino_lift"],
            "dec_sortino_lift": q_dec["sortino_lift"],
        })),
        "",
        "## Sliding-cutoff distribution — 49 monthly cutoffs",
        "",
        _md(slide_summ.reset_index().rename(
            columns={"level_0": "model", "level_1": "arm"}).set_index(
            ["model", "arm"])[["n", "median", "q25", "q75",
                               "frac_positive", "frac_above_0p10", "max"]]),
        "",
        f"![decomp vs baseline]({fig_path.relative_to(PROJECT_ROOT)})",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Caveats",
        "- Same ~1,327-row OOS ceiling: per-cutoff resolution ~±0.5 Sortino, so "
        "the signal is the distributional shift across 49 cutoffs.",
        "- The decomposition's primary value is interpretability (which factor "
        "drives each regime); predictive lift is the secondary question tested here.",
    ]
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


def main() -> int:
    print("=" * 70)
    print("Decomposed-turbulence features vs v5 scalar baseline")
    print("=" * 70)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH)
    turb, binreg = tdf["turbulence"], tdf["regime_smoothed"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))
    thr = float(turb.loc[:DEV_END].dropna().quantile(0.90))
    y = binary_turbulence_entry(turb, threshold=thr, horizon=21)

    # Per-factor decomposition on the same clean 10-factor returns.
    clean = np.log(prices / prices.shift(1)).dropna(how="any")
    contrib = TurbulenceDecomposer(**TURB_PARAMS).fit_transform(clean)

    X_base = build_feature_matrix(prices, turb, regime_smoothed=binreg,
                                  factors=FACTORS_10)
    X_dec = build_feature_matrix_with_decomposition(
        prices, turb, contrib, regime_smoothed=binreg, factors=FACTORS_10)
    n_decomp_cols = X_dec.shape[1] - X_base.shape[1]
    print(f"[decomp] baseline X {X_base.shape}, decomp X {X_dec.shape} "
          f"(+{n_decomp_cols} composition cols), threshold {thr:.2f}")

    q_base = run_quarterly(X_base, y, eq, "baseline")
    q_dec = run_quarterly(X_dec, y, eq, "decomp")
    print("[decomp] baseline quarterly:\n", q_base.round(4).to_string())
    print("[decomp] decomp   quarterly:\n", q_dec.round(4).to_string())

    # Sliding cutoff: decomp (compute) + baseline (reuse 9C cache).
    v59c = pd.read_parquet(V5_9C_PATH)
    slide_base, slide_dec = {}, {}
    for model in SLIDING_MODELS:
        print(f"[decomp] sliding cutoff — decomp arm, {model} ...")
        slide_dec[model] = run_sliding(X_dec, y, eq, model, f"decomp_{model}")
        slide_base[model] = (v59c[v59c["model"] == model]
                             .set_index("cutoff").sort_index())

    rows = {}
    for model in SLIDING_MODELS:
        rows[(model, "baseline")] = summary_statistics(slide_base[model], "sortino_lift")
        rows[(model, "decomp")] = summary_statistics(slide_dec[model], "sortino_lift")
    slide_summ = pd.DataFrame(rows).T
    print("[decomp] sliding-cutoff summary:\n", slide_summ.round(3).to_string())

    q_base.to_parquet(ARTIFACT_DIR / "quarterly_baseline.parquet")
    q_dec.to_parquet(ARTIFACT_DIR / "quarterly_decomp.parquet")
    for model in SLIDING_MODELS:
        slide_dec[model].to_parquet(ARTIFACT_DIR / f"sliding_decomp_{model}.parquet")

    fig_path = render_figure(q_base, q_dec, slide_base, slide_dec)
    print(f"[decomp] saved figure: {fig_path.relative_to(PROJECT_ROOT)}")
    write_checkpoint(q_base, q_dec, slide_summ, fig_path, n_decomp_cols)
    print(f"[decomp] wrote {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
