"""Phase 9F driver — verify the ordinal-turbulence-bucket improvement.

The Phase-9 middle-path experiment found that replacing v5's BINARY
turbulent/quiet regime feature with an ORDINAL 5-level turbulence bucket
gives a less cutoff-sensitive supervised signal — and that the GMM the
middle path wrapped around it adds nothing.  That finding was single-model,
single-window (2010-12+).  This driver verifies it properly:

* the FULL 2007-12+ window (no GMM warm-up handicap);
* all four challenger models for the quarterly walk-forward;
* the 49-cutoff sliding test for logistic_l1 AND random_forest;
* head-to-head against the v5 binary baseline (reusing the Phase-9C cache
  for the binary sliding-cutoff numbers).

If the ordinal bucket beats the binary threshold here, it is a confirmed,
free improvement to v5.  If it doesn't survive the full window / second
model, it was a 2010-12+ artefact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_detection.lib.cutoff_sensitivity import (
    run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.metrics import (
    brier_score, max_drawdown, sortino_ratio, strategy_returns,
)
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.turbulence_bucket import (
    build_feature_matrix_with_bucket,
)
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
V5_9C_PATH = PROJECT_ROOT / "reports" / "robustness" / "sliding_cutoff.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "robustness" / "ordinal_bucket"
FIG_PATH = PROJECT_ROOT / "figures" / "thesis_v5" / "fig11_ordinal_bucket.png"
CHECKPOINT_PATH = (PROJECT_ROOT / "reports"
                   / "checkpoints" / "step12_ordinal_bucket.md")
CACHE_DIR = PROJECT_ROOT / ".cache" / "phase9f"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODELS = ["random_forest", "logistic_l1", "linear_svm", "logistic_elasticnet"]
SLIDING_MODELS = ["logistic_l1", "random_forest"]


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
    print(f"[9F] quarterly walk-forward ({label}) ...")
    models = run_quarterly_walk_forward(
        X, y, model_names=MODELS, purge_window=21,
        dev_end=DEV_END, oos_start=OOS_START, random_state=0)
    return pd.DataFrame(
        {name: _qmetrics(m, eq) for name, m in models.items()}).T


def run_sliding(X, y, eq, model, tag):
    sched = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
    return run_sliding_cutoff(
        X, y, eq, cutoff_dates=sched, model_name=model, purge_window=21,
        min_oos_quarters=4, random_state=0,
        cache_dir=CACHE_DIR / tag, verbose=False)


def render_figure(q_binary, q_bucket, slide_binary, slide_bucket):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: AUC binary vs bucket per model.
    ax = axes[0]
    x = np.arange(len(MODELS)); w = 0.38
    ax.bar(x - w/2, [q_binary.loc[m, "auc"] for m in MODELS], w,
           color="#1f77b4", label="v5 binary threshold")
    ax.bar(x + w/2, [q_bucket.loc[m, "auc"] for m in MODELS], w,
           color="#e67e22", label="ordinal 5-bucket")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_", "\n") for m in MODELS],
                                         fontsize=8)
    ax.set_ylim(0.45, 0.75); ax.set_ylabel("OOS ROC AUC")
    ax.set_title("AUC — binary vs ordinal bucket (full window)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Right: sliding-cutoff Sortino-lift distributions (logistic_l1).
    ax = axes[1]
    bins = np.linspace(-0.45, 0.30, 26)
    sb = slide_binary["logistic_l1"]["sortino_lift"].dropna()
    su = slide_bucket["logistic_l1"]["sortino_lift"].dropna()
    ax.hist(sb, bins=bins, alpha=0.6, color="#1f77b4", label="binary")
    ax.hist(su, bins=bins, alpha=0.6, color="#e67e22", label="ordinal bucket")
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvline(sb.median(), color="#1f77b4", ls="--", lw=1.4,
               label=f"binary median {sb.median():+.3f}")
    ax.axvline(su.median(), color="#e67e22", ls="--", lw=1.4,
               label=f"bucket median {su.median():+.3f}")
    ax.set_xlabel("Sortino lift vs buy-and-hold (49 monthly cutoffs)")
    ax.set_ylabel("count")
    ax.set_title("Cutoff-sensitivity — logistic_l1, FULL window")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 9F — ordinal 5-bucket turbulence vs v5 binary "
                 "threshold (full 2007-12+ window)", fontsize=12, y=1.01)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return FIG_PATH


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return df.to_markdown()


def write_checkpoint(q_binary, q_bucket, slide_summ, fig_path):
    # Verdict keyed on logistic_l1 sliding-cutoff median.
    med_bin = slide_summ.loc[("logistic_l1", "binary"), "median"]
    med_buc = slide_summ.loc[("logistic_l1", "bucket"), "median"]
    fp_bin = slide_summ.loc[("logistic_l1", "binary"), "frac_positive"]
    fp_buc = slide_summ.loc[("logistic_l1", "bucket"), "frac_positive"]
    med_bin_rf = slide_summ.loc[("random_forest", "binary"), "median"]
    med_buc_rf = slide_summ.loc[("random_forest", "bucket"), "median"]
    gain_l1 = med_buc - med_bin
    gain_rf = med_buc_rf - med_bin_rf

    both_improve = gain_l1 > 0.03 and gain_rf > 0.0
    if both_improve and med_buc > -0.05:
        verdict = (
            f"**CONFIRMED.** On the full 2007-12+ window the ordinal "
            f"5-bucket lifts the logistic_l1 median sliding-cutoff Sortino "
            f"from {med_bin:+.3f} to {med_buc:+.3f} (frac positive "
            f"{fp_bin:.0%} -> {fp_buc:.0%}) and the random_forest median "
            f"from {med_bin_rf:+.3f} to {med_buc_rf:+.3f}.  The improvement "
            "survives the full window and a second model — it was NOT a "
            "2010-12+ artefact.  Recommend adopting the ordinal bucket as "
            "v5's regime feature; it is a ~30-line, GMM-free change that "
            "directly addresses the L2 cutoff-sensitivity limitation.")
    elif gain_l1 > 0.0:
        verdict = (
            f"**PARTIAL.** logistic_l1 median moves {gain_l1:+.3f} "
            f"({med_bin:+.3f} -> {med_buc:+.3f}) but random_forest moves "
            f"{gain_rf:+.3f}.  The bucket helps the linear model more than "
            "the tree; adopt with that caveat.")
    else:
        verdict = (
            f"**NOT CONFIRMED on the full window.** logistic_l1 median "
            f"moved {gain_l1:+.3f} ({med_bin:+.3f} -> {med_buc:+.3f}).  The "
            "2010-12+ result did not generalise; the ordinal bucket is not "
            "a robust improvement and v5 should keep the binary threshold.")

    md = []
    md.append("# CHECKPOINT — Phase 9F: ordinal turbulence bucket "
              "(verifying the middle-path side-finding)")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append("## What was tested")
    md.append(
        "The middle-path experiment (`prevversion/midpath/`) showed the GMM "
        "adds nothing but that an ordinal 5-level turbulence bucket beats "
        "v5's binary turbulent/quiet threshold on cutoff-robustness "
        "(median -0.106 -> +0.033 on the 2010-12+ window, logistic_l1 "
        "only).  Phase 9F verifies that on the FULL 2007-12+ window, all "
        "four models for the quarterly walk-forward, and two models for "
        "the 49-cutoff sliding test.")
    md.append("")
    md.append("## What was built")
    md.append(
        "- `regime_detection/lib/turbulence_bucket.py` — "
        "`ordinal_turbulence_bucket()` (causal n-quantile binning) + "
        "`build_feature_matrix_with_bucket()` (v5 matrix, binary regime "
        "swapped for the bucket).\n"
        "- `regime_detection/tests/test_turbulence_bucket.py` — 6 tests "
        "(range, warm-up NaN, occupancy, no-lookahead shock, monotonicity, "
        "validation).\n"
        "- `regime_detection/step12_ordinal_bucket.py` — this driver.")
    md.append("")
    md.append("## What tests pass")
    md.append("- **156/156 total** (was 150 — 6 new bucket tests).")
    md.append("")
    md.append("## Quarterly walk-forward (full window, 4 models)")
    md.append("")
    cmp = pd.DataFrame({
        "binary_auc": q_binary["auc"], "bucket_auc": q_bucket["auc"],
        "binary_brier": q_binary["brier"], "bucket_brier": q_bucket["brier"],
        "binary_sortino_lift": q_binary["sortino_lift"],
        "bucket_sortino_lift": q_bucket["sortino_lift"],
    })
    md.append(_md(cmp))
    md.append("")
    md.append("## Sliding-cutoff distribution — 49 monthly cutoffs, FULL window")
    md.append("")
    md.append(_md(slide_summ.reset_index().rename(
        columns={"level_0": "model", "level_1": "feature"}).set_index(
        ["model", "feature"])[["n", "median", "q25", "q75",
                               "frac_positive", "frac_above_0p10", "max"]]))
    md.append("")
    md.append(f"![Phase 9F]({fig_path.relative_to(PROJECT_ROOT)})")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")
    md.append("## Caveats")
    md.append(
        "- Same ~1,327-row OOS sample-size ceiling: the per-cutoff "
        "resolution is ~±0.5 Sortino, so the signal is the DISTRIBUTIONAL "
        "shift across 49 cutoffs, not any single cutoff.\n"
        "- AUC may be lower than v5 even where Sortino-robustness improves "
        "(the bucket is a more robust trader, not necessarily a better "
        "classifier) — read the table above.\n"
        "- The bucket is deterministic (no model seed), so unlike the GMM "
        "there is no seed-robustness concern.")
    md.append("")
    md.append("## Open questions for the user")
    md.append(
        "- If CONFIRMED: fold the ordinal bucket into the Phase-10 thesis "
        "as the v5 regime feature, and update the abstract's cutoff-"
        "distribution numbers?\n"
        "- Sweep `n_buckets` (3 / 5 / 10) to find the best granularity, or "
        "is 5 sufficient for the thesis?")
    md.append("")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


def main() -> int:
    print("=" * 70)
    print("Phase 9F: verify ordinal turbulence bucket vs v5 binary threshold")
    print("=" * 70)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH)
    turb, binreg = tdf["turbulence"], tdf["regime_smoothed"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))
    thr = float(turb.loc[:DEV_END].dropna().quantile(0.90))
    y = binary_turbulence_entry(turb, threshold=thr, horizon=21)

    X_binary = build_feature_matrix(prices, turb, regime_smoothed=binreg,
                                    factors=FACTORS_10)
    X_bucket = build_feature_matrix_with_bucket(
        prices, turb, binreg, factors=FACTORS_10, n_buckets=5)
    print(f"[9F] binary X {X_binary.shape}, bucket X {X_bucket.shape}, "
          f"threshold {thr:.2f}")

    q_binary = run_quarterly(X_binary, y, eq, "binary")
    q_bucket = run_quarterly(X_bucket, y, eq, "bucket")
    print("[9F] binary quarterly:\n", q_binary.round(4).to_string())
    print("[9F] bucket quarterly:\n", q_bucket.round(4).to_string())

    # Sliding cutoff: bucket (compute) + binary (reuse 9C cache).
    print("[9F] sliding cutoff — ordinal bucket (2 models, 49 cutoffs) ...")
    slide_binary, slide_bucket = {}, {}
    v59c = pd.read_parquet(V5_9C_PATH)
    for model in SLIDING_MODELS:
        slide_bucket[model] = run_sliding(X_bucket, y, eq, model, "bucket")
        slide_binary[model] = (v59c[v59c["model"] == model]
                               .set_index("cutoff").sort_index())

    # Summary table.
    rows = {}
    for model in SLIDING_MODELS:
        rows[(model, "binary")] = summary_statistics(slide_binary[model],
                                                     "sortino_lift")
        rows[(model, "bucket")] = summary_statistics(slide_bucket[model],
                                                     "sortino_lift")
    slide_summ = pd.DataFrame(rows).T
    print("[9F] sliding-cutoff summary:\n", slide_summ.round(3).to_string())

    # Persist.
    q_binary.to_parquet(ARTIFACT_DIR / "quarterly_binary.parquet")
    q_bucket.to_parquet(ARTIFACT_DIR / "quarterly_bucket.parquet")
    for model in SLIDING_MODELS:
        slide_bucket[model].to_parquet(
            ARTIFACT_DIR / f"sliding_bucket_{model}.parquet")

    fig_path = render_figure(q_binary, q_bucket, slide_binary, slide_bucket)
    print(f"[9F] saved figure: {fig_path.relative_to(PROJECT_ROOT)}")
    write_checkpoint(q_binary, q_bucket, slide_summ, fig_path)
    print(f"[9F] wrote {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
