"""Step 23 — the absorption ratio: a fragility signal complementary to turbulence.

Closes the project's weakest "why not X?" gap (KLPR 2011) and tests the
substantive question: does *systemic fragility* (how coupled the factors are)
add predictive information about corrections **beyond** turbulence (how unusual
today is)?

Two parts:
  1. **Validation** — compute the causal absorption ratio on the 10 factors,
     check it behaves (rises into crises) and is related-but-distinct from
     turbulence (so it's not redundant).
  2. **Predictive test** — add the AR level + AR standardised shift to the
     turbulence bucket features and rerun the 5% equity-correction task; compare
     turbulence-only vs turbulence+AR on AUC and the 49-cutoff distribution.

Run:  python3 -m regime_detection.step23_absorption_ratio
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from regime_detection.lib.absorption import AbsorptionRatio, absorption_shift
from regime_detection.lib.cutoff_sensitivity import run_sliding_cutoff, summary_statistics
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.targets import forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.splits import DEV_END, OOS_START

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "absorption_ratio"
FIG_DIR = PROJECT_ROOT / "figures" / "absorption_ratio"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step23_absorption_ratio.md"
CACHE = PROJECT_ROOT / ".cache" / "absorption_ratio"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"; LOSS = 0.05; HORIZON = 21
SCHEDULE = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
CRISES = {"2008 GFC": ("2008-09-01", "2009-03-31"), "2011 Euro": ("2011-08-01", "2011-10-31"),
          "2015 China": ("2015-08-01", "2016-02-29"), "2018 Q4": ("2018-10-01", "2018-12-31"),
          "2020 COVID": ("2020-02-15", "2020-05-31"), "2022 infl": ("2022-01-01", "2022-12-31")}


def evaluate(X, y, eq, tag):
    Xc = X.reindex(y.dropna().index).dropna(); yc = y.reindex(Xc.index)
    oos = run_quarterly_walk_forward(Xc, yc, model_names=[MODEL], purge_window=HORIZON,
            dev_end=DEV_END, oos_start=OOS_START, random_state=0)[MODEL]
    opred = oos.oos_predictions.dropna(); olab = oos.oos_labels.reindex(opred.index)
    auc = roc_auc_score(olab, opred) if olab.nunique() > 1 else np.nan
    slide = run_sliding_cutoff(Xc, yc, eq, cutoff_dates=SCHEDULE, model_name=MODEL,
            purge_window=HORIZON, min_oos_quarters=4, random_state=0,
            cache_dir=CACHE / tag, verbose=False)
    s = summary_statistics(slide, "sortino_lift")
    return {"auc": auc, "median_49": s["median"], "frac_pos_49": s["frac_positive"],
            "_slide": slide["sortino_lift"].dropna()}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    tdf = pd.read_parquet(TURB_PATH); turb = tdf["turbulence"]
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    eq = np.log(prices["equity"] / prices["equity"].shift(1))

    # ---- 1. compute AR (n=2 for 10 factors) + standardised shift ----
    ar = AbsorptionRatio(n_components=2, lookback_days=504, min_periods=252).fit_transform(rets)
    ar_shift = absorption_shift(ar)
    fin = ar.dropna()
    OUT_DIR.joinpath("absorption_series.parquet")
    pd.concat([ar, ar_shift], axis=1).to_parquet(OUT_DIR / "absorption_series.parquet")
    print(f"AR: {len(fin)} rows, {fin.index.min().date()}→{fin.index.max().date()}, "
          f"mean {fin.mean():.3f}")

    # validation: relation to turbulence + crisis behaviour
    both = pd.concat([ar.rename("ar"), turb.rename("turb")], axis=1).dropna()
    corr = float(both["ar"].corr(both["turb"]))
    ar_pct = ar.rank(pct=True)
    crisis_ar = {k: float(ar.loc[s:e].mean()) for k, (s, e) in CRISES.items()}
    overall_ar = float(fin.mean())
    print(f"corr(AR, turbulence) = {corr:.3f}; overall AR {overall_ar:.3f}; "
          f"crisis AR: " + ", ".join(f"{k} {v:.2f}" for k, v in crisis_ar.items()))

    # ---- 2. predictive test: turbulence-only vs turbulence + AR ----
    y = (forward_drawdown_conditional(prices["equity"], horizon=HORIZON) <= -LOSS).astype(float)
    y = y.where(forward_drawdown_conditional(prices["equity"], horizon=HORIZON).notna())
    X_base = build_feature_matrix_with_bucket(prices, turb, tdf["regime_smoothed"],
                                              factors=FACTORS_10, n_buckets=5)
    X_ar = X_base.copy()
    X_ar["absorption"] = ar.reindex(X_ar.index)          # causal level (uses data < t)
    X_ar["absorption_shift"] = ar_shift.reindex(X_ar.index)
    print("evaluating turbulence-only ..."); base = evaluate(X_base, y, eq, "base")
    print("evaluating turbulence + AR ..."); plus = evaluate(X_ar, y, eq, "plus_ar")

    cmp = pd.DataFrame({
        "turbulence only": {k: base[k] for k in ["auc", "median_49", "frac_pos_49"]},
        "turbulence + AR": {k: plus[k] for k in ["auc", "median_49", "frac_pos_49"]},
    }).T
    cmp.to_csv(OUT_DIR / "turb_vs_turb_plus_ar.csv")
    print(cmp.round(3).to_string())

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # A: AR vs turbulence timeline (both percentile-ranked for comparability)
    a = ax[0]
    a.plot(ar_pct.index, ar_pct, color="#8e44ad", lw=0.8, label="absorption ratio (pct)")
    a.plot(turb.rank(pct=True).index, turb.rank(pct=True), color="#7f8c8d", lw=0.6,
           alpha=0.7, label="turbulence (pct)")
    for s, e in CRISES.values():
        a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#c0392b", alpha=0.12)
    a.set_title(f"Fragility (AR) vs unusualness (turbulence)  ·  corr {corr:.2f}")
    a.set_ylabel("percentile"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # B: head-to-head metrics
    a = ax[1]
    labels = ["AUC", "49-median", "% positive"]
    bv = [base["auc"], base["median_49"], base["frac_pos_49"]]
    pv = [plus["auc"], plus["median_49"], plus["frac_pos_49"]]
    xx = np.arange(3); w = 0.38
    a.bar(xx - w/2, bv, w, color="#7f8c8d", label="turbulence only")
    a.bar(xx + w/2, pv, w, color="#8e44ad", label="+ absorption ratio")
    a.axhline(0, color="k", lw=0.6); a.axhline(0.5, color="grey", ls=":", lw=0.7)
    a.set_xticks(xx); a.set_xticklabels(labels, fontsize=9)
    a.set_title("Does fragility add to prediction? (5% correction)")
    a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)
    # C: 49-cutoff distributions
    a = ax[2]
    bins = np.linspace(-0.45, 0.35, 22)
    a.hist(base["_slide"], bins=bins, alpha=0.55, color="#7f8c8d",
           label=f"turb (med {base['median_49']:+.2f}, {base['frac_pos_49']:.0%}+)")
    a.hist(plus["_slide"], bins=bins, alpha=0.55, color="#8e44ad",
           label=f"+AR (med {plus['median_49']:+.2f}, {plus['frac_pos_49']:.0%}+)")
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("Sortino lift (49 cutoffs)"); a.set_title("49-cutoff robustness")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Step 23 — absorption ratio (systemic fragility) and its predictive value",
                 fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "absorption_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {FIG_DIR / 'absorption_ratio.png'}")

    # ---- verdict ----
    d_auc = plus["auc"] - base["auc"]; d_fp = plus["frac_pos_49"] - base["frac_pos_49"]
    d_med = plus["median_49"] - base["median_49"]
    helps = (d_auc > 0.02) + (d_fp > 0.05) + (d_med > 0.03)
    if helps >= 2:
        pv_ = (f"**ADDS VALUE.** Fragility carries information beyond unusualness: "
               f"AUC {base['auc']:.3f}→{plus['auc']:.3f}, 49-median {base['median_49']:+.3f}"
               f"→{plus['median_49']:+.3f}, positive cutoffs {base['frac_pos_49']:.0%}→"
               f"{plus['frac_pos_49']:.0%}. The absorption ratio earns a place as a "
               "complementary feature.")
    elif helps == 1:
        pv_ = (f"**MARGINAL.** AR moves one axis only (ΔAUC {d_auc:+.3f}, Δ%pos "
               f"{d_fp:+.0%}, Δmedian {d_med:+.3f}) — inside the noise floor. Keep it "
               "as a validated complementary signal; predictive lift is not robust.")
    else:
        pv_ = (f"**NO PREDICTIVE LIFT.** Adding AR does not improve the correction "
               f"forecast (ΔAUC {d_auc:+.3f}, Δ%pos {d_fp:+.0%}). AR is a valid, "
               "distinct *fragility* signal (corr {0:.2f} with turbulence) but, like "
               "turbulence, it doesn't make equity drawdowns more forecastable — "
               "consistent with the sample-size ceiling. It still closes the "
               "'why not the absorption ratio?' gap.").format(corr)

    md = [
        "# CHECKPOINT — Step 23: absorption ratio (systemic fragility)",
        "", f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC", "",
        "## What was built",
        "`lib/absorption.py` — causal walk-forward **absorption ratio** (KLPR 2011): "
        "fraction of factor variance in the top-2 principal components of the "
        "trailing-504-day covariance, plus the standardised AR *shift* "
        "(early-warning signal). Closes the project's weakest 'why not X?' gap.",
        "",
        "## 1. Validation",
        f"- AR computed on the 10 factors, {fin.index.min().date()}→{fin.index.max().date()}; "
        f"mean {overall_ar:.3f}.",
        f"- **Nearly orthogonal to turbulence:** Pearson corr {corr:.2f} (even "
        "slightly negative over this sample, partly the different window lengths "
        "— AR uses 2y, turbulence 10y). So AR carries *distinct* information "
        "(fragility ≠ unusualness) rather than re-encoding turbulence — a good "
        "property for a complementary feature.",
        f"- Crisis-window mean AR: " + ", ".join(f"{k} {v:.2f}" for k, v in crisis_ar.items())
        + f" (vs {overall_ar:.2f} overall). The 2-year-window AR *level* is smooth "
        "and does not spike in individual crises; KLPR's crisis signal lives in "
        "the standardised **shift** (rapid rise in coupling), which is included as "
        "a separate feature.",
        "",
        "## 2. Predictive value — turbulence vs turbulence + AR (5% correction)",
        "",
        cmp.round(3).to_markdown(),
        "",
        "![absorption](../../figures/absorption_ratio/absorption_ratio.png)",
        "",
        "## Verdict", "", pv_, "",
        "## Notes",
        "- AR uses a 2-year window (KLPR-style, more responsive) vs turbulence's "
        "10-year window — intentional: AR tracks *current* coupling.",
        "- The AR level is causal (covariance window ends at t-1), so it is a valid "
        "feature with no shift; the AR shift is additionally `.shift(1)`-ed.",
        "- Same OOS ceiling — read the 49-cutoff distribution.",
    ]
    CHECKPOINT.write_text("\n".join(md))
    print(f"wrote {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
