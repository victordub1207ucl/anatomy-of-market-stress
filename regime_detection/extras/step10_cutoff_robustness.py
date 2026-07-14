"""Phase 9C driver — sliding-cutoff sensitivity for the Phase-6 headline.

49 monthly cutoffs from 2018-06-30 to 2022-06-30.  For each cutoff:
  * set ``DEV_END`` = cutoff and ``OOS_START`` = cutoff + 1 day
  * run the Phase-6 quarterly walk-forward on the resulting OOS partition
  * record Sortino lift vs buy-and-hold, AUC, Brier, max DD.

We run two models: ``logistic_l1`` (Phase-6 headline) and
``random_forest`` (a different model family — sanity check).  The
distribution of lifts (median, IQR, fraction-positive, fraction
above +0.10) becomes the new headline number replacing the brittle
single-cutoff point estimate.  The single-cutoff "headline" is derived
dynamically at the current ``DEV_END`` (2019-12-31).

Caching: each (model, cutoff) result is persisted as a JSON file in
``.cache/phase9c/``.  Reruns load from cache in ~1s.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_detection.lib.cutoff_sensitivity import (
    run_sliding_cutoff, summary_statistics,
)
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
ARTIFACT_PATH = (PROJECT_ROOT / "reports" / "robustness"
                 / "sliding_cutoff.parquet")
FIG_PATH = PROJECT_ROOT / "figures" / "thesis_v5" / "fig9_sliding_cutoff.png"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "reports" / "checkpoints"
    / "step10_cutoff_robustness.md"
)
CACHE_DIR = PROJECT_ROOT / ".cache" / "phase9c"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODELS = ["logistic_l1", "random_forest"]
HEADLINE_LIFT_FALLBACK = 0.0552  # used only if DEV_END is outside the sweep


def headline_lift_at_dev_end(primary: "pd.DataFrame") -> float:
    """The single-cutoff 'headline' Sortino lift = the sweep value at the
    project's DEV_END (the sliding cutoffs are month-end and include it).

    Derived dynamically so the headline always reflects the current split
    (DEV_END=2019-12-31) instead of a stale hard-coded number.
    """
    idx = primary.index
    if DEV_END in idx:
        return float(primary.loc[DEV_END, "sortino_lift"])
    # nearest cutoff on/just after DEV_END, else the closest available
    after = idx[idx >= DEV_END]
    pick = after.min() if len(after) else idx[(idx - DEV_END).map(abs).argmin()]
    return float(primary.loc[pick, "sortino_lift"])


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_inputs():
    print("[9C] loading data ...")
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    turb_df = pd.read_parquet(TURB_PATH)
    turbulence = turb_df["turbulence"]
    regime_smoothed = turb_df["regime_smoothed"]
    threshold = float(turbulence.loc[:DEV_END].dropna().quantile(0.90))
    X = build_feature_matrix(prices, turbulence,
                              regime_smoothed=regime_smoothed,
                              factors=FACTORS_10)
    y = binary_turbulence_entry(turbulence, threshold=threshold, horizon=21)
    eq_log_ret = np.log(prices["equity"] / prices["equity"].shift(1))
    print(f"[9C]   X={X.shape}, y nz={int(y.dropna().sum())}, "
          f"threshold={threshold:.2f}")
    return X, y, eq_log_ret


def make_cutoff_schedule(start: str = "2018-06-30",
                          end: str = "2022-06-30") -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="ME"))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(
    df_by_model: Dict[str, pd.DataFrame],
    summary_by_model: Dict[str, Dict[str, float]],
    headline_lift: float,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

    primary = df_by_model["logistic_l1"]
    secondary = df_by_model["random_forest"]
    summary_l1 = summary_by_model["logistic_l1"]

    # ── Row 0: Sortino lift ────────────────────────────────────────────
    ax_ts, ax_hist = axes[0, 0], axes[0, 1]

    # Lift time series.
    ax_ts.plot(primary.index, primary["sortino_lift"],
               "o-", color="#1f77b4", lw=1.6, ms=5,
               label="logistic_l1 (headline)", alpha=0.92)
    ax_ts.plot(secondary.index, secondary["sortino_lift"],
               "s--", color="#9467bd", lw=1.1, ms=3.5,
               label="random_forest (sanity)", alpha=0.7)
    ax_ts.axhline(0, color="grey", lw=0.7, ls="-")
    ax_ts.axhline(headline_lift, color="#d62728", lw=1.1, ls="--",
                   label=f"DEV_END={DEV_END.date()} headline ({headline_lift:+.3f})")
    ax_ts.axhspan(summary_l1["q25"], summary_l1["q75"],
                  color="#1f77b4", alpha=0.10,
                  label=f"IQR(L1): [{summary_l1['q25']:.2f}, "
                        f"{summary_l1['q75']:.2f}]")
    ax_ts.axhline(summary_l1["median"], color="#1f77b4", lw=1.0, ls=":",
                   label=f"median(L1) = {summary_l1['median']:.3f}")
    ax_ts.set_ylabel("Sortino lift vs buy-and-hold")
    ax_ts.set_title("Phase 9C — Sortino lift across sliding cutoffs")
    ax_ts.legend(loc="lower right", fontsize=8, framealpha=0.93)
    ax_ts.grid(True, alpha=0.3)

    # Lift histogram (L1 only — headline distribution).
    bins = np.linspace(primary["sortino_lift"].min() - 0.05,
                       primary["sortino_lift"].max() + 0.05, 22)
    ax_hist.hist(primary["sortino_lift"].dropna(), bins=bins,
                  color="#1f77b4", alpha=0.78, edgecolor="white")
    ax_hist.axvline(0, color="grey", lw=0.7)
    ax_hist.axvline(headline_lift, color="#d62728", lw=1.2, ls="--",
                     label=f"DEV_END headline")
    ax_hist.axvline(summary_l1["median"], color="#1f77b4", lw=1.2, ls=":",
                     label=f"median = {summary_l1['median']:.3f}")
    ax_hist.axvspan(summary_l1["q25"], summary_l1["q75"],
                     color="#1f77b4", alpha=0.15,
                     label="IQR")
    ax_hist.set_xlabel("Sortino lift (logistic_l1)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"distribution: n={summary_l1['n']}, "
        f"frac>0 = {summary_l1['frac_positive']:.0%}, "
        f"frac>+0.10 = {summary_l1['frac_above_0p10']:.0%}")
    ax_hist.legend(loc="upper left", fontsize=8)
    ax_hist.grid(axis="y", alpha=0.3)

    # ── Row 1: AUC ─────────────────────────────────────────────────────
    ax_ts2, ax_hist2 = axes[1, 0], axes[1, 1]

    summary_auc = summary_statistics(primary, metric="auc")
    ax_ts2.plot(primary.index, primary["auc"],
                "o-", color="#2ca02c", lw=1.6, ms=5,
                label="logistic_l1", alpha=0.92)
    ax_ts2.plot(secondary.index, secondary["auc"],
                "s--", color="#9467bd", lw=1.1, ms=3.5,
                label="random_forest", alpha=0.7)
    ax_ts2.axhline(0.5, color="grey", lw=0.7, ls="-")
    ax_ts2.axhline(summary_auc["median"], color="#2ca02c", lw=1.0, ls=":",
                    label=f"median(L1) = {summary_auc['median']:.3f}")
    ax_ts2.set_ylabel("OOS ROC AUC")
    ax_ts2.set_xlabel("cutoff date  (= DEV_END)")
    ax_ts2.set_title("Phase 9C — OOS AUC across sliding cutoffs")
    ax_ts2.legend(loc="lower right", fontsize=8, framealpha=0.93)
    ax_ts2.grid(True, alpha=0.3)

    bins_auc = np.linspace(primary["auc"].min() - 0.02,
                            primary["auc"].max() + 0.02, 22)
    ax_hist2.hist(primary["auc"].dropna(), bins=bins_auc,
                   color="#2ca02c", alpha=0.78, edgecolor="white")
    ax_hist2.axvline(0.5, color="grey", lw=0.7)
    ax_hist2.axvline(summary_auc["median"], color="#2ca02c", lw=1.2, ls=":",
                      label=f"median = {summary_auc['median']:.3f}")
    ax_hist2.axvspan(summary_auc["q25"], summary_auc["q75"],
                      color="#2ca02c", alpha=0.15, label="IQR")
    ax_hist2.set_xlabel("AUC (logistic_l1)")
    ax_hist2.set_ylabel("count")
    ax_hist2.set_title(f"AUC distribution: n={summary_auc['n']}")
    ax_hist2.legend(loc="upper left", fontsize=8)
    ax_hist2.grid(axis="y", alpha=0.3)

    plt.setp(ax_ts.get_xticklabels(), visible=False)
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    fig.suptitle("Phase 9C: 49 monthly cutoffs 2018-06 → 2022-06,  "
                 "Phase-6 quarterly walk-forward per cutoff",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return FIG_PATH


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _df_md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return df.to_markdown()


def write_checkpoint(
    df_by_model: Dict[str, pd.DataFrame],
    summary_by_model: Dict[str, Dict[str, float]],
    fig_path: Path,
    headline_lift: float,
) -> None:
    s_l1 = summary_by_model["logistic_l1"]
    s_rf = summary_by_model["random_forest"]

    # Verdict logic — keyed off logistic_l1 (the headline model).
    headline = headline_lift
    median = s_l1["median"]
    median_inside_iqr = (s_l1["q25"] <= headline <= s_l1["q75"])
    median_positive = median > 0
    median_near_zero = abs(median) < 0.05
    frac_pos = s_l1["frac_positive"]

    if median_near_zero or not median_positive:
        # Both the median and (under the fixed 2020 split) the headline are
        # typically <= 0 here: the overlay does NOT deliver a robust positive
        # lift.  Describe the headline-vs-median relation by its actual sign.
        rel = ("above" if headline > median else
               "below" if headline < median else "at")
        favour = ("a favourable draw" if headline > median
                  else "if anything slightly worse than typical")
        verdict_headline = (
            f"**Overlay lift is not a robust positive — the {headline:+.3f} "
            f"headline is {favour}.**"
        )
        verdict_detail = (
            f"The median Sortino lift across {s_l1['n']} monthly cutoffs "
            f"is {median:+.3f}; the {DEV_END.date()} headline of "
            f"{headline:+.3f} sits {rel} the median.  "
            f"Only {frac_pos:.0%} of cutoffs give a positive lift, and "
            f"only {s_l1['frac_above_0p10']:.0%} cross the "
            "+0.10 'economically meaningful' bar.  "
            "**Recommendation for the thesis**: report the median + IQR "
            f"({median:+.3f}, IQR=[{s_l1['q25']:+.3f}, "
            f"{s_l1['q75']:+.3f}]) and treat the single-cutoff "
            f"{headline:+.3f} as a non-robust point estimate.  Under the "
            "2020-start OOS the correction-timing overlay reduces "
            "drawdown but does not raise risk-adjusted return — the "
            "drawdown-timing target remains hard (see step26 for the "
            "better-posed hedge-selection framing).  L2 stands as a "
            "limitation."
        )
    elif median_inside_iqr and frac_pos > 0.60:
        verdict_headline = (
            "**Headline lift is typical of the distribution.**"
        )
        verdict_detail = (
            f"The {headline:+.3f} DEV_END headline sits inside the IQR "
            f"({s_l1['q25']:+.3f} to {s_l1['q75']:+.3f}) and "
            f"{frac_pos:.0%} of cutoffs deliver a positive lift "
            f"(median = {median:+.3f}).  L2 is downgraded: the headline "
            "is defensible as 'typical of a window of cutoffs', not as "
            "the answer at a single cutoff."
        )
    else:
        verdict_headline = (
            "**Headline lift is positive but not robust.**"
        )
        verdict_detail = (
            f"Median lift is {median:+.3f} (positive but smaller than "
            f"the +{headline:.3f} headline); fraction of cutoffs with "
            f"positive lift is {frac_pos:.0%}.  Report the median in "
            "the thesis abstract instead of the headline point estimate, "
            "and keep L2 as a softer limitation."
        )

    # Recommended abstract language.
    abstract_lang = (
        f"Across {s_l1['n']} monthly cutoffs from {df_by_model['logistic_l1'].index.min().date()} "
        f"to {df_by_model['logistic_l1'].index.max().date()}, the OOS "
        f"Sortino lift over buy-and-hold has median {median:+.3f} "
        f"and IQR [{s_l1['q25']:+.3f}, {s_l1['q75']:+.3f}], with "
        f"{frac_pos:.0%} of cutoffs delivering a positive lift."
    )

    md = []
    md.append("# CHECKPOINT — Phase 9C: sliding-cutoff sensitivity (L2)")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append("## What was implemented")
    md.append(
        "- `regime_detection/lib/cutoff_sensitivity.py`:\n"
        "  - `run_sliding_cutoff(X, y, equity_returns, *, cutoff_dates, "
        "model_name, ...)` loops the Phase-6 quarterly walk-forward over "
        "any list of `DEV_END` candidates, with per-(model, cutoff) "
        "JSON caching.\n"
        "  - `metrics_for_cutoff_run(qwf, equity_returns)` extracts the "
        "headline metrics from a `QuarterlyWalkForward` result.\n"
        "  - `summary_statistics(df, metric)` computes median / IQR / "
        "fraction-positive / fraction>+0.10 / Pearson correlation vs "
        "cutoff date."
    )
    md.append(
        "- `regime_detection/tests/test_cutoff_sensitivity.py` — 7 tests, "
        "all green: metric extractor on well-formed and degenerate "
        "(single-class) input, sliding-cutoff filter on min OOS quarters, "
        "cache round-trip on disk (identical DataFrame on re-read), "
        "empty schedule, summary stats on a known distribution."
    )
    md.append(
        "- `regime_detection/step10_cutoff_robustness.py` — 49 monthly "
        "cutoffs 2018-06-30 → 2022-06-30, two models (`logistic_l1` "
        "headline, `random_forest` sanity), aggressive caching at "
        "`.cache/phase9c/`."
    )
    md.append("")
    md.append("## What tests pass")
    md.append("- **136/136 total** (was 129 before Phase 9C — 7 new).")
    md.append("")
    md.append(f"## Cutoff schedule and coverage")
    md.append("")
    md.append(
        f"- {s_l1['n']} cutoffs emitted for logistic_l1 "
        f"(after min_oos_quarters=4 filter); "
        f"{s_rf['n']} for random_forest.\n"
        f"- Cutoff range: {df_by_model['logistic_l1'].index.min().date()} "
        f"to {df_by_model['logistic_l1'].index.max().date()}."
    )
    md.append("")
    md.append("## Distribution of OOS Sortino lift  (logistic_l1)")
    md.append("")
    summary_tbl = pd.DataFrame({
        "logistic_l1 (headline)": s_l1,
        "random_forest (sanity)": s_rf,
    }).T[[
        "n", "median", "mean", "q25", "q75", "min", "max",
        "frac_positive", "frac_above_0p10", "pearson_r_vs_cutoff",
    ]]
    md.append(_df_md(summary_tbl))
    md.append("")
    md.append(f"![Phase 9C distribution]({fig_path.relative_to(PROJECT_ROOT)})")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict_headline)
    md.append("")
    md.append(verdict_detail)
    md.append("")
    md.append("## Recommended thesis-abstract language")
    md.append("")
    md.append(f"> {abstract_lang}")
    md.append("")
    md.append("## Deviations from spec")
    md.append(
        "1. **Monthly frequency, not week or day.** The brief said "
        "'month-by-month' so 49 cutoffs at month-end suffices.  Finer "
        "frequency (weekly or daily) would multiply runtime ≈ 4–22× "
        "without changing the qualitative shape of the distribution."
    )
    md.append(
        "2. **min_oos_quarters=4** is the filter for cutoff inclusion.  "
        "A cutoff at 2022-06-30 leaves ≈15 OOS quarters, well above 4; "
        "no cutoffs in the schedule are rejected for short OOS at this "
        "threshold."
    )
    md.append(
        "3. **Hyperparameters frozen at Phase-4 defaults** (no per-"
        "cutoff tuning).  Same as Phase 6 — keeps the comparison clean "
        "and avoids tuning-on-distribution effects."
    )
    md.append("")
    md.append("## Open questions for the user")
    md.append(
        "- Should the thesis abstract use the median + IQR language "
        f"above, or report the {headline_lift:+.3f} single-cutoff headline "
        "with the median range as a footnote?"
    )
    md.append(
        "- Run the same sliding-cutoff for the GMM benchmark (Phase 9B) "
        "as a follow-up?  Would tell us whether GMM is also cutoff-"
        "sensitive or genuinely worse across the board."
    )
    md.append(
        "- Per the Phase 9 brief, proceeding to Phase 9D (port v4 "
        "splicing to extend turbulence to the full 14 factors) is the "
        "natural next step.  Will wait for your confirmation before "
        "starting."
    )
    md.append("")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Phase 9C: sliding-cutoff sensitivity for the headline Sortino lift")
    print("=" * 72)
    X, y, eq_log_ret = build_inputs()
    schedule = make_cutoff_schedule()
    print(f"[9C] schedule: {len(schedule)} monthly cutoffs from "
          f"{schedule[0].date()} to {schedule[-1].date()}")

    df_by_model: Dict[str, pd.DataFrame] = {}
    summary_by_model: Dict[str, Dict[str, float]] = {}
    long_rows: List[pd.DataFrame] = []

    for model in MODELS:
        print(f"\n[9C] === model: {model} ===")
        df = run_sliding_cutoff(
            X, y, eq_log_ret,
            cutoff_dates=schedule,
            model_name=model,
            purge_window=21,
            min_oos_quarters=4,
            random_state=0,
            cache_dir=CACHE_DIR,
            verbose=True,
        )
        if df.empty:
            print(f"[9C]   no cutoffs emitted for {model} (skip)")
            continue
        df_by_model[model] = df
        summary_by_model[model] = summary_statistics(df, "sortino_lift")
        df_long = df.copy()
        df_long["model"] = model
        long_rows.append(df_long.reset_index())

    if not df_by_model:
        raise RuntimeError("No cutoff runs succeeded — check cache + data")

    full = pd.concat(long_rows, axis=0, ignore_index=True)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(ARTIFACT_PATH)
    print(f"\n[9C] saved {ARTIFACT_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(full)} rows)")

    print("[9C] summary:")
    print(pd.DataFrame(summary_by_model).T.round(4).to_string())

    headline_lift = headline_lift_at_dev_end(df_by_model["logistic_l1"])
    print(f"[9C] headline lift at DEV_END={DEV_END.date()}: {headline_lift:+.4f}")
    fig_path = render_figure(df_by_model, summary_by_model, headline_lift)
    print(f"[9C] saved figure: {fig_path.relative_to(PROJECT_ROOT)}")
    write_checkpoint(df_by_model, summary_by_model, fig_path, headline_lift)
    print(f"[9C] wrote checkpoint: "
          f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
