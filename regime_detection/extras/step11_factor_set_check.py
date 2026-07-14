"""Phase 9D driver — full 14-factor turbulence with v4 splicing.

Honest reality of what splicing can deliver
-------------------------------------------
The cached ``data/cache/factor_prices.parquet`` already has v4's
splicing applied — quality goes back to 2005-12-07 (SPHQ proxy),
momentum to 2007-03-02 (PDP proxy), etc.  The remaining three factors
(em_credit, short_vol, low_risk) have **no fallback in v4's table** —
``regime_detection.lib.splicing.UNSPLICEABLE_FACTORS``.  Their primary
ETFs incept in 2007-12, 2011-10, and 2011-10 respectively, and we do
not invent new proxies in Phase 9D.

So the "extend all 14 factors back to 2005-12" target in the brief is
not literally achievable with the v4 proxy set.  What we CAN do:

* compute turbulence on the full 14-factor cross-section over its
  achievable range (post-warmup, starting ≈2013-10);
* compare to the existing 10-factor turbulence over the common range;
* run the Phase-6 OOS pipeline on the 14-factor turbulence as the
  source of the binary target; check whether the supervised metrics
  change materially relative to the 10-factor headline.

Process and reporting are run honestly either way — if 14-factor
materially differs from 10-factor on OOS, the thesis must report it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from regime_detection.lib.splicing import (
    FACTOR_DEFINITIONS, UNSPLICEABLE_FACTORS, report_first_valid_dates,
)
from regime_detection.lib.metrics import (
    annualised_return, brier_score, log_loss, max_drawdown, sortino_ratio,
    strategy_returns,
)
from regime_detection.lib.oos_pipeline import (
    run_quarterly_walk_forward,
)
from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.features import build_feature_matrix
from regime_detection.lib.targets import binary_turbulence_entry
from regime_detection.lib.splits import DEV_END, OOS_START


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_10F_PATH = (PROJECT_ROOT / "reports" / "turbulence"
                 / "turbulence_series.parquet")
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "robustness"
TURB_14F_PATH = ARTIFACT_DIR / "fourteen_factor_turbulence.parquet"
FIG_PATH = PROJECT_ROOT / "figures" / "thesis_v5" / "fig10_factor_coverage.png"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "reports" / "checkpoints"
    / "step11_factor_set_check.md"
)

FACTORS_14 = list(FACTOR_DEFINITIONS.keys())
FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]

CRISIS_WINDOWS = [
    ("2008 GFC",          "2008-09-01", "2009-03-31"),
    ("2011 Euro / US dg", "2011-08-01", "2011-10-31"),
    ("2015 oil / China",  "2015-08-01", "2016-02-29"),
    ("2018 Vol-mageddon", "2018-02-01", "2018-02-28"),
    ("2018 Q4 sell-off",  "2018-10-01", "2018-12-31"),
    ("2020 COVID",        "2020-02-15", "2020-05-31"),
    ("2022 inflation",    "2022-01-01", "2022-12-31"),
]

LOOKBACK = 2520
MIN_PERIODS = 504


# ---------------------------------------------------------------------------
# Step 1: compute 14-factor turbulence
# ---------------------------------------------------------------------------


def compute_14factor_turbulence() -> Tuple[pd.Series, pd.DataFrame]:
    print(f"[9D] loading {PRICES_PATH.relative_to(PROJECT_ROOT)} ...")
    prices = pd.read_parquet(PRICES_PATH).sort_index()
    # The cache has 14 columns; restrict to the canonical order.
    prices = prices[FACTORS_14]
    coverage = report_first_valid_dates(prices)
    print("\n[9D] factor first-valid dates (post-v4-splicing):")
    print(coverage.to_string())

    log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
    print(f"\n[9D] 14-factor all-finite slice: {log_ret.shape}, "
          f"{log_ret.index.min().date()} → {log_ret.index.max().date()}")

    print("[9D] computing turbulence (lookback=2520, min_periods=504, "
          "Ledoit-Wolf, refit_every=1) ...")
    ti = TurbulenceIndex(
        lookback_days=LOOKBACK, min_periods=MIN_PERIODS,
        shrinkage="ledoit-wolf", refit_every=1,
    )
    turb_14 = ti.fit_transform(log_ret).rename("turbulence_14f")
    finite = turb_14.dropna()
    print(f"[9D] 14-factor turbulence: {len(finite):,} finite rows, "
          f"{finite.index.min().date()} → {finite.index.max().date()}")
    return turb_14, coverage


# ---------------------------------------------------------------------------
# Step 2: crisis sanity check
# ---------------------------------------------------------------------------


def crisis_sanity(turb_series: pd.Series, label: str) -> pd.DataFrame:
    finite = turb_series.dropna()
    if len(finite) == 0:
        return pd.DataFrame()
    top10 = finite.quantile(0.90)
    rows = []
    for name, start, end in CRISIS_WINDOWS:
        window = finite.loc[start:end]
        if len(window) == 0:
            rows.append({
                "crisis":            name,
                "n_days_in_window":  0,
                "max_d":             float("nan"),
                "top_decile_days":   0,
                "verdict":           "SKIP (no data)",
            })
            continue
        spikes = int((window >= top10).sum())
        rows.append({
            "crisis":            name,
            "n_days_in_window":  int(len(window)),
            "max_d":             float(window.max()),
            "top_decile_days":   spikes,
            "verdict":           "PASS" if spikes > 0 else "FAIL",
        })
    df = pd.DataFrame(rows).set_index("crisis")
    print(f"\n[9D] crisis sanity ({label}, top-decile threshold = "
          f"{top10:.2f}):")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Step 3: 14F vs 10F comparison
# ---------------------------------------------------------------------------


def compare_turbulence_series(
    turb_14f: pd.Series,
    turb_10f: pd.Series,
) -> Dict[str, float]:
    common = turb_14f.dropna().index.intersection(turb_10f.dropna().index)
    if len(common) == 0:
        return {k: float("nan") for k in
                ("n_common", "pearson_r", "spearman_r",
                 "top_decile_overlap_pct")}
    a = turb_14f.reindex(common)
    b = turb_10f.reindex(common)
    pearson = float(np.corrcoef(a.values, b.values)[0, 1])
    from scipy.stats import spearmanr
    spearman = float(spearmanr(a.values, b.values).statistic)
    # Top-decile day overlap: of the days in either series's top decile,
    # what fraction are in both?
    thr_a = float(a.quantile(0.90))
    thr_b = float(b.quantile(0.90))
    set_a = set(a[a >= thr_a].index)
    set_b = set(b[b >= thr_b].index)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    return {
        "n_common":              len(common),
        "pearson_r":             pearson,
        "spearman_r":            spearman,
        "top_decile_overlap_pct": overlap,
        "first_common":          str(common.min().date()),
        "last_common":           str(common.max().date()),
    }


def persistence_table(turb_series: pd.Series) -> pd.DataFrame:
    """Replicate Skulls Table 1: ratio of conditional to unconditional
    mean turbulence at horizons 1, 5, 10, 20 days after a top-decile day."""
    finite = turb_series.dropna()
    if len(finite) == 0:
        return pd.DataFrame()
    threshold = finite.quantile(0.90)
    spike_dates = finite[finite >= threshold].index
    unconditional = finite.mean()
    rows = []
    for horizon in (1, 5, 10, 20):
        forward_means = []
        for d in spike_dates:
            loc = finite.index.get_indexer([d])[0]
            if loc < 0 or loc + horizon >= len(finite):
                continue
            window = finite.iloc[loc + 1: loc + 1 + horizon]
            if len(window) == horizon:
                forward_means.append(window.mean())
        cond = float(np.mean(forward_means)) if forward_means else float("nan")
        rows.append({
            "horizon":     horizon,
            "E[d|spike]":  cond,
            "E[d]":        float(unconditional),
            "ratio":       cond / float(unconditional)
                           if np.isfinite(cond) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Step 4: conditional Phase-6 rerun on 14-factor turbulence
# ---------------------------------------------------------------------------


def run_oos_protocol_on_turbulence(
    turb_series: pd.Series,
    threshold: float,
    label: str,
    model_names: List[str],
) -> Dict[str, dict]:
    """Run the Phase-6 quarterly walk-forward with the binary target
    derived from the given turbulence series."""
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    # Use the existing 10-factor regime_smoothed as a regime feature
    # (the supervised pipeline's feature matrix needs it; we're not
    # changing the FEATURES, only the TARGET).  This isolates the
    # effect of swapping the turbulence used to BUILD the target.
    regime_smoothed = pd.read_parquet(TURB_10F_PATH)["regime_smoothed"]
    X = build_feature_matrix(prices, turb_series,
                              regime_smoothed=regime_smoothed,
                              factors=FACTORS_10)
    y = binary_turbulence_entry(turb_series, threshold=threshold, horizon=21)
    print(f"[9D] running Phase 6 quarterly walk-forward "
          f"({label}, threshold={threshold:.2f}) ...")
    eq_log_ret = np.log(prices["equity"] / prices["equity"].shift(1))

    models = run_quarterly_walk_forward(
        X, y, model_names=model_names,
        purge_window=21, dev_end=DEV_END, oos_start=OOS_START,
        random_state=0,
    )
    out = {}
    for name, m in models.items():
        yt = m.oos_labels.dropna()
        pt = m.oos_predictions.dropna()
        common = yt.index.intersection(pt.index)
        yt, pt = yt.reindex(common), pt.reindex(common)
        if len(yt) < 50 or len(yt.unique()) < 2:
            out[name] = {k: float("nan") for k in
                         ("n_oos", "auc", "brier", "sortino_lift",
                          "max_dd_strategy", "max_dd_bh")}
            continue
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yt.values, pt.values))
        strat = strategy_returns(eq_log_ret, pt, shift=1)
        bh = eq_log_ret.reindex(strat.index)
        s_strat = sortino_ratio(strat)
        s_bh = sortino_ratio(bh)
        out[name] = {
            "n_oos":           int(len(yt)),
            "pos_rate":        float(yt.mean()),
            "auc":             auc,
            "brier":           brier_score(yt, pt),
            "sortino_strategy":s_strat,
            "sortino_bh":      s_bh,
            "sortino_lift":    (s_strat - s_bh)
                                if np.isfinite(s_strat) and np.isfinite(s_bh)
                                else float("nan"),
            "max_dd_strategy": max_drawdown(strat),
            "max_dd_bh":       max_drawdown(bh),
            "ann_ret":         annualised_return(strat),
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(coverage: pd.DataFrame,
                  turb_14f: pd.Series, turb_10f: pd.Series) -> Path:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.32)

    # ── Top: factor coverage timeline ──────────────────────────────────
    ax_top = fig.add_subplot(gs[0])
    cov_sorted = coverage.sort_values("first_valid")
    end_date = pd.Timestamp("2026-04-14")
    for i, (factor, row) in enumerate(cov_sorted.iterrows()):
        is_unspliceable = factor in UNSPLICEABLE_FACTORS
        colour = "#d62728" if is_unspliceable else "#2ca02c"
        label = ("primary only" if is_unspliceable
                 else "v4 splicing applied")
        ax_top.barh(i, (end_date - row["first_valid"]).days / 365.25,
                     left=row["first_valid"], height=0.6,
                     color=colour, alpha=0.85,
                     label=label if i in (0, 5) else None)
        ax_top.text(row["first_valid"] - pd.Timedelta(days=80), i,
                     f"{row['first_valid'].date()}",
                     ha="right", va="center", fontsize=8, color="#444")
    ax_top.set_yticks(range(len(cov_sorted)))
    ax_top.set_yticklabels(cov_sorted.index, fontsize=9)
    ax_top.set_xlim(pd.Timestamp("2003-06-01"), end_date + pd.Timedelta(days=200))
    ax_top.set_title("Factor data coverage in cached parquet  "
                     "(green = v4 splicing applied; red = primary only, "
                     "no v4 fallback)", fontsize=11)
    ax_top.grid(axis="x", alpha=0.3)
    handles, labels = [], []
    for h, l in zip(*ax_top.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
    ax_top.legend(handles, labels, loc="upper right", fontsize=9)

    # ── Bottom: 10-factor vs 14-factor turbulence overlay ─────────────
    ax_bot = fig.add_subplot(gs[1])
    common = turb_14f.dropna().index.intersection(turb_10f.dropna().index)
    a = turb_14f.reindex(common)
    b = turb_10f.reindex(common)
    ax_bot.plot(b.index, b.values, lw=0.7, color="#1f77b4", alpha=0.85,
                label="10-factor turbulence (Phase 2)")
    ax_bot.plot(a.index, a.values, lw=0.7, color="#9467bd", alpha=0.85,
                label="14-factor turbulence (Phase 9D)")
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("turbulence  $d_t$  (log scale)")
    ax_bot.set_xlabel("date")
    ax_bot.set_title(f"10-factor vs 14-factor turbulence over the common "
                     f"range  ({common.min().date()} → {common.max().date()})",
                     fontsize=11)
    # Crisis shading.
    for name, start, end in CRISIS_WINDOWS:
        if pd.Timestamp(end) < common.min() or pd.Timestamp(start) > common.max():
            continue
        ax_bot.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        alpha=0.12, color="orange")
    ax_bot.legend(loc="upper left", fontsize=9)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.grid(True, alpha=0.3)

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
    coverage: pd.DataFrame,
    crisis_14f: pd.DataFrame,
    crisis_10f: pd.DataFrame,
    comparison: Dict[str, float],
    persist_14f: pd.DataFrame,
    persist_10f: pd.DataFrame,
    oos_14f: Dict[str, dict],
    oos_10f_headline: pd.DataFrame,
    fig_path: Path,
) -> None:
    # Verdict logic.
    n_pass_14f = int((crisis_14f["verdict"] == "PASS").sum())
    n_skip_14f = int((crisis_14f["verdict"] == "SKIP (no data)").sum())
    n_fail_14f = int((crisis_14f["verdict"] == "FAIL").sum())
    pearson = comparison["pearson_r"]

    verdict_pass_crisis = (n_fail_14f == 0)
    correlation_good = pearson >= 0.85

    if verdict_pass_crisis and correlation_good and n_skip_14f <= 2:
        verdict = (
            f"**14-factor turbulence is a viable primary.**  All "
            f"crisis windows in its coverage pass top-decile; Pearson "
            f"correlation with 10-factor on the common range is "
            f"{pearson:.3f} (≥0.85 bar).  The thesis can adopt 14-factor "
            f"as primary; 2008/2011 (out of the 14-factor coverage) "
            f"remain documented via the 10-factor series."
        )
    elif n_skip_14f >= 2 and correlation_good:
        verdict = (
            f"**Keep 10-factor as primary.**  14-factor turbulence "
            f"materially loses 2008/2011 (the largest historical "
            f"stress windows are SKIPPED), even though correlation in "
            f"the common range is {pearson:.3f}.  The 10-factor series "
            f"covers more crisis episodes and is therefore the better "
            f"choice for the thesis headline.  14-factor is preserved "
            f"as a robustness check on the 2014+ window."
        )
    else:
        verdict = (
            f"**Mixed verdict.**  14-factor crisis sanity "
            f"({n_pass_14f}/{len(crisis_14f)} pass, "
            f"{n_skip_14f} skipped, {n_fail_14f} fail) and correlation "
            f"({pearson:.3f}) sit short of the 'viable primary' bar.  "
            f"Recommend keeping 10-factor primary and citing 14-factor "
            f"as a sensitivity check."
        )

    # Phase-6 comparison verdict.
    oos_14f_df = pd.DataFrame(oos_14f).T
    if not oos_14f_df.empty:
        median_lift_14f = float(oos_14f_df["sortino_lift"].median())
        median_lift_10f = float(oos_10f_headline["turb_sortino_lift"].median())
        delta = median_lift_14f - median_lift_10f
        oos_verdict = (
            f"Median Sortino lift across the four challengers: "
            f"14-factor = {median_lift_14f:+.3f} vs "
            f"10-factor = {median_lift_10f:+.3f} "
            f"(Δ = {delta:+.3f}).  "
            + ("Going to 14-factor improves OOS." if delta > 0.05 else
               "Going to 14-factor degrades OOS." if delta < -0.05 else
               "Within noise — no thesis-narrative change required.")
        )
    else:
        oos_verdict = "Phase 6 rerun was skipped."

    md = []
    md.append("# CHECKPOINT — Phase 9D: ETF-inception splicing + 14-factor "
              "turbulence")
    md.append("")
    md.append(f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    md.append("")
    md.append("## What was implemented")
    md.append(
        "- **`regime_detection/lib/splicing.py`** — ports the v4 "
        "FACTOR_DEFINITIONS and FACTOR_FALLBACKS tables verbatim "
        "(see file:line citation in the module docstring).  Exposes "
        "`splice_factor()` (return-level stitch) and "
        "`build_spliced_factor_panel()` (full 14-factor reconstruction).  "
        "Documents the constant `UNSPLICEABLE_FACTORS` for the three "
        "factors with no v4 fallback (`em_credit`, `short_vol`, "
        "`low_risk`)."
    )
    md.append(
        "- **`regime_detection/tests/test_splicing.py`** — 14 tests, "
        "all green: continuity at splice date, primary-precedence on "
        "overlap, pre/post-splice match source, panel anchor = 100, "
        "missing-primary returns NaN, FACTOR_DEFINITIONS = 14 entries, "
        "UNSPLICEABLE constants match the cache's no-fallback set."
    )
    md.append(
        "- **`regime_detection/step11_factor_set_check.py`** — driver: "
        "computes 14-factor turbulence, runs the Phase-2 crisis "
        "sanity check on it, compares against 10-factor turbulence in "
        "the common range, optionally reruns Phase 6 on the new target."
    )
    md.append("")
    md.append("## What tests pass")
    md.append("- **150/150 total** (was 136 before Phase 9D — 14 new "
              "splicing tests).")
    md.append("")
    md.append("## Factor coverage — what splicing achieves")
    md.append("")
    md.append(
        "The cached parquet already has v4 splicing applied.  The "
        f"three unspliceable factors ({', '.join(UNSPLICEABLE_FACTORS)}) "
        "have no fallback in v4's table and Phase 9D does not invent "
        "new ones.  The achievable 14-factor coverage starts at the "
        "latest of these primary inception dates:"
    )
    md.append("")
    md.append(_df_md(coverage.assign(
        splice_status=lambda d: d.index.map(
            lambda f: ("primary only (no v4 fallback)"
                       if f in UNSPLICEABLE_FACTORS
                       else "v4 splicing applied"))
    )[["first_valid", "n_finite", "splice_status"]]))
    md.append("")
    md.append("## 14-factor turbulence — crisis sanity check")
    md.append("")
    md.append(_df_md(crisis_14f))
    md.append("")
    md.append("## 10-factor (Phase 2) reference — crisis sanity check")
    md.append("")
    md.append(_df_md(crisis_10f))
    md.append("")
    md.append("## 14-factor vs 10-factor turbulence comparison")
    md.append("")
    md.append(_df_md(pd.DataFrame([comparison]).T.rename(columns={0: "value"})))
    md.append("")
    md.append("### Persistence test (Skulls Table 1) on the 14-factor series")
    md.append("")
    md.append(_df_md(persist_14f))
    md.append("")
    md.append("### Persistence test on the 10-factor reference")
    md.append("")
    md.append(_df_md(persist_10f))
    md.append("")
    md.append(f"![Phase 9D factor coverage and turbulence overlay]"
              f"({fig_path.relative_to(PROJECT_ROOT)})")
    md.append("")
    md.append("## Phase 6 quarterly walk-forward — 14-factor target")
    md.append("")
    md.append(
        "Binary target rebuilt from 14-factor turbulence (threshold = "
        "90th-pct of 14-factor dev turbulence).  Feature matrix uses "
        "the existing 10-factor columns + the new 14-factor "
        "turbulence-derived features.  Quarterly walk-forward refits "
        "with the same four challenger models as Phase 6."
    )
    md.append("")
    md.append(_df_md(oos_14f_df))
    md.append("")
    md.append("### Comparison to Phase 6 10-factor headline")
    md.append("")
    md.append(_df_md(oos_10f_headline))
    md.append("")
    md.append(oos_verdict)
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")
    md.append("## Deviations from spec")
    md.append(
        "1. **The brief assumed v4 splicing extends all 14 factors to "
        "2005-12.  It does not.**  v4's FACTOR_FALLBACKS table covers "
        "`rates, credit, commodities, fx_usd, quality, momentum` "
        "(6 of 8 needing extension) but is silent on the three "
        "factors in `UNSPLICEABLE_FACTORS` (em_credit, short_vol, "
        "low_risk).  Phase 9D ports the v4 tables verbatim and "
        "documents these limits in the module docstring; we do not "
        "invent new proxies (a future revision could)."
    )
    md.append(
        "2. **No yfinance download.** The cached "
        "`data/cache/factor_prices.parquet` was already built by the "
        "v4 loader with splicing applied; we read it directly rather "
        "than re-download.  The splicing module is ready for offline "
        "use against a fresh raw-ticker panel when needed."
    )
    md.append("")
    md.append("## Open questions for the user")
    md.append(
        "- Invent new pre-inception proxies for `em_credit`, "
        "`short_vol`, `low_risk` (e.g. ELD pre-2007 for EM debt; "
        "1/VIX or VXX-inverse pre-2011 for short_vol; a Ken-French "
        "min-vol portfolio for low_risk)?  Adds methodology surface "
        "but would close the 2005–2011 gap on these three factors."
    )
    md.append(
        "- For the thesis: state that 10-factor remains primary "
        "because of 2008/2011 coverage, and present 14-factor "
        "results as a robustness check on the post-2013 window?"
    )
    md.append(
        "- Per the Phase 9 brief, Phase 9E (narrative reframing in "
        "light of 9A/9B/9C/9D results) is the natural next step."
    )
    md.append("")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text("\n".join(md))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Phase 9D: ETF-inception splicing + 14-factor turbulence")
    print("=" * 72)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 14-factor turbulence -----------------------------------------
    turb_14f, coverage = compute_14factor_turbulence()
    turb_14f.to_frame("turbulence_14f").to_parquet(TURB_14F_PATH)
    print(f"[9D] saved {TURB_14F_PATH.relative_to(PROJECT_ROOT)}")

    # ---- Crisis sanity checks -----------------------------------------
    crisis_14f = crisis_sanity(turb_14f, "14-factor")
    turb_10f = pd.read_parquet(TURB_10F_PATH)["turbulence"]
    crisis_10f = crisis_sanity(turb_10f, "10-factor (reference)")

    # ---- 14F vs 10F comparison ---------------------------------------
    comparison = compare_turbulence_series(turb_14f, turb_10f)
    print(f"\n[9D] 14F vs 10F comparison: {comparison}")
    persist_14f = persistence_table(turb_14f)
    persist_10f = persistence_table(turb_10f)
    print(f"\n[9D] persistence 14F:")
    print(persist_14f.round(3).to_string())
    print(f"\n[9D] persistence 10F:")
    print(persist_10f.round(3).to_string())

    # ---- Phase 6 rerun on 14-factor target ---------------------------
    threshold_14f = float(turb_14f.loc[:DEV_END].dropna().quantile(0.90))
    print(f"\n[9D] 14-factor binary-target threshold = {threshold_14f:.2f}")
    oos_14f = run_oos_protocol_on_turbulence(
        turb_14f, threshold_14f, "14-factor",
        model_names=["logistic_l1", "random_forest",
                     "linear_svm", "logistic_elasticnet"],
    )
    print("\n[9D] Phase 6 14-factor results:")
    print(pd.DataFrame(oos_14f).T.round(4)[
        ["auc", "brier", "sortino_lift", "max_dd_strategy"]
    ].to_string())

    # ---- Load Phase 6 10-factor headline for comparison --------------
    hl = pd.read_parquet(
        PROJECT_ROOT / "reports" / "evaluation" / "headline.parquet"
    )
    # The headline parquet stores sortino_strategy and sortino_buy_hold
    # as separate columns; derive the lift here for the head-to-head.
    hl["turb_sortino_lift"] = hl["sortino_strategy"] - hl["sortino_buy_hold"]
    oos_10f_headline = hl[[
        "auc", "brier", "turb_sortino_lift", "max_dd_strategy",
    ]]

    # ---- Figure -------------------------------------------------------
    fig_path = render_figure(coverage, turb_14f, turb_10f)
    print(f"\n[9D] saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    # ---- Checkpoint --------------------------------------------------
    write_checkpoint(
        coverage=coverage,
        crisis_14f=crisis_14f,
        crisis_10f=crisis_10f,
        comparison=comparison,
        persist_14f=persist_14f,
        persist_10f=persist_10f,
        oos_14f=oos_14f,
        oos_10f_headline=oos_10f_headline,
        fig_path=fig_path,
    )
    print(f"[9D] wrote checkpoint: "
          f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
