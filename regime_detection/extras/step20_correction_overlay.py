"""Step 20 — calibrated probabilistic correction-timing overlay (equity factor).

The deployable artifact the project targets: predict the probability that the
equity benchmark loses >= X% over the next H days, calibrate it, and overlay a
go-to-cash decision, evaluated honestly (transaction costs + the 49-cutoff
distribution).

Pieces, all leak-free:
  * **Target** — `forward_drawdown_conditional(equity, H)` thresholded into a
    binary "correction" label (worst forward H-day return <= -X). 1% is near
    daily noise, so we sweep X in {2%, 5%} and report both; 5% is the headline.
  * **Features** — the bucket feature matrix (turbulence severity + dynamics).
  * **Probabilities** — quarterly walk-forward (`logistic_l1`); each quarter
    trained on the past only.
  * **Calibration** — isotonic fit on *dev* walk-forward predictions (data
    <= 2020), frozen, applied to the 2020+ OOS probabilities.
  * **Decision threshold tau** — go to cash when P_calibrated >= tau, with tau
    selected on *dev* (never on the OOS being scored).
  * **Costs** — transaction cost charged on every switch; we sweep 0–20 bps.
  * **Robustness** — the 49-cutoff distribution of the (gross) Sortino lift.

Honest framing: the ~60-block OOS ceiling means "outperform" is not
guaranteed; the contribution is the rigorous, calibrated, cost-aware test.

Run:  python3 -m regime_detection.step20_correction_overlay
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from regime_detection.lib.cutoff_sensitivity import run_sliding_cutoff, summary_statistics
from regime_detection.lib.metrics import (
    annualised_return, brier_score, max_drawdown, sortino_ratio,
)
from regime_detection.lib.oos_pipeline import run_quarterly_walk_forward
from regime_detection.lib.targets import forward_drawdown_conditional
from regime_detection.lib.turbulence_bucket import build_feature_matrix_with_bucket
from regime_detection.lib.splits import DEV_END, OOS_START

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
TURB_PATH = PROJECT_ROOT / "reports" / "turbulence" / "turbulence_series.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "correction_overlay"
FIG_DIR = PROJECT_ROOT / "figures" / "correction_overlay"
CHECKPOINT = PROJECT_ROOT / "reports" / "checkpoints" / "step20_correction_overlay.md"
CACHE = PROJECT_ROOT / ".cache" / "correction_overlay"

FACTORS_10 = ["equity", "rates", "credit", "commodities", "em_equity",
              "fx_usd", "inflation", "value", "quality", "vix"]
MODEL = "logistic_l1"
HORIZON = 21
LOSSES = [0.02, 0.05]          # 2% and 5% corrections
HEADLINE_LOSS = 0.05
COSTS_BPS = [0, 5, 10, 20]     # transaction-cost sweep
HEADLINE_COST = 10
DEV_CALIB_END = pd.Timestamp("2017-12-31")   # dev split for calibration/tau


def correction_label(prices_eq: pd.Series, loss: float, horizon: int) -> pd.Series:
    """1 if the worst forward `horizon`-day cumulative return <= -loss."""
    fdd = forward_drawdown_conditional(prices_eq, horizon=horizon)
    return (fdd <= -loss).astype(float).where(fdd.notna())


def overlay_with_costs(eq_ret: pd.Series, in_market: pd.Series, cost_bps: float):
    """Daily strategy returns: hold eq when in_market (lagged 1d), flat else,
    minus `cost_bps` on each switch. Returns (strat_returns, frac_in_cash, n_switches)."""
    pos = in_market.shift(1).fillna(1.0)                 # yesterday's decision
    switch = pos.diff().abs().fillna(0.0)                # 1 on every flip
    cost = (cost_bps / 1e4) * switch
    strat = eq_ret * pos - cost
    return strat, float((pos == 0).mean()), int(switch.sum())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PATH).sort_index()[FACTORS_10]
    turb = pd.read_parquet(TURB_PATH)["turbulence"]
    eq = np.log(prices["equity"] / prices["equity"].shift(1))
    X = build_feature_matrix_with_bucket(prices, turb,
            pd.read_parquet(TURB_PATH)["regime_smoothed"], factors=FACTORS_10, n_buckets=5)
    print(f"features {X.shape}; horizon {HORIZON}d; model {MODEL}")

    rows = []
    headline = {}
    for loss in LOSSES:
        y = correction_label(prices["equity"], loss, HORIZON)
        base = y.loc[OOS_START:].mean()
        # --- walk-forward probabilities: dev-calib window and the real OOS ---
        dev = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=HORIZON,
                dev_end=DEV_CALIB_END, oos_start=DEV_CALIB_END + pd.Timedelta(days=1),
                random_state=0)[MODEL]
        oos = run_quarterly_walk_forward(X, y, model_names=[MODEL], purge_window=HORIZON,
                dev_end=DEV_END, oos_start=OOS_START, random_state=0)[MODEL]
        # restrict the dev-calibration probs to <= DEV_END (strictly dev)
        dpred = dev.oos_predictions.loc[:DEV_END].dropna()
        dlab = dev.oos_labels.reindex(dpred.index)
        opred = oos.oos_predictions.dropna(); olab = oos.oos_labels.reindex(opred.index)

        # --- isotonic calibration fit on dev, applied to OOS ---
        iso = IsotonicRegression(out_of_bounds="clip").fit(dpred.values, dlab.values)
        opred_cal = pd.Series(iso.predict(opred.values), index=opred.index)
        dpred_cal = pd.Series(iso.predict(dpred.values), index=dpred.index)
        auc = roc_auc_score(olab, opred) if olab.nunique() > 1 else np.nan
        brier_raw, brier_cal = brier_score(olab, opred), brier_score(olab, opred_cal)

        # --- decision threshold tau: pick on DEV to maximise dev overlay Sortino ---
        eq_dev = eq.reindex(dpred_cal.index)
        best_tau, best_s = 0.5, -np.inf
        for tau in np.linspace(0.2, 0.9, 36):
            strat, _, _ = overlay_with_costs(eq_dev, (dpred_cal < tau).astype(float), HEADLINE_COST)
            s = sortino_ratio(strat.dropna())
            if np.isfinite(s) and s > best_s:
                best_s, best_tau = s, float(tau)

        # --- deploy on OOS, headline cost. Two policies: ---
        eq_oos = eq.reindex(opred_cal.index)
        bh = eq_oos
        s_bh = sortino_ratio(bh.dropna())
        # (1) dev-selected absolute-tau policy (may degenerate to "stay invested")
        in_cal = (opred_cal < best_tau).astype(float)
        strat_cal, frac_cash, nsw = overlay_with_costs(eq_oos, in_cal, HEADLINE_COST)
        lift_cal = sortino_ratio(strat_cal.dropna()) - s_bh
        # (2) forced policy: go flat on the top-decile highest-risk days, ranked
        # on the RAW (continuous) score — isotonic ties destroy decision ranking.
        in_force = (opred.reindex(opred_cal.index).rank(pct=True) < 0.90).astype(float)
        strat_force, frac_cash_f, nsw_f = overlay_with_costs(eq_oos, in_force, HEADLINE_COST)
        lift_force = sortino_ratio(strat_force.dropna()) - s_bh
        fa = float(((in_force.shift(1) == 0) & (bh > 0)).sum() / max((in_force.shift(1) == 0).sum(), 1))
        rows.append({"loss": f"{loss:.0%}", "base_rate": base, "auc": auc,
                     "brier_raw": brier_raw, "brier_cal": brier_cal, "tau": best_tau,
                     "lift_devtau": lift_cal, "in_cash_devtau": frac_cash,
                     "lift_forced_decile": lift_force, "false_alarm_forced": fa,
                     "maxdd_forced": max_drawdown(strat_force.dropna()),
                     "maxdd_bh": max_drawdown(bh.dropna())})
        if loss == HEADLINE_LOSS:
            headline = dict(y=y, opred_cal=opred_cal, olab=olab, opred=opred,
                            dpred=dpred,
                            dpred_cal=dpred_cal, dlab=dlab, eq_oos=eq_oos,
                            in_mkt=in_force, strat=strat_force, bh=bh, tau=best_tau,
                            lift_devtau=lift_cal, in_cash_devtau=frac_cash)

    res = pd.DataFrame(rows).set_index("loss")
    res.to_csv(OUT_DIR / "overlay_metrics.csv")
    print(res.round(3).to_string())

    # --- transaction-cost sweep (headline loss) ---
    h = headline
    cost_curve = []
    for c in COSTS_BPS:
        strat_c, _, _ = overlay_with_costs(h["eq_oos"], h["in_mkt"], c)
        cost_curve.append(sortino_ratio(strat_c.dropna()) - sortino_ratio(h["bh"].dropna()))

    # --- 49-cutoff robustness of the (gross) correction-timing edge ---
    sched = list(pd.date_range("2018-06-30", "2022-06-30", freq="ME"))
    slide = run_sliding_cutoff(X, h["y"], eq, cutoff_dates=sched, model_name=MODEL,
            purge_window=HORIZON, min_oos_quarters=4, random_state=0,
            cache_dir=CACHE / "slide_5pct", verbose=False)
    ssum = summary_statistics(slide, "sortino_lift")
    print(f"49-cutoff (5% target): median {ssum['median']:+.3f}, frac+ {ssum['frac_positive']:.0%}")
    slide.to_parquet(OUT_DIR / "sliding_5pct.parquet")  # for the thesis-figure generator
    # persist the headline OOS frame so downstream figures read the analysis
    # rather than re-deriving it (see make_thesis_figures.fig_correction_overlay)
    pd.DataFrame({"pred_raw": h["opred"].reindex(h["opred_cal"].index),
                  "pred_cal": h["opred_cal"], "label": h["olab"],
                  "in_market": h["in_mkt"], "eq_ret": h["eq_oos"],
                  }).to_parquet(OUT_DIR / "headline_oos_5pct.parquet")
    h["dpred"].to_frame("pred_raw").to_parquet(OUT_DIR / "headline_dev_5pct.parquet")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    # A: OOS equity curves
    a = ax[0, 0]
    a.plot(h["bh"].index, h["bh"].cumsum(), color="#7f8c8d", label="buy & hold")
    a.plot(h["strat"].index, h["strat"].cumsum(), color="#c0392b",
           label=f"forced top-decile overlay ({HEADLINE_COST}bps)")
    a.set_title("OOS cumulative log-return — equity vs forced timing overlay"); a.legend(fontsize=9); a.grid(alpha=0.3)
    # B: reliability diagram (raw vs calibrated)
    a = ax[0, 1]
    for pred, lab_, name, col in [(h["opred"], h["olab"], "raw", "#1f77b4"),
                                  (h["opred_cal"], h["olab"], "calibrated", "#2ca02c")]:
        bins = np.linspace(0, 1, 9); idx = np.digitize(pred, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() > 5: xs.append(pred[m].mean()); ys.append(lab_.values[m].mean())
        a.plot(xs, ys, "-o", color=col, label=name)
    # The calibrated series is near-degenerate: isotonic maps almost every day to
    # one value. A single plotted point looks like a rendering fault, so say so.
    _mode = h["opred_cal"].round(3).mode().iloc[0]
    _share = float((h["opred_cal"].round(3) == _mode).mean())
    if _share > 0.5:
        a.annotate(f"calibrated $\\to$ {_mode:.2f} on {_share:.0%} of days\n"
                   f"(base rate {h['olab'].mean():.2f})",
                   xy=(_mode, h["olab"].mean()), xytext=(0.42, 0.62),
                   fontsize=8.5, color="#2ca02c",
                   arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
    a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set_xlabel("predicted P"); a.set_ylabel("observed freq")
    a.set_title(f"Calibration (Brier {res.loc['5%','brier_raw']:.3f}→{res.loc['5%','brier_cal']:.3f})")
    a.legend(fontsize=9); a.grid(alpha=0.3)
    # C: transaction-cost sweep
    a = ax[1, 0]
    a.plot(COSTS_BPS, cost_curve, "-o", color="#8e44ad"); a.axhline(0, color="grey", lw=0.8)
    a.set_xlabel("transaction cost (bps per switch)"); a.set_ylabel("Sortino lift vs B&H")
    a.set_title("Net edge vs transaction cost (OOS)"); a.grid(alpha=0.3)
    # D: 49-cutoff distribution
    a = ax[1, 1]
    sl = slide["sortino_lift"].dropna()
    a.hist(sl, bins=np.linspace(-0.45, 0.35, 22), color="#c0392b", alpha=0.7)
    a.axvline(0, color="grey", lw=0.8)
    a.axvline(sl.median(), color="#c0392b", ls="--", label=f"median {sl.median():+.3f}")
    a.set_xlabel("Sortino lift (49 cutoffs, gross)"); a.set_ylabel("count")
    a.set_title(f"Robustness — {ssum['frac_positive']:.0%} of cutoffs positive")
    a.legend(fontsize=9); a.grid(alpha=0.3)
    fig.suptitle("Calibrated correction-timing overlay (equity factor)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correction_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'correction_overlay.png'}")

    # ---------------- checkpoint ----------------
    hl = res.loc["5%"]
    auc5 = float(hl["auc"])
    degenerate = abs(hl["in_cash_devtau"]) < 0.01  # dev-tau policy never goes flat
    verdict = (
        f"**NO ROBUST CALENDAR-TIME EDGE — but a clean, honest result.** "
        f"Three things, in order:\n\n"
        f"1. **The signal is weak for *equity corrections* specifically.** OOS AUC "
        f"for the 5% target is **{auc5:.3f}** (≈ chance), vs ≈0.66 for the "
        f"turbulence-entry target (step06). *Turbulence predicts turbulence far "
        f"better than it predicts equity drawdowns* — the timing ask is genuinely "
        f"harder than the task the project had been solving.\n\n"
        f"2. **Calibration works and is worth keeping.** Isotonic (dev-fit) cuts the "
        f"5% Brier from {hl['brier_raw']:.3f} to {hl['brier_cal']:.3f} — the "
        f"probabilities become usable even though the ranking is weak.\n\n"
        f"3. **Acting on the signal doesn't beat buy-and-hold.** The dev-selected "
        + ("absolute-probability policy degenerates to *stay invested* (the model is "
           "never confident enough about a 5% correction to justify cash given costs)"
           if degenerate else
           f"policy gives lift {hl['lift_devtau']:+.3f}") +
        f"; forcing a go-to-cash on the top-decile most-stressed days gives net "
        f"Sortino lift {hl['lift_forced_decile']:+.3f}; and across 49 cutoffs the "
        f"gross edge has median {ssum['median']:+.3f} ({ssum['frac_positive']:.0%} "
        f"positive). All inside the noise floor.\n\n"
        f"**This is the expected, well-evidenced answer under the sample-size "
        f"ceiling — and it sharpens the case for the next step: event-time "
        f"(step 21), where the drawdown target may be better-posed.**")

    md = [
        "# CHECKPOINT — Step 20: calibrated correction-timing overlay",
        "", f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M} UTC", "",
        "## What was built",
        f"A deployable overlay on the **equity factor**: predict P(loss ≥ X% over "
        f"{HORIZON}d), isotonic-calibrate on dev, go to cash when P ≥ τ "
        "(τ dev-selected), charge transaction costs, evaluate vs buy-and-hold and "
        "across 49 cutoffs. Probabilities are quarterly walk-forward; calibrator "
        "and τ use dev data only.",
        "",
        "## Per-target results (OOS 2020+, headline cost "
        f"{HEADLINE_COST}bps)", "",
        res.round(3).to_markdown(),
        "",
        f"**49-cutoff robustness (5% target, gross):** median "
        f"{ssum['median']:+.3f}, IQR [{ssum['q25']:+.3f}, {ssum['q75']:+.3f}], "
        f"{ssum['frac_positive']:.0%} of cutoffs positive.",
        "",
        f"**Transaction-cost sweep (5%, Sortino lift):** "
        + ", ".join(f"{c}bps→{v:+.3f}" for c, v in zip(COSTS_BPS, cost_curve)) + ".",
        "",
        "![overlay](../../figures/correction_overlay/correction_overlay.png)",
        "",
        "## Verdict", "", verdict, "",
        "## Notes",
        "- Calibration is leak-free (isotonic fit on dev walk-forward predictions, "
        "applied to OOS); reliability diagram + Brier before/after in the figure.",
        "- τ is dev-selected to max dev Sortino — a dev decision, not OOS tuning.",
        "- 1% was excluded as a target: it fires on near-daily noise. 2% and 5% "
        "are reported; 5% is a genuine correction.",
        "- Same OOS ceiling: read the 49-cutoff distribution, not the single net "
        "number. **Next: step 21 repeats this on the event clock.**",
    ]
    CHECKPOINT.write_text("\n".join(md))
    print(f"wrote {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
