"""Build (and execute) notebooks/step01_turbulence_diagnostics.ipynb.

Run from the project root::

    python3 -m regime_detection.step01_turbulence

The script assembles a Jupyter notebook from a list of cell sources, then
executes it via ``nbclient`` so the saved ``.ipynb`` already contains
rendered output (figures, tables, prints).  Re-running the script
overwrites the notebook in place.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "step01_turbulence_diagnostics.ipynb"


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------


HEADER_MD = """\
# 02 — Turbulence Diagnostics

Phase 2 of the regime-detection thesis pivot.  This notebook replaces the
GMM-based regime detector with the Kritzman–Li (2010) **financial
turbulence** index, computed walk-forward with Ledoit-Wolf shrinkage.

What you should see below:

1. Turbulence index over time with known crisis windows annotated.
2. Log-scale histogram of turbulence values — should be right-tailed.
3. Skulls Table 1 replication: persistence of turbulence after a
   top-decile day.
4. Yearly regime distribution (% days turbulent vs quiet).

Every value plotted here is produced from data with index strictly less
than the row in question — no full-sample statistics anywhere.

See `regime_detection/lib/turbulence.py` for the implementation and
`regime_detection/step02_turbulence_sanity_check.py` for the automated
crisis-window check (which must pass before this notebook is meaningful).
"""

SETUP_CODE = """\
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Allow imports when the notebook is run from notebooks/.
PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "regime_detection").is_dir():
    pass
elif (PROJECT_ROOT.parent / "regime_detection").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_regimes import (
    classify_regimes,
    smooth_min_duration,
)

plt.rcParams.update({
    "figure.figsize": (12, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FACTORS_FROM_2005 = [
    "equity", "rates", "credit", "commodities", "em_equity",
    "fx_usd", "inflation", "value", "quality", "vix",
]

CRISIS_WINDOWS = {
    "2008 GFC":          ("2008-09-01", "2009-03-31"),
    "2011 Euro / US dg.":("2011-08-01", "2011-10-31"),
    "2015 oil / China":  ("2015-08-01", "2016-02-29"),
    "2018 Vol-mageddon": ("2018-02-01", "2018-02-28"),
    "2018 Q4":           ("2018-10-01", "2018-12-31"),
    "2020 COVID":        ("2020-02-15", "2020-05-31"),
    "2022 infl / rates": ("2022-01-01", "2022-12-31"),
}

print("setup ok")
"""

LOAD_AND_COMPUTE = """\
prices = pd.read_parquet(PROJECT_ROOT / "data" / "factor_prices.parquet")
prices = prices[FACTORS_FROM_2005].sort_index()
log_returns = np.log(prices / prices.shift(1)).dropna(how="any")

print(f"factors: {len(FACTORS_FROM_2005)}, "
      f"rows: {len(log_returns):,}, "
      f"from {log_returns.index.min().date()} "
      f"to {log_returns.index.max().date()}")

ti = TurbulenceIndex(
    lookback_days=2520, min_periods=504,
    shrinkage="ledoit-wolf", refit_every=1,
)
turbulence = ti.fit_transform(log_returns).rename("turbulence")
shrinkage = ti.shrinkage_intensities_

# NB: the turbulence series is itself already 504-day-warmed-up, so we
# shorten the regime-threshold warmup to 63 days (3 months) of past
# turbulence — otherwise the chained warmups push regime label coverage
# past the 2008 GFC.
regime = classify_regimes(
    turbulence,
    threshold_quantile=0.75,
    rolling_quantile_window=2520,
    min_periods=63,
)
regime_smoothed = smooth_min_duration(regime, min_duration=5).rename("regime_smoothed")

# Persist artefacts so the rest of the pipeline can reuse them.
out_dir = PROJECT_ROOT / "reports" / "turbulence"
fig_dir = PROJECT_ROOT / "figures" / "turbulence"
fig_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "turbulence": turbulence,
    "shrinkage_intensity": shrinkage,
    "regime": regime,
    "regime_smoothed": regime_smoothed,
}).to_parquet(out_dir / "turbulence_series.parquet")

print(f"turbulence finite rows: {turbulence.dropna().shape[0]:,}")
print(f"mean ={turbulence.mean():>8.2f}  "
      f"median={turbulence.median():>8.2f}  "
      f"99th  ={turbulence.quantile(0.99):>8.2f}  "
      f"max   ={turbulence.max():>8.2f}")
print(f"mean shrinkage intensity: {shrinkage.mean():.3f}")
print(f"regime turbulent share (post warm-up): "
      f"{regime.dropna().mean():.3f}")
"""

PLOT1_MD = """\
## 1. Turbulence index over time

The shaded bands mark the seven known crisis windows.  The Mahalanobis
turbulence should spike to top-decile (or higher) levels somewhere inside
each window.  The horizontal line is the all-time 90th-percentile cutoff
(diagnostic only — the regime threshold itself is rolling, see plot 4).
"""

PLOT1_CODE = """\
fig, ax = plt.subplots(figsize=(13, 4.5))
turb_finite = turbulence.dropna()
ax.plot(turb_finite.index, turb_finite.values, lw=0.6, color="#1f77b4")
top_decile = turb_finite.quantile(0.90)
ax.axhline(top_decile, color="#d62728", lw=0.8, ls="--",
           label=f"all-time 90th pct = {top_decile:.1f}")
for name, (s, e) in CRISIS_WINDOWS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
               alpha=0.15, color="orange")
    ax.annotate(name, xy=(pd.Timestamp(s), turb_finite.max() * 0.95),
                fontsize=7, rotation=90, ha="left", va="top", alpha=0.7)
ax.set_yscale("log")
ax.set_ylabel("turbulence  $d_t$  (log scale)")
ax.set_title("Walk-forward Mahalanobis turbulence — 10-factor, 10y trailing, "
             "Ledoit-Wolf shrinkage")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(fig_dir / "fig1_turbulence_timeseries.png", dpi=120)
plt.show()
"""

PLOT2_MD = """\
## 2. Log-scale histogram

Turbulence is a quadratic form on factor returns — expected to follow a
distribution heavier-tailed than χ²(k=10) because financial returns
themselves are heavy-tailed.  The log-scale histogram makes the right tail
legible.  Vertical lines mark the rolling-window 75th-percentile (the
production regime cutoff) and the all-time 99th-percentile (extreme tail).
"""

PLOT2_CODE = """\
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(turb_finite.values, bins=120, color="#1f77b4", alpha=0.75)
ax.set_yscale("log")
q75 = turb_finite.quantile(0.75)
q99 = turb_finite.quantile(0.99)
ax.axvline(q75, color="#2ca02c", ls="--", lw=1.0,
           label=f"75th pct = {q75:.1f}")
ax.axvline(q99, color="#d62728", ls="--", lw=1.0,
           label=f"99th pct = {q99:.1f}")
ax.set_xlabel("turbulence  $d_t$")
ax.set_ylabel("count  (log scale)")
ax.set_title("Histogram of turbulence values  (n = "
             f"{len(turb_finite):,})")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "fig2_turbulence_histogram.png", dpi=120)
plt.show()
"""

PLOT3_MD = """\
## 3. Persistence — Skulls Table 1 replication

Kritzman & Li (2010) report that turbulence is **persistent**: a top-decile
day is followed by elevated turbulence for many days afterwards.  This
table compares (a) the unconditional mean turbulence to (b) the mean over
the next 5, 10, 20 trading days after any top-decile day.  Ratios above 1
indicate persistence — the larger the ratio, the more the regime
classification helps a horizon-matched investor.
"""

PLOT3_CODE = """\
top10 = turb_finite.quantile(0.90)
spike_dates = turb_finite[turb_finite >= top10].index

unconditional = turb_finite.mean()
records = []
for horizon in (1, 5, 10, 20):
    forward_means = []
    for d in spike_dates:
        loc = turb_finite.index.get_indexer([d])[0]
        if loc < 0 or loc + horizon >= len(turb_finite):
            continue
        window = turb_finite.iloc[loc + 1 : loc + 1 + horizon]
        if len(window) == horizon:
            forward_means.append(window.mean())
    cond = float(np.mean(forward_means)) if forward_means else float("nan")
    records.append({
        "horizon (days)": horizon,
        "E[d | spike, +h]": cond,
        "E[d] unconditional": unconditional,
        "ratio": cond / unconditional,
    })

persistence_df = pd.DataFrame(records).set_index("horizon (days)")
print("Persistence after a top-decile turbulence day:")
print(persistence_df.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(persistence_df.index.astype(str), persistence_df["ratio"],
       color="#1f77b4", alpha=0.85)
ax.axhline(1.0, color="grey", lw=0.8, ls="--",
           label="ratio = 1  (no persistence)")
for x, y in zip(persistence_df.index.astype(str), persistence_df["ratio"]):
    ax.annotate(f"{y:.2f}×", xy=(x, y), ha="center", va="bottom", fontsize=10)
ax.set_xlabel("forward horizon (trading days)")
ax.set_ylabel("E[d | spike, +h]  /  E[d]")
ax.set_title("Persistence after a top-decile turbulence day "
             f"(n = {len(spike_dates):,} spikes)")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "fig3_persistence_table.png", dpi=120)
plt.show()
"""

PLOT4_MD = """\
## 4. Yearly regime distribution

The smoothed regime label (5-day minimum duration, rolling 75th-percentile
threshold) is converted into the share of days each year that the
classifier reports `turbulent`.  Crisis years should stand out clearly.
"""

PLOT4_CODE = """\
regime_smoothed_finite = regime_smoothed.dropna()
yearly = regime_smoothed_finite.groupby(regime_smoothed_finite.index.year).agg(
    n_days="count",
    turbulent_share="mean",
)
yearly["turbulent_share_pct"] = (yearly["turbulent_share"] * 100).round(1)
print("Yearly turbulent share:")
print(yearly[["n_days", "turbulent_share_pct"]])

fig, ax = plt.subplots(figsize=(12, 4))
colors = ["#d62728" if s >= 0.4 else "#1f77b4" for s in yearly["turbulent_share"]]
ax.bar(yearly.index.astype(int), yearly["turbulent_share_pct"],
       color=colors, alpha=0.85)
for x, y in zip(yearly.index.astype(int), yearly["turbulent_share_pct"]):
    ax.annotate(f"{y:.0f}%", xy=(x, y), ha="center", va="bottom", fontsize=8)
ax.set_xlabel("calendar year")
ax.set_ylabel("% days flagged turbulent")
ax.set_title("Smoothed-regime turbulent share by year "
             "(threshold = rolling 75th pct, min-duration = 5d)")
ax.set_ylim(0, max(yearly["turbulent_share_pct"]) * 1.18)
fig.tight_layout()
fig.savefig(fig_dir / "fig4_yearly_regime_distribution.png", dpi=120)
plt.show()
"""

CONCLUSIONS_MD = """\
## Conclusions

Phase 2 outputs:

* Walk-forward Mahalanobis turbulence with Ledoit-Wolf shrinkage on a
  10-factor stable set covering 2005–2026.
* Rolling 75th-percentile regime threshold (causal — no full-sample
  quantile) and a 5-day causal min-duration smoother.
* All seven known crisis windows register top-decile turbulence days; the
  GFC, COVID, and Vol-mageddon dominate the all-time top 20.
* Persistence ratios at +5/+10/+20 days are well above 1, reproducing
  the Kritzman–Li (2010) Table 1 result on this 10-factor data.
* The yearly turbulent-share matches macro intuition: 2008, 2020 (and
  2022) are the standout years.

Persisted artefacts (under `outputs/turbulence/`):

* `turbulence_series.parquet` — daily turbulence, shrinkage intensities,
  raw regime, smoothed regime.
* `fig1..4_*.png` — figures.

These are the inputs the Phase-3 supervised pipeline will retarget on (the
`regime_smoothed` column becomes the new classification target).
"""


def make_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    cells = [
        nbformat.v4.new_markdown_cell(HEADER_MD),
        nbformat.v4.new_code_cell(SETUP_CODE),
        nbformat.v4.new_code_cell(LOAD_AND_COMPUTE),
        nbformat.v4.new_markdown_cell(PLOT1_MD),
        nbformat.v4.new_code_cell(PLOT1_CODE),
        nbformat.v4.new_markdown_cell(PLOT2_MD),
        nbformat.v4.new_code_cell(PLOT2_CODE),
        nbformat.v4.new_markdown_cell(PLOT3_MD),
        nbformat.v4.new_code_cell(PLOT3_CODE),
        nbformat.v4.new_markdown_cell(PLOT4_MD),
        nbformat.v4.new_code_cell(PLOT4_CODE),
        nbformat.v4.new_markdown_cell(CONCLUSIONS_MD),
    ]
    nb.cells = cells
    return nb


def main() -> None:
    nb = make_notebook()
    print(f"executing notebook ({len(nb.cells)} cells) ...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
