"""Decompose turbulence into per-factor contributions and visualise drivers.

Run from the project root::

    python3 -m regime_detection.step14_turbulence_decomposition

Loads the cached factor prices (10-factor stable subset), computes the
walk-forward turbulence decomposition, verifies the per-factor
contributions reconstruct ``d_t`` exactly, writes the contributions to
``outputs/decomposition/turbulence_contributions.parquet``, and renders:

  * ``figures/decomposition/crisis_drivers.png`` — which factors drove each
    named crisis (the headline interpretability figure);
  * ``figures/decomposition/driver_timeline.png`` — monthly per-factor
    contribution share across 2007-2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from regime_detection.lib.turbulence import TurbulenceIndex
from regime_detection.lib.turbulence_decomposition import TurbulenceDecomposer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTOR_PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "decomposition"
FIG_DIR = PROJECT_ROOT / "figures" / "decomposition"

FACTORS_10 = [
    "equity", "rates", "credit", "commodities", "em_equity",
    "fx_usd", "inflation", "value", "quality", "vix",
]

CRISIS_WINDOWS = {
    "2008 GFC":          ("2008-09-01", "2009-03-31"),
    "2011 Euro/US dg":   ("2011-08-01", "2011-10-31"),
    "2015 China/oil":    ("2015-08-01", "2016-02-29"),
    "2018 Volmageddon":  ("2018-02-01", "2018-02-28"),
    "2018 Q4 selloff":   ("2018-10-01", "2018-12-31"),
    "2020 COVID":        ("2020-02-15", "2020-05-31"),
    "2022 inflation":    ("2022-01-01", "2022-12-31"),
}

TURB_PARAMS = dict(
    lookback_days=2520, min_periods=504,
    shrinkage="ledoit-wolf", refit_every=1,
)


def load_clean_returns() -> pd.DataFrame:
    prices = pd.read_parquet(FACTOR_PRICES_PATH).sort_index()[FACTORS_10]
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_ret.dropna(how="any")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 10-factor log returns ...")
    clean = load_clean_returns()
    print(f"  → {clean.shape}, {clean.index.min().date()} → {clean.index.max().date()}")

    print("Computing turbulence and its per-factor decomposition (walk-forward) ...")
    d_t = TurbulenceIndex(**TURB_PARAMS).fit_transform(clean)
    contrib = TurbulenceDecomposer(**TURB_PARAMS).fit_transform(clean)

    # ---- validation: contributions must reconstruct d_t exactly ----
    recon = contrib.sum(axis=1)
    both = pd.concat([d_t.rename("d_t"), recon.rename("recon")], axis=1).dropna()
    max_abs_err = float((both["d_t"] - both["recon"]).abs().max())
    rel_err = max_abs_err / float(both["d_t"].abs().max())
    print(f"  → reconstruction check on {len(both)} rows: "
          f"max|d_t - sum(contrib)| = {max_abs_err:.3e} "
          f"(relative {rel_err:.3e})")
    assert max_abs_err < 1e-6, "decomposition does not reconstruct d_t!"
    print("  ✓ per-factor contributions sum to d_t exactly.")

    finite = contrib.dropna(how="all")
    finite.to_parquet(OUT_DIR / "turbulence_contributions.parquet")
    print(f"  → saved contributions: {OUT_DIR / 'turbulence_contributions.parquet'}")

    # ---- crisis driver table + figure (mean contribution SHARE per crisis) ----
    rows = []
    for name, (s, e) in CRISIS_WINDOWS.items():
        win = finite.loc[s:e]
        if len(win) == 0:
            continue
        mean_c = win.mean(axis=0)                 # mean signed contribution per factor
        share = 100.0 * mean_c / mean_c.sum()     # % of mean d_t over the window
        rows.append(pd.Series(share, name=name))
    share_df = pd.DataFrame(rows)                 # crises × factors, in %

    print("\nTop-3 drivers per crisis (share of mean turbulence):")
    for name in share_df.index:
        top = share_df.loc[name].sort_values(ascending=False).head(3)
        desc = ", ".join(f"{f} {v:.0f}%" for f, v in top.items())
        print(f"  {name:18s} → {desc}")

    share_df.to_csv(OUT_DIR / "crisis_driver_shares.csv")

    # Figure 1 — crisis × factor driver heatmap
    fig, ax = plt.subplots(figsize=(11, 5.2))
    vmax = float(np.nanmax(np.abs(share_df.values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(share_df.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(share_df.columns)))
    ax.set_xticklabels(share_df.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(share_df.index)))
    ax.set_yticklabels(share_df.index)
    for i in range(share_df.shape[0]):
        for j in range(share_df.shape[1]):
            v = share_df.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.6 * vmax else "white")
    ax.set_title("What drives each crisis — per-factor share of mean turbulence (%)")
    fig.colorbar(im, ax=ax, label="contribution share (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crisis_drivers.png", dpi=140)
    plt.close(fig)
    print(f"  → saved figure: {FIG_DIR / 'crisis_drivers.png'}")

    # Figure 2 — full monthly driver timeline (share of monthly mean d_t)
    monthly = finite.resample("ME").mean()
    monthly_share = 100.0 * monthly.div(monthly.sum(axis=1), axis=0)
    monthly_share = monthly_share.dropna(how="all")
    fig, ax = plt.subplots(figsize=(14, 4.6))
    vmax2 = float(np.nanpercentile(np.abs(monthly_share.values), 98))
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0.0, vmax=vmax2)
    im = ax.imshow(monthly_share.T.values, aspect="auto", cmap="RdBu_r", norm=norm2,
                   extent=[0, len(monthly_share), len(FACTORS_10), 0])
    ax.set_yticks(np.arange(len(FACTORS_10)) + 0.5)
    ax.set_yticklabels(monthly_share.columns)
    # x ticks: year boundaries
    years = monthly_share.index.year
    tick_pos = [i for i in range(1, len(years)) if years[i] != years[i - 1]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(monthly_share.index[i].year) for i in tick_pos],
                       rotation=0, fontsize=8)
    ax.set_title("Turbulence drivers over time — monthly per-factor share of d_t (%)")
    fig.colorbar(im, ax=ax, label="share (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "driver_timeline.png", dpi=140)
    plt.close(fig)
    print(f"  → saved figure: {FIG_DIR / 'driver_timeline.png'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
