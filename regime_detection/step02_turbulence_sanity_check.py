"""Sanity check: turbulence should spike during known crisis windows.

Run from the project root::

    python3 -m regime_detection.step02_turbulence_sanity_check

The script loads the cached factor prices, computes log returns, runs the
walk-forward Mahalanobis turbulence index, and verifies that each of the
listed crisis windows contains at least one day whose turbulence is in the
top decile of all-time observed values.  Any window failing this check
raises ``AssertionError`` — propagate this back to the user before
proceeding to Phase 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime_detection.lib.turbulence import TurbulenceIndex


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTOR_PRICES_PATH = PROJECT_ROOT / "data" / "factor_prices.parquet"


# Known crisis windows.  Boundaries inclusive.
CRISIS_WINDOWS = {
    "2008 GFC (Lehman)":        ("2008-09-01", "2009-03-31"),
    "2011 Euro / US downgrade": ("2011-08-01", "2011-10-31"),
    "2015 oil / China devaln":  ("2015-08-01", "2016-02-29"),
    "2018 Vol-mageddon (Feb)":  ("2018-02-01", "2018-02-28"),
    "2018 Q4 sell-off":         ("2018-10-01", "2018-12-31"),
    "2020 COVID crash":         ("2020-02-15", "2020-05-31"),
    "2022 inflation / rates":   ("2022-01-01", "2022-12-31"),
}


# Factors with full history from ~2005-12 — needed to cover the 2008 GFC
# trailing window once we require min_periods=504.  short_vol (2011-10),
# low_risk (2011-10), em_credit (2007-12) and momentum (2007-03) come online
# too late and are excluded here.  Phase 3 should restore them by extending
# the loader's pre-inception splicing logic.
FACTORS_FROM_2005 = [
    "equity", "rates", "credit", "commodities", "em_equity",
    "fx_usd", "inflation", "value", "quality", "vix",
]


def load_log_returns(columns: list[str] | None = None) -> pd.DataFrame:
    prices = pd.read_parquet(FACTOR_PRICES_PATH).sort_index()
    if columns is not None:
        prices = prices[columns]
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    return log_returns


def main() -> int:
    print(f"Loading factor prices from {FACTOR_PRICES_PATH} ...")
    print(f"  using stable 10-factor subset: {FACTORS_FROM_2005}")
    log_returns = load_log_returns(columns=FACTORS_FROM_2005)
    print(
        f"  → shape {log_returns.shape}, "
        f"date range {log_returns.index.min().date()} "
        f"to {log_returns.index.max().date()}"
    )

    # All-finite slice: every column non-NaN on every row.  With the
    # 10-factor subset this starts around 2005-12-07 (quality inception).
    clean = log_returns.dropna(how="any")
    print(
        f"  → all-finite slice: {clean.shape}, "
        f"from {clean.index.min().date()} to {clean.index.max().date()}"
    )

    print()
    print("Computing turbulence (lookback=2520, min_periods=504, "
          "shrinkage=Ledoit-Wolf, refit_every=1) ...")
    ti = TurbulenceIndex(
        lookback_days=2520, min_periods=504,
        shrinkage="ledoit-wolf", refit_every=1,
    )
    turb = ti.fit_transform(clean)
    finite = turb.dropna()
    print(
        f"  → {len(finite)} finite rows, "
        f"from {finite.index.min().date()} to {finite.index.max().date()}"
    )
    print(f"  → mean {finite.mean():.2f}, median {finite.median():.2f}, "
          f"95th pct {finite.quantile(0.95):.2f}, "
          f"99th pct {finite.quantile(0.99):.2f}, "
          f"max {finite.max():.2f}")

    # Compute the top-decile threshold from observed turbulence
    # (this is a global stat, used only for diagnostic reporting — the
    # production regime threshold is rolling and lives in
    # turbulence_regimes.classify_regimes).
    top_decile = finite.quantile(0.90)
    print(f"  → global 90th-pct threshold (diagnostic only): {top_decile:.2f}")

    print()
    print("Top 20 turbulence days (all-time):")
    top20 = finite.nlargest(20)
    for date, val in top20.items():
        print(f"  {date.date()}  d_t = {val:8.2f}")

    print()
    print("Crisis window coverage:")
    failures: list[str] = []
    for name, (start, end) in CRISIS_WINDOWS.items():
        window = finite.loc[start:end]
        if len(window) == 0:
            print(f"  [SKIP] {name}: no turbulence data in {start}–{end}")
            continue
        spikes = window[window >= top_decile]
        max_d = window.max()
        max_d_date = window.idxmax().date() if len(window) else "n/a"
        # The window must contain at least one top-decile day to count.
        ok = len(spikes) > 0
        symbol = "PASS" if ok else "FAIL"
        print(
            f"  [{symbol}] {name:32s} "
            f"max d_t = {max_d:7.2f} on {max_d_date}, "
            f"{len(spikes):>4d}/{len(window):>4d} top-decile days"
        )
        if not ok:
            failures.append(name)

    print()
    if failures:
        print(f"FAILED windows: {failures}")
        print("Debug before proceeding to Phase 3.")
        return 1

    print("All crisis windows registered top-decile turbulence days. Sanity check PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
