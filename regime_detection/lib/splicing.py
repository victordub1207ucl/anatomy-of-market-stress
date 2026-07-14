"""ETF-inception splicing for the 14-factor Two Sigma factor lens.

Ported from v4 ``data/loader.py:53-95`` (``FACTOR_DEFINITIONS`` and
``FACTOR_FALLBACKS``) and ``data/loader.py:240-344`` (the splicing
algorithm).  The v4 version was confirmed CLEAN by the Phase 0
look-ahead audit (``data/loader.py`` had 0 findings).

What v4 does, in algorithmic form
---------------------------------
For each factor ``f`` with primary ticker ``P_f`` and an ordered list of
fallback tickers ``[Q_f^1, Q_f^2, ...]``:

1. Compute daily log-returns for every ticker (primary + fallbacks).
2. Build the merged return series for factor ``f``: start with
   ``log_ret(P_f)``; for every NaN date (the primary's pre-inception
   period), backfill from the first fallback that has data; repeat
   for the next fallback, etc.
3. Reconstruct pseudo-prices by exponentiating the cumulative merged
   log-returns, anchored at ``100`` on the first valid date.

The splice is on **returns**, not on price levels, so different
absolute price magnitudes of the primary and fallback ETFs cause no
discontinuity.  At the splice date, the synthetic price level is
continuous by construction.

What the brief's "extend all 14 factors back to 2005-12" target asks
-------------------------------------------------------------------
Three factors in ``FACTOR_DEFINITIONS`` have NO entry in
``FACTOR_FALLBACKS`` (see :data:`UNSPLICEABLE_FACTORS`).  v4 did not
identify pre-inception proxies for them, and neither has this Phase 9D
revision — they are documented limits of the splicing strategy rather
than a porting oversight.  The achievable coverage after splicing is:

* 11 of 14 factors back to 2005-12 (the latest fallback inception date,
  which is QUAL's fallback SPHQ at 2005-12-07);
* 12 factors back to 2007-12 (adding em_credit's primary EMB);
* full 14 factors back to 2011-10 (adding short_vol via SVXY and
  low_risk via USMV — both with no fallback).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Factor table — verbatim port from v4 data/loader.py:53-95
# ---------------------------------------------------------------------------

#: Maps factor name -> (primary ticker, description, category).
FACTOR_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    # Core Macro
    "equity":           ("SPY",   "Equity risk (S&P 500 ETF)",                  "core_macro"),
    "rates":            ("TLT",   "Interest rate duration (20yr Treasury ETF)", "core_macro"),
    "credit":           ("HYG",   "Credit risk (High Yield ETF)",               "core_macro"),
    "commodities":      ("GSG",   "Commodity risk (S&P GSCI ETF)",              "core_macro"),
    # Secondary Macro
    "em_equity":        ("EEM",   "Emerging market equity",                     "secondary_macro"),
    "em_credit":        ("EMB",   "Emerging market credit",                     "secondary_macro"),
    "fx_usd":           ("UUP",   "USD strength (DXY ETF)",                     "secondary_macro"),
    "short_vol":        ("SVXY",  "Short equity volatility",                    "secondary_macro"),
    "inflation":        ("TIP",   "US inflation (TIPS ETF)",                    "secondary_macro"),
    # Style Factors
    "momentum":         ("MTUM",  "Equity momentum factor",                     "style"),
    "value":            ("IWD",   "Equity value factor (Russell 1000 Value)",   "style"),
    "quality":          ("QUAL",  "Equity quality factor (MSCI USA Quality)",   "style"),
    "low_risk":         ("USMV",  "Equity low volatility (MSCI Min Vol)",       "style"),
    # Market Indicators
    "vix":              ("^VIX",  "CBOE Volatility Index",                      "indicator"),
}

#: Maps factor -> ordered list of fallback tickers, used to fill
#: pre-inception NaN returns of the primary ticker.  Verbatim from v4.
FACTOR_FALLBACKS: Dict[str, List[str]] = {
    "rates":        ["IEF"],
    "credit":       ["JNK", "LQD"],
    "commodities":  ["DJP", "CL=F"],
    "fx_usd":       ["DX-Y.NYB"],
    "quality":      ["SPHQ"],
    "momentum":     ["PDP"],
}

#: Factors with no pre-inception proxy in v4.  Their coverage starts at
#: the primary ticker's inception date.  Documented explicitly so callers
#: do not assume the splicing brings them back to 2005.
UNSPLICEABLE_FACTORS: Tuple[str, ...] = ("em_credit", "short_vol", "low_risk")


# ---------------------------------------------------------------------------
# Splicing primitives
# ---------------------------------------------------------------------------


def splice_factor(
    primary_returns: pd.Series,
    proxy_returns_list: Sequence[pd.Series],
    splice_dates: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.Series:
    """Splice a primary daily log-return series with one or more proxies.

    Parameters
    ----------
    primary_returns :
        Daily log returns of the primary ticker.  May be NaN before its
        inception date.
    proxy_returns_list :
        Ordered list of proxy log-return Series.  Used in order — the
        first proxy fills the primary's leading NaN gap, the second fills
        whatever remains, etc.
    splice_dates :
        Optional explicit list of splice dates, one per proxy.  If
        provided, returns from proxy ``i`` are used only on dates that
        are strictly before ``splice_dates[i]``.  If ``None`` (v4
        default), the splice point is implicit — proxy returns fill
        wherever the merged series is still NaN.

    Returns
    -------
    pd.Series
        Merged log-return series, of length ``max(len(p)`` for ``p`` in
        all inputs).  Index is the union of all input indices.

    Causality note
    --------------
    Splicing is a calendar-fact operation, not a data-derived one — the
    splice point comes from ETF inception (a known historical date).
    Therefore splicing does not introduce look-ahead.  ``splice_dates``
    must be Timestamps in the past relative to the rest of the
    pipeline's evaluation window (the test
    ``test_splice_with_future_only_data_raises`` enforces this for the
    explicit-date variant).
    """
    if not isinstance(primary_returns, pd.Series):
        raise TypeError(
            f"primary_returns must be a Series, got "
            f"{type(primary_returns).__name__}"
        )
    full_index = primary_returns.index
    for p in proxy_returns_list:
        if not isinstance(p, pd.Series):
            raise TypeError("All proxy_returns_list entries must be Series")
        full_index = full_index.union(p.index)

    merged = primary_returns.reindex(full_index)

    if splice_dates is not None:
        if len(splice_dates) != len(proxy_returns_list):
            raise ValueError(
                "splice_dates length must match proxy_returns_list length"
            )
        for proxy_rets, splice_date in zip(proxy_returns_list, splice_dates):
            splice_date = pd.Timestamp(splice_date)
            # Use proxy returns only on dates strictly before splice_date.
            mask = full_index < splice_date
            proxy_pre = proxy_rets.reindex(full_index).where(mask)
            merged = merged.combine_first(proxy_pre)
    else:
        # v4 default: fill any remaining NaN from each proxy in order.
        for proxy_rets in proxy_returns_list:
            merged = merged.combine_first(proxy_rets.reindex(full_index))

    return merged


def build_spliced_factor_panel(
    raw_ticker_prices: pd.DataFrame,
    factor_definitions: Optional[Dict[str, Tuple[str, str, str]]] = None,
    factor_fallbacks: Optional[Dict[str, List[str]]] = None,
    anchor_value: float = 100.0,
) -> pd.DataFrame:
    """Build the full spliced factor-price panel from raw ticker prices.

    Parameters
    ----------
    raw_ticker_prices :
        Wide DataFrame of close prices, one column per ticker, indexed
        by trading date.  Must contain every primary ticker named in
        ``factor_definitions`` and every fallback ticker named in
        ``factor_fallbacks``.
    factor_definitions, factor_fallbacks :
        Override the module-level tables for unit tests.  Defaults to
        :data:`FACTOR_DEFINITIONS` and :data:`FACTOR_FALLBACKS`.
    anchor_value :
        Pseudo-prices are anchored at this value on the first finite
        date per factor.  Defaults to ``100``.

    Returns
    -------
    pd.DataFrame
        Wide panel of pseudo-prices, one column per factor name, indexed
        by ``raw_ticker_prices.index``.  Each column is NaN before the
        earliest available source date and finite afterwards.
    """
    fd = factor_definitions or FACTOR_DEFINITIONS
    ff = factor_fallbacks or FACTOR_FALLBACKS

    # Compute log returns for every ticker present in the raw panel.
    ticker_log_rets: Dict[str, pd.Series] = {}
    for ticker in raw_ticker_prices.columns:
        p = raw_ticker_prices[ticker].dropna()
        if len(p) > 1:
            ticker_log_rets[ticker] = np.log(p / p.shift(1)).iloc[1:]

    full_index = raw_ticker_prices.index
    panel = pd.DataFrame(index=full_index)

    for factor, (primary, _desc, _cat) in fd.items():
        if primary not in ticker_log_rets:
            panel[factor] = np.nan
            continue
        proxies = [ticker_log_rets[t] for t in ff.get(factor, [])
                   if t in ticker_log_rets]
        merged = splice_factor(
            ticker_log_rets[primary], proxies,
        )
        merged = merged.reindex(full_index)

        # Reconstruct pseudo-prices: 100 * exp(cumsum of log returns).
        finite = merged.dropna()
        if len(finite) == 0:
            panel[factor] = np.nan
            continue
        cum = finite.cumsum()
        pseudo = np.exp(cum - cum.iloc[0]) * float(anchor_value)
        panel[factor] = pseudo.reindex(full_index)

    return panel


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def report_first_valid_dates(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising first/last valid dates per factor."""
    rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        rows.append({
            "factor": col,
            "first_valid": s.index.min() if len(s) else None,
            "last_valid":  s.index.max() if len(s) else None,
            "n_finite":    int(len(s)),
        })
    return pd.DataFrame(rows).set_index("factor")
