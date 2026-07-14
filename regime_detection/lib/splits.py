"""Development/out-of-sample split contract.

The thesis cutoff is hard: every model, scaler, selector, and explainer must
be fit on data with index strictly <= DEV_END.  The OOS window is reserved
for true held-out evaluation and must never be visible during fitting.

Fixed-split constraint (decisive, set 2026-06): the out-of-sample test set must
begin on 1 January 2020.  This deliberately brings the COVID-19 crash
(Feb-Mar 2020) into the held-out window, so any de-risking / hedging overlay
is judged on the single largest stress event of the modern sample rather than
being able to learn from it.  Accordingly DEV_END = 2019-12-31 and
OOS_START = 2020-01-01.  (Prior to this constraint the split was
2020-12-31 / 2021-01-01; OOS-dependent results computed under the old split
must be regenerated to comply.)

This module provides:
    DEV_END / OOS_START  — the cutoff timestamps
    get_dev_test_split() — split a DataFrame / Series into (dev, oos)
    assert_no_oos_contamination() — runtime guard used by walk-forward
                                    objects and pipeline glue code

The intent is that any object that fits on data calls
``assert_no_oos_contamination(data.index)`` before fitting, so that a
look-ahead bug raises loudly rather than silently biasing metrics.
"""

from __future__ import annotations

from typing import Tuple, Union

import pandas as pd

DEV_END: pd.Timestamp = pd.Timestamp("2019-12-31")
OOS_START: pd.Timestamp = pd.Timestamp("2020-01-01")

Frame = Union[pd.DataFrame, pd.Series]


def get_dev_test_split(df: Frame) -> Tuple[Frame, Frame]:
    """Return ``(dev, oos)`` where ``dev.index <= DEV_END`` and
    ``oos.index >= OOS_START``.

    Parameters
    ----------
    df :
        A DataFrame or Series with a ``DatetimeIndex``.

    Returns
    -------
    (dev, oos) :
        Two slices of the same type as ``df``.  Either slice may be empty
        if the input does not straddle the cutoff.

    Raises
    ------
    TypeError
        If ``df`` does not have a ``DatetimeIndex``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "get_dev_test_split requires a DatetimeIndex; "
            f"got {type(df.index).__name__}"
        )
    dev = df.loc[:DEV_END]
    oos = df.loc[OOS_START:]
    return dev, oos


def assert_no_oos_contamination(
    index: pd.DatetimeIndex,
    *,
    max_fit_date: pd.Timestamp = DEV_END,
    label: str = "fit",
) -> None:
    """Raise ``AssertionError`` if ``index`` contains any timestamp
    strictly greater than ``max_fit_date``.

    Intended to be called by any object immediately before it fits on a
    time-indexed dataset.  Walk-forward classes pass the relevant
    sub-window index here so an off-by-one slip in the caller surfaces
    immediately.

    Parameters
    ----------
    index :
        Timestamps that the caller is about to fit on.
    max_fit_date :
        Latest timestamp allowed in the fit set.  Defaults to ``DEV_END``.
    label :
        Free-text label included in the assertion message (e.g. the name
        of the calling class) — purely diagnostic.
    """
    if len(index) == 0:
        return
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    latest = index.max()
    if latest > max_fit_date:
        raise AssertionError(
            f"Look-ahead violation in {label!r}: fit set extends to "
            f"{latest.date()}, which is after the allowed cutoff "
            f"{max_fit_date.date()}."
        )
