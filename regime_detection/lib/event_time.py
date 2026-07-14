"""Event-time conversion (Czasonis, Kritzman & Turkington 2023).

Reference
---------
Czasonis, M., Kritzman, M. and Turkington, D. (2023), "Event Time",
*Journal of Portfolio Management* 49(7).

Intuition
---------
Calendar time advances uniformly: every trading day counts as one unit.
**Event time** advances proportional to the cumulative Mahalanobis
turbulence observed across markets — so it speeds up during crises and
slows down during quiet periods.  One *event unit* is defined as a fixed
amount of cumulative turbulence, called the ``intensity_threshold``.

Causality
---------
The conversion is strictly causal: at any date ``t`` the event boundaries
up to ``t`` depend only on turbulence values with index ``<= t``.  No
future information is used.

Overflow handling
-----------------
A single day can deliver more turbulence than ``threshold``.  Two
semantics are possible:

* *Absorb*: the event ends on that day with ``cumulative_intensity >=
  threshold``; the excess is recorded inside the event but not carried.
* *Carry*: each event has cumulative_intensity ≈ ``threshold``; the
  excess starts the next event's accumulator.

This implementation uses **absorb** — it keeps every event a contiguous
calendar window with ``cumulative_intensity >= threshold`` and prevents
spurious zero-turbulence "events" on quiet days that follow a spike.
:func:`get_overlapping_event_returns` uses the matching semantics: the
event window starting at day ``i`` ends at the smallest ``j > i`` such
that turbulence accumulated over ``(i, j]`` reaches ``threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class _CalibrationDiagnostics:
    """Bookkeeping returned by :meth:`EventTimeConverter.calibrate`."""

    threshold: float
    mean_trading_days: float
    n_events: int
    n_iterations: int


class EventTimeConverter:
    """Walk-forward event-time converter.

    Parameters
    ----------
    intensity_threshold :
        Cumulative Mahalanobis turbulence that defines one event unit.

    See Also
    --------
    EventTimeConverter.calibrate :
        Solve for an ``intensity_threshold`` that yields a target average
        event length on a given turbulence series.
    get_overlapping_event_returns :
        Companion utility that produces a Series of one-event-unit asset
        returns starting at every calendar day (analogue of rolling
        21-day returns).
    """

    EVENT_COLUMNS = [
        "event_id",
        "start_date",
        "end_date",
        "calendar_days_in_event",
        "trading_days_in_event",
        "cumulative_intensity",
    ]

    def __init__(self, intensity_threshold: float) -> None:
        threshold = float(intensity_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                f"intensity_threshold must be a positive finite number, "
                f"got {intensity_threshold!r}"
            )
        self.intensity_threshold = threshold

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def fit_transform(self, daily_turbulence: pd.Series) -> pd.DataFrame:
        """Convert a daily turbulence series into disjoint event windows.

        Parameters
        ----------
        daily_turbulence :
            Series indexed by trading date.  ``NaN`` rows (e.g. the
            initial Mahalanobis warm-up) are dropped before accumulation.
            Negative values are not allowed — turbulence is a quadratic
            form and is non-negative by construction.

        Returns
        -------
        pd.DataFrame
            One row per event, with columns ``event_id``, ``start_date``,
            ``end_date``, ``calendar_days_in_event``,
            ``trading_days_in_event``, ``cumulative_intensity``.
        """
        if not isinstance(daily_turbulence, pd.Series):
            raise TypeError(
                f"daily_turbulence must be a pandas Series, got "
                f"{type(daily_turbulence).__name__}"
            )
        if not isinstance(daily_turbulence.index, pd.DatetimeIndex):
            raise TypeError(
                "daily_turbulence must have a DatetimeIndex; got "
                f"{type(daily_turbulence.index).__name__}"
            )
        s = daily_turbulence.dropna()
        if len(s) == 0:
            return pd.DataFrame(columns=self.EVENT_COLUMNS)
        if (s.values < 0).any():
            raise ValueError(
                "daily_turbulence contains negative values; Mahalanobis "
                "turbulence is non-negative by construction."
            )

        vals = s.values.astype(float)
        dates = s.index
        n = len(vals)
        threshold = self.intensity_threshold

        events = []
        event_cum = 0.0
        start_idx = 0
        event_id = 0
        for i in range(n):
            event_cum += vals[i]
            if event_cum >= threshold:
                events.append({
                    "event_id": event_id,
                    "start_date": dates[start_idx],
                    "end_date": dates[i],
                    "calendar_days_in_event": int(
                        (dates[i] - dates[start_idx]).days
                    ),
                    "trading_days_in_event": int(i - start_idx + 1),
                    "cumulative_intensity": float(event_cum),
                })
                event_id += 1
                event_cum = 0.0
                start_idx = i + 1

        return pd.DataFrame(events, columns=self.EVENT_COLUMNS)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        daily_turbulence: pd.Series,
        target_mean_trading_days: float = 21.0,
        tolerance: float = 0.25,
        max_iter: int = 60,
        verbose: bool = False,
    ) -> Tuple["EventTimeConverter", _CalibrationDiagnostics]:
        """Find an ``intensity_threshold`` whose mean event length matches
        the target.

        Bisection over the threshold.  Mean event length is monotonically
        increasing in the threshold, so bisection is guaranteed to
        converge.

        Parameters
        ----------
        daily_turbulence :
            Series used to calibrate.  Use the same series the converter
            will be applied to (e.g. the full dev-window turbulence).
        target_mean_trading_days :
            Desired mean number of trading days per event.  Defaults to
            ``21`` to match the existing 21-day forward-drawdown horizon.
        tolerance :
            Bisection stops when ``|mean_event_days - target| < tolerance``.
        max_iter :
            Hard cap on bisection iterations.
        verbose :
            If True, print each iteration.

        Returns
        -------
        (converter, diagnostics) :
            ``converter`` is a fitted :class:`EventTimeConverter`;
            ``diagnostics`` contains the chosen threshold and observed
            mean event length.
        """
        s = daily_turbulence.dropna()
        if len(s) < target_mean_trading_days * 4:
            raise ValueError(
                f"Need at least {int(target_mean_trading_days * 4)} non-NaN "
                f"turbulence values to calibrate; got {len(s)}"
            )

        # Bracket: low threshold → events too short; high threshold →
        # events too long.  Start at [0.1 * mean, 10 * mean * target].
        mean_turb = float(s.mean())
        lo = mean_turb * 0.1
        hi = mean_turb * target_mean_trading_days * 10.0

        def mean_event_days(threshold: float) -> float:
            conv = cls(intensity_threshold=threshold)
            df = conv.fit_transform(s)
            if len(df) == 0:
                return float("inf")  # threshold too high → no events
            return float(df["trading_days_in_event"].mean())

        # Ensure the bracket actually brackets the target.
        for _ in range(20):
            if mean_event_days(lo) < target_mean_trading_days:
                break
            lo *= 0.5
        for _ in range(20):
            if mean_event_days(hi) > target_mean_trading_days:
                break
            hi *= 2.0

        last_mid = (lo + hi) / 2.0
        last_mer = float("nan")
        for it in range(max_iter):
            mid = (lo + hi) / 2.0
            mer = mean_event_days(mid)
            if verbose:
                print(f"[calibrate] iter={it:>3d} threshold={mid:.4f} "
                      f"mean_event_days={mer:.3f}")
            last_mid, last_mer = mid, mer
            if abs(mer - target_mean_trading_days) < tolerance:
                break
            if mer < target_mean_trading_days:
                lo = mid
            else:
                hi = mid

        conv = cls(intensity_threshold=last_mid)
        events = conv.fit_transform(s)
        diag = _CalibrationDiagnostics(
            threshold=last_mid,
            mean_trading_days=last_mer,
            n_events=len(events),
            n_iterations=it + 1,
        )
        return conv, diag


# ---------------------------------------------------------------------------
# Overlapping-event-window utility
# ---------------------------------------------------------------------------


def get_overlapping_event_returns(
    prices: pd.Series,
    daily_turbulence: pd.Series,
    intensity_threshold: float,
    log_returns: bool = True,
) -> pd.Series:
    """One-event-unit asset returns starting at every calendar day.

    For each calendar day ``i`` in the intersection of ``prices`` and
    ``daily_turbulence``, find the smallest ``j > i`` such that the
    cumulative turbulence over ``(i, j]`` exceeds ``intensity_threshold``
    and return ``log(p[j] / p[i])`` (or the arithmetic equivalent).

    Parameters
    ----------
    prices :
        Asset price series (close-to-close).
    daily_turbulence :
        Daily Mahalanobis turbulence.  ``NaN`` rows are dropped before
        accumulation; output is ``NaN`` at those dates.
    intensity_threshold :
        Cumulative turbulence per event unit.  Use
        :meth:`EventTimeConverter.calibrate` to choose this.
    log_returns :
        ``True`` (default) returns ``log(p[j] / p[i])``; ``False`` returns
        the arithmetic return.

    Returns
    -------
    pd.Series
        Indexed by the start date ``i``.  Rows whose event window extends
        past the end of the data are ``NaN``.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a Series, got {type(prices).__name__}")
    if not isinstance(daily_turbulence, pd.Series):
        raise TypeError(
            f"daily_turbulence must be a Series, got "
            f"{type(daily_turbulence).__name__}"
        )
    threshold = float(intensity_threshold)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError(
            f"intensity_threshold must be a positive finite number, "
            f"got {intensity_threshold!r}"
        )

    s = daily_turbulence.dropna()
    common = prices.index.intersection(s.index)
    if len(common) == 0:
        return pd.Series([], dtype=float, name="event_return")
    p = prices.reindex(common).values.astype(float)
    t = s.reindex(common).values.astype(float)
    if (t < 0).any():
        raise ValueError(
            "daily_turbulence contains negative values on the common index."
        )
    cum = np.cumsum(t)
    n = len(common)
    out = np.full(n, np.nan)
    for i in range(n):
        target = cum[i] + threshold
        j = int(np.searchsorted(cum, target, side="left"))
        if j >= n:
            continue
        if not (np.isfinite(p[i]) and np.isfinite(p[j])):
            continue
        if log_returns:
            out[i] = np.log(p[j] / p[i])
        else:
            out[i] = p[j] / p[i] - 1.0
    return pd.Series(out, index=common, name="event_return")
