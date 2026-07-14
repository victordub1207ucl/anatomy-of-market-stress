"""Interpretation on the event clock (step27).

Fuses the project's strongest technical result — the event-time re-clocking of
Czasonis, Kritzman & Turkington (2023), where time advances by cumulative
Mahalanobis turbulence — with its most novel interpretive results: crisis
*composition predictability* (step17) and *causal drivers* of turbulence
(step16).  The question is whether measuring the run-up and the lead-lag
structure in **event time** (uniform information content) rather than **calendar
time** (uniform clock) sharpens the interpretation.

Two re-clocked objects:

1.  ``runup_composition_eventtime`` — the pre-spike composition computed over the
    window holding ``event_units`` of cumulative turbulence before an episode,
    instead of a fixed number of calendar days.  Paired with the calendar
    baseline (``runup_composition_calendar``) and a shared permutation core
    (``predictability_from_runup``) so the two clocks are directly comparable.

2.  ``resample_to_event_grid`` — bins daily returns / turbulence into disjoint
    one-event windows, yielding an event-indexed panel on which the existing
    Granger machinery can be re-run (calendar vs event clock).

All windows are strictly pre-episode / trailing, hence causal.  The event
threshold must be calibrated on development turbulence only.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Run-up composition on the two clocks
# --------------------------------------------------------------------------- #
def runup_composition_calendar(
    contributions: pd.DataFrame,
    episode_starts: Iterable[pd.Timestamp],
    lead_window: int = 21,
) -> pd.DataFrame:
    """Normalised mean per-factor contribution over the ``lead_window`` rows
    ending the day before each episode start (the step17 calendar pre-vector)."""
    rows, idx = [], []
    starts = pd.DatetimeIndex(list(episode_starts))
    for start in starts:
        loc = contributions.index.searchsorted(start)
        seg = contributions.iloc[max(0, loc - lead_window):loc].dropna(how="all")
        if seg.empty:
            continue
        v = seg.mean(axis=0)
        if v.sum() <= 0:
            continue
        rows.append((v / v.sum()).values)
        idx.append(start)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="start"),
                       columns=contributions.columns)
    out.attrs["mean_runup_days"] = float(lead_window)
    return out


def runup_composition_eventtime(
    contributions: pd.DataFrame,
    daily_turbulence: pd.Series,
    episode_starts: Iterable[pd.Timestamp],
    threshold: float,
    event_units: float = 1.0,
) -> pd.DataFrame:
    """Event-clock analogue: the pre-vector is the window holding ``event_units
    × threshold`` of cumulative turbulence immediately before each episode.

    Because turbulence is elevated going into a spike, this window is typically
    *shorter in calendar days* than the matched calendar window — it normalises
    the run-up by information content rather than elapsed time.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    target = float(event_units) * float(threshold)
    turb = daily_turbulence.reindex(contributions.index).to_numpy(dtype=float)
    rows, idx, ndays = [], [], []
    starts = pd.DatetimeIndex(list(episode_starts))
    for start in starts:
        loc = int(contributions.index.searchsorted(start))
        acc = 0.0
        j = loc - 1
        while j >= 0 and acc < target:
            t = turb[j]
            if np.isfinite(t):
                acc += t
            j -= 1
        lo = j + 1
        seg = contributions.iloc[lo:loc].dropna(how="all")
        if seg.empty:
            continue
        v = seg.mean(axis=0)
        if v.sum() <= 0:
            continue
        rows.append((v / v.sum()).values)
        idx.append(start)
        ndays.append(loc - lo)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="start"),
                       columns=contributions.columns)
    out.attrs["mean_runup_days"] = float(np.mean(ndays)) if ndays else float("nan")
    return out


def predictability_from_runup(
    runup: pd.DataFrame,
    composition: pd.DataFrame,
    *,
    n_perm: int = 10000,
    random_state: int = 0,
) -> Dict[str, float]:
    """Cosine similarity between each episode's run-up vector and its realised
    composition, vs a permutation null that shuffles realised compositions.

    Identical statistical core to ``crisis_typology.composition_predictability``
    so calendar and event-clock run-ups are directly comparable.
    """
    common = runup.index.intersection(composition.index)
    P = runup.loc[common].to_numpy(dtype=float)
    R = composition.loc[common].to_numpy(dtype=float)
    if len(common) == 0:
        return {"n_episodes": 0, "mean_cosine_observed": float("nan"),
                "mean_cosine_null": float("nan"), "p_value": float("nan"),
                "mean_runup_days": float(runup.attrs.get("mean_runup_days", float("nan")))}

    def _cos(a, b):
        na = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return (a * b).sum(axis=1) / np.where(na == 0, np.nan, na)

    obs = float(np.nanmean(_cos(P, R)))
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.nanmean(_cos(P, R[rng.permutation(len(R))]))
    return {
        "n_episodes": int(len(common)),
        "mean_cosine_observed": obs,
        "mean_cosine_null": float(null.mean()),
        "p_value": float((null >= obs).mean()),
        "mean_runup_days": float(runup.attrs.get("mean_runup_days", float("nan"))),
    }


# --------------------------------------------------------------------------- #
# Event-grid resampling (for re-running causality on the event clock)
# --------------------------------------------------------------------------- #
def event_windows(daily_turbulence: pd.Series, threshold: float) -> pd.DataFrame:
    """Disjoint one-event windows ('absorb' semantics): accumulate daily
    turbulence forward, cut a window each time the running sum reaches
    ``threshold``.  Returns a frame with integer start/end positions and the
    end timestamp of each event.  Causal: window boundaries up to t depend only
    on turbulence <= t.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    t = daily_turbulence.to_numpy(dtype=float)
    rows = []
    acc, start = 0.0, 0
    for i in range(len(t)):
        if np.isfinite(t[i]):
            acc += t[i]
        if acc >= threshold:
            rows.append((start, i))
            acc, start = 0.0, i + 1
    return pd.DataFrame(rows, columns=["start_pos", "end_pos"])


def resample_to_event_grid(
    returns: pd.DataFrame,
    turbulence: pd.Series,
    threshold: float,
) -> Dict[str, pd.DataFrame]:
    """Aggregate daily returns (sum) and turbulence (mean) into disjoint event
    windows.  Returns ``{"returns": <event x factor>, "turbulence": <event>}``
    indexed by each event's end timestamp — a uniform-information panel for
    re-running Granger causality on the event clock.
    """
    rets = returns.dropna(how="all")
    turb = turbulence.reindex(rets.index)
    wins = event_windows(turb, threshold)
    r_arr = rets.to_numpy(dtype=float)
    t_arr = turb.to_numpy(dtype=float)
    idx = rets.index
    r_rows, t_rows, ends = [], [], []
    for _, w in wins.iterrows():
        a, b = int(w["start_pos"]), int(w["end_pos"])
        sl_r = r_arr[a:b + 1]
        sl_t = t_arr[a:b + 1]
        r_rows.append(np.nansum(sl_r, axis=0))
        t_rows.append(float(np.nanmean(sl_t)))
        ends.append(idx[b])
    ends = pd.DatetimeIndex(ends, name="event_end")
    return {
        "returns": pd.DataFrame(r_rows, index=ends, columns=rets.columns),
        "turbulence": pd.Series(t_rows, index=ends, name="turbulence"),
    }
