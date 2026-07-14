"""Sliding-cutoff sensitivity for the headline Sortino lift.

An earlier three-point cutoff sensitivity reported that the headline
single-cutoff Sortino lift flipped sign at adjacent cutoffs.  This
module replaces the three points with a configurable schedule of
cutoffs (default monthly), so the headline lift is reported as a
distribution rather than a point estimate.  The single-cutoff
"headline" is taken at the current ``DEV_END`` (2019-12-31).

The protocol per cutoff:

1. set ``DEV_END`` = cutoff, ``OOS_START`` = cutoff + 1 day;
2. run the Phase-6 quarterly walk-forward
   (``run_quarterly_walk_forward``) over the resulting OOS partition;
3. compute Sortino lift, AUC, Brier, max DD from the resulting
   predictions.

Cutoffs that leave fewer than ``min_oos_quarters`` quarters of OOS
data are skipped.  Per-(model, cutoff) results are cached as JSON in
``cache_dir`` so reruns are cheap.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from regime_detection.lib.metrics import (
    annualised_return, brier_score, log_loss, max_drawdown, sortino_ratio,
    strategy_returns,
)
from regime_detection.lib.oos_pipeline import (
    QuarterlyWalkForward, run_quarterly_walk_forward,
)


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def metrics_for_cutoff_run(
    qwf: QuarterlyWalkForward,
    equity_returns: pd.Series,
) -> Dict[str, float]:
    """Compute the per-cutoff metrics from a single QuarterlyWalkForward
    output."""
    y = qwf.oos_labels.dropna()
    p = qwf.oos_predictions.dropna()
    common = y.index.intersection(p.index)
    y, p = y.reindex(common), p.reindex(common)
    if len(y) < 50 or len(y.unique()) < 2:
        return {
            "n_oos":           int(len(y)),
            "n_refits":        int(len(qwf.refit_log)),
            "pos_rate":        float(y.mean()) if len(y) else float("nan"),
            "auc":             float("nan"),
            "brier":           float("nan"),
            "log_loss":        float("nan"),
            "sortino_strategy":float("nan"),
            "sortino_buy_hold":float("nan"),
            "sortino_lift":    float("nan"),
            "max_dd_strategy": float("nan"),
            "max_dd_buy_hold": float("nan"),
            "ann_ret_strategy":float("nan"),
        }
    auc = _safe_auc(y.values, p.values)
    strat = strategy_returns(equity_returns, p, shift=1)
    bh = equity_returns.reindex(strat.index)
    s_strat = sortino_ratio(strat)
    s_bh = sortino_ratio(bh)
    return {
        "n_oos":            int(len(y)),
        "n_refits":         int(len(qwf.refit_log)),
        "pos_rate":         float(y.mean()),
        "auc":              auc,
        "brier":            brier_score(y, p),
        "log_loss":         log_loss(y, p),
        "sortino_strategy": s_strat,
        "sortino_buy_hold": s_bh,
        "sortino_lift":     (s_strat - s_bh)
                            if (np.isfinite(s_strat) and np.isfinite(s_bh))
                            else float("nan"),
        "max_dd_strategy":  max_drawdown(strat),
        "max_dd_buy_hold":  max_drawdown(bh),
        "ann_ret_strategy": annualised_return(strat),
    }


def _is_pre_existing_cache(cache_dir: Optional[Path], model: str,
                            cutoff: pd.Timestamp) -> Optional[Path]:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{model}_cutoff_{cutoff.date()}.json"
    return path if path.exists() else None


def _write_cache(cache_dir: Path, model: str, cutoff: pd.Timestamp,
                  metrics: Dict[str, float]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{model}_cutoff_{cutoff.date()}.json"
    # Override the cutoff value with the bare ISO date so json.dump
    # doesn't introduce a "00:00:00" component via default=str on the
    # Timestamp held by metrics["cutoff"].  Model name is encoded in
    # the filename and deliberately NOT a column in the cached payload,
    # so a cache hit yields a row schema identical to a fresh run.
    payload = {**metrics, "cutoff": str(cutoff.date())}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _read_cache(path: Path) -> Dict:
    with open(path) as f:
        d = json.load(f)
    d["cutoff"] = pd.Timestamp(d["cutoff"])
    return d


def run_sliding_cutoff(
    X: pd.DataFrame,
    y: pd.Series,
    equity_returns: pd.Series,
    *,
    cutoff_dates: Sequence[pd.Timestamp],
    model_name: str = "logistic_l1",
    purge_window: int = 21,
    min_oos_quarters: int = 4,
    random_state: int = 0,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the quarterly walk-forward repeatedly with sliding ``DEV_END``.

    Parameters
    ----------
    X, y, equity_returns :
        Inputs to the supervised pipeline.  Computed once by the caller
        (typically the Phase 4 feature matrix + binary turbulence target +
        equity log-returns).
    cutoff_dates :
        Iterable of timestamps to use as ``DEV_END``.  ``OOS_START`` for
        each cutoff is ``cutoff + 1 day``.
    model_name :
        Key into ``MODEL_BUILDERS`` (typically ``"logistic_l1"`` or
        ``"random_forest"``).
    min_oos_quarters :
        Skip any cutoff that leaves fewer than this many calendar
        quarters of OOS data before the end of ``X``.
    cache_dir :
        Optional directory to persist per-cutoff JSON.  Subsequent calls
        with the same ``(model, cutoff)`` pair load from cache and skip
        the quarterly walk-forward.

    Returns
    -------
    pd.DataFrame
        Indexed by ``cutoff`` (DatetimeIndex), one row per emitted
        cutoff, columns include ``sortino_lift``, ``auc``, ``brier``,
        ``max_dd_strategy``, etc.
    """
    end_of_data = X.index.max()
    rows: List[Dict] = []

    for k, cutoff in enumerate(cutoff_dates):
        cutoff = pd.Timestamp(cutoff)
        if cutoff <= X.index.min():
            continue
        if cutoff >= end_of_data:
            continue
        # OOS quarters available between cutoff and end of data.
        days_remaining = (end_of_data - cutoff).days
        if days_remaining < min_oos_quarters * 91:
            if verbose:
                print(f"  [{k:>2d}] cutoff={cutoff.date()}: skip "
                      f"({days_remaining}d remaining)")
            continue

        # Cache lookup.
        cached_path = _is_pre_existing_cache(cache_dir, model_name, cutoff)
        if cached_path is not None:
            row = _read_cache(cached_path)
            rows.append(row)
            if verbose:
                print(f"  [{k:>2d}] cutoff={cutoff.date()}: cache hit "
                      f"(lift={row.get('sortino_lift', float('nan')):.3f})")
            continue

        oos_start = cutoff + pd.Timedelta(days=1)
        try:
            result = run_quarterly_walk_forward(
                X, y, model_names=[model_name],
                purge_window=purge_window, dev_end=cutoff,
                oos_start=oos_start, random_state=random_state,
            )[model_name]
        except Exception as exc:
            if verbose:
                print(f"  [{k:>2d}] cutoff={cutoff.date()}: FAILED ({exc})")
            continue

        metrics = metrics_for_cutoff_run(result, equity_returns)
        metrics["cutoff"] = cutoff
        rows.append(metrics)

        if cache_dir is not None:
            _write_cache(Path(cache_dir), model_name, cutoff, metrics)

        if verbose:
            print(
                f"  [{k:>2d}] cutoff={cutoff.date()}: "
                f"n_oos={metrics['n_oos']:>4d} "
                f"AUC={metrics['auc']:.3f} "
                f"lift={metrics['sortino_lift']:+.3f}"
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("cutoff").sort_index()
    return df


def summary_statistics(
    df: pd.DataFrame,
    metric: str = "sortino_lift",
) -> Dict[str, float]:
    """Compute the headline summary statistics on a metric column."""
    s = df[metric].dropna()
    if len(s) == 0:
        return {
            "n":                       0,
            "median":                  float("nan"),
            "mean":                    float("nan"),
            "q25":                     float("nan"),
            "q75":                     float("nan"),
            "min":                     float("nan"),
            "max":                     float("nan"),
            "frac_positive":           float("nan"),
            "frac_above_0p10":         float("nan"),
            "pearson_r_vs_cutoff":     float("nan"),
        }
    cutoff_days = (df.dropna(subset=[metric]).index
                   - df.dropna(subset=[metric]).index.min()).days.values
    r = float(np.corrcoef(cutoff_days, s.values)[0, 1]) if len(s) > 2 \
        else float("nan")
    return {
        "n":                       int(len(s)),
        "median":                  float(s.median()),
        "mean":                    float(s.mean()),
        "q25":                     float(s.quantile(0.25)),
        "q75":                     float(s.quantile(0.75)),
        "min":                     float(s.min()),
        "max":                     float(s.max()),
        "frac_positive":           float((s > 0).mean()),
        "frac_above_0p10":         float((s > 0.10).mean()),
        "pearson_r_vs_cutoff":     r,
    }
