"""Quarterly walk-forward refit for OOS supervised forecasting.

Realistic-deployment protocol::

    at each quarter-end t_q in [DEV_END, end_of_data]:
        train_on = X.loc[: t_q - purge_window]
        scaler   = StandardScaler().fit(train_on)
        estimator = MODEL_BUILDERS[name].fit(scaler.transform(train_on))
        predict for rows in (t_q, t_{q+1}]

Hyperparameters are FROZEN at the Phase-4 defaults — no per-quarter CV.
That matches a real deployment where the model is retuned rarely and
refit often.  The output is one prediction series per model, every value
of which was produced by a model trained *only* on past data.

Outputs (per model)::

    oos_predictions : pd.Series   # P(target=1) for every OOS date
    oos_labels      : pd.Series   # ground-truth binary labels
    refit_log       : pd.DataFrame # one row per quarterly refit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_detection.lib.train import MODEL_BUILDERS, _predict_proba
from regime_detection.lib.splits import (
    DEV_END, OOS_START, assert_no_oos_contamination,
)


@dataclass
class QuarterlyWalkForward:
    """Per-model output of the quarterly walk-forward pipeline."""

    name: str
    oos_predictions: pd.Series
    oos_labels: pd.Series
    refit_log: pd.DataFrame
    feature_columns: List[str]


def _quarter_ends(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Quarter-end dates between ``start`` (inclusive) and ``end``
    (inclusive)."""
    return pd.date_range(start=start, end=end, freq="QE")


def run_quarterly_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_names: List[str],
    purge_window: int = 21,
    dev_end: pd.Timestamp = DEV_END,
    oos_start: pd.Timestamp = OOS_START,
    random_state: int = 0,
) -> Dict[str, QuarterlyWalkForward]:
    """Run quarterly walk-forward refit for every model in ``model_names``.

    Parameters
    ----------
    X, y :
        Feature matrix and binary target — already aligned, NaNs dropped
        by the caller.
    model_names :
        Subset of :data:`regime_detection.lib.train.MODEL_BUILDERS`
        keys to refit each quarter.
    purge_window :
        Number of trading days to drop from the right edge of the
        training set before each fit (so the most-recent training rows
        do not have label-look-ahead into the prediction window).
    dev_end, oos_start :
        Cutoffs.  The first quarterly refit uses ``dev_end`` as its
        as-of date and predicts ``(dev_end, first_quarter_end]``.
    random_state :
        Forwarded to every model builder.

    Returns
    -------
    dict[str, QuarterlyWalkForward]
    """
    # Drop rows with any NaN; align X and y.
    aligned = X.join(y.rename("__y"), how="inner").dropna()
    X_all = aligned[X.columns]
    y_all = aligned["__y"].astype(float)

    # Sequence of refit dates: dev_end + every quarter end up to the
    # end of the data.  The OOS prediction window for refit k spans
    # (refit_dates[k], refit_dates[k+1]] (or (refit_dates[-1], end_of_data]
    # for the last one).
    end_of_data = X_all.index.max()
    refit_dates = [pd.Timestamp(dev_end)] + list(
        _quarter_ends(pd.Timestamp(oos_start), end_of_data),
    )
    # Deduplicate / sort.
    refit_dates = sorted(set(refit_dates))

    # Per-model prediction containers.
    preds: Dict[str, List[pd.Series]] = {name: [] for name in model_names}
    refit_rows: Dict[str, List[dict]] = {name: [] for name in model_names}

    for k in range(len(refit_dates)):
        asof = refit_dates[k]
        next_end = (
            refit_dates[k + 1] if k + 1 < len(refit_dates) else end_of_data
        )
        if next_end <= asof:
            continue

        train_end = asof - pd.Timedelta(days=purge_window)
        train_mask = X_all.index <= train_end
        if train_mask.sum() < 252:
            continue

        X_train = X_all.loc[train_mask]
        y_train = y_all.loc[train_mask]
        assert_no_oos_contamination(
            X_train.index, max_fit_date=train_end, label="QuarterlyWalkForward",
        )

        pred_mask = (X_all.index > asof) & (X_all.index <= next_end)
        X_pred = X_all.loc[pred_mask]
        y_pred = y_all.loc[pred_mask]
        if len(X_pred) == 0:
            continue

        scaler = StandardScaler().fit(X_train.values)
        X_train_s = scaler.transform(X_train.values)
        X_pred_s = scaler.transform(X_pred.values)

        for name in model_names:
            est = MODEL_BUILDERS[name](random_state=random_state)
            est.fit(X_train_s, y_train.values.astype(int))
            probs = _predict_proba(est, X_pred_s)
            preds[name].append(pd.Series(probs, index=X_pred.index))
            refit_rows[name].append({
                "asof":              asof,
                "train_end":         train_end,
                "train_n":           int(len(X_train)),
                "pred_window_end":   next_end,
                "pred_n":            int(len(X_pred)),
                "train_pos_rate":    float(y_train.mean()),
                "pred_pos_rate":     float(y_pred.mean()),
            })

    out: Dict[str, QuarterlyWalkForward] = {}
    for name in model_names:
        if not preds[name]:
            out[name] = QuarterlyWalkForward(
                name=name,
                oos_predictions=pd.Series(dtype=float, name="proba"),
                oos_labels=pd.Series(dtype=float, name="y_true"),
                refit_log=pd.DataFrame(),
                feature_columns=list(X.columns),
            )
            continue
        proba = pd.concat(preds[name]).sort_index()
        # Ground-truth labels at the same dates.
        labels = y_all.reindex(proba.index)
        out[name] = QuarterlyWalkForward(
            name=name,
            oos_predictions=proba.rename("proba"),
            oos_labels=labels.rename("y_true"),
            refit_log=pd.DataFrame(refit_rows[name]),
            feature_columns=list(X.columns),
        )
    return out
