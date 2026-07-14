"""Purged Group Time-Series Split with embargo.

Reference
---------
de Prado, M. L. (2018), *Advances in Financial Machine Learning*,
chapter 7.

What this enforces
------------------
The classical ``TimeSeriesSplit`` from scikit-learn produces expanding
training folds with a forward-walking test fold, but it does **not**
purge training samples whose forward-looking *label* overlaps the test
fold.  When the label is built from future returns, the last
``forward_window`` training rows leak future information into the fold.

This class operates in strict expanding-window mode:

* training fold = ``[0, test_start - forward_window - embargo)`` —
  contiguous block of past data only, with a left-side gap of
  ``forward_window + embargo`` rows so neither the label horizon nor any
  embargo buffer reaches into the test fold;
* test fold = contiguous block of rows, advancing chronologically.

``forward_window`` should equal the supervised target's look-ahead
horizon (e.g. ``21`` for :func:`binary_turbulence_entry(horizon=21)`).
``embargo`` is an *additional* defensive buffer on top of that — useful
if features use rolling statistics whose window may briefly straddle the
test boundary.  Together they bound the worst-case leakage to zero.

API mirrors :class:`sklearn.model_selection.BaseCrossValidator` so the
splitter plugs into ``cross_val_score`` / ``GridSearchCV`` directly.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


class PurgedGroupTimeSeriesSplit:
    """Time-series CV with purge + embargo.

    Parameters
    ----------
    n_splits :
        Number of test folds.
    forward_window :
        Look-ahead horizon used to build the labels — training rows whose
        own forward window overlaps the test fold are purged.  Pass the
        same value used in the target builder (e.g. ``21`` for
        ``binary_turbulence_entry(horizon=21)``).
    embargo :
        Additional defensive buffer added to the left-side gap on top of
        ``forward_window``.  Total purge = ``forward_window + embargo``
        rows before each test fold start.  No training rows are ever
        taken from after the test fold — this is strict expanding-window.
    min_train_size :
        Minimum number of rows required in the training fold; folds whose
        training partition is smaller than this are skipped.
    """

    def __init__(
        self,
        n_splits: int = 5,
        forward_window: int = 21,
        embargo: int = 21,
        min_train_size: int = 252,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if forward_window < 0:
            raise ValueError(f"forward_window must be >= 0, got {forward_window}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
        self.n_splits = int(n_splits)
        self.forward_window = int(forward_window)
        self.embargo = int(embargo)
        self.min_train_size = int(min_train_size)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y: Optional[pd.Series] = None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs.

        The input is split chronologically.  The test folds are
        contiguous, equal-sized blocks tiling the second half of the
        sample; the training fold expands from the start of the sample
        up to the test-fold start (minus the purge), with the embargo
        applied to any training rows after the test fold (none in the
        default expanding-window mode).
        """
        n = len(X)
        if n < self.n_splits + 1:
            raise ValueError(
                f"n_splits={self.n_splits} requires at least "
                f"{self.n_splits + 1} samples; got {n}"
            )

        fold_size = n // (self.n_splits + 1)  # +1 keeps the first chunk for warm-up
        for k in range(self.n_splits):
            test_start = (k + 1) * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end)
            # Total left-side gap: purge for label horizon + extra embargo
            # buffer.  Pure expanding-window: no training rows are ever
            # taken from after the test fold.
            purge_start = test_start - self.forward_window - self.embargo
            train_idx = np.arange(0, max(0, purge_start))
            if len(train_idx) < self.min_train_size:
                continue
            yield train_idx, test_idx
