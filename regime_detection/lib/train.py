"""Phase-4 training pipeline.

Strict dev / OOS protocol::

    dev = X.loc[:DEV_END]
    oos = X.loc[OOS_START:]

* Cross-validation runs only on ``dev`` via
  :class:`PurgedGroupTimeSeriesSplit`.  Inside each fold the feature
  scaler is fit on the fold's training partition only.
* The final model is trained on the full ``dev`` partition and then
  evaluated **once** on ``oos``.
* OOS labels are never used to choose hyper-parameters, model family,
  feature set, or threshold.

Outputs (per model):

* ``estimator``           — the final fitted estimator on full dev.
* ``scaler``              — the StandardScaler fit on full dev (used to
                            transform OOS features once at inference).
* ``cv_scores``           — DataFrame with per-fold AUC, AP and avg
                            precision-recall at top-decile threshold.
* ``oos_predictions``     — Series of OOS predicted probabilities.
* ``oos_labels``          — Aligned OOS true labels (Series).
* ``feature_columns``     — Column names the model consumes.
* ``hyperparams``         — Estimator parameters at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from regime_detection.lib.pgts import PurgedGroupTimeSeriesSplit
from regime_detection.lib.splits import (
    DEV_END, OOS_START, assert_no_oos_contamination,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _rf(random_state: int = 0):
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def _logistic_l1(random_state: int = 0):
    return LogisticRegression(
        penalty="l1", solver="liblinear",
        C=1.0, class_weight="balanced",
        max_iter=5_000, random_state=random_state,
    )


def _linear_svm(random_state: int = 0):
    base = LinearSVC(
        C=1.0, class_weight="balanced", dual="auto",
        max_iter=10_000, random_state=random_state,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def _logistic_elasticnet(random_state: int = 0):
    return LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        C=1.0, class_weight="balanced",
        max_iter=10_000, random_state=random_state,
    )


MODEL_BUILDERS: Dict[str, Callable[..., Any]] = {
    "random_forest":         _rf,
    "logistic_l1":           _logistic_l1,
    "linear_svm":            _linear_svm,
    "logistic_elasticnet":   _logistic_elasticnet,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainedModel:
    """Everything Phase-4 evaluation needs from a single model."""

    name: str
    estimator: Any
    scaler: StandardScaler
    feature_columns: List[str]
    cv_scores: pd.DataFrame
    oos_predictions: pd.Series
    oos_labels: pd.Series
    hyperparams: Dict[str, Any]
    fit_date_max: pd.Timestamp
    random_state: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _predict_proba(estimator, X: np.ndarray) -> np.ndarray:
    """Return the positive-class probability vector for any estimator."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p.ravel()
    if hasattr(estimator, "decision_function"):
        # Map decision_function to (0, 1) via sigmoid for monotone ranking.
        df = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-df))
    raise RuntimeError(
        f"Estimator {type(estimator).__name__} exposes neither "
        f"predict_proba nor decision_function"
    )


def _top_decile_precision_recall(y_true, y_score, decile: float = 0.10):
    """Precision and recall when labelling the top ``decile`` of scores
    as positive."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]
    if len(y_true) == 0:
        return float("nan"), float("nan")
    cutoff = np.quantile(y_score, 1.0 - decile)
    predicted_pos = y_score >= cutoff
    true_pos = (y_true > 0.5)
    tp = int((predicted_pos & true_pos).sum())
    fp = int((predicted_pos & ~true_pos).sum())
    fn = int((~predicted_pos & true_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def _safe_auc(y_true, y_score) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    if finite.sum() < 2:
        return float("nan")
    y_true, y_score = np.asarray(y_true)[finite], np.asarray(y_score)[finite]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def train_one_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    forward_window: int = 21,
    embargo: int = 21,
    n_splits: int = 5,
    random_state: int = 0,
    dev_end: pd.Timestamp = DEV_END,
    oos_start: pd.Timestamp = OOS_START,
) -> TrainedModel:
    """Train one challenger model end-to-end.

    Parameters
    ----------
    name :
        Key into :data:`MODEL_BUILDERS`.
    X :
        Feature matrix indexed by date.  NaNs are dropped row-wise before
        training; both ``X`` and ``y`` must be already aligned.
    y :
        Binary target (0/1) aligned with ``X``.
    forward_window, embargo, n_splits :
        Parameters of :class:`PurgedGroupTimeSeriesSplit`.
    random_state :
        Forwarded to the model builder.
    dev_end / oos_start :
        Override the global cutoffs if needed.

    Returns
    -------
    TrainedModel
    """
    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model {name!r}; available: {list(MODEL_BUILDERS)}"
        )

    # Align and drop NaNs.
    Z = X.join(y.rename("__y"), how="inner").dropna()
    X_aligned = Z.drop(columns="__y")
    y_aligned = Z["__y"].astype(float)

    dev_mask = X_aligned.index <= dev_end
    oos_mask = X_aligned.index >= oos_start
    X_dev, y_dev = X_aligned.loc[dev_mask], y_aligned.loc[dev_mask]
    X_oos, y_oos = X_aligned.loc[oos_mask], y_aligned.loc[oos_mask]

    # Hard guard — fail loudly if either partition is contaminated.
    assert_no_oos_contamination(
        X_dev.index, max_fit_date=dev_end, label=f"{name}.dev",
    )
    if len(X_dev) == 0:
        raise ValueError(f"{name}: dev partition is empty.")
    if len(X_oos) == 0:
        raise ValueError(f"{name}: OOS partition is empty.")

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits, forward_window=forward_window, embargo=embargo,
        min_train_size=252,
    )

    fold_rows = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_dev)):
        X_tr, y_tr = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx]
        X_te, y_te = X_dev.iloc[te_idx], y_dev.iloc[te_idx]
        scaler = StandardScaler().fit(X_tr.values)
        Xtr_s = scaler.transform(X_tr.values)
        Xte_s = scaler.transform(X_te.values)
        est = MODEL_BUILDERS[name](random_state=random_state)
        est.fit(Xtr_s, y_tr.values.astype(int))
        scores = _predict_proba(est, Xte_s)
        auc = _safe_auc(y_te.values, scores)
        ap = float(average_precision_score(y_te.values, scores)) \
            if len(np.unique(y_te)) > 1 else float("nan")
        prec_td, rec_td = _top_decile_precision_recall(y_te.values, scores)
        brier = float(brier_score_loss(y_te.values, scores)) \
            if len(np.unique(y_te)) > 1 else float("nan")
        fold_rows.append({
            "fold":            fold,
            "train_n":         len(tr_idx),
            "test_n":          len(te_idx),
            "test_start":      X_te.index.min().date(),
            "test_end":        X_te.index.max().date(),
            "test_pos_rate":   float(y_te.mean()),
            "auc":             auc,
            "avg_precision":   ap,
            "prec_top_decile": prec_td,
            "rec_top_decile":  rec_td,
            "brier":           brier,
        })
    cv_scores = pd.DataFrame(fold_rows)

    # Fit final model on full dev partition.
    final_scaler = StandardScaler().fit(X_dev.values)
    X_dev_s = final_scaler.transform(X_dev.values)
    final_est = MODEL_BUILDERS[name](random_state=random_state)
    final_est.fit(X_dev_s, y_dev.values.astype(int))

    # Single OOS evaluation pass.
    X_oos_s = final_scaler.transform(X_oos.values)
    oos_pred = _predict_proba(final_est, X_oos_s)
    oos_predictions = pd.Series(oos_pred, index=X_oos.index, name="proba")
    oos_labels = y_oos.rename("y_true")

    return TrainedModel(
        name=name,
        estimator=final_est,
        scaler=final_scaler,
        feature_columns=list(X_aligned.columns),
        cv_scores=cv_scores,
        oos_predictions=oos_predictions,
        oos_labels=oos_labels,
        hyperparams=final_est.get_params(),
        fit_date_max=X_dev.index.max(),
        random_state=random_state,
        extras={
            "dev_rows":           int(len(X_dev)),
            "oos_rows":           int(len(X_oos)),
            "dev_pos_rate":       float(y_dev.mean()),
            "oos_pos_rate":       float(y_oos.mean()),
            "forward_window":     int(forward_window),
            "embargo":            int(embargo),
        },
    )


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    forward_window: int = 21,
    embargo: int = 21,
    n_splits: int = 5,
    random_state: int = 0,
) -> Dict[str, TrainedModel]:
    """Train every model in :data:`MODEL_BUILDERS`."""
    out: Dict[str, TrainedModel] = {}
    for name in MODEL_BUILDERS:
        out[name] = train_one_model(
            name, X, y,
            forward_window=forward_window, embargo=embargo,
            n_splits=n_splits, random_state=random_state,
        )
    return out
