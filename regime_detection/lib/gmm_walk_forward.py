"""Walk-forward Gaussian Mixture Model regime detector.

Algorithmic skeleton extracted from v4 ``models/gmm_regime.py:224-427``
(``AdaptiveGMMDetector.fit_walk_forward``), reimplemented minimally:

* uses only Phase 1 walk-forward utilities + sklearn primitives;
* drops the v4 anchor-based labelling (which mixed full-sample anchor
  episodes into the regime mapping — replaced here by Hungarian-algorithm
  alignment across refits);
* drops the v4 economic post-processing (regime names, transition
  matrices, per-factor stats) — the Phase 9B comparison consumes only
  integer labels as features;
* fixes ``k=5`` per window (no per-window k-selection), matching v4's
  effective behaviour and removing one source of variance.

Causality contract: at refit date ``t``, the GMM is fit on
``X.iloc[:t]`` only.  The fitted scaler, PCA and GMM are then used to
predict labels for ``X.iloc[t:t + refit_freq]``.  No future data appears
in any per-window fit — the Phase 0 audit specifically confirmed
``v4 gmm_regime.py:306-315`` is clean; this reimplementation preserves
that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from regime_detection.lib.label_alignment import (
    align_labels_across_refits,
)
from regime_detection.lib.splits import assert_no_oos_contamination


@dataclass
class WalkForwardGMMResult:
    """Outputs of one walk-forward GMM run."""

    labels: pd.Series                                # integer 0..k-1 per row
    probabilities: pd.DataFrame                       # columns prob_0..prob_{k-1}
    cluster_means_history: List[np.ndarray]            # one (k, d_pca) per refit
    refit_dates: List[pd.Timestamp]                   # refit cut-over dates
    permutations_applied: List[dict]                  # alignment perm per refit
    n_components: int
    pca_variance_kept: float


class WalkForwardGMM:
    """Minimal walk-forward GMM regime detector.

    Parameters
    ----------
    n_components :
        Number of clusters per refit.  Default ``5`` matches v4.
    refit_freq :
        Number of rows between refits (default ``252`` ≈ 1 year).
    min_train :
        Minimum rows required before the first refit (default ``756``
        ≈ 3 years).
    pca_variance :
        Fraction of variance retained by the per-window PCA
        (default ``0.95`` — matches v4's choice).
    covariance_type :
        Forwarded to :class:`sklearn.mixture.GaussianMixture` (default
        ``"full"`` — matches the corrected v4 choice from Phase 0).
    n_init :
        Number of EM restarts per fit.
    max_iter :
        EM iterations cap.
    reg_covar :
        Covariance regularisation, prevents singular covariances on
        small clusters.
    random_state :
        Forwarded to PCA and GMM; controls EM initialisation.
    align_labels :
        If ``True`` (default), apply Hungarian-algorithm alignment across
        refits so the same regime keeps the same integer label.
    """

    def __init__(
        self,
        n_components: int = 5,
        refit_freq: int = 252,
        min_train: int = 756,
        pca_variance: float = 0.95,
        covariance_type: str = "full",
        n_init: int = 5,
        max_iter: int = 200,
        reg_covar: float = 1e-3,
        random_state: int = 0,
        align_labels: bool = True,
    ) -> None:
        if n_components < 2:
            raise ValueError(f"n_components must be >= 2, got {n_components}")
        if refit_freq < 1:
            raise ValueError(f"refit_freq must be >= 1, got {refit_freq}")
        if min_train < n_components * 10:
            raise ValueError(
                f"min_train ({min_train}) too small for n_components "
                f"({n_components})"
            )
        if not 0.0 < pca_variance <= 1.0:
            raise ValueError(f"pca_variance must lie in (0, 1]")
        self.n_components = int(n_components)
        self.refit_freq = int(refit_freq)
        self.min_train = int(min_train)
        self.pca_variance = float(pca_variance)
        self.covariance_type = covariance_type
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.align_labels = bool(align_labels)

    # ------------------------------------------------------------------

    def fit_transform(self, X: pd.DataFrame) -> WalkForwardGMMResult:
        """Run walk-forward refits over ``X`` and return integer labels +
        per-cluster probabilities for the entire OOS region."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must have a DatetimeIndex")
        X_clean = X.dropna()
        if len(X_clean) < self.min_train + self.refit_freq:
            raise ValueError(
                f"Not enough rows for walk-forward: have {len(X_clean)}, "
                f"need at least {self.min_train + self.refit_freq}"
            )

        n = len(X_clean)
        refit_starts = list(range(self.min_train, n, self.refit_freq))

        label_chunks: List[pd.Series] = []
        prob_chunks: List[pd.DataFrame] = []
        cluster_means_history: List[np.ndarray] = []
        refit_dates: List[pd.Timestamp] = []

        for win_idx, start in enumerate(refit_starts):
            pred_end = min(start + self.refit_freq, n)
            X_train = X_clean.iloc[:start]
            X_pred = X_clean.iloc[start:pred_end]
            if len(X_train) == 0 or len(X_pred) == 0:
                continue

            # Hard guard — refuse to fit on anything that includes the
            # OOS predict window's first row or later.
            assert_no_oos_contamination(
                X_train.index,
                max_fit_date=X_clean.index[start - 1],
                label=f"WalkForwardGMM[win={win_idx}]",
            )

            scaler = StandardScaler().fit(X_train.values)
            X_train_s = scaler.transform(X_train.values)
            X_pred_s = scaler.transform(X_pred.values)

            pca = PCA(n_components=self.pca_variance,
                      random_state=self.random_state)
            X_train_pca = pca.fit_transform(X_train_s)
            X_pred_pca = pca.transform(X_pred_s)

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                random_state=self.random_state + win_idx,
            )
            gmm.fit(X_train_pca)

            raw_labels = gmm.predict(X_pred_pca)
            raw_probs = gmm.predict_proba(X_pred_pca)

            label_chunks.append(pd.Series(
                raw_labels.astype(int),
                index=X_pred.index,
                name="regime",
            ))
            prob_chunks.append(pd.DataFrame(
                raw_probs,
                index=X_pred.index,
                columns=[f"prob_{j}" for j in range(self.n_components)],
            ))
            # Cluster means in the PCA space — that's the space the
            # Hungarian alignment runs in (per-refit consistent).
            cluster_means_history.append(gmm.means_.copy())
            refit_dates.append(pd.Timestamp(X_clean.index[start - 1]))

        if not label_chunks:
            raise RuntimeError(
                "No refit windows produced — check min_train vs sample size"
            )

        # ---- Hungarian alignment across refits ---------------------
        if self.align_labels and len(label_chunks) > 1:
            # The PCA-space means change shape across refits because PCA
            # n_components depends on the training window.  Hungarian
            # alignment requires equal-shape means: pad / truncate to
            # the common minimum number of PCA components per pair.
            min_pca = min(m.shape[1] for m in cluster_means_history)
            trimmed_means = [m[:, :min_pca] for m in cluster_means_history]
            aligned_chunks, perms = align_labels_across_refits(
                label_chunks, trimmed_means,
            )
            # Also remap the probability column order to match the new
            # label convention.
            aligned_probs: List[pd.DataFrame] = [prob_chunks[0].copy()]
            for w in range(1, len(prob_chunks)):
                perm = perms[w]
                # perm maps OLD label -> NEW label.  Re-order columns so
                # column `prob_{new_label}` is the OLD `prob_{old_label}`.
                k = self.n_components
                new_cols = [None] * k
                for old_j, new_j in perm.items():
                    new_cols[new_j] = prob_chunks[w][f"prob_{old_j}"].values
                aligned_probs.append(pd.DataFrame(
                    np.column_stack(new_cols),
                    index=prob_chunks[w].index,
                    columns=[f"prob_{j}" for j in range(k)],
                ))
        else:
            aligned_chunks = label_chunks
            aligned_probs = prob_chunks
            perms = [{j: j for j in range(self.n_components)}
                     for _ in label_chunks]

        labels_all = pd.concat(aligned_chunks).rename("gmm_regime")
        probs_all = pd.concat(aligned_probs)
        return WalkForwardGMMResult(
            labels=labels_all,
            probabilities=probs_all,
            cluster_means_history=cluster_means_history,
            refit_dates=refit_dates,
            permutations_applied=perms,
            n_components=self.n_components,
            pca_variance_kept=self.pca_variance,
        )


def fit_walk_forward_gmm(X: pd.DataFrame, **kwargs) -> WalkForwardGMMResult:
    """Convenience wrapper: ``WalkForwardGMM(**kwargs).fit_transform(X)``."""
    return WalkForwardGMM(**kwargs).fit_transform(X)
