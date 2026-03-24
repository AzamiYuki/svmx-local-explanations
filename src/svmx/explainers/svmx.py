"""
SVM-X: Local explanation via linear SVM surrogates.

Implements Algorithm 1 from Xu et al. (Symmetry 2025).
Given a black-box model M and a target record x_t, SVM-X:
  1. Generates weighted neighbourhood samples around x_t.
  2. Fits a linear SVM surrogate on (samples, M(samples)).
  3. Extracts normalised feature weights as the explanation.

Note: The paper formulates surrogate training as a weighted MSE minimisation
(Eq. 9).  This implementation uses sklearn's SVC with sample_weight, which
optimises a hinge-loss objective instead.  The two are not identical, but
both produce a linear decision boundary influenced by sample proximity.
This is a pragmatic approximation suitable for reproduction purposes.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from typing import Callable, Optional

from .local_sampling import generate_neighbourhood, compute_sample_weights


class SVMXExplainer:
    """Model-agnostic local explainer using a linear SVM surrogate."""

    def __init__(
        self,
        n_samples: int = 2000,
        top_k: int = 5,
        svm_C: float = 1.0,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_samples : int
            Number of perturbed samples to generate around the target record.
        top_k : int
            Number of top features to return in the explanation.
        svm_C : float
            Regularisation parameter for the linear SVM surrogate.
        random_state : int
            Seed for reproducibility of sampling and SVM fitting.
        """
        self.n_samples = n_samples
        self.top_k = top_k
        self.svm_C = svm_C
        self.random_state = random_state

        # Populated after .explain()
        self.surrogate_: Optional[SVC] = None
        self.weights_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        target_record: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
        feature_stats: dict,
    ) -> dict:
        """
        Generate a local explanation for *target_record* under model *predict_fn*.

        Parameters
        ----------
        target_record : np.ndarray, shape (n_features,)
            The instance to explain.
        predict_fn : callable
            Black-box classifier: X (n, d) -> labels (n,).
        predict_proba_fn : callable
            Black-box probability output: X (n, d) -> probabilities (n, C).
        feature_stats : dict
            Per-feature metadata for the sampler.  Required keys:
              "ranges"           : list — (min, max) or array of categories per feature
              "categorical_mask" : bool array, True for one-hot / categorical columns

        Returns
        -------
        dict with keys:
            top_k_indices   : np.ndarray — indices of top-K features by |weight|
            top_k_weights   : np.ndarray — corresponding normalised weights
            all_weights     : np.ndarray — full normalised weight vector
            surrogate       : fitted sklearn SVC object
            neighbours      : np.ndarray — generated neighbourhood (n_samples, d)
            sample_weights  : np.ndarray — distance-based weights (n_samples,)
            surrogate_labels: np.ndarray — black-box labels for the neighbours
        """
        x_t = np.asarray(target_record, dtype=np.float64).ravel()

        # Step 1 — generate neighbourhood (Section 3.1)
        neighbours = generate_neighbourhood(
            x_t,
            n_samples=self.n_samples,
            feature_stats=feature_stats,
            random_state=self.random_state,
        )

        # Step 2 — distance-based sample weights (Eq. 7 and 8)
        sample_weights = compute_sample_weights(
            x_t, neighbours, predict_proba_fn
        )

        # Step 3 — obtain black-box predictions for the neighbourhood
        surrogate_labels = predict_fn(neighbours)

        # Step 4 — fit local SVM surrogate (Section 3.2)
        self.surrogate_ = self._fit_local_surrogate(
            neighbours, surrogate_labels, sample_weights
        )

        # Step 5 — extract and rank feature weights (Section 3.3)
        self.weights_ = self._extract_feature_weights()
        top_idx, top_w = self._select_top_k(self.weights_, self.top_k)

        return {
            "top_k_indices": top_idx,
            "top_k_weights": top_w,
            "all_weights": self.weights_,
            "surrogate": self.surrogate_,
            "neighbours": neighbours,
            "sample_weights": sample_weights,
            "surrogate_labels": surrogate_labels,
        }

    def predict_surrogate(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels with the fitted surrogate."""
        if self.surrogate_ is None:
            raise RuntimeError("Call .explain() before .predict_surrogate().")
        return self.surrogate_.predict(X)

    def predict_proba_surrogate(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with the fitted surrogate."""
        if self.surrogate_ is None:
            raise RuntimeError("Call .explain() before .predict_proba_surrogate().")
        return self.surrogate_.predict_proba(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_local_surrogate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> SVC:
        """
        Train a linear SVM on the neighbourhood, weighted by proximity.

        If the black-box assigns a single class to every neighbour (degenerate
        case), we flip a tiny fraction of labels so sklearn can fit a boundary.
        The flipped samples receive near-zero weight so they barely affect the
        decision surface.
        """
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            n_flip = max(1, len(y) // 50)
            y = y.copy()
            y[:n_flip] = 1 - y[:n_flip]
            sample_weights = sample_weights.copy()
            sample_weights[:n_flip] *= 0.01

        svc = SVC(
            kernel="linear",
            C=self.svm_C,
            probability=True,
            random_state=self.random_state,
            max_iter=5000,
        )
        svc.fit(X, y, sample_weight=sample_weights)
        return svc

    def _extract_feature_weights(self) -> np.ndarray:
        """
        Extract the weight vector from the linear SVM and L1-normalise it.

        For a linear kernel the decision function is f(x) = w^T x + b.
        The raw coefficients w are in svc.coef_.  We normalise by the sum of
        absolute values so weights are interpretable as relative importances
        (positive = pushes toward class 1, negative = toward class 0).
        """
        raw_w = self.surrogate_.coef_.ravel()
        abs_sum = np.abs(raw_w).sum()
        if abs_sum == 0:
            return np.zeros_like(raw_w)
        return raw_w / abs_sum

    @staticmethod
    def _select_top_k(
        weights: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return indices and weights of the top-K features by |weight|."""
        k = min(k, len(weights))
        ranked = np.argsort(np.abs(weights))[::-1]
        top_idx = ranked[:k]
        return top_idx, weights[top_idx]