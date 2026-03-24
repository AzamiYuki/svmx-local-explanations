"""
Local fidelity evaluation for explanation methods.

Measures how faithfully a surrogate model reproduces the black-box model's
behaviour in the neighbourhood of a target record.  Implements the metrics
from Section 4.1.3 of Xu et al. (Symmetry 2025):
  - Accuracy  : fraction of local samples with matching predicted class.
  - Recall    : recall of the target-record class among local samples.
  - F1 Score  : harmonic mean of precision and recall.
  - MSE       : mean squared error between predicted probabilities (Eq. 10).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from typing import Callable


def evaluate_fidelity(
    neighbours: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    surrogate_predict_fn: Callable[[np.ndarray], np.ndarray],
    predict_proba_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    surrogate_proba_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """
    Compute fidelity metrics between a black-box and its local surrogate.

    The black-box predictions on *neighbours* serve as the ground truth;
    the surrogate predictions are the values under test.

    Parameters
    ----------
    neighbours : np.ndarray, shape (n_samples, n_features)
        Perturbed samples in the neighbourhood of the target record.
    predict_fn : callable
        Black-box model: X -> class labels.
    surrogate_predict_fn : callable
        Surrogate model: X -> class labels.
    predict_proba_fn : callable or None
        Black-box: X -> (n, C) probability matrix.  Required for MSE.
    surrogate_proba_fn : callable or None
        Surrogate: X -> (n, C) probability matrix.  Required for MSE.

    Returns
    -------
    metrics : dict[str, float]
        Always contains "accuracy", "recall", "f1".
        Contains "mse" only when both probability functions are provided.
    """
    y_blackbox = predict_fn(neighbours)
    y_surrogate = surrogate_predict_fn(neighbours)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_blackbox, y_surrogate)),
        "recall": float(recall_score(y_blackbox, y_surrogate, zero_division=0.0)),
        "f1": float(f1_score(y_blackbox, y_surrogate, zero_division=0.0)),
    }

    if predict_proba_fn is not None and surrogate_proba_fn is not None:
        p_bb = predict_proba_fn(neighbours)
        p_surr = surrogate_proba_fn(neighbours)
        metrics["mse"] = float(np.mean((p_bb - p_surr) ** 2))

    return metrics


def summarize_metrics(metrics: dict[str, float]) -> str:
    """
    Format a metrics dict as a human-readable one-line summary.

    Example output:
        "accuracy=0.912 | f1=0.934 | recall=0.958 | mse=0.048"
    """
    parts = []
    for key in ("accuracy", "f1", "recall", "mse"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)


def evaluate_weight_stability(
    target_record: np.ndarray,
    explain_fn: Callable[[np.ndarray], np.ndarray],
    n_neighbours: int = 100,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> float:
    """
    Weighted mean variance of explanations (Eq. 11 in the paper).

    Generates *n_neighbours* slight perturbations of the target record,
    obtains a full explanation weight vector for each, and returns the mean
    per-feature variance.  Lower values indicate more stable explanations.

    This function is intended for post-hoc stability analysis.  It requires
    a callable *explain_fn* that accepts a single record and returns a
    weight vector (np.ndarray of shape (n_features,)).  In practice this
    callable wraps SVMXExplainer.explain() and extracts "all_weights".

    Parameters
    ----------
    target_record : np.ndarray, shape (n_features,)
    explain_fn : callable
        record (n_features,) -> weight vector (n_features,).
    n_neighbours : int
        Number of perturbed copies to evaluate.
    noise_scale : float
        Std-dev of Gaussian noise added to create perturbations.
    random_state : int

    Returns
    -------
    mean_var : float
        Mean per-feature variance across the perturbed explanations.
    """
    rng = np.random.RandomState(random_state)
    x_t = target_record.ravel()
    d = x_t.shape[0]

    weight_matrix = np.empty((n_neighbours, d))
    for i in range(n_neighbours):
        perturbed = x_t + rng.normal(0, noise_scale, size=d)
        weight_matrix[i] = explain_fn(perturbed)

    per_feature_var = np.var(weight_matrix, axis=0)
    return float(np.mean(per_feature_var))