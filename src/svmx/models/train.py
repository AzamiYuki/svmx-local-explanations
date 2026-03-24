"""
Model training and evaluation helpers.

Provides fit_model() and evaluate_model() for use by experiment runners.
Keeps training logic separate from experiment orchestration.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def fit_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit a sklearn-compatible model and return it.

    Parameters
    ----------
    model : sklearn-compatible estimator (unfitted).
    X_train : np.ndarray, shape (n_samples, n_features)
    y_train : np.ndarray, shape (n_samples,)

    Returns
    -------
    model : the same estimator, now fitted.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    """
    Compute basic classification metrics for a fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator.
    X_eval : np.ndarray, shape (n_samples, n_features)
    y_eval : np.ndarray, shape (n_samples,)

    Returns
    -------
    metrics : dict with "accuracy" and "f1".
    """
    y_pred = model.predict(X_eval)
    return {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0.0)),
    }