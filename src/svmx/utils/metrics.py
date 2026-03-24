"""
General-purpose metric helpers for binary classification.

These are thin wrappers around sklearn, kept separate from the
surrogate-specific fidelity evaluation in evaluation/fidelity.py.
Use these for target-model evaluation (train/test accuracy, F1)
rather than for surrogate-vs-blackbox comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard binary classification metrics.

    Returns dict with: accuracy, precision, recall, f1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
    }


def probability_mse(
    p_true: np.ndarray,
    p_pred: np.ndarray,
) -> float:
    """
    Mean squared error between two probability arrays.

    Works for both (n,) vectors (single-class probability) and
    (n, C) matrices (full probability distribution).
    """
    return float(np.mean((np.asarray(p_true) - np.asarray(p_pred)) ** 2))