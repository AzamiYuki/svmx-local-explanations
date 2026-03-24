"""
Model registry: unified entry point for building target models by name.

This registry intentionally exposes only a minimal default interface for
the repository's experiment runners. If model-specific hyperparameters are
needed, the individual builder functions in each model module should be used
directly.
"""

from __future__ import annotations

from .random_forest import build_random_forest
from .logistic_regression import build_logistic_regression
from .decision_tree import build_decision_tree
from .xgboost_model import build_xgboost

_BUILDERS = {
    "rf": build_random_forest,
    "lr": build_logistic_regression,
    "dt": build_decision_tree,
    "xgb": build_xgboost,
}

SUPPORTED_MODELS = list(_BUILDERS.keys())


def build_model(name: str, random_state: int = 42):
    """
    Return an unfitted sklearn-compatible classifier by short name.

    Supported names: "rf", "lr", "dt", "xgb".

    This function provides a minimal default configuration intended for
    lightweight experiments and repository consistency. More specific
    hyperparameter control should use the model-specific builder directly.

    Raises
    ------
    ValueError
        If *name* is not in SUPPORTED_MODELS.
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model {name!r}. Supported: {SUPPORTED_MODELS}"
        )
    return _BUILDERS[name](random_state=random_state)