"""
XGBoost target model for binary tabular classification.

If xgboost is unavailable in the current environment, this module falls back
to a RandomForestClassifier so the repository remains runnable. This fallback
is only for lightweight local execution and is not intended for model
benchmarking.
"""


from __future__ import annotations


def build_xgboost(
    n_estimators: int = 100,
    random_state: int = 42,
):
    """
    Build an XGBoost classifier with sensible defaults.

    Returns an unfitted estimator compatible with the sklearn API
    (predict / predict_proba).

    If xgboost is not installed, falls back to RandomForestClassifier so the
    minimal pipeline can still run. This fallback should not be treated as an
    actual XGBoost benchmark result.
    """
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        print("[WARN] xgboost not installed — falling back to RandomForest")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )