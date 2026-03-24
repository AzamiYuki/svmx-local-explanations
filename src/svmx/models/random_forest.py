"""Random Forest target model for binary tabular classification."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with sensible defaults for tabular data.

    Returns an unfitted sklearn estimator that supports predict() and
    predict_proba().
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )