"""Logistic Regression target model for binary tabular classification."""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    random_state: int = 42,
) -> LogisticRegression:
    """
    Build a Logistic Regression classifier with sensible defaults.

    max_iter is set high enough for one-hot-encoded tabular datasets
    (Adult ~100 features) to converge reliably.
    """
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )