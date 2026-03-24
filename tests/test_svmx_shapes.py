"""
Tests for src.svmx.explainers.svmx

Covers:
  - explain() returns all expected dictionary keys
  - Array shapes are mutually consistent
  - top_k length respects the configured k
  - Surrogate is a fitted SVC object
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.svmx.data.preprocess import preprocess
from src.svmx.explainers.svmx import SVMXExplainer


# ── Fixtures ──────────────────────────────────────────────────────────

def _build_fixture(n: int = 200, seed: int = 0, top_k: int = 5, n_samples: int = 300):
    """Return (explainer, result, X, feature_stats) from a small Adult run."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "fnlwgt": rng.randint(10000, 500000, n),
        "education-num": rng.randint(1, 17, n),
        "capital-gain": rng.randint(0, 100000, n),
        "capital-loss": rng.randint(0, 5000, n),
        "hours-per-week": rng.randint(1, 99, n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov"], n),
        "education": rng.choice(["Bachelors", "Masters", "HS-grad"], n),
        "marital-status": rng.choice(["Married", "Single", "Divorced"], n),
        "occupation": rng.choice(["Tech", "Sales", "Craft"], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black", "Asian"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["US", "Mexico", "India"], n),
        "income": rng.choice(["<=50K", ">50K"], n),
    })
    X, y, stats, _ = preprocess(df, dataset_name="adult")

    rf = RandomForestClassifier(n_estimators=30, random_state=seed)
    rf.fit(X, y)

    explainer = SVMXExplainer(
        n_samples=n_samples, top_k=top_k, random_state=seed
    )
    result = explainer.explain(X[0], rf.predict, rf.predict_proba, stats)
    return explainer, result, X, stats


# ── Tests ─────────────────────────────────────────────────────────────

class TestExplainReturnKeys:

    EXPECTED_KEYS = {
        "top_k_indices",
        "top_k_weights",
        "all_weights",
        "surrogate",
        "neighbours",
        "sample_weights",
        "surrogate_labels",
    }

    def test_all_keys_present(self):
        _, result, _, _ = _build_fixture()
        assert set(result.keys()) == self.EXPECTED_KEYS


class TestExplainShapes:

    def test_neighbours_shape(self):
        _, result, X, _ = _build_fixture(n_samples=200)
        assert result["neighbours"].shape == (200, X.shape[1])

    def test_sample_weights_length(self):
        _, result, _, _ = _build_fixture(n_samples=200)
        assert result["sample_weights"].shape == (200,)

    def test_surrogate_labels_length(self):
        _, result, _, _ = _build_fixture(n_samples=200)
        assert result["surrogate_labels"].shape == (200,)

    def test_all_weights_length(self):
        _, result, X, _ = _build_fixture()
        assert result["all_weights"].shape == (X.shape[1],)


class TestTopK:

    def test_top_k_respects_config(self):
        _, result, _, _ = _build_fixture(top_k=3)
        assert len(result["top_k_indices"]) == 3
        assert len(result["top_k_weights"]) == 3

    def test_top_k_clamps_to_feature_count(self):
        """If top_k > n_features, return all features instead of crashing."""
        _, result, X, _ = _build_fixture(top_k=999)
        assert len(result["top_k_indices"]) == X.shape[1]

    def test_top_k_indices_are_valid(self):
        _, result, X, _ = _build_fixture(top_k=5)
        assert all(0 <= i < X.shape[1] for i in result["top_k_indices"])


class TestSurrogateObject:

    def test_surrogate_is_fitted(self):
        explainer, result, X, _ = _build_fixture()
        # A fitted SVC should be able to predict
        preds = explainer.predict_surrogate(X[:5])
        assert preds.shape == (5,)