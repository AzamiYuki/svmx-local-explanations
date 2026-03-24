"""
Tests for src.svmx.explainers.local_sampling.compute_sample_weights

Covers:
  - Weights are strictly positive (exp(-x) > 0 for all finite x)
  - Length matches number of neighbours
  - Mean-normalised: average weight is close to 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.svmx.data.preprocess import preprocess
from src.svmx.explainers.local_sampling import (
    generate_neighbourhood,
    compute_sample_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _build_fixture(n_data: int = 200, n_neighbours: int = 300, seed: int = 0):
    """Return (x_t, neighbours, weights) from a small Adult run."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n_data),
        "fnlwgt": rng.randint(10000, 500000, n_data),
        "education-num": rng.randint(1, 17, n_data),
        "capital-gain": rng.randint(0, 100000, n_data),
        "capital-loss": rng.randint(0, 5000, n_data),
        "hours-per-week": rng.randint(1, 99, n_data),
        "workclass": rng.choice(["Private", "Self-emp", "Gov"], n_data),
        "education": rng.choice(["Bachelors", "Masters", "HS-grad"], n_data),
        "marital-status": rng.choice(["Married", "Single", "Divorced"], n_data),
        "occupation": rng.choice(["Tech", "Sales", "Craft"], n_data),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n_data),
        "race": rng.choice(["White", "Black", "Asian"], n_data),
        "sex": rng.choice(["Male", "Female"], n_data),
        "native-country": rng.choice(["US", "Mexico", "India"], n_data),
        "income": rng.choice(["<=50K", ">50K"], n_data),
    })
    X, y, stats, _ = preprocess(df, dataset_name="adult")

    rf = RandomForestClassifier(n_estimators=30, random_state=seed)
    rf.fit(X, y)

    x_t = X[0]
    neighbours = generate_neighbourhood(
        x_t, n_samples=n_neighbours, feature_stats=stats, random_state=seed
    )
    weights = compute_sample_weights(x_t, neighbours, rf.predict_proba)
    return x_t, neighbours, weights


# ── Tests ─────────────────────────────────────────────────────────────

class TestWeightProperties:

    def test_strictly_positive(self):
        _, _, weights = _build_fixture()
        assert np.all(weights > 0), "Weights must be strictly positive"

    def test_length_matches_neighbours(self):
        n_neighbours = 250
        _, neighbours, weights = _build_fixture(n_neighbours=n_neighbours)
        assert weights.shape == (n_neighbours,)
        assert weights.shape[0] == neighbours.shape[0]

    def test_mean_normalised(self):
        """After mean-normalisation, average weight should be ~1.0."""
        _, _, weights = _build_fixture()
        assert abs(weights.mean() - 1.0) < 1e-6, (
            f"Mean weight = {weights.mean():.6f}, expected ~1.0"
        )

    def test_no_nan_or_inf(self):
        _, _, weights = _build_fixture()
        assert np.all(np.isfinite(weights)), "Weights contain NaN or Inf"