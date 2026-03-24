"""
Tests for src.svmx.data.preprocess

Covers:
  - No missing values after preprocessing
  - Output shape consistency
  - Reproducibility (same seed -> identical output)
  - Binary target encoding
  - Normalisation range for continuous features
  - Scaler reuse between train and test
  - Column alignment via expected_columns
"""

import numpy as np
import pandas as pd

from src.svmx.data.preprocess import preprocess


# ── Synthetic data factories ──────────────────────────────────────────

def _make_adult_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic Adult-like dataframe."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "fnlwgt": rng.randint(10000, 500000, n),
        "education-num": rng.randint(1, 17, n),
        "capital-gain": rng.randint(0, 100000, n),
        "capital-loss": rng.randint(0, 5000, n),
        "hours-per-week": rng.randint(1, 99, n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov", "?"], n),
        "education": rng.choice(["Bachelors", "Masters", "HS-grad"], n),
        "marital-status": rng.choice(["Married", "Single", "Divorced"], n),
        "occupation": rng.choice(["Tech", "Sales", "?", "Craft"], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black", "Asian"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["US", "Mexico", "India"], n),
        "income": rng.choice(["<=50K", ">50K"], n),
    })


def _make_bank_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic Bank-Marketing-like dataframe."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "balance": rng.randint(-5000, 100000, n),
        "day": rng.randint(1, 31, n),
        "duration": rng.randint(0, 4000, n),
        "campaign": rng.randint(1, 50, n),
        "pdays": rng.choice([-1, 30, 90, 180], n),
        "previous": rng.randint(0, 10, n),
        "job": rng.choice(["admin.", "technician", "services"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": rng.choice(["yes", "no"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "month": rng.choice(["jan", "feb", "mar", "apr"], n),
        "poutcome": rng.choice(["success", "failure", "unknown"], n),
        "y": rng.choice(["yes", "no"], n),
    })


# ── Adult dataset tests ───────────────────────────────────────────────

class TestAdultPreprocessing:

    def test_no_missing_values(self):
        df = _make_adult_df()
        X, y, _, _ = preprocess(df, dataset_name="adult")
        assert not np.isnan(X).any(), "X contains NaN after preprocessing"
        assert not np.isnan(y).any(), "y contains NaN after preprocessing"

    def test_output_shapes(self):
        n = 150
        df = _make_adult_df(n=n)
        X, y, stats, _ = preprocess(df, dataset_name="adult")
        assert X.shape[0] == n
        assert y.shape == (n,)
        assert X.shape[1] == len(stats["feature_names"])

    def test_reproducibility(self):
        df = _make_adult_df(seed=42)
        X1, y1, _, _ = preprocess(df, dataset_name="adult")
        X2, y2, _, _ = preprocess(df, dataset_name="adult")
        np.testing.assert_array_equal(X1, X2, err_msg="X differs across runs")
        np.testing.assert_array_equal(y1, y2, err_msg="y differs across runs")

    def test_target_encoding(self):
        df = _make_adult_df()
        _, y, _, _ = preprocess(df, dataset_name="adult")
        assert set(np.unique(y)).issubset({0, 1})

    def test_normalisation_range(self):
        df = _make_adult_df()
        X, _, stats, _ = preprocess(df, dataset_name="adult")
        cat_mask = stats["categorical_mask"]
        cont_cols = X[:, ~cat_mask]
        assert cont_cols.min() >= -1e-9, "Continuous feature below 0"
        assert cont_cols.max() <= 1.0 + 1e-9, "Continuous feature above 1"


# ── Bank dataset tests ────────────────────────────────────────────────

class TestBankPreprocessing:

    def test_no_missing_values(self):
        df = _make_bank_df()
        X, y, _, _ = preprocess(df, dataset_name="bank")
        assert not np.isnan(X).any()

    def test_output_shapes(self):
        n = 100
        df = _make_bank_df(n=n)
        X, y, stats, _ = preprocess(df, dataset_name="bank")
        assert X.shape[0] == n
        assert y.shape == (n,)

    def test_feature_stats_structure(self):
        df = _make_bank_df()
        _, _, stats, _ = preprocess(df, dataset_name="bank")
        assert "ranges" in stats
        assert "categorical_mask" in stats
        assert "feature_names" in stats
        assert len(stats["ranges"]) == len(stats["feature_names"])


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_missing_value_imputation(self):
        """Ensure '?' values in Adult are properly replaced."""
        df = _make_adult_df(n=50)
        df.loc[0, "workclass"] = "?"
        df.loc[1, "occupation"] = "?"
        X, _, _, _ = preprocess(df, dataset_name="adult")
        assert not np.isnan(X).any(), "'?' values not imputed"

    def test_scaler_reuse(self):
        """Train scaler on train split, reuse on test split."""
        df_train = _make_adult_df(n=300, seed=1)
        df_test = _make_adult_df(n=100, seed=2)
        X_train, _, stats_train, scaler = preprocess(
            df_train, dataset_name="adult", fit_scaler=True
        )
        train_columns = stats_train["feature_names"]
        X_test, _, _, _ = preprocess(
            df_test, dataset_name="adult",
            scaler=scaler, fit_scaler=False,
            expected_columns=train_columns,
        )
        assert X_test.shape[1] == X_train.shape[1], "Feature count mismatch"

    def test_expected_columns_alignment(self):
        """Column alignment should add missing columns and drop extras."""
        df_train = _make_adult_df(n=200, seed=10)
        X_train, _, stats_train, scaler = preprocess(
            df_train, dataset_name="adult", fit_scaler=True
        )
        train_cols = stats_train["feature_names"]

        # Build a test set that may have different one-hot columns
        df_test = _make_adult_df(n=50, seed=99)
        X_test, _, _, _ = preprocess(
            df_test, dataset_name="adult",
            scaler=scaler, fit_scaler=False,
            expected_columns=train_cols,
        )
        assert list(stats_train["feature_names"]) == train_cols
        assert X_test.shape[1] == len(train_cols)