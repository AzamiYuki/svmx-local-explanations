"""
Preprocessing utilities for tabular datasets (UCI Adult, Bank Marketing).

Handles:
  - Missing value imputation (mode for categorical, median for continuous).
  - One-hot encoding of categorical columns.
  - Min-max normalisation of continuous columns.
  - Train/test column alignment via the *expected_columns* parameter.
  - Feature-stats extraction for the SVM-X sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


# ── Column definitions ────────────────────────────────────────────────

ADULT_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
ADULT_CONTINUOUS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]
ADULT_TARGET = "income"

BANK_CATEGORICAL = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
BANK_CONTINUOUS = [
    "age", "balance", "day", "duration", "campaign",
    "pdays", "previous",
]
BANK_TARGET = "y"


# ── Public API ────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    dataset_name: str = "adult",
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
    expected_columns: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, dict, MinMaxScaler]:
    """
    End-to-end preprocessing: impute -> encode -> align -> normalise.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (must include the target column).
    dataset_name : {"adult", "bank"}
        Selects column definitions.
    scaler : MinMaxScaler or None
        Pass the training scaler when processing test data.
    fit_scaler : bool
        True for training data, False for test data.
    expected_columns : list[str] or None
        Column order from the training set.  When provided the encoded
        dataframe is aligned to these columns (missing columns filled
        with 0, extra columns dropped).  This prevents one-hot mismatch
        between train and test splits.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_encoded_features)
    y : np.ndarray, shape (n_samples,)
    feature_stats : dict
        Metadata for the SVM-X sampler:
          "ranges"           : list — (min, max) or unique values per column
          "categorical_mask" : np.ndarray of bool
          "feature_names"    : list of str
    scaler : fitted MinMaxScaler
    """
    cat_cols, cont_cols, target_col = _get_column_defs(dataset_name)
    df = df.copy()

    # 1. Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Handle missing values
    df = _impute_missing(df, cat_cols, cont_cols)

    # 3. Separate target
    y = _encode_target(df[target_col], dataset_name)

    # 4. One-hot encode categoricals
    feature_cols = [c for c in cat_cols + cont_cols if c in df.columns]
    df_encoded = pd.get_dummies(df[feature_cols], columns=cat_cols, dtype=float)

    # 5. Align columns to training set (avoids one-hot mismatch)
    if expected_columns is not None:
        df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0.0)

    # 6. Normalise continuous features
    if scaler is None:
        scaler = MinMaxScaler()
    cont_present = [c for c in cont_cols if c in df_encoded.columns]
    if fit_scaler:
        df_encoded[cont_present] = scaler.fit_transform(df_encoded[cont_present])
    else:
        df_encoded[cont_present] = scaler.transform(df_encoded[cont_present])

    feature_names = list(df_encoded.columns)
    X = df_encoded.values.astype(np.float64)

    # 7. Build feature stats for the sampler
    feature_stats = _build_feature_stats(X, feature_names, cat_cols)

    return X, y, feature_stats, scaler


# ── Internals ─────────────────────────────────────────────────────────

def _get_column_defs(name: str):
    if name == "adult":
        return ADULT_CATEGORICAL, ADULT_CONTINUOUS, ADULT_TARGET
    elif name == "bank":
        return BANK_CATEGORICAL, BANK_CONTINUOUS, BANK_TARGET
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose 'adult' or 'bank'.")


def _impute_missing(
    df: pd.DataFrame,
    cat_cols: list[str],
    cont_cols: list[str],
) -> pd.DataFrame:
    """Replace '?' and NaN: mode for categorical, median for continuous."""
    df = df.replace("?", np.nan)

    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    for col in cont_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def _encode_target(series: pd.Series, dataset_name: str) -> np.ndarray:
    """Binary-encode the target column."""
    if dataset_name == "adult":
        return (series.str.strip().str.startswith(">50K")).astype(int).values
    elif dataset_name == "bank":
        return (series.str.lower() == "yes").astype(int).values
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


def _build_feature_stats(
    X: np.ndarray,
    feature_names: list[str],
    original_cat_cols: list[str],
) -> dict:
    """
    Build sampler metadata from the final encoded matrix.

    One-hot columns (whose name starts with an original categorical column
    prefix) are marked as categorical with range [0, 1].  Continuous columns
    store their observed (min, max).
    """
    n_features = X.shape[1]
    cat_mask = np.zeros(n_features, dtype=bool)
    ranges: list = []

    for i, name in enumerate(feature_names):
        is_ohe = any(name.startswith(cat + "_") for cat in original_cat_cols)
        if is_ohe:
            cat_mask[i] = True
            ranges.append(np.array([0.0, 1.0]))
        else:
            cat_mask[i] = False
            col_min, col_max = float(X[:, i].min()), float(X[:, i].max())
            ranges.append((col_min, col_max))

    return {
        "ranges": ranges,
        "categorical_mask": cat_mask,
        "feature_names": feature_names,
    }