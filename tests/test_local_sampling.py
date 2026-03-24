"""
Neighbourhood sampling and distance-based weighting for SVM-X.

Implements the local sample generation strategy from Section 3.1:
  - Perturb a random subset of features in x_t to create each neighbour.
  - Weight each neighbour using the prediction-probability distance (Eq. 7-8).

Designed for one-hot-encoded tabular data.  Categorical features are handled
as GROUPS: when perturbing a one-hot group, exactly one column is set to 1
and all others to 0, so every generated sample represents a valid category.

NOTE ON INDEPENDENT SAMPLING (why we need groups):
  Treating each one-hot column independently (e.g. flipping each to 0 or 1)
  produces invalid feature vectors — a sample could have multiple 1s in the
  same categorical group (encoding two categories simultaneously) or all 0s
  (encoding no category).  These vectors lie outside the data manifold and
  degrade the surrogate's approximation of the true local decision boundary.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from typing import Callable


# ------------------------------------------------------------------
# One-hot group inference
# ------------------------------------------------------------------

def build_one_hot_groups(
    feature_names: list[str],
    categorical_mask: np.ndarray,
) -> dict[str, list[int]]:
    """
    Infer one-hot column groups from encoded feature names.

    pd.get_dummies produces names like "education_Bachelors",
    "marital-status_Married".  The group key is the part before the last
    underscore (i.e. the original column name).

    Parameters
    ----------
    feature_names : list[str]
        Column names of the encoded feature matrix.
    categorical_mask : np.ndarray of bool
        True for columns that are one-hot indicators.

    Returns
    -------
    groups : dict[str, list[int]]
        Mapping from group prefix to sorted list of column indices.
        Only groups with ≥ 2 columns are included (a lone binary column
        does not need group-aware sampling).
    """
    prefix_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, name in enumerate(feature_names):
        if not categorical_mask[i]:
            continue
        # Split at the last underscore to recover the original column name.
        # e.g. "marital-status_Married" -> prefix "marital-status"
        if "_" in name:
            prefix = name.rsplit("_", 1)[0]
        else:
            prefix = name
        prefix_to_indices[prefix].append(i)

    # Only keep groups with 2+ columns (single binary columns are fine as-is)
    return {k: sorted(v) for k, v in prefix_to_indices.items() if len(v) >= 2}


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_feature_stats(feature_stats: dict, n_features: int) -> None:
    """
    Check that *feature_stats* has the required keys and consistent lengths.

    Raises ValueError with a descriptive message on any mismatch.
    """
    required_keys = {"ranges", "categorical_mask"}
    missing = required_keys - set(feature_stats.keys())
    if missing:
        raise ValueError(f"feature_stats is missing keys: {missing}")

    if len(feature_stats["ranges"]) != n_features:
        raise ValueError(
            f"len(ranges)={len(feature_stats['ranges'])} != n_features={n_features}"
        )
    cat_mask = np.asarray(feature_stats["categorical_mask"])
    if cat_mask.shape[0] != n_features:
        raise ValueError(
            f"len(categorical_mask)={cat_mask.shape[0]} != n_features={n_features}"
        )


# ------------------------------------------------------------------
# Neighbourhood generation
# ------------------------------------------------------------------

def generate_neighbourhood(
    x_t: np.ndarray,
    n_samples: int,
    feature_stats: dict,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate *n_samples* perturbed copies of the target record *x_t*.

    Perturbation strategy:
      1. Build the set of "perturbable units": each continuous feature is
         one unit; each one-hot group is one unit.
      2. For each sample, randomly choose how many units to perturb and
         which ones.
      3. Continuous units: draw uniformly from [min, max].
         Categorical (group) units: activate exactly one column in the
         group uniformly at random, zero out the rest.

    Parameters
    ----------
    x_t : np.ndarray, shape (n_features,)
    n_samples : int
    feature_stats : dict
        "ranges"           : list — (min, max) for continuous features.
        "categorical_mask" : bool array, True for one-hot columns.
        "feature_names"    : list of str (needed for group inference).
    random_state : int

    Returns
    -------
    neighbours : np.ndarray, shape (n_samples, n_features)
    """
    rng = np.random.RandomState(random_state)
    d = x_t.shape[0]

    validate_feature_stats(feature_stats, d)

    ranges = feature_stats["ranges"]
    cat_mask = np.asarray(feature_stats["categorical_mask"])
    feature_names = feature_stats.get("feature_names")

    # --- Build perturbable units ---
    # Each unit is either ("continuous", index) or ("group", [indices]).
    ohe_groups: dict[str, list[int]] = {}
    if feature_names is not None:
        ohe_groups = build_one_hot_groups(feature_names, cat_mask)

    # Track which feature indices are covered by a group
    grouped_indices: set[int] = set()
    for indices in ohe_groups.values():
        grouped_indices.update(indices)

    units: list[tuple] = []
    for j in range(d):
        if j in grouped_indices:
            continue  # handled as part of its group
        if cat_mask[j]:
            # Lone binary indicator (no group partner) — treat as binary unit
            units.append(("binary", j))
        else:
            units.append(("continuous", j))

    for prefix, indices in ohe_groups.items():
        units.append(("group", indices))

    n_units = len(units)

    # --- Generate samples ---
    neighbours = np.tile(x_t, (n_samples, 1)).astype(np.float64)

    for i in range(n_samples):
        n_perturb = rng.randint(1, n_units + 1)
        chosen = rng.choice(n_units, size=n_perturb, replace=False)

        for u in chosen:
            kind = units[u][0]
            payload = units[u][1]

            if kind == "continuous":
                j = payload
                lo, hi = ranges[j]
                if lo < hi:
                    neighbours[i, j] = rng.uniform(lo, hi)

            elif kind == "binary":
                j = payload
                neighbours[i, j] = float(rng.randint(0, 2))

            elif kind == "group":
                col_indices = payload
                # Zero out the whole group, then activate exactly one
                for idx in col_indices:
                    neighbours[i, idx] = 0.0
                active = rng.choice(col_indices)
                neighbours[i, active] = 1.0

    return neighbours


# ------------------------------------------------------------------
# Distance-based weighting (Eq. 7-8)
# ------------------------------------------------------------------

def compute_sample_weights(
    x_t: np.ndarray,
    neighbours: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Assign importance weights based on the paper's probability-distance metric.

    Distance (Eq. 8):
        Dist = (||P_neighbour - P_boundary||_2 - ||P_xt - P_boundary||_2)^2
    where P_boundary = (1/C, ..., 1/C) for a C-class problem.

    Weight (Eq. 7):
        pi(neighbour) = exp(-Dist)

    Neighbours whose predicted probability is similar to x_t's receive higher
    weight, focusing the surrogate on the locally relevant region.

    Parameters
    ----------
    x_t : np.ndarray, shape (n_features,)
    neighbours : np.ndarray, shape (n_samples, n_features)
    predict_proba_fn : callable
        Black-box probability output: X (n, d) -> (n, C).

    Returns
    -------
    weights : np.ndarray, shape (n_samples,), mean-normalised.
    """
    p_xt = predict_proba_fn(x_t.reshape(1, -1))       # (1, C)
    p_neighbours = predict_proba_fn(neighbours)         # (N, C)

    n_classes = p_xt.shape[1]
    p_boundary = np.full(n_classes, 1.0 / n_classes)

    dist_xt = np.linalg.norm(p_xt.ravel() - p_boundary)
    dist_neighbours = np.linalg.norm(p_neighbours - p_boundary, axis=1)

    dist = (dist_neighbours - dist_xt) ** 2   # Eq. 8
    weights = np.exp(-dist)                    # Eq. 7

    # Mean-normalise so the total weight scale matches n_samples
    weights = weights / weights.mean()
    return weights


# ------------------------------------------------------------------
# Fallback: Euclidean-kernel weighting (LIME-style, for comparison)
# ------------------------------------------------------------------

def compute_euclidean_weights(
    x_t: np.ndarray,
    neighbours: np.ndarray,
    kernel_width: float = 0.75,
) -> np.ndarray:
    """
    Euclidean-distance exponential kernel, similar to LIME's default.

    Useful as a baseline or when the black-box does not expose probabilities.
    """
    dists = np.linalg.norm(neighbours - x_t, axis=1)
    weights = np.sqrt(np.exp(-(dists ** 2) / (kernel_width ** 2)))
    return weights