"""
Neighbourhood sampling and distance-based weighting for SVM-X.

Implements the local sample generation strategy from Section 3.1:
  - Perturb a random subset of features in x_t to create each neighbour.
  - Weight each neighbour using the prediction-probability distance (Eq. 7-8).

Designed for one-hot-encoded tabular data. Categorical features are handled
as groups: when perturbing a one-hot group, exactly one column is set to 1
and all others to 0, so every generated sample remains a valid category.
"""
# Updated: Sample exactly one active column per one-hot group to prevent invalid states.
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


def build_one_hot_groups(
    feature_names: list[str],
    categorical_mask: np.ndarray,
) -> dict[str, list[int]]:
    """
    Infer one-hot groups from encoded feature names produced by pd.get_dummies.

    Example:
        education_Bachelors
        education_Masters
    -> group key: "education"
    """
    prefix_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, name in enumerate(feature_names):
        if not categorical_mask[i]:
            continue
        if "_" in name:
            prefix = name.rsplit("_", 1)[0]
        else:
            prefix = name
        prefix_to_indices[prefix].append(i)

    return {k: sorted(v) for k, v in prefix_to_indices.items() if len(v) >= 2}


def validate_feature_stats(feature_stats: dict, n_features: int) -> None:
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


def generate_neighbourhood(
    x_t: np.ndarray,
    n_samples: int,
    feature_stats: dict,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate n_samples perturbed copies of x_t.

    Strategy:
      1. Build perturbable units:
         - one continuous feature = one unit
         - one one-hot group = one unit
         - lone binary indicator = one unit
      2. Randomly choose how many units to perturb
      3. Continuous units: sample uniformly in [min, max]
         Group units: activate exactly one column in the group
         Binary units: sample 0/1
    """
    rng = np.random.RandomState(random_state)
    d = x_t.shape[0]

    validate_feature_stats(feature_stats, d)

    ranges = feature_stats["ranges"]
    cat_mask = np.asarray(feature_stats["categorical_mask"])
    feature_names = feature_stats.get("feature_names")

    ohe_groups: dict[str, list[int]] = {}
    if feature_names is not None:
        ohe_groups = build_one_hot_groups(feature_names, cat_mask)

    grouped_indices: set[int] = set()
    for indices in ohe_groups.values():
        grouped_indices.update(indices)

    units: list[tuple[str, object]] = []

    for j in range(d):
        if j in grouped_indices:
            continue
        if cat_mask[j]:
            units.append(("binary", j))
        else:
            units.append(("continuous", j))

    for _, indices in ohe_groups.items():
        units.append(("group", indices))

    n_units = len(units)
    neighbours = np.tile(x_t, (n_samples, 1)).astype(np.float64)

    for i in range(n_samples):
        n_perturb = rng.randint(1, n_units + 1)
        chosen = rng.choice(n_units, size=n_perturb, replace=False)

        for unit_id in chosen:
            kind, payload = units[unit_id]

            if kind == "continuous":
                j = int(payload)
                lo, hi = ranges[j]
                if lo < hi:
                    neighbours[i, j] = rng.uniform(lo, hi)

            elif kind == "binary":
                j = int(payload)
                neighbours[i, j] = float(rng.randint(0, 2))

            elif kind == "group":
                col_indices = payload
                for idx in col_indices:
                    neighbours[i, idx] = 0.0
                active = rng.choice(col_indices)
                neighbours[i, active] = 1.0

    return neighbours


def compute_sample_weights(
    x_t: np.ndarray,
    neighbours: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Distance-based weighting from Eq. 7-8 in the paper.
    """
    p_xt = predict_proba_fn(x_t.reshape(1, -1))
    p_neighbours = predict_proba_fn(neighbours)

    n_classes = p_xt.shape[1]
    p_boundary = np.full(n_classes, 1.0 / n_classes)

    dist_xt = np.linalg.norm(p_xt.ravel() - p_boundary)
    dist_neighbours = np.linalg.norm(p_neighbours - p_boundary, axis=1)

    dist = (dist_neighbours - dist_xt) ** 2
    weights = np.exp(-dist)

    weights = weights / weights.mean()
    return weights


def compute_euclidean_weights(
    x_t: np.ndarray,
    neighbours: np.ndarray,
    kernel_width: float = 0.75,
) -> np.ndarray:
    """
    LIME-style Euclidean kernel weighting baseline.
    """
    dists = np.linalg.norm(neighbours - x_t, axis=1)
    weights = np.sqrt(np.exp(-(dists ** 2) / (kernel_width ** 2)))
    return weights
