"""
Neighbourhood sampling and distance-based weighting for SVM-X.

Implements the local sample generation strategy from Section 3.1:
  - Perturb a random subset of features in x_t to create each neighbour.
  - Weight each neighbour using the prediction-probability distance (Eq. 7-8).

Designed for one-hot-encoded tabular data: categorical indicators are sampled
as binary {0, 1}, continuous columns within their observed [min, max] range.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


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

    For each sample:
      1. Randomly choose how many features to perturb (uniform in [1, d]).
      2. Select that many features uniformly at random.
      3. For categorical (one-hot) columns: flip to 0 or 1 with equal probability.
         For continuous columns: draw uniformly from [min, max].

    Parameters
    ----------
    x_t : np.ndarray, shape (n_features,)
    n_samples : int
    feature_stats : dict
        "ranges"           : list — (min, max) for continuous, or array([0, 1])
                             for one-hot binary features.
        "categorical_mask" : bool array, True where feature is a one-hot indicator.
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

    neighbours = np.tile(x_t, (n_samples, 1)).astype(np.float64)

    for i in range(n_samples):
        n_perturb = rng.randint(1, d + 1)
        perturb_idx = rng.choice(d, size=n_perturb, replace=False)

        for j in perturb_idx:
            if cat_mask[j]:
                # One-hot indicator: sample binary {0, 1}
                neighbours[i, j] = float(rng.randint(0, 2))
            else:
                # Continuous: uniform within observed range
                lo, hi = ranges[j]
                if lo < hi:
                    neighbours[i, j] = rng.uniform(lo, hi)
                # If lo == hi the feature is constant; leave it unchanged.

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