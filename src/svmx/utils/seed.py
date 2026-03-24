"""Reproducibility helper: seed all relevant RNGs in one call."""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python stdlib, NumPy, and (optionally) PyTorch.

    Call this at the start of any experiment script to ensure
    reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not installed; skip silently