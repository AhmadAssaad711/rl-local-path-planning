"""Shared seeding helpers."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Build a numpy random generator and synchronise Python's RNG."""
    if seed is None:
        return np.random.default_rng()
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
