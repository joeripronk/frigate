"""Z-score normalization for search distance."""

import math

import numpy as np

from frigate.embeddings.util_cython import (
    cython_normalize,
    cython_normalize_inplace,
    cython_update_stats,
)

try:
    from frigate.embeddings.util_cython import cython_normalize_inplace
except ImportError:
    cython_normalize_inplace = None


class ZScoreNormalization:
    def __init__(self, scale_factor: float = 1.0, bias: float = 0.0):
        """Initialize with optional scaling and bias adjustments."""
        """scale_factor adjusts the magnitude of each score"""
        """bias will artificially shift the entire distribution upwards"""
        self.n = 0
        self.mean = 0
        self.m2 = 0
        self.scale_factor = scale_factor
        self.bias = bias

    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def stddev(self):
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    def normalize(self, distances: list[float], save_stats: bool):
        if save_stats:
            self._update(distances)
        if self.stddev == 0:
            return distances

        # Pass as numpy array for Cython typed memory view
        distances_arr = np.asarray(distances, dtype=np.float64)
        return cython_normalize(
            distances_arr, self.mean, self.stddev, self.scale_factor, self.bias
        )

    def _update(self, distances: list[float]):
        # Pass as numpy array for Cython typed memory view
        distances_arr = np.asarray(distances, dtype=np.float64)
        n, mean, m2 = cython_update_stats(distances_arr)
        self.n = n
        self.mean = mean
        self.m2 = m2

    def to_dict(self):
        return {
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
        }

    def from_dict(self, data: dict):
        self.n = data["n"]
        self.mean = data["mean"]
        self.m2 = data["m2"]
        return self
