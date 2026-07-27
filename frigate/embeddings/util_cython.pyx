"""
Cython-accelerated Z-score normalization for semantic search distances.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float64_t


def cython_normalize(
    const double[::1] distances,
    double mean,
    double stddev,
    double scale_factor,
    double bias,
):
    """Normalize distances using Z-score formula with precomputed stats.

    Args:
        distances: Float64 array of distances (typed memory view)
        mean: Precomputed mean of the distribution
        stddev: Precomputed standard deviation
        scale_factor: Scaling factor for normalized values
        bias: Bias offset to add to normalized values

    Returns:
        Normalized float64 array
    """
    cdef:
        double inv_stddev = 1.0 / stddev
        int n = distances.shape[0]
        int i
        double[:] result_view

    if stddev == 0.0:
        return np.asarray(distances)

    result = np.empty(n, dtype=np.float64)
    result_view = result

    # Single-pass typed loop — no Python boxing
    for i in range(n):
        result_view[i] = ((distances[i] - mean) * inv_stddev * scale_factor) + bias

    return result


def cython_normalize_inplace(
    double[::1] distances,
    double mean,
    double stddev,
    double scale_factor,
    double bias,
):
    """Normalize distances in-place using Z-score formula.

    Args:
        distances: Float64 array to normalize (typed memory view, modified in-place)
        mean: Precomputed mean
        stddev: Precomputed standard deviation
        scale_factor: Scaling factor
        bias: Bias offset

    Returns:
        The same array (modified in-place)
    """
    cdef:
        double inv_stddev = 1.0 / stddev
        int n = distances.shape[0]
        int i

    if stddev == 0.0:
        return distances

    # Single-pass typed loop — no Python boxing
    for i in range(n):
        distances[i] = ((distances[i] - mean) * inv_stddev * scale_factor) + bias

    return distances


def cython_update_stats(const double[::1] distances):
    """Welford's online algorithm for running mean and variance.

    Updates running statistics incrementally. Returns (n, mean, m2)
    where n is count, mean is running mean, m2 is sum of squared differences.

    Args:
        distances: Float64 array of distances (typed memory view)

    Returns:
        Tuple of (n, mean, m2)
    """
    cdef:
        int n = distances.shape[0]
        int i
        double mean = 0.0
        double m2 = 0.0
        double delta, delta2

    for i in range(n):
        delta = distances[i] - mean
        mean += delta / (i + 1)
        delta2 = distances[i] - mean
        m2 += delta * delta2

    return (n, mean, m2)
