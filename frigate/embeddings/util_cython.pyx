"""
Cython-accelerated Z-score normalization for semantic search distances.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True


def cython_normalize(
    distances,
    double mean,
    double stddev,
    double scale_factor,
    double bias,
):
    """Normalize distances using Z-score formula with precomputed stats.

    Args:
        distances: List of float distances to normalize
        mean: Precomputed mean of the distribution
        stddev: Precomputed standard deviation
        scale_factor: Scaling factor for normalized values
        bias: Bias offset to add to normalized values

    Returns:
        List of normalized floats
    """
    if stddev == 0.0:
        return distances

    return [
        (x - mean) / stddev * scale_factor + bias
        for x in distances
    ]


def cython_normalize_inplace(distances, double mean, double stddev, double scale_factor, double bias):
    """Normalize distances in-place using Z-score formula.

    Args:
        distances: List of floats to normalize (modified in-place)
        mean: Precomputed mean
        stddev: Precomputed standard deviation
        scale_factor: Scaling factor
        bias: Bias offset

    Returns:
        The same list (modified in-place)
    """
    if stddev == 0.0:
        return distances

    for i in range(len(distances)):
        distances[i] = (distances[i] - mean) / stddev * scale_factor + bias

    return distances


def cython_update_stats(distances):
    """Welford's online algorithm for running mean and variance.

    Updates running statistics incrementally. Returns (n, mean, m2)
    where n is count, mean is running mean, m2 is sum of squared differences.

    Args:
        distances: List of float distances to update statistics with

    Returns:
        Tuple of (n, mean, m2)
    """
    n = 0
    mean = 0.0
    m2 = 0.0

    for x in distances:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    return (n, mean, m2)
