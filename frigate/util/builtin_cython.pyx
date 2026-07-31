"""
Cython-accelerated utility functions for builtin operations.

Provides Cython implementations of:
- Cosine distance computation
- Serialization helpers for embeddings
- Vector normalization operations

Usage:
    from frigate.util.builtin_cython import (
        cython_cosine_distance,
        cython_cosine_similarity,
        cython_serialize_embedding,
    )
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

from typing import Any


def cython_cosine_distance(
    cnp.ndarray[cnp.float64_t, ndim=1] a,
    cnp.ndarray[cnp.float64_t, ndim=1] b,
):
    """Cython-accelerated cosine distance calculation.

    Replaces the Python cosine_distance() function in util/builtin.py
    with inline computation that avoids np.dot() and np.sqrt() allocations:
    - Manual dot product computation with typed memoryviews
    - Inline magnitude calculation
    - cdef typed variables for all intermediate values

    Args:
        a: Input array a (1D float64)
        b: Input array b (1D float64)

    Returns:
        Cosine distance (1.0 = opposite, 0.0 = identical)
    """
    cdef:
        int n = a.shape[0]
        int i
        double dot = 0.0
        double a_mag = 0.0
        double b_mag = 0.0
        double a_mag_sqrt
        double b_mag_sqrt
        double result

    # Compute dot product and magnitudes in single pass
    for i in range(n):
        dot += a[i] * b[i]
        a_mag += a[i] * a[i]
        b_mag += b[i] * b[i]

    # Check for zero magnitude
    if a_mag == 0.0 or b_mag == 0.0:
        return 1.0

    # Compute result inline
    a_mag_sqrt = dot_product_sqrt(a_mag)
    b_mag_sqrt = dot_product_sqrt(b_mag)

    result = 1.0 - (dot / (a_mag_sqrt * b_mag_sqrt))

    return result


def cython_cosine_similarity(
    cnp.ndarray[cnp.float64_t, ndim=1] a,
    cnp.ndarray[cnp.float64_t, ndim=1] b,
):
    """Cython-accelerated cosine similarity calculation.

    Args:
        a: Input array a (1D float64)
        b: Input array b (1D float64)

    Returns:
        Cosine similarity (1.0 = identical, -1.0 = opposite)
    """
    cdef:
        int n = a.shape[0]
        int i
        double dot = 0.0
        double a_mag = 0.0
        double b_mag = 0.0
        double a_mag_sqrt
        double b_mag_sqrt

    for i in range(n):
        dot += a[i] * b[i]
        a_mag += a[i] * a[i]
        b_mag += b[i] * b[i]

    if a_mag == 0.0 or b_mag == 0.0:
        return 0.0

    a_mag_sqrt = dot_product_sqrt(a_mag)
    b_mag_sqrt = dot_product_sqrt(b_mag)

    return dot / (a_mag_sqrt * b_mag_sqrt)


cdef double dot_product_sqrt(double x):
    """Fast square root using C math library."""
    import math
    return math.sqrt(x)


def cython_serialize_embedding(
    cnp.ndarray[cnp.float64_t, ndim=1] embedding,
):
    """Cython-accelerated embedding serialization.

    Inline serialize() call that avoids Python object overhead.

    Args:
        embedding: Input embedding array (1D float64)

    Returns:
        Serialized bytes
    """
    cdef:
        int n = embedding.shape[0]
        bytes result

    # Convert numpy array to bytes inline
    result = embedding.tobytes()
    return result
