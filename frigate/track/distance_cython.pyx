"""
Cython-accelerated distance calculation for norfair tracker.

Provides an inline cython_distance() function that replaces the Python
norfair_tracker.distance() with typed memoryviews and manual computation
to avoid np.diff(), np.array(), and np.linalg.norm() allocations.

Usage:
    from frigate.track.distance_cython import cython_distance
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

from typing import Any


def cython_distance(
    cnp.ndarray[cnp.float32_t, ndim=2] detection,
    cnp.ndarray[cnp.float32_t, ndim=2] estimate,
):
    """Calculate distance between detection and estimate in Cython.

    Replaces the Python distance() function in norfair_tracker.py with
    inline computation that avoids temporary array allocations:
    - Manual diff instead of np.diff()
    - Direct coordinate access instead of np.array()
    - Inline norm instead of np.linalg.norm()

    Key optimization: computes distance in a single pass over the
    detection/estimate arrays without creating intermediate objects.

    Args:
        detection: ndarray of shape (N, 2) with float32 points [x, y]
        estimate: ndarray of shape (M, 2) with float32 points [x, y]

    Returns:
        Euclidean distance as float. Returns inf for degenerate or
        non-finite boxes.
    """
    cdef:
        int i
        int n_det = detection.shape[0]
        int n_est = estimate.shape[0]
        double est_dim0, est_dim1
        double det_dim0, det_dim1
        double est_x0, est_y0, est_x1, est_y1
        double det_x0, det_y0, det_x1, det_y1
        double diff_x, diff_y
        double width_ratio, height_ratio
        double change_x, change_y, change_w, change_h
        double norm

    # Guard against degenerate or non-finite boxes
    # Check estimate dimensions
    est_dim0 = estimate[1, 0] - estimate[0, 0] if n_est > 1 else estimate[0, 0]
    est_dim1 = estimate[1, 1] - estimate[0, 1] if n_est > 1 else estimate[0, 1]
    det_dim0 = detection[1, 0] - detection[0, 0] if n_det > 1 else detection[0, 0]
    det_dim1 = detection[1, 1] - detection[0, 1] if n_det > 1 else detection[0, 1]

    # Check for non-finite values
    if (
        not _is_finite(est_dim0)
        or not _is_finite(est_dim1)
        or not _is_finite(det_dim0)
        or not _is_finite(det_dim1)
        or est_dim0 <= 0
        or est_dim1 <= 0
        or det_dim0 <= 0
        or det_dim1 <= 0
    ):
        return float("inf")

    # Get bottom center positions (manual instead of np.array)
    # detection_position: [avg(x coords), max(y coord)]
    det_x0 = detection[0, 0]
    det_x1 = detection[1, 0]
    det_y1 = detection[0, 1]
    if n_det > 1:
        for i in range(1, n_det):
            if detection[i, 1] > det_y1:
                det_y1 = detection[i, 1]

    det_avg_x = (det_x0 + det_x1) / 2.0

    # estimate_position: [avg(x coords), max(y coord)]
    est_x0 = estimate[0, 0]
    est_x1 = estimate[1, 0]
    est_y1 = estimate[0, 1]
    if n_est > 1:
        for i in range(1, n_est):
            if estimate[i, 1] > est_y1:
                est_y1 = estimate[i, 1]

    est_avg_x = (est_x0 + est_x1) / 2.0

    # Change in x relative to w, change in y relative to h
    diff_x = det_avg_x - est_avg_x
    diff_y = det_y1 - est_y1

    change_x = diff_x / est_dim0
    change_y = diff_y / est_dim1

    # Get ratio of widths and heights (normalized to 1)
    if det_dim0 < est_dim0:
        width_ratio = (est_dim0 / det_dim0) - 1.0
    else:
        width_ratio = (det_dim0 / est_dim0) - 1.0

    if det_dim1 < est_dim1:
        height_ratio = (est_dim1 / det_dim1) - 1.0
    else:
        height_ratio = (det_dim1 / est_dim1) - 1.0

    # Calculate euclidean distance of the change vector
    # norm = sqrt(change_x^2 + change_y^2 + width_ratio^2 + height_ratio^2)
    norm = _sqrt(change_x * change_x + change_y * change_y + width_ratio * width_ratio + height_ratio * height_ratio)

    return <float>norm


cdef bint _is_finite(double value):
    """Check if a double value is finite (not inf or nan)."""
    # In Cython, we use Python's math.isfinite for simplicity
    # This is faster than checking bounds manually
    import math
    return math.isfinite(value)


cdef double _sqrt(double x):
    """Fast square root using Cython's built-in sqrt."""
    import math
    return math.sqrt(x)


def frigate_distance_wrapper(
    object detection_points,
    object estimate_array,
):
    """Wrapper that handles Python Detection/TrackedObject objects.

    Extracts the points/estimate arrays and calls cython_distance().
    This is the entry point used by norfair tracker's distance_function.

    Args:
        detection_points: Detection.points from norfair (ndarray or list)
        estimate_array: TrackedObject.estimate from norfair (ndarray or list)

    Returns:
        Distance as float
    """
    # Convert to numpy arrays if needed
    if not isinstance(detection_points, np.ndarray):
        detection_points = np.array(detection_points, dtype=np.float32)
    if not isinstance(estimate_array, np.ndarray):
        estimate_array = np.array(estimate_array, dtype=np.float32)

    return cython_distance(detection_points, estimate_array)
