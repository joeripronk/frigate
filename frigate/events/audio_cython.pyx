"""
Cython-accelerated post-processing for audio detection results.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from typing import Any


def cython_select_top_k(
    scores,
    class_ids,
    int k,
    double min_confidence,
):
    """Select top-K detection classes with scores above min_confidence.

    Args:
        scores: Array of raw scores from the model
        class_ids: Array of class IDs partitioned by top-K
        k: Number of top classes to select
        min_confidence: Minimum score threshold

    Returns:
        Tuple of (filtered class_ids, filtered scores) as numpy arrays
    """
    result_count = 0
    for i in range(k):
        if scores[class_ids[i]] < min_confidence:
            break
        result_count += 1

    if result_count == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    selected_ids = np.empty(result_count, dtype=np.int64)
    selected_scores = np.empty(result_count, dtype=np.float32)

    for i in range(result_count):
        selected_ids[i] = class_ids[i]
        selected_scores[i] = float(scores[class_ids[i]])

    return (selected_ids, selected_scores)


def cython_build_detections(
    selected_ids,
    selected_scores,
    int count,
):
    """Build detection output array from filtered results.

    Args:
        selected_ids: Array of selected class IDs
        selected_scores: Array of selected scores
        count: Number of valid detections

    Returns:
        Detection array of shape (count, 6) with [class_id, score, -1, -1, -1, -1]
    """
    detections = np.zeros((count, 6), dtype=np.float32)

    for i in range(count):
        detections[i, 0] = selected_ids[i]
        detections[i, 1] = selected_scores[i]

    return detections


def cython_detect_raw(
    res,
    int count,
    double min_confidence,
):
    """Full Cython audio detection post-processing pipeline.

    Combines top-K selection and detection array building into one function
    to avoid intermediate allocations.

    Args:
        res: Raw scores from the model
        count: Number of top-20 candidates to consider
        min_confidence: Minimum score threshold

    Returns:
        Detection array of shape (count, 6) or empty if no detections
    """
    n = len(res)
    k = count if count < n else n
    if k <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    class_ids = np.argpartition(-res, k - 1)[:k]
    sorted_order = np.argsort(-res[class_ids])
    class_ids = class_ids[sorted_order]
    non_zero_mask = res > min_confidence
    class_ids = class_ids[non_zero_mask[class_ids]]
    scores = res[class_ids]

    if len(scores) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    detections = np.zeros((len(scores), 6), dtype=np.float32)

    for i in range(len(scores)):
        detections[i, 0] = class_ids[i]
        detections[i, 1] = scores[i]

    return detections
