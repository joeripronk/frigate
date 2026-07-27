"""
Cython-accelerated post-processing for audio detection results.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport qsort

# Typedefs for convenience
ctypedef cnp.float32_t float32_t
ctypedef cnp.int64_t int64_t


cpdef cython_select_top_k(
    const float[::1] scores,
    const int64_t[::1] class_ids,
    int k,
    double min_confidence,
):
    """Select top-K detection classes with scores above min_confidence.

    Args:
        scores: Float32 array of raw scores (typed memory view)
        class_ids: Int64 array of class IDs partitioned by top-K
        k: Number of top classes to select
        min_confidence: Minimum score threshold

    Returns:
        Tuple of (filtered class_ids, filtered scores) as numpy arrays
    """
    cdef:
        int result_count = 0
        int i
        float score

    for i in range(k):
        if class_ids[i] < 0 or class_ids[i] >= scores.shape[0]:
            break
        score = scores[class_ids[i]]
        if score < min_confidence:
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
        selected_scores[i] = scores[class_ids[i]]

    return (selected_ids, selected_scores)


cpdef cython_build_detections(
    const int64_t[::1] selected_ids,
    const float[::1] selected_scores,
    int count,
):
    """Build detection output array from filtered results.

    Args:
        selected_ids: Array of selected class IDs (typed memory view)
        selected_scores: Array of selected scores (typed memory view)
        count: Number of valid detections

    Returns:
        Detection array of shape (count, 6) with [class_id, score, -1, -1, -1, -1]
    """
    cdef:
        float[:, :] detections_view
        int i

    detections = np.zeros((count, 6), dtype=np.float32)
    detections_view = detections

    # Write class_id and score in one pass
    for i in range(count):
        detections_view[i, 0] = <float> selected_ids[i]
        detections_view[i, 1] = selected_scores[i]

    return detections


cpdef cython_detect_raw(
    const float[::1] res,
    int count,
    double min_confidence,
):
    """Full Cython audio detection post-processing pipeline.

    Combines top-K selection and detection array building into one function
    to avoid intermediate allocations.  All hot loops use typed memory views
    so they never box into Python objects.

    Args:
        res: Float32 array of raw scores (typed memory view)
        count: Number of top-20 candidates to consider
        min_confidence: Minimum score threshold

    Returns:
        Detection array of shape (N, 6) or empty if no detections
    """
    cdef:
        float[:, :] detections_view
        int n, k
        cnp.ndarray[int64_t, ndim=1] class_ids_arr
        cnp.ndarray[int64_t, ndim=1] sorted_order_arr
        cnp.ndarray[int64_t, ndim=1] class_ids_filtered
        cnp.ndarray[float32_t, ndim=1] scores_filtered
        int score_count, i

    n = res.shape[0]
    k = count if count < n else n
    if k <= 0:
        return np.zeros((0, 6), dtype=np.float32)

    # argpartition + argsort on numpy arrays (still the best approach here)
    class_ids_arr = np.argpartition(-np.asarray(res, dtype=np.float32), k - 1)[:k]
    sorted_order_arr = np.argsort(-np.asarray(res, dtype=np.float32)[class_ids_arr])
    class_ids_arr = class_ids_arr[sorted_order_arr]

    # Filter by confidence threshold using numpy boolean indexing
    class_ids_filtered = class_ids_arr[np.asarray(res, dtype=np.float32)[class_ids_arr] > min_confidence]
    scores_filtered = np.asarray(res, dtype=np.float32)[class_ids_filtered]

    if scores_filtered.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    score_count = scores_filtered.shape[0]

    detections = np.zeros((score_count, 6), dtype=np.float32)
    detections_view = detections

    # TYPED LOOP — no Python object overhead
    for i in range(score_count):
        detections_view[i, 0] = <float> class_ids_filtered[i]
        detections_view[i, 1] = <float> scores_filtered[i]

    return detections


# ---------------------------------------------------------------------------
# Batch helper: process multiple audio frames at once
# ---------------------------------------------------------------------------

cpdef cython_detect_raw_batch(
    const float[:, :] scores_batch,
    const int[::1] counts,
    double min_confidence,
):
    """Process multiple audio detection batches in one call.

    Each row in scores_batch is a separate audio frame's score vector.
    counts[i] specifies how many top-K candidates to consider for row i.

    Args:
        scores_batch: 2D float32 array, shape (batch_size, num_classes)
        counts: Int array of top-K counts per row
        min_confidence: Minimum score threshold

    Returns:
        List of detection arrays (one per input frame)
    """
    cdef:
        int batch_size = scores_batch.shape[0]
        int n, k, i, j
        list results = []
        cnp.ndarray[int64_t, ndim=1] class_ids_arr
        cnp.ndarray[int64_t, ndim=1] sorted_order_arr
        cnp.ndarray[int64_t, ndim=1] class_ids_filtered
        cnp.ndarray[float32_t, ndim=1] scores_filtered
        int score_count
        float[:, :] detections_view

    for i in range(batch_size):
        n = scores_batch.shape[1]
        k = counts[i] if counts[i] < n else n
        if k <= 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        row = scores_batch[i]
        row_arr = np.asarray(row, dtype=np.float32)

        class_ids_arr = np.argpartition(-row_arr, k - 1)[:k]
        sorted_order_arr = np.argsort(-row_arr[class_ids_arr])
        class_ids_arr = class_ids_arr[sorted_order_arr]

        class_ids_filtered = class_ids_arr[row_arr[class_ids_arr] > min_confidence]
        scores_filtered = row_arr[class_ids_filtered]

        if scores_filtered.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        score_count = scores_filtered.shape[0]
        detections = np.zeros((score_count, 6), dtype=np.float32)
        detections_view = detections

        for j in range(score_count):
            detections_view[j, 0] = <float> class_ids_filtered[j]
            detections_view[j, 1] = <float> scores_filtered[j]

        results.append(detections)

    return results
