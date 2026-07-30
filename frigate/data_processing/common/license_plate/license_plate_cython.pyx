"""
Cython-accelerated license plate recognition operations.

Provides Cython-optimized implementations of:
- Plate variant clustering with Jaro-Winkler string similarity
- Sequential box merging for license plate detection
- Box sorting with insertion-sort pattern
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmax, fmin

# Import Jaro-Winkler from rapidfuzz
from rapidfuzz.distance import JaroWinkler


def cython_cluster_plate_variants(
    list plates,
    double cluster_threshold,
):
    """Cython version of _get_cluster_rep plate clustering.

    Clusters plate text variants using Jaro-Winkler similarity and
    returns the representative from the best (largest/highest-confidence) cluster.

    Args:
        plates: List of dicts with keys 'plate' (str), 'conf' (float)
        cluster_threshold: Minimum average similarity to merge into cluster

    Returns:
        Tuple of (best_plate: str, best_conf: float, char_confs: list, area: int)
    """
    cdef:
        int n = len(plates)
        int i, j
        list clusters = []
        list cluster
        list sims
        double sim, avg_sim
        bint merged
        double max_conf
        int best_cluster_idx
        list best_cluster
        dict rep
        list char_confs = []
        int area = 0

    if n == 0:
        return ("", 0.0, [], 0)

    if n == 1:
        p = plates[0]
        return (p["plate"], p["conf"], p.get("char_confidences", []), p.get("area", 0))

    for i in range(n):
        plate = plates[i]
        plate_text = plate["plate"]
        merged = False

        for j in range(len(clusters)):
            cluster = clusters[j]
            sims = []
            for item in cluster:
                sim = JaroWinkler.similarity(plate_text, item["plate"])
                sims.append(sim)

            if len(sims) > 0:
                avg_sim = sum(sims) / len(sims)
                if avg_sim >= cluster_threshold:
                    cluster.append(plate)
                    merged = True
                    break

        if not merged:
            clusters.append([plate])

    if not clusters:
        return ("", 0.0, [], 0)

    # Find best cluster: largest size, tiebroken by max conf
    best_cluster_idx = 0
    best_size = len(clusters[0])
    best_max_conf = 0.0
    for j in range(len(clusters)):
        cluster = clusters[j]
        max_conf = 0.0
        for item in cluster:
            if item["conf"] > max_conf:
                max_conf = item["conf"]
        size = len(cluster)
        if size > best_size or (size == best_size and max_conf > best_max_conf):
            best_size = size
            best_max_conf = max_conf
            best_cluster_idx = j

    best_cluster = clusters[best_cluster_idx]

    # Find representative: highest conf in best cluster
    rep = best_cluster[0]
    for item in best_cluster[1:]:
        if item["conf"] > rep["conf"]:
            rep = item

    return (
        rep["plate"],
        rep["conf"],
        rep.get("char_confidences", []),
        rep.get("area", 0),
    )


def cython_merge_nearby_boxes(
    boxes_arr,
    double plate_width,
    double gap_fraction=0.1,
    double min_overlap_fraction=-0.2,
):
    """Cython version of _merge_nearby_boxes.

    Merges bounding boxes that are likely part of the same license plate
    based on proximity, with a dynamic max_gap based on plate width.

    Args:
        boxes_arr: Numpy array of shape (n, 4, 2) - boxes with 4 corners each
        plate_width: Width of entire license plate for gap calculation
        gap_fraction: Fraction of plate width for max gap (default 0.1)
        min_overlap_fraction: Fraction for min overlap (default -0.2)

    Returns:
        Merged boxes as numpy array of shape (m, 4, 2)
    """
    cdef:
        int n = boxes_arr.shape[0]
        double max_gap, min_overlap
        int i
        double current_right, next_left, horizontal_gap
        double current_top, current_bottom, next_top, next_bottom
        list result = []
        object current_box
        object next_box
        object merged_points
        object new_box

    if n == 0:
        return np.empty((0, 4, 2), dtype=boxes_arr.dtype)

    max_gap = plate_width * gap_fraction
    min_overlap = plate_width * min_overlap_fraction

    # Sort boxes by top-left x coordinate (index 0 of corner 0)
    indices = np.argsort(boxes_arr[:, 0, 0])
    sorted_boxes = boxes_arr[indices]

    boxes_view = sorted_boxes

    # Start with first box
    current_box = sorted_boxes[0].copy()

    for i in range(1, n):
        next_box = sorted_boxes[i]

        # Calculate horizontal gap
        current_right = np.max(current_box[:, 0])
        next_left = np.min(next_box[:, 0])
        horizontal_gap = next_left - current_right

        # Check vertical alignment
        current_top = np.min(current_box[:, 1])
        current_bottom = np.max(current_box[:, 1])
        next_top = np.min(next_box[:, 1])
        next_bottom = np.max(next_box[:, 1])

        # Check if boxes should be merged
        if min_overlap <= horizontal_gap <= max_gap and fmax(
            current_top, next_top
        ) <= fmin(current_bottom, next_bottom):
            merged_points = np.vstack((current_box, next_box))
            new_box = np.array(
                [
                    [
                        np.min(merged_points[:, 0]),
                        np.min(merged_points[:, 1]),
                    ],
                    [
                        np.max(merged_points[:, 0]),
                        np.min(merged_points[:, 1]),
                    ],
                    [
                        np.max(merged_points[:, 0]),
                        np.max(merged_points[:, 1]),
                    ],
                    [
                        np.min(merged_points[:, 0]),
                        np.max(merged_points[:, 1]),
                    ],
                ]
            )
            current_box = new_box
        else:
            result.append(current_box)
            current_box = next_box.copy()

    result.append(current_box)
    return np.array(result, dtype=np.int32)


def cython_sort_boxes(boxes_arr):
    """Cython version of _sort_boxes.

    Sorts boxes by vertical position first, then horizontally for
    boxes within 5 pixels vertically.

    Args:
        boxes_arr: Numpy array of shape (n, 4, 2) - boxes with 4 corners each

    Returns:
        Sorted boxes as numpy array of shape (n, 4, 2)
    """
    cdef:
        int n = boxes_arr.shape[0]
        int i, j
        object temp
        double v1, v2

    if n <= 1:
        return boxes_arr.copy()

    # Sort by vertical position first, then horizontal
    sort_indices = np.lexsort((boxes_arr[:, 0, 0], boxes_arr[:, 0, 1]))
    sorted_boxes = boxes_arr[sort_indices].copy()

    # Insertion sort for boxes close in vertical position
    for i in range(n - 1):
        for j in range(i, -1, -1):
            v1 = sorted_boxes[j + 1, 0, 1]
            v2 = sorted_boxes[j, 0, 1]
            if fabs(v1 - v2) < 5 and sorted_boxes[j + 1, 0, 0] < sorted_boxes[j, 0, 0]:
                temp = sorted_boxes[j].copy()
                sorted_boxes[j] = sorted_boxes[j + 1]
                sorted_boxes[j + 1] = temp
            else:
                break

    return sorted_boxes
