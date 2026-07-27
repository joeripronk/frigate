"""Cython-accelerated detection loops.

This module provides Cython-optimized implementations of common detection
post-processing loops that run per-frame on every camera.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import logging

import cv2
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libcpp cimport bool

logger = logging.getLogger(__name__)


# --- 1. convert_detection_boxes ---
# video/detect.py:detect() loop — converts normalized detections to frame coords

def convert_detection_boxes(
    int width,
    int height,
    float region_x0,
    float region_y0,
    float region_size,
    list raw_detections,
) -> list:
    """Convert normalized detection boxes to frame coordinates.

    Equivalent to the loop in video/detect.py:detect() but Cython-optimized.

    Args:
        width: Frame width
        height: Frame height
        region_x0: Region x offset (frame coord)
        region_y0: Region y offset (frame coord)
        region_size: Region width/height (same for both, square region)
        raw_detections: List of (label, score, (y0_norm, x0_norm, y1_norm, x1_norm))

    Returns:
        List of (label, score, (x_min, y_min, x_max, y_max), area, ratio)
    """
    results: list = []

    # Precompute limits
    cdef int w = width - 1
    cdef int h = height - 1
    cdef float r_x0 = region_x0
    cdef float r_y0 = region_y0
    cdef float r_size = region_size

    for d in raw_detections:
        # d = (label, score, (y0_norm, x0_norm, y1_norm, x1_norm))
        label = d[0]
        score = d[1]
        y0_n = d[2][0]
        x0_n = d[2][1]
        y1_n = d[2][2]
        x1_n = d[2][3]

        # Convert normalized box to frame coords
        x_min = <int>(max(0.0, x0_n * r_size + r_x0))
        y_min = <int>(max(0.0, y0_n * r_size + r_y0))
        x_max = <int>(min(<float>w, x1_n * r_size + r_x0))
        y_max = <int>(min(<float>h, y1_n * r_size + r_y0))

        # Skip objects outside the frame
        if x_min >= w or y_min >= h:
            continue

        w_box = x_max - x_min
        h_box = y_max - y_min
        area = w_box * h_box
        ratio = w_box / max(1.0, <float>h_box)

        results.append((label, score, (x_min, y_min, x_max, y_max), area, ratio))

    return results


# --- 2. filter_raw_detections ---
# object_detection/base.py:BaseLocalDetector.detect() loop

def filter_raw_detections(
    list raw_detections,
    dict labels,
    float threshold,
) -> list:
    """Filter raw detections by threshold and convert to final format.

    Equivalent to the loop in BaseLocalDetector.detect() but Cython-optimized.

    Args:
        raw_detections: List of (label_id, score, y0, x0, y1, x1) from detect_raw
        labels: Label map {int: str}
        threshold: Minimum score threshold

    Returns:
        List of (label, score, (y0, x0, y1, x1))
    """
    results: list = []
    cdef int label_count = len(labels)
    cdef float th = threshold

    for d in raw_detections:
        label_id = <int>d[0]
        score = d[1]

        if label_id < 0 or label_id >= label_count:
            continue
        if score < th:
            break

        results.append((
            labels[label_id],
            score,
            (d[2], d[3], d[4], d[5]),
        ))

    return results


# --- 3. filter_from_shared_memory ---
# object_detection/base.py:RemoteObjectDetector.detect() loop

def filter_from_shared_memory(
    shm_detections,
    dict labels,
    float threshold,
) -> list:
    """Filter detections from shared memory by threshold.

    Equivalent to the loop in RemoteObjectDetector.detect() but Cython-optimized.

    Args:
        shm_detections: (N, 6) float32 array from shared memory: [label_id, score, y0, x0, y1, x1]
        labels: Label map {int: str}
        threshold: Minimum score threshold

    Returns:
        List of (label, score, (y0, x0, y1, x1))
    """
    results: list = []
    cdef int n = shm_detections.shape[0]
    cdef int label_count = len(labels)
    cdef float th = threshold
    cdef int i
    cdef float score
    cdef int label_id

    for i in range(n):
        label_id = <int>shm_detections[i, 0]
        score = shm_detections[i, 1]

        if label_id < 0 or label_id >= label_count:
            continue
        if score < th:
            break

        results.append((
            labels[label_id],
            score,
            (
                shm_detections[i, 2],
                shm_detections[i, 3],
                shm_detections[i, 4],
                shm_detections[i, 5],
            ),
        ))

    return results


# --- 4. overlap_consolidate ---
# util/object.py:get_consolidated_object_detections() loop

def overlap_consolidate(
    list sorted_by_area,
    dict consolidation_map,
    float default_threshold,
) -> list:
    """Consolidate overlapping detections by area-based overlap threshold.

    Equivalent to the inner loop of get_consolidated_object_detections() but Cython-optimized.

    Args:
        sorted_by_area: List of detections sorted by area (smallest to largest), each is (label, score, box, area, ratio, region)
        consolidation_map: Dict mapping label to threshold
        default_threshold: Default consolidation threshold

    Returns:
        List of non-overlapping detections
    """
    results: list = []
    cdef int n = len(sorted_by_area)
    cdef float default_th = default_threshold
    cdef float current_area
    cdef float to_check_area
    cdef float overlap_thresh
    cdef int i
    cdef int j

    for i in range(n):
        current_detection = sorted_by_area[i]
        current_area = current_detection[3]
        current_box = current_detection[2]
        overlap = 0

        for j in range(i + 1, n):
            to_check = sorted_by_area[j]
            to_check_box = to_check[2]

            # Skip if area ratio < 5%
            if current_area < 0.05 * to_check[3]:
                continue

            # Compute intersection
            ix0 = max(current_box[0], to_check_box[0])
            iy0 = max(current_box[1], to_check_box[1])
            ix1 = min(current_box[2], to_check_box[2])
            iy1 = min(current_box[3], to_check_box[3])

            if ix0 >= ix1 or iy0 >= iy1:
                continue

            intersect_w = ix1 - ix0
            intersect_h = iy1 - iy0
            intersect_area = intersect_w * intersect_h

            # Compute overlap threshold for this label
            overlap_thresh = consolidation_map.get(current_detection[0], default_th)

            if intersect_area / current_area > overlap_thresh:
                overlap = 1
                break

        if overlap == 0:
            results.append(sorted_by_area[i])

    return results


# --- 5. is_object_filtered_batch ---
# util/object.py:is_object_filtered for a batch of detections

def is_object_filtered_batch(
    list labels,
    list scores,
    list boxes,
    list areas,
    list ratios,
    list objects_to_track,
    dict object_filters,
) -> list:
    """Check which objects are filtered out.

    Args:
        labels: List of label strings
        scores: List of scores
        boxes: List of (x_min, y_min, x_max, y_max) tuples
        areas: List of areas
        ratios: List of width/height ratios
        objects_to_track: Set/list of tracked labels
        object_filters: Filter config dict

    Returns:
        List of booleans (True = filtered out)
    """
    results: list = []

    for i in range(len(labels)):
        object_name = labels[i]
        object_score = scores[i]
        object_box = boxes[i]
        object_area = areas[i]
        object_ratio = ratios[i]

        if object_name not in objects_to_track:
            results.append(True)
            continue

        if object_name in object_filters:
            obj_settings = object_filters[object_name]

            if obj_settings.min_area > object_area:
                results.append(True)
                continue

            if obj_settings.max_area < object_area:
                results.append(True)
                continue

            if obj_settings.min_score > object_score:
                results.append(True)
                continue

            if obj_settings.min_ratio > object_ratio:
                results.append(True)
                continue

            if obj_settings.max_ratio < object_ratio:
                results.append(True)
                continue

            if obj_settings.rasterized_mask is not None:
                y_location = min(<int>object_box[3], len(obj_settings.rasterized_mask) - 1)
                x_location = min(
                    <int>((object_box[2] + object_box[0]) / 2.0),
                    len(obj_settings.rasterized_mask[0]) - 1,
                )

                if obj_settings.rasterized_mask[y_location][x_location] == 0:
                    results.append(True)
                    continue

        results.append(False)

    return results


# --- 6. box_intersects_any_vectorized ---

def box_intersects_any_vectorized(
    box_a,
    boxes,
):
    """Vectorized check if box_a intersects any of boxes.

    Args:
        box_a: (1, 4) array [x_min, y_min, x_max, y_max]
        boxes: (N, 4) array of boxes

    Returns:
        Boolean array of length N
    """
    cdef int n = boxes.shape[0]
    cdef cnp.ndarray result = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return result

    # Use vectorized numpy operations
    a0 = box_a[0]
    result[:] = (
        (a0[2] < boxes[:, 0]) | (a0[0] > boxes[:, 2]) |
        (a0[1] > boxes[:, 3]) | (a0[3] < boxes[:, 1])
    ) == 0

    return result


def box_inside_any_vectorized(
    box_a,
    boxes,
):
    """Vectorized check if box_a is inside any of boxes.

    Args:
        box_a: (1, 4) array [x_min, y_min, x_max, y_max]
        boxes: (N, 4) array of boxes

    Returns:
        Boolean array of length N
    """
    cdef int n = boxes.shape[0]
    cdef cnp.ndarray result = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return result

    # Use vectorized numpy operations
    a = box_a[0]
    result[:] = (
        (a[0] >= boxes[:, 0]) & (a[1] >= boxes[:, 1]) &
        (a[2] <= boxes[:, 2]) & (a[3] <= boxes[:, 3])
    )

    return result


# --- 7. cython_standalone_motion_boxes ---
# video/detect.py: standalone_motion_boxes filter

def cython_standalone_motion_boxes(
    motion_boxes,
    regions,
):
    """Filter motion boxes that are inside any region.

    Args:
        motion_boxes: (M, 4) array of motion boxes
        regions: (N, 4) array of regions

    Returns:
        Filtered array of motion boxes not inside any region
    """
    cdef int m = motion_boxes.shape[0]
    cdef int n = regions.shape[0]

    if m == 0 or n == 0:
        return motion_boxes

    # For each motion box, check if it's inside any region
    result_boxes: list = []
    cdef int i, j

    for i in range(m):
        mb = motion_boxes[i]
        inside = False

        for j in range(n):
            reg = regions[j]
            if mb[0] >= reg[0] and mb[1] >= reg[1] and mb[2] <= reg[2] and mb[3] <= reg[3]:
                inside = True
                break

        if not inside:
            result_boxes.append(motion_boxes[i])

    if len(result_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    return np.array(result_boxes, dtype=np.float32)


# --- 8. cython_reduce_overlapping_detections ---

def cython_reduce_overlapping_detections(
    list group,
    cnp.ndarray frame_shape,
    float threshold,
) -> list:
    """Apply non-maxima suppression to a group of detections with the same label.

    Args:
        group: List of detections (label, score, box, area, ratio, region)
        frame_shape: (height, width) for clipping check
        threshold: NMS threshold (already applied via cv2.dnn.NMSBoxes)

    Returns:
        Selected detections after NMS
    """
    import numpy as np

    if not group:
        return []

    cdef int n = len(group)
    cdef float h = frame_shape[0]
    cdef float w = frame_shape[1]

    boxes = []
    confidences = []

    for o in group:
        box = o[2]
        boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        # Apply clipping factor
        if box[2] > w or box[3] > h:
            confidences.append(0.6)
        else:
            confidences.append(o[1])

    if not boxes:
        return []

    # Use cv2.dnn.NMSBoxes
    indices = cv2.dnn.NMSBoxes(
        [int(b[0]) for b in boxes],
        confidences,
        0.5,
        threshold,
    )

    selected = []
    for idx in indices:
        idx = idx if isinstance(idx, np.int32) else idx[0]
        selected.append(group[idx])

    return selected
