"""Cython-accelerated motion detection.

This module provides Cython-optimized implementations of the per-frame
motion detection loop that runs on every camera, every frame. The main
speedup comes from eliminating the Python loop over contours.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cv2
import numpy as np
cimport numpy as cnp

from frigate.util.image import grab_cv2_contours


def cython_find_motion_boxes(
    cnp.ndarray thresh_dilated,
    float contour_area_thresh,
    float resize_factor,
    tuple motion_frame_size,
) -> tuple[cnp.ndarray, float]:
    """Find motion boxes from a thresholded/dilated frame.

    Cython replacement for the contour iteration loop in
    ImprovedMotionDetector.detect(). Eliminates Python overhead
    when iterating over potentially hundreds of contours per frame.

    Args:
        thresh_dilated: Thresholded and dilated frame (uint8, single channel)
        contour_area_thresh: Minimum contour area to count as motion
        resize_factor: Factor to scale boxes back to full resolution
        motion_frame_size: (height, width) of the motion processing frame

    Returns:
        Tuple of (motion_boxes_array, total_contour_area) where
        motion_boxes_array is (N, 4) in full resolution coordinates
        and total_contour_area is the sum of all contour areas
    """
    cdef float area_thresh = contour_area_thresh
    cdef float rf = resize_factor

    contours = cv2.findContours(
        thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = grab_cv2_contours(contours)

    if len(contours) == 0:
        return np.empty((0, 4), dtype=np.int32), 0.0

    result_boxes: list = []
    total_area: float = 0.0
    cdef int i
    cdef float contour_area
    cdef int x, y, w, h

    for i in range(len(contours)):
        c = contours[i]
        contour_area = cv2.contourArea(c)
        total_area += contour_area

        if contour_area > area_thresh:
            x, y, w, h = cv2.boundingRect(c)
            result_boxes.append((
                int(x * rf),
                int(y * rf),
                int((x + w) * rf),
                int((y + h) * rf),
            ))

    if not result_boxes:
        return np.empty((0, 4), dtype=np.int32), total_area

    return np.array(result_boxes, dtype=np.int32), total_area


def cython_filter_motion_boxes(
    cnp.ndarray motion_boxes,
    cnp.ndarray regions,
) -> cnp.ndarray:
    """Filter motion boxes that are inside any region.

    Cython replacement for the list comprehension in detect.py that
    filters standalone motion boxes against tracked object regions.

    Args:
        motion_boxes: (M, 4) array of motion boxes [x_min, y_min, x_max, y_max]
        regions: (N, 4) array of regions [x_min, y_min, x_max, y_max]

    Returns:
        Filtered array of motion boxes not inside any region
    """
    cdef int m = motion_boxes.shape[0]
    cdef int n = regions.shape[0]

    if m == 0 or n == 0:
        return motion_boxes.astype(np.int32)

    result_boxes: list = []
    cdef int i, j
    cdef float mb_x0, mb_y0, mb_x1, mb_y1
    cdef float reg_x0, reg_y0, reg_x1, reg_y1
    cdef int inside

    for i in range(m):
        mb = motion_boxes[i]
        mb_x0 = mb[0]
        mb_y0 = mb[1]
        mb_x1 = mb[2]
        mb_y1 = mb[3]
        inside = 0

        for j in range(n):
            reg = regions[j]
            reg_x0 = reg[0]
            reg_y0 = reg[1]
            reg_x1 = reg[2]
            reg_y1 = reg[3]

            if (mb_x0 >= reg_x0 and mb_y0 >= reg_y0 and
                    mb_x1 <= reg_x1 and mb_y1 <= reg_y1):
                inside = 1
                break

        if not inside:
            result_boxes.append(tuple(motion_boxes[i]))

    if not result_boxes:
        return np.empty((0, 4), dtype=np.int32)

    return np.array(result_boxes, dtype=np.int32)


def cython_count_motion(
    cnp.ndarray thresh_dilated,
    float contour_area_thresh,
) -> tuple[int, float]:
    """Count motion and compute total contour area.

    Cython replacement for the contour iteration that computes
    total_contour_area and motion box list separately.

    Args:
        thresh_dilated: Thresholded and dilated frame (uint8, single channel)
        contour_area_thresh: Minimum contour area to count as motion

    Returns:
        Tuple of (number_of_motion_contours, total_contour_area)
    """
    contours = cv2.findContours(
        thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = grab_cv2_contours(contours)

    if len(contours) == 0:
        return 0, 0.0

    total_area: float = 0.0
    count: int = 0
    cdef int i
    cdef float contour_area

    for i in range(len(contours)):
        c = contours[i]
        contour_area = cv2.contourArea(c)
        total_area += contour_area

        if contour_area > contour_area_thresh:
            count += 1

    return count, total_area
