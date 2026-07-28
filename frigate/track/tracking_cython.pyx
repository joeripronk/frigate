"""
Cython-accelerated tracking operations for norfair and centroid trackers.

Provides Cython implementations of the tracking module's hot paths:
- Detection grouping by label (norfair_tracker)
- Bounding box clamping (norfair_tracker)
- Frame time and motionless count updates (centroid_tracker)
- Disappeared object counting (centroid_tracker)

Usage:
    from frigate.track.tracking_cython import (
        cython_group_detections_by_label,
        cython_clamp_box,
        cython_update_frame_times_and_motionless,
        cython_count_disappeared,
    )
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from typing import Any


def cython_group_detections_by_label(object detections):
    """Group detections by label in a single Cython pass.

    Replaces the Python loop in norfair_tracker.process_detections() that
    groups detections by label into a dict.

    Args:
        detections: List of detection tuples (label, score, box, area, ratio, region)

    Returns:
        Dict mapping label -> list of detection tuples
    """
    cdef:
        dict by_label = {}
        tuple det
        str label

    for det in detections:
        label = det[0]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(det)

    return by_label


def cython_clamp_box(
    int x0, int y0, int x1, int y1,
    int width, int height,
):
    """Clamp a bounding box to image boundaries in Cython.

    Replaces the Python clamping loop in norfair_tracker.process_detections().

    Args:
        x0: Left X coordinate
        y0: Top Y coordinate
        x1: Right X coordinate
        y1: Bottom Y coordinate
        width: Image width
        height: Image height

    Returns:
        Clamped box tuple (x0, y0, x1, y1)
    """
    cdef int cx0, cy0, cx1, cy1

    cx0 = x0 if x0 > 0 else 0
    cy0 = y0 if y0 > 0 else 0
    cx1 = x1 if x1 < width - 1 else width - 1
    cy1 = y1 if y1 < height - 1 else height - 1

    if cx0 < cx1 and cy0 < cy1:
        return (cx0, cy0, cx1, cy1)
    return None


def cython_clamp_boxes(
    object boxes,
    int width,
    int height,
):
    """Clamp multiple bounding boxes to image boundaries.

    Args:
        boxes: List of box tuples (x0, y0, x1, y1)
        width: Image width
        height: Image height

    Returns:
        List of clamped box tuples (only valid boxes)
    """
    cdef:
        list result = []
        tuple box
        int x0, y0, x1, y1
        int cx0, cy0, cx1, cy1

    for box in boxes:
        x0, y0, x1, y1 = box
        cx0 = x0 if x0 > 0 else 0
        cy0 = y0 if y0 > 0 else 0
        cx1 = x1 if x1 < width - 1 else width - 1
        cy1 = y1 if y1 < height - 1 else height - 1

        if cx0 < cx1 and cy0 < cy1:
            result.append((cx0, cy0, cx1, cy1))

    return result


def cython_update_frame_times_and_motionless(
    object tracked_objects,
    double frame_time,
    object is_expired,
    object deregister,
):
    """Update frame times and motionless counts for all tracked objects.

    Replaces the loop in centroid_tracker.update_frame_times().
    Also handles expiration checks in the same pass.

    Args:
        tracked_objects: Dict of tracked objects
        frame_time: Current frame timestamp
        is_expired: Function to check if object is expired
        deregister: Function to deregister an expired object

    Returns:
        List of IDs that were deregistered (expired)
    """
    cdef:
        list expired_ids = []
        str obj_id
        dict obj

    for obj_key in tracked_objects:
        obj_id = obj_key
        obj = tracked_objects[obj_key]
        obj["frame_time"] = frame_time
        obj["motionless_count"] += 1

        if is_expired(obj_id):
            deregister(obj_id)
            expired_ids.append(obj_id)

    return expired_ids


def cython_count_disappeared(
    object tracked_objects,
    object disappeared,
    object detection_groups,
    object max_disappeared,
    object deregister,
):
    """Count disappeared objects and deregister expired ones.

    Replaces the loop in centroid_tracker.match_and_update() that
    iterates over tracked_objects to count disappeared objects.

    Args:
        tracked_objects: Dict of tracked objects
        disappeared: Dict of disappeared counts
        detection_groups: Dict mapping label -> list of detections
        max_disappeared: Maximum disappeared count before deregistration
        deregister: Function to deregister an object

    Returns:
        Tuple of (updated_disappeared_counts, deregistered_ids)
    """
    cdef:
        list deregistered = []
        dict obj
        str obj_id
        str label
        int disp_count
        int max_d

    max_d = max_disappeared

    for obj_key in tracked_objects:
        obj = tracked_objects[obj_key]
        obj_id = obj["id"]
        label = obj["label"]

        if label not in detection_groups:
            disp_count = disappeared[obj_id] + 1
            disappeared[obj_id] = disp_count

            if disp_count >= max_d:
                deregister(obj_id)
                deregistered.append(obj_id)

    return (disappeared, deregistered)


def cython_get_tracking_data(
    object tracked_objects,
    object get_stationary_threshold,
    object update,
    object frame_time,
):
    """Extract tracking data and perform updates for all tracked objects.

    Consolidates the loop in norfair_tracker.process_detections() that
    updates or registers tracked objects.

    Args:
        tracked_objects: Dict of current tracked objects
        get_stationary_threshold: Function to get threshold for label
        update: Function to update an existing object
        frame_time: Current frame timestamp

    Returns:
        Tuple of (active_ids, new_registrations, updates_performed)
    """
    cdef:
        list active_ids = []
        list new_registrations = []
        list updates = []

    # This is a simplified version - the full implementation
    # would need access to the tracker's register/update methods
    # and the new object data from norfair tracks.
    # This function serves as a template for batching.
    return (active_ids, new_registrations, updates)
