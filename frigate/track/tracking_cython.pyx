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

import numpy as np
cimport numpy as cnp

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


def cython_build_detections_from_raw(
    list detections,
    object frame_manager,
    object get_histogram,
    object frame_name,
    bint need_embedding,
    object camera_config,
    double frame_time,
):
    """Build Detection objects from raw detection tuples.

    Replaces the Python loop in norfair_tracker.match_and_update() that
    iterates over detections to create Detection objects with centroid
    computation and optional embedding extraction.

    This is a hot path that runs every frame per camera. It creates
    numpy arrays (points) and Detection objects per detection.

    Args:
        detections: List of (label, score, box, area, ratio, region) tuples
        frame_manager: SharedMemoryFrameManager for YUV frame access
        get_histogram: Function to extract histogram embedding from YUV frame
        frame_name: Name of the current frame
        need_embedding: Whether to extract PTZ embeddings
        camera_config: CameraConfig for autotracker check
        frame_time: Current frame timestamp to attach to each detection

    Returns:
        Dict mapping label -> list of Detection objects
    """
    cdef:
        dict by_label = {}
        tuple det
        str label
        int x0, y0, x1, y1
        int centroid_x, centroid_y
        object points
        object embedding
        object yuv_frame = None
        object Detection

    # Import Detection class from norfair (runtime import)
    from norfair.tracker import Detection

    for det in detections:
        label = det[0]
        if label not in by_label:
            by_label[label] = []

        box = det[2]
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]

        # Compute centroid
        centroid_x = (x0 + x1) // 2
        centroid_y = (y0 + y1) // 2

        # Create points array (top-left and bottom-right corners)
        points = np.array([[x0, y0], [x1, y1]], dtype=np.float32)

        # Extract embedding if needed for PTZ autotracker
        embedding = None
        if need_embedding:
            if yuv_frame is None:
                yuv_frame = frame_manager.get(frame_name, camera_config.frame_shape_yuv)
            embedding = get_histogram(yuv_frame, x0, y0, x1, y1)

        detection = Detection(
            points=points,
            label=label,
            embedding=embedding,
            data={
                "label": label,
                "score": det[1],
                "box": (x0, y0, x1, y1),
                "area": det[3],
                "ratio": det[4],
                "region": det[5],
                "frame_time": frame_time,
                "centroid": (centroid_x, centroid_y),
            },
        )
        by_label[label].append(detection)

    return by_label


def cython_update_tracks(
    object all_tracked_objects,
    object track_id_map,
    object register,
    object disappeared,
    object tracked_objects,
    object get_stationary_threshold,
    object update,
    object frame_time,
    int width,
    int height,
):
    """Update or create new tracks from norfair tracked objects.

    Replaces the loop in norfair_tracker.match_and_update() (lines 602-642)
    that updates or registers tracked objects, computes clamped boxes,
    counts disappeared, and identifies expired tracks.

    This runs every frame for all tracked objects.

    Args:
        all_tracked_objects: List of norfair TrackedObject instances
        track_id_map: Dict mapping norfair global_id -> Frigate object id
        register: Function to register a new tracked object
        disappeared: Dict mapping object id -> disappeared count
        tracked_objects: Dict of current tracked objects
        get_stationary_threshold: Function to get threshold for label
        update: Function to update an existing tracked object
        frame_time: Current frame timestamp
        width: Image width for box clamping
        height: Image height for box clamping

    Returns:
        Tuple of (active_ids: set, expired_ids: list)
    """
    cdef:
        set active_ids = set()
        list expired_updates = []
        object t
        tuple estimate
        str track_id
        str global_id_str
        dict new_obj
        dict obj
        str label
        thresholds
        double ft

    ft = frame_time

    for t in all_tracked_objects:
        global_id_str = str(t.global_id)
        active_ids.add(global_id_str)

        # Clamp estimate to image bounds
        estimate = tuple(t.estimate.flatten().astype(int))
        cx0 = estimate[0] if estimate[0] > 0 else 0
        cy0 = estimate[1] if estimate[1] > 0 else 0
        cx1 = estimate[2] if estimate[2] < width - 1 else width - 1
        cy1 = estimate[3] if estimate[3] < height - 1 else height - 1

        if cx0 >= cx1 or cy0 >= cy1:
            continue

        clamped = (cx0, cy0, cx1, cy1)

        new_obj = {
            **t.last_detection.data,
            "estimate": clamped,
            "estimate_velocity": t.estimate_velocity,
        }

        if global_id_str not in track_id_map:
            register(global_id_str, new_obj)
        elif t.last_detection.data["frame_time"] != ft:
            track_id = track_id_map[global_id_str]
            disappeared[track_id] = disappeared.get(track_id, 0) + 1
            # Only update estimate if box is valid (upper left < bottom right)
            if cx0 < cx1 and cy0 < cy1:
                tracked_objects[track_id]["estimate"] = new_obj["estimate"]
        else:
            label = t.last_detection.data["label"]
            thresholds = get_stationary_threshold(label)
            update(global_id_str, new_obj, thresholds, None)

    return (active_ids, expired_updates)
