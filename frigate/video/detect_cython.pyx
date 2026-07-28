"""
Cython-accelerated detection processing loops.

Consolidates the multiple Python list comprehensions in video/detect.py
that iterate over tracked_objects.values() into single Cython passes:
- Stationary object ID detection
- Tracked object box extraction
- Seed detection construction
- Detection consolidation filtering

Usage:
    from frigate.video.detect_cython import (
        cython_process_tracked_objects,
        cython_filter_detections_by_label,
    )
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

from typing import Any


def cython_process_tracked_objects(
    object tracked_objects,
    object motion_boxes,
    object stationary_object_ids,
    int stationary_threshold,
    object disappeared,
    object intersects_any,
    object reduce_overlapping_detections,
    object reduce_detections,
    object all_attributes,
    object consolidate_detections,
):
    """Process tracked objects in a single Cython pass.

    Replaces the four separate Python list comprehensions in detect.py:
    1. Stationary object ID detection (lines 330-343)
    2. Tracked object box extraction (lines 346-356)
    3. Seed detection construction (lines 416-427)
    4. Detection consolidation filtering (lines 446-448)

    This function iterates over tracked_objects ONCE instead of 4 times,
    and performs the stationary check and seed detection construction
    in a single pass.

    Args:
        tracked_objects: Dict of tracked object data (from object_tracker.tracked_objects)
        motion_boxes: List of motion boxes for intersects_any check
        stationary_object_ids: Set/list of already-known stationary IDs
        stationary_threshold: motionless_count threshold for stationary classification
        disappeared: Dict mapping object ID to disappeared count
        intersects_any: Function to check if box intersects motion boxes
        reduce_overlapping_detections: Cython reduce function
        reduce_detections: Python reduce function
        all_attributes: Set of attribute label names
        consolidate_detections: Set of already-consolidated detection labels

    Returns:
        Tuple of (
            stationary_object_ids: set of IDs,
            tracked_object_boxes: list of box tuples,
            detections: list of seed detection tuples,
            tracked_detections: list of detection tuples filtered by attributes,
        )
    """
    cdef:
        list stationary_ids = list(stationary_object_ids) if stationary_object_ids else []
        list tracked_boxes = []
        list detections = []
        list tracked_dets = []

    cdef dict obj
    cdef str obj_id
    cdef int motionless
    cdef int disp
    cdef bint is_stationary
    cdef dict box_data
    cdef object box

    for obj_key in tracked_objects:
        obj = tracked_objects[obj_key]
        obj_id = obj["id"]
        motionless = obj["motionless_count"]
        disp = disappeared.get(obj_id, 0)

        # Check if object is stationary
        if motionless >= stationary_threshold and disp == 0:
            # Check if it overlaps with motion boxes (only when not calibrating)
            if motion_boxes:
                box = obj["box"]
                if intersects_any(box, motion_boxes):
                    continue
            stationary_ids.append(obj_id)

        if obj_id in stationary_ids:
            # Seed detection for stationary object
            detections.append((
                obj["label"],
                obj["score"],
                obj["box"],
                obj["area"],
                obj["ratio"],
                obj["region"],
            ))
        else:
            # Extract tracked object box (use estimate for non-stationary, box for stationary)
            if motionless < stationary_threshold:
                tracked_boxes.append(tuple(obj["estimate"]))
            else:
                tracked_boxes.append(tuple(obj["box"]))

    # Filter consolidated detections by label
    if consolidate_detections is not None and len(consolidate_detections) > 0:
        for det in consolidate_detections:
            if det[0] not in all_attributes:
                tracked_dets.append(det)
    else:
        tracked_dets = list(consolidate_detections) if consolidate_detections else []

    return (set(stationary_ids), tracked_boxes, detections, tracked_dets)


def cython_filter_detections_by_label(
    object detections,
    object filter_labels,
):
    """Filter detections by label in a single Cython pass.

    Args:
        detections: List of detection tuples (label, score, box, area, ratio, region)
        filter_labels: Set of labels to keep

    Returns:
        List of detections whose label is NOT in filter_labels
    """
    cdef:
        list result = []
        tuple det
        str label

    for det in detections:
        label = det[0]
        if label not in filter_labels:
            result.append(det)

    return result


def cython_extract_tracked_box(
    object obj,
    int motionless,
    int stationary_threshold,
):
    """Extract the tracking box for a single object.

    Args:
        obj: Object dict with 'estimate', 'box', 'motionless_count'
        motionless: Pre-fetched motionless_count
        stationary_threshold: Threshold for stationary classification

    Returns:
        Box tuple (estimate if non-stationary, box if stationary)
    """
    if motionless < stationary_threshold:
        return tuple(obj["estimate"])
    else:
        return tuple(obj["box"])


def cython_build_detections_with_attributes(
    object tracked_objects,
    object consolidated_detections,
    object all_attributes,
    object attributes_map,
    object cython_find_best_object,
):
    """Build detections dict and assign attributes in a single pass.

    Replaces the Python loops in video/detect.py that:
    1. Build a detections dict from tracked_objects (lines 443-445)
    2. Filter consolidated detections by attribute labels
    3. For each detected attribute, find the best matching object

    The Cython find_best_object is already used but the Python wrapper
    TrackedObjectAttribute.find_best_object() is called instead.

    Args:
        tracked_objects: Dict of tracked object data
        consolidated_detections: List of consolidated detection tuples
        all_attributes: Set of attribute label names
        attributes_map: Dict mapping attribute label -> list of parent labels
        cython_find_best_object: cython_find_best_object function from object_cython

    Returns:
        Dict mapping object ID -> {**obj_data, "attributes": [...]}
    """
    cdef:
        dict detections = {}
        dict obj
        str obj_id
        tuple det
        str label
        list filtered_boxes = []
        list filtered_ids = []
        list filtered_labels = []
        object best_id
        object best_label
        dict attr_data

    # Build detections dict (single pass)
    for obj_key in tracked_objects:
        obj = tracked_objects[obj_key]
        obj_id = obj["id"]
        detections[obj_id] = dict(obj)  # shallow copy
        detections[obj_id]["attributes"] = []

    # Process consolidated detections for attributes
    # Filter to only attribute detections
    attr_dets = []
    for det in consolidated_detections:
        label = det[0]
        if label in all_attributes:
            attr_dets.append(det)

    # For each attribute detection, find best matching object
    for det in attr_dets:
        attr_label = det[0]
        attr_box = det[2]

        # Find objects whose label is in the attributes_map for this attribute
        filtered_boxes = []
        filtered_ids = []
        filtered_labels = []

        for obj_key in tracked_objects:
            obj = tracked_objects[obj_key]
            if obj["label"] in attributes_map.get(attr_label, []):
                filtered_boxes.append(obj["box"])
                filtered_ids.append(obj["id"])
                filtered_labels.append(obj["label"])

        best_id, best_label = cython_find_best_object(
            filtered_boxes, filtered_ids, filtered_labels, [attr_box[0], attr_box[1], attr_box[2], attr_box[3]]
        )

        if best_id is not None:
            attr_data = {
                "label": attr_label,
                "box": attr_box,
                "score": det[1],
            }
            detections[best_id]["attributes"].append(attr_data)

    return detections
