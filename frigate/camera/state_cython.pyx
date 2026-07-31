"""
Cython-accelerated camera state updates.

Provides cython_camera_state_update() that optimizes the CameraState.update()
method in camera/state.py by consolidating set operations and reducing Python
object allocations:
- Pre-allocated new_ids/updated_ids/removed_ids sets
- Inline is_better_thumbnail logic
- Reduced dict lookups for frame cache operations

Usage:
    from frigate.camera.state_cython import cython_camera_state_update
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

from typing import Any


def cython_camera_state_update(
    object tracked_objects,
    object current_detections,
    object frame_cache,
    object frame_manager,
    object config,
    object camera_config,
    object best_objects,
    object callbacks,
    object send_mqtt_snapshot,
    double frame_time,
    object current_frame,
    list motion_boxes,
    list regions,
    object name,
):
    """Consolidated Cython implementation of CameraState.update().

    Replaces the pure-Python loop in camera/state.py:CameraState.update()
    with a single Cython pass that minimizes Python object allocations:
    - Pre-allocated new_ids/updated_ids/removed_ids sets
    - Inline is_better_thumbnail logic
    - Reduced frame cache operations
    - Consolidated camera_activity building

    This runs every frame after detections arrive.

    Args:
        tracked_objects: Dict of current tracked objects (may be modified in place)
        current_detections: Dict of detections for this frame
        frame_cache: Dict mapping frame_time -> {frame, object_id}
        frame_manager: FrameManager for YUV frame access
        config: FrigateConfig for model data
        camera_config: CameraConfig for camera-specific settings
        best_objects: Dict of object_type -> TrackedObject (best objects)
        callbacks: Dict of event_type -> list of callback functions
        send_mqtt_snapshot: Function to send MQTT snapshot
        frame_time: Current frame timestamp
        current_frame: Current YUV frame (numpy array or None)
        motion_boxes: List of motion box tuples
        regions: List of region tuples
        name: Camera name for logging

    Returns:
        Tuple of (new_objects, updated_objects, removed_objects)
    """
    cdef:
        set new_ids = set()
        set updated_ids = set()
        set removed_ids = set()
        set current_ids = set(current_detections.keys())
        set previous_ids = set(tracked_objects.keys())
        dict new_obj
        dict obj
        str obj_id
        object type_module
        object object_type
        object thumb_update
        object significant_update
        object path_update
        object autotracker_update
        double frame_time_val
        object obj_area
        object obj_label
        int publish_threshold
        object current_best
        object now
        bint is_better
        object frame_cache_items
        str cache_key
        dict cache_val
        object thumb_time
        object obj_thumb_data
        list thumb_frames_to_delete
        object camera_activity
        list activity_objects
        dict activity_obj
        str best_obj_type
        bint obj_active
        bint obj_false_positive
        object obj_sub_label
        object current_zones
        int i

    frame_time_val = frame_time

    # Step 1: Compute set operations (new/updated/removed)
    new_ids = current_ids.difference(previous_ids)
    updated_ids = current_ids.intersection(previous_ids)
    removed_ids = previous_ids.difference(current_ids)

    # Step 2: Process new objects
    for obj_id in new_ids:
        new_obj = tracked_objects[obj_id] = current_detections[obj_id]
        object_type = new_obj["label"]
        new_obj["id"] = obj_id

        # Add to frame cache if current_frame is readable
        if current_frame is not None:
            if frame_time_val not in frame_cache:
                frame_cache[frame_time_val] = {
                    "frame": np.copy(current_frame),
                    "object_id": obj_id,
                }

        # Save thumbnail data
        new_obj["thumbnail_data"] = {
            "frame_time": frame_time_val,
            "box": new_obj.get("box"),
            "area": new_obj.get("area", 0),
            "region": new_obj.get("region"),
            "score": new_obj.get("score", 0),
            "attributes": new_obj.get("attributes", []),
            "current_estimated_speed": 0,
            "velocity_angle": 0,
            "path_data": [],
            "recognized_license_plate": None,
            "recognized_license_plate_score": None,
        }

        # Call start callbacks
        for c in callbacks.get("start", []):
            c(name, new_obj, frame_time_val)

    # Step 3: Process updated objects
    for obj_id in updated_ids:
        updated_obj = tracked_objects[obj_id]
        obj = updated_obj.obj_data if hasattr(updated_obj, 'obj_data') else updated_obj
        thumb_update, significant_update, path_update, autotracker_update = (
            updated_obj.update(
                frame_time_val, current_detections[obj_id], current_frame is not None
            )
            if hasattr(updated_obj, 'update')
            else (False, False, False, False)
        )

        if autotracker_update or significant_update:
            for c in callbacks.get("autotrack", []):
                c(name, updated_obj, frame_time_val)

        if thumb_update and current_frame is not None:
            thumb_data = getattr(updated_obj, 'thumbnail_data', None)
            if (
                thumb_data is not None
                and thumb_data["frame_time"] == frame_time_val
                and frame_time_val not in frame_cache
            ):
                frame_cache[frame_time_val] = {
                    "frame": np.copy(current_frame),
                    "object_id": obj_id,
                }

            if hasattr(updated_obj, 'last_updated'):
                updated_obj.last_updated = frame_time_val

        # Determine publish threshold
        obj_area = obj.get("area", 0)
        obj_label = obj.get("label")
        publish_threshold = 5

        if (
            obj_label == "person"
            and hasattr(camera_config, 'face_recognition_min_obj_area')
            and camera_config.face_recognition_min_obj_area > 0
            and obj_area >= camera_config.face_recognition_min_obj_area
            and obj.get("sub_label") is None
        ) or (
            obj_label in ("car", "motorcycle")
            and hasattr(camera_config, 'lpr_min_obj_area')
            and camera_config.lpr_min_obj_area > 0
            and obj_area >= camera_config.lpr_min_obj_area
            and obj.get("sub_label") is None
            and obj.get("recognized_license_plate") is None
        ):
            publish_threshold = 1

        if (
            (frame_time_val - getattr(updated_obj, 'last_published', frame_time_val)) > publish_threshold
            and getattr(updated_obj, 'last_updated', frame_time_val) > getattr(updated_obj, 'last_published', frame_time_val)
        ) or significant_update or path_update:
            for c in callbacks.get("update", []):
                c(name, updated_obj, frame_time_val)
            if hasattr(updated_obj, 'last_published'):
                updated_obj.last_published = frame_time_val

    # Step 4: Process removed objects
    for obj_id in removed_ids:
        removed_obj = tracked_objects[obj_id]
        obj = removed_obj.obj_data if hasattr(removed_obj, 'obj_data') else removed_obj
        if "end_time" not in obj:
            obj["end_time"] = frame_time_val
            for c in callbacks.get("end", []):
                c(name, removed_obj, frame_time_val)

    # Step 5: Build camera_activity
    camera_activity = {
        "motion": len(motion_boxes) > 0,
        "objects": [],
    }

    for obj_id, obj in tracked_objects.items():
        obj_data = obj.obj_data if hasattr(obj, 'obj_data') else obj
        object_type = obj_data.get("label", "unknown")
        obj_active = obj.is_active() if hasattr(obj, 'is_active') else True

        if not getattr(obj, 'false_positive', False):
            label = object_type
            sub_label = None

            sub_label_data = obj_data.get("sub_label")
            if sub_label_data:
                if sub_label_data[0] in config.model.all_attributes if hasattr(config, 'model') else False:
                    label = sub_label_data[0]
                else:
                    label = f"{object_type}-verified"
                    sub_label = sub_label_data[0]

            activity_obj = {
                "id": obj_data.get("id", obj_id),
                "label": label,
                "stationary": not obj_active,
                "area": obj_data.get("area", 0),
                "ratio": obj_data.get("ratio", 0),
                "score": obj_data.get("score", 0),
                "sub_label": sub_label,
                "current_zones": getattr(obj, 'current_zones', []),
            }
            camera_activity["objects"].append(activity_obj)

        # Check best object
        if (
            current_frame is not None
            and not getattr(obj, 'false_positive', False)
        ):
            thumb_data = getattr(obj, 'thumbnail_data', None)
            if thumb_data and thumb_data["frame_time"] == frame_time_val:
                if object_type in best_objects:
                    current_best = best_objects[object_type]
                    import datetime
                    now = datetime.datetime.now().timestamp()

                    is_better = False
                    current_thumb = getattr(current_best, 'thumbnail_data', None)
                    if (
                        current_thumb is not None
                        and thumb_data is not None
                    ):
                        # Inline is_better_thumbnail logic
                        is_better = (
                            (getattr(current_best, 'score', 0) < getattr(obj, 'score', 0)) or
                            (now - current_thumb["frame_time"]) > camera_config.best_image_timeout if hasattr(camera_config, 'best_image_timeout') else False
                        )

                    if is_better:
                        send_mqtt_snapshot(obj, object_type)
                else:
                    send_mqtt_snapshot(obj, object_type)

    # Step 6: Callback camera_activity
    for c in callbacks.get("camera_activity", []):
        c(name, camera_activity)

    # Step 7: Cleanup thumbnail frame cache
    current_thumb_frames = set()
    current_best_frames = set()

    for obj in tracked_objects.values():
        thumb_data = getattr(obj, 'thumbnail_data', None)
        if thumb_data and thumb_data["frame_time"] is not None:
            current_thumb_frames.add(thumb_data["frame_time"])

    for obj in best_objects.values():
        thumb_data = getattr(obj, 'thumbnail_data', None)
        if thumb_data and thumb_data["frame_time"] is not None:
            current_best_frames.add(thumb_data["frame_time"])

    thumb_frames_to_delete = [
        t for t in frame_cache.keys()
        if t not in current_thumb_frames and t not in current_best_frames
    ]

    for t in thumb_frames_to_delete:
        del frame_cache[t]

    return (new_ids, updated_ids, removed_ids)


def cython_is_better_thumbnail(
    object current_best,
    object new_obj,
    object frame_shape,
):
    """Inline is_better_thumbnail logic for Cython.

    Args:
        current_best: Current best object with thumbnail_data
        new_obj: New object to compare
        frame_shape: Camera frame shape (height, width)

    Returns:
        True if new_obj is better than current_best
    """
    cdef:
        object current_thumb
        object new_thumb
        double now
        double timeout
        bint is_better

    current_thumb = getattr(current_best, 'thumbnail_data', None)
    new_thumb = getattr(new_obj, 'thumbnail_data', None)

    if current_thumb is None or new_thumb is None:
        return False

    import datetime
    now = datetime.datetime.now().timestamp()
    timeout = getattr(current_best, 'thumbnail_data', {}).get(
        'timeout', 30.0
    )

    # Compare scores and age
    is_better = (
        new_thumb["score"] > current_thumb.get("score", 0)
        or (now - current_thumb["frame_time"]) > timeout
    )

    return is_better
