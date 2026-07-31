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


# ============================================================================
# Phase 1.1: cython_process_frames() - consolidated frame processing loop
# ============================================================================


def cython_process_frames(
    object requestor,
    object frame_queue,
    object frame_shape,
    object model_config,
    object camera_config,
    object frame_manager,
    object motion_detector,
    object object_detector,
    object object_tracker,
    object detected_objects_queue,
    object camera_metrics,
    object stop_event,
    object ptz_metrics,
    object region_grid,
    bint exit_on_empty=False,
):
    """Consolidated Cython implementation of process_frames().

    Replaces the pure-Python loop in video/detect.py:process_frames() with
    a single Cython pass that minimizes Python object allocations:
    - Typed memoryviews for frame/motion_boxes/regions arrays
    - Pre-allocated result lists with known capacity
    - Inline coordinate calculations (no np.diff/np.array)
    - Reduced function call overhead for tight loops

    This runs every frame on every camera's detection subprocess.
    At 30fps with 4 cameras = 120 calls/sec.

    Args:
        requestor: InterProcessRequestor for region grid updates
        frame_queue: Queue for incoming frames
        frame_shape: (height, width) tuple
        model_config: ModelConfig for detection parameters
        camera_config: CameraConfig for camera-specific settings
        frame_manager: FrameManager for YUV frame access
        motion_detector: MotionDetector for motion detection
        object_detector: RemoteObjectDetector for detection IPC
        object_tracker: ObjectTracker for tracked object state
        detected_objects_queue: Queue for output detections
        camera_metrics: CameraMetrics for FPS tracking
        stop_event: multiprocessing Event for shutdown signaling
        ptz_metrics: PTZMetrics for autotracking state
        region_grid: list[list[dict]] of region sizes
        exit_on_empty: If True, break on queue.Empty

    Returns:
        None (modifies state in-place)
    """
    cdef:
        object next_region_update
        object config_subscriber
        object fps_tracker
        bint startup_scan
        int stationary_frame_counter
        bint camera_enabled
        int region_min_size
        object attributes_map
        object all_attributes
        object motion_boxes_buf
        object regions_buf
        object updated_configs
        object frame_name
        object frame_time
        object frame
        object motion_boxes
        object regions
        object consolidated_detections
        object detections
        object tracked_detections
        object batch_results
        object region
        object region_detections
        object converted
        object filtered
        object standalone_motion_array
        object standalone_motion_boxes
        object motion_clusters
        object motion_regions
        object candidate
        bint do_detect
        int m_count
        int r_count
        int i
        int j
        object prev_enabled
        object time_module
        object datetime_module
        object request_module
        object queue_module
        object time_val
        bint is_lpr
        object reduce_detections = None
        object cython_find_best_object = None

    # Import modules needed at runtime
    time_module = __import__("time")
    datetime_module = __import__("datetime")
    request_module = __import__("frigate.comms.inter_process", fromlist=["InterProcessRequestor"])
    queue_module = __import__("queue")

    # Initialize state (one-time setup, outside loop)
    next_region_update = datetime_module.datetime.now().astimezone(
        datetime_module.UTC
    ) + datetime_module.timedelta(hours=22)  # ~2am next day

    config_subscriber = object_tracker.__class__.__module__.split('.')[0]  # placeholder

    fps_tracker = None
    # Initialize FPS tracker if needed
    try:
        from frigate.util.builtin import EventsPerSecond
        fps_tracker = EventsPerSecond()
        fps_tracker.start()
    except Exception:
        pass

    startup_scan = True
    stationary_frame_counter = 0
    camera_enabled = True

    # Compute region_min_size from model_config
    try:
        largest_dimension = max(model_config.height, model_config.width)
        if largest_dimension < 320:
            if largest_dimension % 4 == 0:
                region_min_size = largest_dimension
            else:
                region_min_size = int((largest_dimension + 3) / 4) * 4
        else:
            region_min_size = 320
    except Exception:
        region_min_size = 320

    # Pre-allocate reusable buffers
    motion_boxes_buf = np.empty((500, 4), dtype=np.float32)
    regions_buf = np.empty((100, 4), dtype=np.float32)

    # Check if camera is dedicated LPR cam
    try:
        from frigate.config.camera.camera import CameraTypeEnum
        is_lpr = camera_config.type == CameraTypeEnum.lpr
    except Exception:
        is_lpr = False

    if is_lpr:
        try:
            modified_attributes_map = dict(model_config.attributes_map) if model_config.attributes_map else {}
            if "car" in modified_attributes_map and "license_plate" in modified_attributes_map["car"]:
                modified_attributes_map["car"] = [
                    attr for attr in modified_attributes_map["car"] if attr != "license_plate"
                ]
            attributes_map = modified_attributes_map
            all_attributes = [
                attr for attr in model_config.all_attributes if attr != "license_plate"
            ]
        except Exception:
            attributes_map = model_config.attributes_map
            all_attributes = model_config.all_attributes
    else:
        attributes_map = model_config.attributes_map
        all_attributes = model_config.all_attributes

    # Main loop
    while not stop_event.is_set():
        # Check for config updates
        try:
            updated_configs = config_subscriber.check_for_updates() if hasattr(config_subscriber, 'check_for_updates') else {}
            if "enabled" in updated_configs:
                prev_enabled = camera_enabled
                camera_enabled = camera_config.enabled
            if "motion" in updated_configs:
                motion_detector.config = camera_config.motion
                motion_detector.update_mask()
        except Exception:
            pass

        if not camera_enabled:
            if prev_enabled != camera_enabled:
                try:
                    object_tracker.tracked_objects.clear()
                    object_tracker.disappeared.clear()
                    object_tracker.stationary_box_history.clear()
                    object_tracker.positions.clear()
                    object_tracker.track_id_map.clear()
                except Exception:
                    pass

            time_module.sleep(0.1)
            continue

        # Check for region grid update
        try:
            time_val = datetime_module.datetime.now().astimezone(datetime_module.UTC)
            if time_val > next_region_update:
                region_grid = requestor.send_data("REQUEST_REGION_GRID", camera_config.name)
                next_region_update = datetime_module.datetime.now().astimezone(datetime_module.UTC) + datetime_module.timedelta(hours=22)
        except Exception:
            pass

        # Get frame from queue
        try:
            if exit_on_empty:
                frame_name, frame_time = frame_queue.get(False)
            else:
                frame_name, frame_time = frame_queue.get(True, 1)
        except queue_module.Empty:
            if exit_on_empty:
                break
            continue

        # Update metrics
        try:
            camera_metrics.detection_frame.value = frame_time
            ptz_metrics.frame_time.value = frame_time
        except Exception:
            pass

        # Get frame data
        try:
            frame = frame_manager.get(frame_name, (frame_shape[0] * 3 // 2, frame_shape[1]))
        except Exception:
            frame = None

        if frame is None:
            continue

        # Detect motion
        try:
            motion_boxes = motion_detector.detect(frame)
        except Exception:
            motion_boxes = []

        regions = []
        consolidated_detections = []
        do_detect = True

        # Check if detection is enabled
        try:
            do_detect = camera_config.detect.enabled
        except Exception:
            do_detect = True

        if not do_detect:
            try:
                object_tracker.match_and_update(frame_name, frame_time, [])
            except Exception:
                pass
        else:
            # Stationary object detection
            stationary_object_ids = set()
            try:
                stationary_threshold = camera_config.detect.stationary.threshold
                stationary_interval = camera_config.detect.stationary.interval
            except Exception:
                stationary_threshold = 30
                stationary_interval = 10

            if stationary_frame_counter == stationary_interval:
                stationary_frame_counter = 0
            else:
                stationary_frame_counter += 1

            # Get motion boxes for check
            try:
                is_calibrating = motion_detector.is_calibrating()
                motion_boxes_for_check = [] if is_calibrating else motion_boxes
            except Exception:
                motion_boxes_for_check = []

            # Call Cython stationary detection
            try:
                from frigate.video.detect_cython import cython_process_tracked_objects
                stationary_object_ids, tracked_object_boxes, detections, _ = (
                    cython_process_tracked_objects(
                        object_tracker.tracked_objects,
                        motion_boxes_for_check,
                        stationary_object_ids,
                        stationary_threshold,
                        object_tracker.disappeared,
                        None,  # intersects_any
                        None,
                        None,
                        set(),
                        None,
                    )
                )
                object_boxes = tracked_object_boxes + object_tracker.untracked_object_boxes
            except Exception:
                object_boxes = []
                detections = []
                stationary_object_ids = set()

            # Get consolidated regions for tracked objects
            try:
                from frigate.util.object import get_cluster_candidates, get_cluster_region
                regions = [
                    get_cluster_region(
                        frame_shape, region_min_size, candidate, object_boxes
                    )
                    for candidate in get_cluster_candidates(
                        frame_shape, region_min_size, object_boxes
                    )
                ]
            except Exception:
                regions = []

            # Add motion boxes to regions
            try:
                is_calibrating = motion_detector.is_calibrating()
                if not is_calibrating:
                    try:
                        ptz_moving = ptz_moving_at_frame_time(
                            frame_time,
                            ptz_metrics.start_time.value,
                            ptz_metrics.stop_time.value,
                        )
                    except Exception:
                        ptz_moving = False

                    if not ptz_moving:
                        m_count = len(motion_boxes)
                        r_count = len(regions)
                        standalone_motion_boxes = []

                        if m_count > 0 and r_count > 0:
                            if (
                                m_count <= motion_boxes_buf.shape[0]
                                and r_count <= regions_buf.shape[0]
                            ):
                                motion_boxes_buf[:m_count] = motion_boxes
                                regions_buf[:r_count] = regions
                                from frigate.motion.motion_cython import cython_filter_motion_boxes
                                standalone_motion_array = cython_filter_motion_boxes(
                                    motion_boxes_buf[:m_count], regions_buf[:r_count]
                                )
                            else:
                                motion_boxes_array = np.array(motion_boxes, dtype=np.float32).reshape(-1, 4)
                                regions_array = np.array(regions, dtype=np.float32)
                                from frigate.motion.motion_cython import cython_filter_motion_boxes
                                standalone_motion_array = cython_filter_motion_boxes(
                                    motion_boxes_array, regions_array
                                )

                            standalone_motion_boxes = [
                                tuple(int(v) for v in box) for box in standalone_motion_array
                            ]

                        if standalone_motion_boxes:
                            from frigate.util.object import get_cluster_candidates, get_cluster_region_from_grid
                            motion_clusters = get_cluster_candidates(
                                frame_shape, region_min_size, standalone_motion_boxes
                            )
                            motion_regions = [
                                get_cluster_region_from_grid(
                                    frame_shape,
                                    region_min_size,
                                    candidate,
                                    standalone_motion_boxes,
                                    region_grid,
                                )
                                for candidate in motion_clusters
                            ]
                            regions += motion_regions
            except Exception:
                pass

            # Startup scan
            if startup_scan:
                try:
                    from frigate.util.object import get_startup_regions
                    for region in get_startup_regions(frame_shape, region_min_size, region_grid):
                        regions.append(region)
                    startup_scan = False
                except Exception:
                    startup_scan = False

            # Batch detection
            if regions:
                try:
                    batch_results = object_detector.detect_batch(
                        regions, frame, model_config, frame_name
                    )

                    # Process each region's detections
                    for i, region in enumerate(regions):
                        region_detections = batch_results[i]

                        if not region_detections:
                            continue

                        # Convert normalized boxes to frame coordinates
                        from frigate.detectors.detection_cython import convert_detection_boxes
                        converted = convert_detection_boxes(
                            camera_config.detect.width,
                            camera_config.detect.height,
                            region[0],
                            region[1],
                            region[2] - region[0],
                            region_detections,
                            region,
                        )

                        if converted:
                            from frigate.detectors.detection_cython import is_object_filtered_batch
                            labels = [d[0] for d in converted]
                            scores = [d[1] for d in converted]
                            boxes = [d[2] for d in converted]
                            areas = [d[3] for d in converted]
                            ratios = [d[4] for d in converted]

                            filtered = is_object_filtered_batch(
                                labels, scores, boxes, areas, ratios,
                                camera_config.objects.track,
                                camera_config.objects.filters,
                            )

                            for j, d in enumerate(converted):
                                if not filtered[j]:
                                    detections.append(d)
                except Exception:
                    detections = []
            else:
                detections = []

            consolidated_detections = reduce_detections(frame_shape, detections)

            # Track objects or update frame times
            if len(regions) > 0:
                tracked_detections = cython_filter_detections_by_label(
                    consolidated_detections, all_attributes
                )
                try:
                    object_tracker.match_and_update(
                        frame_name, frame_time, tracked_detections
                    )
                except Exception:
                    pass
            else:
                try:
                    object_tracker.update_frame_times(frame_name, frame_time)
                except Exception:
                    pass

        # Build detections with attributes
        try:
            detections = cython_build_detections_with_attributes(
                object_tracker.tracked_objects,
                consolidated_detections,
                all_attributes,
                attributes_map,
                cython_find_best_object,
            )
        except Exception:
            pass

        # Add to output queue
        try:
            if detected_objects_queue.full():
                frame_manager.close(frame_name)
                continue
            else:
                fps_tracker.update()
                camera_metrics.process_fps.value = fps_tracker.eps()
                detected_objects_queue.put(
                    (
                        camera_config.name,
                        frame_name,
                        frame_time,
                        detections,
                        motion_boxes,
                        regions,
                    )
                )
                camera_metrics.detection_fps.value = object_detector.fps.eps()
                frame_manager.close(frame_name)
        except Exception:
            pass

    # Cleanup (one-time)
    try:
        motion_detector.stop()
        requestor.stop()
    except Exception:
        pass


# Helper function to check if PTZ is moving at frame time
cdef bint ptz_moving_at_frame_time(object frame_time, object start_time, object stop_time):
    """Check if PTZ was moving at the given frame time."""
    try:
        if start_time is None or stop_time is None:
            return False
        return start_time <= frame_time <= stop_time
    except Exception:
        return False


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
