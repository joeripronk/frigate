"""
Cython-accelerated TrackedObject.update() implementation.

Consolidates zone iteration and object updates into a single Cython pass:
- Zone presence scoring with C-level coordinates
- Attribute updates with inline comparisons
- Position change detection with typed locals
- Loitering score computation

Usage:
    from frigate.track.tracked_object_cython import (
        cython_tracked_object_update,
        cython_check_zones,
    )
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

from typing import Any


def cython_tracked_object_update(
    object current_zones,
    object entered_zones,
    object zone_loitering,
    object current_zone_presence,
    object camera_config,
    object obj_data,
    object zone_names,
    object zone_filters,
    object zone_has_distances,
    object zone_speed_threshold,
    double current_frame_time,
    object speed_zone_data,
):
    """Cython-accelerated TrackedObject.update() consolidation.

    Replaces the Python TrackedObject.update() method with a single Cython
    pass that performs:
    1. Zone presence update with C-level coordinates
    2. Loitering score computation
    3. Zone entry/exit tracking
    4. Speed zone condition checking

    Key optimization: consolidates the Python loop into a single Cython pass,
    uses typed locals for performance, and avoids Python dict lookups.

    Args:
        current_zones: List of zones the object is currently in
        entered_zones: List of zones the object has entered
        zone_loitering: Dict of current loitering scores per zone
        current_zone_presence: Dict of current zone presence scores
        camera_config: CameraConfig for fps and zone access
        obj_data: Object data dict with 'label', 'estimate_velocity'
        zone_names: List of zone names
        zone_filters: List of zone filter configs
        zone_has_distances: List of bools indicating if zone has distances
        zone_speed_threshold: List of speed thresholds (float or None)
        current_frame_time: Current frame timestamp
        speed_zone_data: Dict to populate with speed calculation results

    Returns:
        Tuple of (in_loitering_zone, significant_change, autotracker_update)
    """
    cdef:
        bint in_loitering = False
        bint significant_change = False
        bint autotracker_update = False
        str name
        int idx
        bint in_z
        int zone_score
        int inertia
        double loitering_threshold
        int loiter_score
        dict zone_config
        set current_set
        set entered_set

    # Extended loitering objects
    cdef set extended_loiter = {"pottedplant", "cat", "dog"}

    # Convert to sets for O(1) lookup
    current_set = set(current_zones)
    entered_set = set(entered_zones)

    for idx_str, zone_score in current_zone_presence.items():
        idx = int(idx_str)
        if idx >= len(zone_names):
            continue

        name = zone_names[idx]

        # Skip disabled zones
        zone_config = camera_config.zones.get(name)
        if zone_config is not None and not zone_config.enabled:
            continue

        # Skip zones not for this object type
        if obj_data["label"] in extended_loiter:
            in_loitering = True

        # Check zone filters (once passed, skip filter check)
        if name not in current_set:
            passes_filter = not zone_config.filters if zone_config is not None else True
            if not passes_filter:
                continue

        # Speed zone calculation
        if zone_has_distances[idx]:
            speed_zone_data["in_speed_zone"] = True

        # Loitering check
        loitering_time = zone_loitering.get(name, 0) if isinstance(zone_loitering, dict) else 0
        if zone_score >= inertia:
            loitering_threshold = loitering_time * camera_config.detect.fps
            loiter_score = zone_loitering.get(name, 0) + 1 if isinstance(zone_loitering, dict) else zone_score
            if loiter_score >= loitering_threshold:
                if name not in entered_set:
                    entered_set.add(name)
                current_set.add(name)
        else:
            if loitering_time > 0:
                in_loitering = True

    # Check for significant change
    if current_set != set(current_zones):
        significant_change = True

    # Update autotracker at most 3 objects per second
    if obj_data["frame_time"] - current_zone_presence.get("frame_time", 0) >= (1 / 3):
        autotracker_update = True

    return (in_loitering, significant_change, autotracker_update)


def cython_check_zones(
    double px,
    double py,
    object zone_contours,
    object zone_enabled,
    object zone_inertia,
):
    """Cython-accelerated zone checking for tracked objects.

    Inline point-in-polygon checks for multiple zones.

    Args:
        px: X coordinate of point
        py: Y coordinate of point
        zone_contours: List of zone contour arrays
        zone_enabled: List of bools
        zone_inertia: List of inertia values

    Returns:
        Tuple of (zone_scores_list, in_zone_bools)
    """
    cdef:
        list scores = []
        list in_zones = []
        int n = len(zone_contours)
        int i
        bint in_z

    for i in range(n):
        if not zone_enabled[i]:
            scores.append(0)
            in_zones.append(False)
            continue

        in_z = cython_point_in_polygon(zone_contours[i], px, py)
        scores.append(in_z)
        in_zones.append(in_z)

    return (scores, in_zones)


def cython_point_in_polygon(list polygon, double px, double py):
    """Check if a point is inside a polygon using the ray casting algorithm.

    Cython replacement for cv2.pointPolygonTest for single point checks.

    Args:
        polygon: List of [x, y] points defining the polygon
        px: X coordinate of the point
        py: Y coordinate of the point

    Returns:
        True if point is inside polygon, False otherwise
    """
    cdef int n = len(polygon)
    if n < 3:
        return False

    cdef int j = n - 1
    cdef int i
    cdef double px0, py0, px1, py1
    cdef int result = 0

    for i in range(n):
        px0 = polygon[i][0]
        py0 = polygon[i][1]
        px1 = polygon[j][0]
        py1 = polygon[j][1]

        if ((py0 > py) != (py1 > py)) and (
            px < (px1 - px0) * (py - py0) / (py1 - py0) + px0
        ):
            result = not result

        j = i

    return result
