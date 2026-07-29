import logging
import random
import string
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import distance as dist

from frigate.config import DetectConfig
from frigate.track import ObjectTracker
from frigate.track.tracking_cython import (
    cython_centroid_build_centroids,
    cython_centroid_compute_assignment,
    cython_centroid_match_and_update,
    cython_update_frame_times_and_motionless,
)
from frigate.util.image import intersection_over_union

logger = logging.getLogger(__name__)


class CentroidTracker(ObjectTracker):
    def __init__(self, config: DetectConfig):
        self.tracked_objects: dict[str, dict[str, Any]] = {}
        self.untracked_object_boxes: list[tuple[int, int, int, int]] = []
        self.disappeared: dict[str, Any] = {}
        self.positions: dict[str, Any] = {}
        self.max_disappeared = config.max_disappeared
        self.detect_config = config

    def register(self, obj: dict[str, Any]) -> None:
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        id = f"{obj['frame_time']}-{rand_id}"
        obj["id"] = id
        obj["start_time"] = obj["frame_time"]
        obj["motionless_count"] = 0
        obj["position_changes"] = 0
        self.tracked_objects[id] = obj
        self.disappeared[id] = 0
        self.positions[id] = {
            "xmins": [],
            "ymins": [],
            "xmaxs": [],
            "ymaxs": [],
            "xmin": 0,
            "ymin": 0,
            "xmax": self.detect_config.width,
            "ymax": self.detect_config.height,
        }

    def deregister(self, id: str) -> None:
        del self.tracked_objects[id]
        del self.disappeared[id]

    # tracks the current position of the object based on the last N bounding boxes
    # returns False if the object has moved outside its previous position
    def update_position(self, id: str, box: tuple[int, int, int, int]) -> bool:
        position = self.positions[id]
        position_box = (
            position["xmin"],
            position["ymin"],
            position["xmax"],
            position["ymax"],
        )

        xmin, ymin, xmax, ymax = box

        iou = intersection_over_union(position_box, box)

        # if the iou drops below the threshold
        # assume the object has moved to a new position and reset the computed box
        if iou < 0.6:
            self.positions[id] = {
                "xmins": [xmin],
                "ymins": [ymin],
                "xmaxs": [xmax],
                "ymaxs": [ymax],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
            return False

        # if there are less than 10 entries for the position, add the bounding box
        # and recompute the position box
        if len(position["xmins"]) < 10:
            position["xmins"].append(xmin)
            position["ymins"].append(ymin)
            position["xmaxs"].append(xmax)
            position["ymaxs"].append(ymax)
            # by using percentiles here, we hopefully remove outliers
            position["xmin"] = np.percentile(position["xmins"], 15)
            position["ymin"] = np.percentile(position["ymins"], 15)
            position["xmax"] = np.percentile(position["xmaxs"], 85)
            position["ymax"] = np.percentile(position["ymaxs"], 85)

        return True

    def is_expired(self, id: str) -> bool:
        obj = self.tracked_objects[id]
        # get the max frames for this label type or the default
        max_frames = self.detect_config.stationary.max_frames.objects.get(
            obj["label"], self.detect_config.stationary.max_frames.default
        )

        # if there is no max_frames for this label type, continue
        if max_frames is None:
            return False

        # if the object has exceeded the max_frames setting, deregister
        if (
            obj["motionless_count"] - self.detect_config.stationary.threshold
            > max_frames
        ):
            return True

        return False

    def update(self, id: str, new_obj: dict[str, Any]) -> None:
        self.disappeared[id] = 0
        # update the motionless count if the object has not moved to a new position
        if self.update_position(id, new_obj["box"]):
            self.tracked_objects[id]["motionless_count"] += 1
            if self.is_expired(id):
                self.deregister(id)
                return
        else:
            # register the first position change and then only increment if
            # the object was previously stationary
            if (
                self.tracked_objects[id]["position_changes"] == 0
                or self.tracked_objects[id]["motionless_count"]
                >= self.detect_config.stationary.threshold
            ):
                self.tracked_objects[id]["position_changes"] += 1
            self.tracked_objects[id]["motionless_count"] = 0

        self.tracked_objects[id].update(new_obj)

    def update_frame_times(self, frame_name: str, frame_time: float) -> None:
        cython_update_frame_times_and_motionless(
            self.tracked_objects,
            frame_time,
            self.is_expired,
            self.deregister,
        )

    def match_and_update(
        self,
        frame_name: str,
        frame_time: float,
        detections: list[tuple[Any, Any, Any, Any, Any, Any]],
    ) -> None:
        # Use Cython for detection grouping and disappeared counting
        detection_groups, expired_ids = cython_centroid_match_and_update(
            detections,
            self.tracked_objects,
            self.disappeared,
            self.max_disappeared,
            self.register,
            self.update,
            self.deregister,
            self.is_expired,
            frame_time,
        )

        if len(detections) == 0:
            return

        # Process each label group with Cython-accelerated assignment
        for label, group in detection_groups.items():
            current_objects = [
                o for o in self.tracked_objects.values() if o["label"] == label
            ]
            current_ids = [o["id"] for o in current_objects]

            if len(current_objects) == 0:
                # No existing objects for this label, register all new detections
                for obj in group:
                    self.register(obj)
                continue

            # Compute centroids using Cython
            current_centroids = cython_centroid_build_centroids(current_objects)

            # Compute centroids of new objects in Python (still fast)
            for obj in group:
                centroid_x = int((obj["box"][0] + obj["box"][2]) / 2.0)
                centroid_y = int((obj["box"][1] + obj["box"][3]) / 2.0)
                obj["centroid"] = (centroid_x, centroid_y)

            new_centroids = cython_centroid_build_centroids(group)

            # Compute assignment using Cython
            rows, cols = cython_centroid_compute_assignment(
                current_centroids, new_centroids
            )

            if len(rows) == 0:
                # No matches found, register all as new, deregister all existing
                for obj in group:
                    self.register(obj)
                for row_idx in range(len(current_ids)):
                    id = current_ids[row_idx]
                    if self.disappeared.get(id, 0) >= self.max_disappeared:
                        self.deregister(id)
                    else:
                        self.disappeared[id] = self.disappeared.get(id, 0) + 1
                continue

            # Apply matches using Cython
            unused_rows = set(range(len(current_ids))).difference(rows)
            unused_cols = set(range(len(group))).difference(cols)

            # Update matched objects
            for row, col in zip(rows, cols):
                object_id = current_ids[row]
                self.update(object_id, group[col])

            # Handle unmatched current objects (disappeared)
            for row in unused_rows:
                if row < len(current_ids):
                    id = current_ids[row]
                    if self.disappeared.get(id, 0) >= self.max_disappeared:
                        self.deregister(id)
                    else:
                        self.disappeared[id] = self.disappeared.get(id, 0) + 1

            # Handle unmatched new detections (register as new)
            for col in unused_cols:
                if col < len(group):
                    self.register(group[col])
