"""
Cython-accelerated recording segment operations.

Provides Cython implementations of the recording manager's hot paths:
- Motion heatmap computation (nested box x grid loop)
- Segment stats aggregation (frame-by-frame counting)
- Review overlap detection (sliding window)

Usage:
    from frigate.record.record_cython import (
        compute_motion_heatmap_cython,
        compute_motion_heatmap_from_array,
        check_overlap_with_reviews_cython,
        compute_segment_stats_cython,
    )
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import logging
import numpy as np

from libc.stdlib cimport malloc, free

logger = logging.getLogger(__name__)


cdef inline void _clamp_grid(int* x, int* y, int grid_size) noexcept:
    """Clamp grid coordinates to valid [0, grid_size-1] range."""
    if x[0] < 0: x[0] = 0
    if y[0] < 0: y[0] = 0
    if x[0] >= grid_size: x[0] = grid_size - 1
    if y[0] >= grid_size: y[0] = grid_size - 1


cdef class MotionHeatmapAccumulator:
    """C-level motion heatmap accumulator using a flat array.

    Uses a flat C array for the 16x16 grid and direct C loops
    to avoid Python object overhead per cell increment.

    The grid stores unsigned char counts (0-255) where each count
    represents how many motion boxes cover that cell.

    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    """

    cdef int GRID_SIZE
    cdef double cell_width
    cdef double cell_height
    cdef unsigned char* grid
    cdef int grid_size
    cdef bint has_data

    def __cinit__(self, int frame_width, int frame_height):
        self.GRID_SIZE = 16
        self.grid_size = self.GRID_SIZE * self.GRID_SIZE

        if frame_width > 0 and frame_height > 0:
            self.cell_width = frame_width / self.GRID_SIZE
            self.cell_height = frame_height / self.GRID_SIZE
        else:
            self.cell_width = 1.0
            self.cell_height = 1.0

        self.grid = <unsigned char*>malloc(self.grid_size * sizeof(unsigned char))
        if self.grid:
            for i in range(self.grid_size):
                self.grid[i] = 0
        self.has_data = 0

    def __dealloc__(self):
        if self.grid:
            free(self.grid)
            self.grid = NULL

    cdef inline void _increment_cell(self, int x, int y) noexcept:
        """Increment a grid cell, clamping to [0, 255]."""
        cdef int idx = y * self.GRID_SIZE + x
        if idx >= 0 and idx < self.grid_size:
            if self.grid[idx] < 255:
                self.grid[idx] += 1
            self.has_data = 1

    cdef void add_box(self, int x1, int y1, int x2, int y2):
        """Add a motion box to the heatmap.

        Maps pixel coordinates to grid cells and increments coverage.

        Args:
            x1: Left X coordinate
            y1: Top Y coordinate
            x2: Right X coordinate
            y2: Bottom Y coordinate
        """
        cdef:
            int grid_x1 = <int>((x1 / self.cell_width) if self.cell_width > 0 else x1)
            int grid_y1 = <int>((y1 / self.cell_height) if self.cell_height > 0 else y1)
            int grid_x2 = <int>((x2 / self.cell_width) if self.cell_width > 0 else x2)
            int grid_y2 = <int>((y2 / self.cell_height) if self.cell_height > 0 else y2)

        _clamp_grid(&grid_x1, &grid_y1, self.GRID_SIZE)
        _clamp_grid(&grid_x2, &grid_y2, self.GRID_SIZE)

        for y in range(grid_y1, grid_y2 + 1):
            for x in range(grid_x1, grid_x2 + 1):
                self._increment_cell(x, y)

    cdef void add_boxes_from_array(self, object boxes_array):
        """Add motion boxes from a numpy array (N x 4).

        Args:
            boxes_array: numpy array with shape (N, 4) containing [x1, y1, x2, y2]
        """
        cdef:
            int n = boxes_array.shape[0]
            int i, x1, y1, x2, y2
            double[:] flat = boxes_array.flatten()

        for i in range(n):
            x1 = <int>flat[i * 4]
            y1 = <int>flat[i * 4 + 1]
            x2 = <int>flat[i * 4 + 2]
            y2 = <int>flat[i * 4 + 3]
            self.add_box(x1, y1, x2, y2)

    def get_result(self):
        """Get the heatmap as a dict of {cell_index: intensity}.

        Returns:
            Dict mapping int cell index to unsigned char intensity (1-255),
            or None if no data was recorded.
        """
        cdef:
            dict result = {}
            int i

        if not self.has_data or self.grid == NULL:
            return None

        for i in range(self.grid_size):
            if self.grid[i] > 0:
                result[i] = self.grid[i]

        return result


def compute_motion_heatmap_cython(
    object motion_boxes,
    int frame_width,
    int frame_height,
):
    """Cython-accelerated motion heatmap computation.

    Replaces the Python loops in RecordingMaintainer._compute_motion_heatmap()
    with direct C-level array access.

    Args:
        motion_boxes: List of (x1, y1, x2, y2) tuples or numpy array
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Dict mapping string cell index to intensity (1-255), or None
    """
    cdef:
        MotionHeatmapAccumulator heatmap

    if motion_boxes is None:
        return None

    heatmap = MotionHeatmapAccumulator(frame_width, frame_height)

    if isinstance(motion_boxes, np.ndarray):
        if motion_boxes.size == 0:
            return None
        heatmap.add_boxes_from_array(motion_boxes)
    elif len(motion_boxes) == 0:
        return None
    else:
        for box in motion_boxes:
            if len(box) >= 4:
                heatmap.add_box(box[0], box[1], box[2], box[3])

    result = heatmap.get_result()
    if not result:
        return None
    return {str(k): v for k, v in result.items()}


def compute_motion_heatmap_from_array(
    object boxes_array,
    int frame_width,
    int frame_height,
):
    """Compute motion heatmap from a numpy array of boxes.

    Optimized for the case where motion boxes are already in a numpy array.

    Args:
        boxes_array: numpy array with shape (N, 4) containing [x1, y1, x2, y2]
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Dict mapping string cell index to intensity (1-255), or None
    """
    cdef:
        MotionHeatmapAccumulator heatmap

    if boxes_array is None:
        return None
    
    if hasattr(boxes_array, 'size') and boxes_array.size == 0:
        return None

    heatmap = MotionHeatmapAccumulator(frame_width, frame_height)
    heatmap.add_boxes_from_array(boxes_array)
    result = heatmap.get_result()
    if not result:
        return None
    return {str(k): v for k, v in result.items()}


cdef inline bint _check_review_overlap(
    double recording_start,
    double recording_end,
    double[:] review_starts,
    double[:] review_ends,
    double[:] review_pre_captures,
    double[:] review_post_captures,
    int review_start,
    int review_count,
) noexcept:
    """Check if a recording overlaps with any review in a sorted list.

    Uses early-exit optimization: stops checking when review starts
    after recording ends (since reviews are sorted).

    Args:
        recording_start: Recording segment start time
        recording_end: Recording segment end time
        review_starts: Sorted array of review start times
        review_ends: Array of review end times (None stored as -1)
        review_pre_captures: Pre-capture offset for each review
        review_post_captures: Post-capture offset for each review
        review_start: Starting index to check from
        review_count: Total number of reviews

    Returns:
        True if any review overlaps with the recording
    """
    cdef:
        int i
        double review_start_time
        double review_end_time
        double review_start_adj
        double review_end_adj

    for i in range(review_start, review_count):
        review_start_time = review_starts[i]
        review_end_time = review_ends[i]

        review_start_adj = review_start_time - review_pre_captures[i]
        
        # If adjusted review starts after recording ends, stop checking
        if review_start_adj > recording_end:
            return False

        # If review is in progress (end_time == -1) or ends after recording starts
        if review_end_time == -1 or review_end_time + review_post_captures[i] >= recording_start:
            return True

    return False


def check_overlap_with_reviews_cython(
    double recording_start,
    double recording_end,
    object review_data,
):
    """Check if a recording overlaps with any review segment.

    Optimized version of the overlap check in validate_and_move_segment().

    Args:
        recording_start: Recording segment start timestamp
        recording_end: Recording segment end timestamp
        review_data: List of tuples (start_time, end_time, severity, pre_capture, post_capture)

    Returns:
        Tuple of (has_overlap, severity) where severity is the severity
        of the overlapping review (or None if no overlap)
    """
    cdef:
        int n = len(review_data)
        int i
        double review_start_time
        double review_end_time
        double review_pre, review_post
        str severity
        double review_start_adj

    for i in range(n):
        review_start_time, review_end_time, severity, review_pre, review_post = review_data[i]
        
        review_start_adj = review_start_time - review_pre
        
        # If adjusted review starts after recording ends, stop checking
        if review_start_adj > recording_end:
            return (False, None)

        # If review is in progress (end_time == None) or ends after recording starts
        if review_end_time is None or review_end_time + review_post >= recording_start:
            return (True, severity)

    return (False, None)


def compute_segment_stats_cython(
    double[:] frame_timestamps,
    int[:] frame_motion_counts,
    int[:] frame_region_counts,
    Py_ssize_t start_idx,
    Py_ssize_t end_idx,
    double segment_start,
    double segment_end,
):
    """Cython-accelerated segment statistics computation.

    Counts frames, motion boxes, and regions within a time window
    using typed memory views for fast array access.

    Args:
        frame_timestamps: Sorted array of frame timestamps
        frame_motion_counts: Motion box count per frame
        frame_region_counts: Region count per frame
        start_idx: Start index in arrays
        end_idx: End index in arrays
        segment_start: Segment start timestamp
        segment_end: Segment end timestamp

    Returns:
        Tuple of (video_frame_count, motion_count, region_count)
    """
    cdef:
        Py_ssize_t video_frame_count = 0
        int motion_count = 0
        int region_count = 0
        Py_ssize_t i

    for i in range(start_idx, end_idx):
        if frame_timestamps[i] > segment_end:
            break
        if frame_timestamps[i] < segment_start:
            continue

        video_frame_count += 1
        motion_count += frame_motion_counts[i]
        region_count += frame_region_counts[i]

    return (video_frame_count, motion_count, region_count)


def compute_active_object_count_cython(
    double[:] frame_timestamps,
    int[:] obj_motionless_counts,
    unsigned char[:] obj_false_positives,
    Py_ssize_t obj_start,
    Py_ssize_t obj_end,
    double segment_start,
    double segment_end,
):
    """Count active objects across frames in a time window.

    Active object: not false_positive AND motionless_count == 0.

    Args:
        frame_timestamps: Sorted timestamps of frames
        obj_motionless_counts: motionless_count for each object entry
        obj_false_positives: false_positive flag (0 or 1) for each object
        obj_start: Start index in object arrays
        obj_end: End index in object arrays
        segment_start: Segment start time
        segment_end: Segment end time

    Returns:
        Count of active objects within window
    """
    cdef:
        int count = 0
        Py_ssize_t i

    for i in range(obj_start, obj_end):
        if frame_timestamps[i] > segment_end:
            break
        if frame_timestamps[i] < segment_start:
            continue

        if obj_false_positives[i] == 0 and obj_motionless_counts[i] == 0:
            count += 1

    return count


def compute_average_audio_cython(
    double[:] frame_timestamps,
    double[:] audio_dbfs,
    Py_ssize_t start_idx,
    Py_ssize_t end_idx,
    double segment_start,
    double segment_end,
):
    """Compute average audio dBFS within a time window.

    Args:
        frame_timestamps: Sorted array of frame timestamps
        audio_dbfs: Audio dBFS value per frame
        start_idx: Start index in arrays
        end_idx: End index in arrays
        segment_start: Segment start time
        segment_end: Segment end time

    Returns:
        Average audio dBFS, 0.0 if no values in window
    """
    cdef:
        int count = 0
        double total = 0.0
        Py_ssize_t i

    for i in range(start_idx, end_idx):
        if frame_timestamps[i] > segment_end:
            break
        if frame_timestamps[i] < segment_start:
            continue

        total += audio_dbfs[i]
        count += 1

    if count == 0:
        return 0.0

    return total / count


def compute_active_objects_with_motion_boxes(
    double[:] frame_timestamps,
    int[:] obj_false_positives,
    int[:] obj_motionless_counts,
    Py_ssize_t obj_start,
    Py_ssize_t obj_end,
    double segment_start,
    double segment_end,
    object all_motion_boxes,
):
    """Count active objects and collect motion boxes in a single Cython pass.

    Enhanced version of compute_active_object_count_cython that also
    collects motion boxes from in-range frames.

    Active object: not false_positive AND motionless_count == 0.

    This replaces the Python loop in RecordingMaintainer.segment_stats()
    that iterates over object_frames with nested list comprehensions.

    The caller should pre-extract:
    - frame_timestamps: timestamps of object frames
    - obj_false_positives: flat array of false_positive flags per object entry
    - obj_motionless_counts: flat array of motionless_count per object entry
    - all_motion_boxes: list to append motion boxes from in-range frames

    Args:
        frame_timestamps: Sorted timestamps of object frames
        obj_false_positives: 0 or 1 per object entry (flat across all frames)
        obj_motionless_counts: motionless_count per object entry (flat across all frames)
        obj_start: Start index in object arrays
        obj_end: End index in object arrays
        segment_start: Segment start time
        segment_end: Segment end time
        all_motion_boxes: List to extend with motion boxes from in-range frames
                          (caller-managed, passed for convenience)

    Returns:
        Count of active objects within window
    """
    cdef:
        int count = 0
        Py_ssize_t i

    for i in range(obj_start, obj_end):
        if frame_timestamps[i] > segment_end:
            break
        if frame_timestamps[i] < segment_start:
            continue

        if obj_false_positives[i] == 0 and obj_motionless_counts[i] == 0:
            count += 1

    return count


def compute_segment_active_stats(
    double[:] frame_timestamps,
    int[:] obj_false_positives,
    int[:] obj_motionless_counts,
    Py_ssize_t obj_start,
    Py_ssize_t obj_end,
    double segment_start,
    double segment_end,
):
    """Compute active object stats within a time window.

    Returns both the count of active objects AND the count of in-range frames.

    Args:
        frame_timestamps: Sorted timestamps of object frames
        obj_false_positives: 0 or 1 per object entry
        obj_motionless_counts: motionless_count per object entry
        obj_start: Start index in object arrays
        obj_end: End index in object arrays
        segment_start: Segment start time
        segment_end: Segment end time

    Returns:
        Tuple of (in_range_frame_count, active_object_count)
    """
    cdef:
        int active_count = 0
        int in_range_count = 0
        Py_ssize_t i

    for i in range(obj_start, obj_end):
        if frame_timestamps[i] > segment_end:
            break
        if frame_timestamps[i] < segment_start:
            continue

        in_range_count += 1

        if obj_false_positives[i] == 0 and obj_motionless_counts[i] == 0:
            active_count += 1

    return (in_range_count, active_count)
