"""
Cython-accelerated object clustering and bounding box operations.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from typing import Any


def cython_intersection(box_a, box_b):
    """Calculate intersection rectangle of two bounding boxes.

    Args:
        box_a: [x_min, y_min, x_max, y_max]
        box_b: [x_min, y_min, x_max, y_max]

    Returns:
        [x_min, y_min, x_max, y_max] or None if no overlap
    """
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    if x_min >= x_max or y_min >= y_max:
        return None

    return (x_min, y_min, x_max, y_max)


def cython_area(box):
    """Calculate area of a bounding box.

    Args:
        box: [x_min, y_min, x_max, y_max]

    Returns:
        Area as int
    """
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def cython_box_inside(b1, b2):
    """Check if b2 is inside b1.

    Args:
        b1: [x_min, y_min, x_max, y_max] - outer box
        b2: [x_min, y_min, x_max, y_max] - inner box candidate

    Returns:
        True if b2 is inside b1
    """
    return b2[0] >= b1[0] and b2[1] >= b1[1] and b2[2] <= b1[2] and b2[3] <= b1[3]


def cython_intersection_over_union(box_a, box_b):
    """Calculate IoU between two boxes.

    Args:
        box_a: [x_min, y_min, x_max, y_max]
        box_b: [x_min, y_min, x_max, y_max]

    Returns:
        IoU value between 0 and 1
    """
    intersect = cython_intersection(box_a, box_b)

    if intersect is None:
        return 0.0

    inter_area = max(0, intersect[2] - intersect[0] + 1) * max(
        0, intersect[3] - intersect[1] + 1
    )

    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    return inter_area / (box_a_area + box_b_area - inter_area)


def cython_reduce_boxes(boxes, double iou_threshold):
    """Reduce overlapping boxes by merging clusters with high IoU.

    Args:
        boxes: List of [x_min, y_min, x_max, y_max] boxes
        iou_threshold: IoU threshold above which boxes are merged

    Returns:
        List of merged box tuples
    """
    clusters = []

    for box in boxes:
        matched = 0
        for cluster in clusters:
            if cython_intersection_over_union(box, cluster) > iou_threshold:
                matched = 1
                if box[0] < cluster[0]:
                    cluster[0] = box[0]
                if box[1] < cluster[1]:
                    cluster[1] = box[1]
                if box[2] > cluster[2]:
                    cluster[2] = box[2]
                if box[3] > cluster[3]:
                    cluster[3] = box[3]

        if not matched:
            clusters.append([box[0], box[1], box[2], box[3]])

    return [tuple(c) for c in clusters]


def cython_get_cluster_region(frame_shape, int min_region, cluster, boxes):
    """Calculate bounding region for a cluster of boxes.

    Args:
        frame_shape: (height, width)
        min_region: Minimum region size
        cluster: List of box indices
        boxes: List of all boxes [x_min, y_min, x_max, y_max]

    Returns:
        Cluster region [x_min, y_min, x_max, y_max]
    """
    min_x = frame_shape[1]
    min_y = frame_shape[0]
    max_x = 0
    max_y = 0

    for b in cluster:
        box = boxes[b]
        if box[0] < min_x:
            min_x = box[0]
        if box[1] < min_y:
            min_y = box[1]
        if box[2] > max_x:
            max_x = box[2]
        if box[3] > max_y:
            max_y = box[3]

    return _cython_calculate_region(
        frame_shape, min_x, min_y, max_x, max_y, min_region, 1.35
    )


def cython_get_cluster_region_from_grid(frame_shape, int min_region, cluster, boxes):
    """Calculate bounding region for a cluster using min/max of coordinates.

    Args:
        frame_shape: (height, width)
        min_region: Minimum region size
        cluster: List of box indices
        boxes: List of all boxes [x_min, y_min, x_max, y_max]

    Returns:
        Cluster region [x_min, y_min, x_max, y_max]
    """
    min_x = frame_shape[1]
    min_y = frame_shape[0]
    max_x = 0
    max_y = 0

    for b in cluster:
        box = boxes[b]
        if box[0] < min_x:
            min_x = box[0]
        if box[1] < min_y:
            min_y = box[1]
        if box[2] > max_x:
            max_x = box[2]
        if box[3] > max_y:
            max_y = box[3]

    return _cython_calculate_region(
        frame_shape, min_x, min_y, max_x, max_y, min_region, 1.0
    )


def cython_get_cluster_boundary(box, int min_region):
    """Calculate the max region boundary for clustering.

    Args:
        box: [x_min, y_min, x_max, y_max]
        min_region: Minimum region size

    Returns:
        [x_min, y_min, x_max, y_max] boundary
    """
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    max_region_area = abs(box_width * box_height) / 0.1
    max_region_size = max(min_region, int(max_region_area**0.5))

    centroid_x = box[0] + box_width / 2
    centroid_y = box[1] + box_height / 2

    max_x_dist = int(max_region_size - box_width / 2 * 1.1)
    max_y_dist = int(max_region_size - box_height / 2 * 1.1)

    return [
        int(centroid_x - max_x_dist),
        int(centroid_y - max_y_dist),
        int(centroid_x + max_x_dist),
        int(centroid_y + max_y_dist),
    ]


def cython_get_cluster_candidates(frame_shape, int min_region, boxes):
    """Cluster nearby boxes together.

    O(n^2) clustering algorithm that groups overlapping boxes.

    Args:
        frame_shape: (height, width)
        min_region: Minimum region size
        boxes: List of [x_min, y_min, x_max, y_max] boxes

    Returns:
        List of cluster lists, each containing box indices
    """
    cluster_candidates = []
    used = [False] * len(boxes)

    for current_idx in range(len(boxes)):
        if used[current_idx]:
            continue

        cluster = [current_idx]
        used[current_idx] = True
        boundary = cython_get_cluster_boundary(boxes[current_idx], min_region)

        for compare_idx in range(len(boxes)):
            if used[compare_idx]:
                continue

            if not cython_box_inside(boundary, boxes[compare_idx]):
                continue

            potential = cluster + [compare_idx]
            region = cython_get_cluster_region(
                frame_shape, min_region, potential, boxes
            )

            should_cluster = True
            if (region[2] - region[0]) > min_region:
                for bi in potential:
                    box_area = cython_area(boxes[bi])
                    region_area = cython_area(region)
                    if box_area / region_area < 0.05:
                        should_cluster = False
                        break

            if should_cluster:
                cluster.append(compare_idx)
                used[compare_idx] = True

        cluster_candidates.append(cluster)

    # Deduplicate clusters
    seen = set()
    unique = []
    for cluster in cluster_candidates:
        key = tuple(sorted(cluster))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))

    return unique


cdef list _cython_calculate_region(
    frame_shape, double xmin, double ymin, double xmax, double ymax,
    int min_region, double multiplier
):
    """Internal region calculation helper."""
    size = int(((max(xmax - xmin, ymax - ymin) * multiplier)) // 4 * 4)
    if size < min_region:
        size = min_region

    x_offset = int((xmax - xmin) / 2.0 + xmin - size / 2.0)
    if x_offset < 0:
        x_offset = 0
    elif x_offset > (frame_shape[1] - size):
        x_offset = max(0, (frame_shape[1] - size))

    y_offset = int((ymax - ymin) / 2.0 + ymin - size / 2.0)
    if y_offset < 0:
        y_offset = 0
    elif y_offset > (frame_shape[0] - size):
        y_offset = max(0, (frame_shape[0] - size))

    return [x_offset, y_offset, x_offset + size, y_offset + size]
