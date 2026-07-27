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
    a0, a1, a2, a3 = box_a
    b0, b1, b2, b3 = box_b
    x_min = max(a0, b0)
    y_min = max(a1, b1)
    x_max = min(a2, b2)
    y_max = min(a3, b3)

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
    b0, b1, b2, b3 = box
    return (b2 - b0 + 1) * (b3 - b1 + 1)


def cython_box_inside(b1, b2):
    """Check if b2 is inside b1.

    Args:
        b1: [x_min, y_min, x_max, y_max] - outer box
        b2: [x_min, y_min, x_max, y_max] - inner box candidate

    Returns:
        True if b2 is inside b1
    """
    b10, b11, b12, b13 = b1
    b20, b21, b22, b23 = b2
    return b20 >= b10 and b21 >= b11 and b22 <= b12 and b23 <= b13


def cython_intersection_over_union(box_a, box_b):
    """Calculate IoU between two boxes.

    Args:
        box_a: [x_min, y_min, x_max, y_max]
        box_b: [x_min, y_min, x_max, y_max]

    Returns:
        IoU value between 0 and 1
    """
    a0, a1, a2, a3 = box_a
    b0, b1, b2, b3 = box_b
    ix = max(a0, b0)
    iy = max(a1, b1)
    ix2 = min(a2, b2)
    iy2 = min(a3, b3)

    if ix >= ix2 or iy >= iy2:
        return 0.0

    inter_area = (ix2 - ix + 1) * (iy2 - iy + 1)

    if inter_area == 0:
        return 0.0

    box_a_area = (a2 - a0 + 1) * (a3 - a1 + 1)
    box_b_area = (b2 - b0 + 1) * (b3 - b1 + 1)

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
        b0, b1, b2, b3 = box
        matched = 0
        for cluster in clusters:
            if cython_intersection_over_union(box, cluster) > iou_threshold:
                matched = 1
                if b0 < cluster[0]:
                    cluster[0] = b0
                if b1 < cluster[1]:
                    cluster[1] = b1
                if b2 > cluster[2]:
                    cluster[2] = b2
                if b3 > cluster[3]:
                    cluster[3] = b3

        if not matched:
            clusters.append([b0, b1, b2, b3])

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
        bx = boxes[b]
        b0, b1, b2, b3 = bx
        if b0 < min_x:
            min_x = b0
        if b1 < min_y:
            min_y = b1
        if b2 > max_x:
            max_x = b2
        if b3 > max_y:
            max_y = b3

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
        bx = boxes[b]
        b0, b1, b2, b3 = bx
        if b0 < min_x:
            min_x = b0
        if b1 < min_y:
            min_y = b1
        if b2 > max_x:
            max_x = b2
        if b3 > max_y:
            max_y = b3

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
    b0, b1, b2, b3 = box
    box_width = b2 - b0
    box_height = b3 - b1
    max_region_area = abs(box_width * box_height) / 0.1
    max_region_size = max(min_region, int(max_region_area**0.5))

    centroid_x = b0 + box_width / 2
    centroid_y = b1 + box_height / 2

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

            bx = boxes[compare_idx]
            if not cython_box_inside(boundary, bx):
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
