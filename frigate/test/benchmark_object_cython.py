"""
Benchmarks: Cython-accelerated vs pure-Python object clustering and image ops.

Run: python3 -u -m unittest frigate.test.benchmark_object_cython

All pure-Python reference implementations are embedded here — no Frigate
dependencies needed — so the benchmark runs in any Python 3.12 env.
"""

import unittest
import time
import numpy as np


# ── Pure-Python reference implementations ──────────────────────────────────────

def py_area(box):
    """Calculate area of a bounding box (Python reference)."""
    b0, b1, b2, b3 = box
    return (b2 - b0 + 1) * (b3 - b1 + 1)


def py_intersection(box_a, box_b):
    """Intersection of two boxes (Python reference)."""
    a0, a1, a2, a3 = box_a
    b0, b1, b2, b3 = box_b
    ix, iy = max(a0, b0), max(a1, b1)
    ix2, iy2 = min(a2, b2), min(a3, b3)
    if ix >= ix2 or iy >= iy2:
        return None
    return (ix, iy, ix2, iy2)


def py_iou(box_a, box_b):
    """Intersection-over-union (Python reference)."""
    a0, a1, a2, a3 = box_a
    b0, b1, b2, b3 = box_b
    ix, iy = max(a0, b0), max(a1, b1)
    ix2, iy2 = min(a2, b2), min(a3, b3)
    if ix >= ix2 or iy >= iy2:
        return 0.0
    inter_area = (ix2 - ix + 1) * (iy2 - iy + 1)
    if inter_area == 0:
        return 0.0
    return inter_area / ((a2 - a0 + 1) * (a3 - a1 + 1) + (b2 - b0 + 1) * (b3 - b1 + 1) - inter_area)


def py_box_inside(b1, b2):
    """Check if b2 is inside b1 (Python reference)."""
    b10, b11, b12, b13 = b1
    b20, b21, b22, b23 = b2
    return b20 >= b10 and b21 >= b11 and b22 <= b12 and b23 <= b13


def py_reduce_boxes(boxes, iou_threshold=0.0):
    """Reduce overlapping boxes by merging high-IoU clusters (Python reference)."""
    clusters = []
    for box in boxes:
        b0, b1, b2, b3 = box
        matched = 0
        for cluster in clusters:
            if py_iou(box, cluster) > iou_threshold:
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


def py_get_cluster_boundary(box, min_region):
    """Cluster boundary for clustering (Python reference)."""
    b0, b1, b2, b3 = box
    bw, bh = b2 - b0, b3 - b1
    max_region_size = max(min_region, int((bw * bh / 0.1) ** 0.5))
    cx, cy = b0 + bw / 2, b1 + bh / 2
    mx = int(max_region_size - bw / 2 * 1.1)
    my = int(max_region_size - bh / 2 * 1.1)
    return [int(cx - mx), int(cy - my), int(cx + mx), int(cy + my)]


def py_calculate_region(frame_shape, xmin, ymin, xmax, ymax, min_region, multiplier=1.35):
    """Internal region calculation (Python reference)."""
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


def py_get_cluster_region(frame_shape, min_region, cluster, boxes):
    """Calculate bounding region for a cluster of boxes (Python reference)."""
    min_x = frame_shape[1]
    min_y = frame_shape[0]
    max_x = 0
    max_y = 0
    for idx in cluster:
        bx = boxes[idx]
        b0, b1, b2, b3 = bx
        if b0 < min_x:
            min_x = b0
        if b1 < min_y:
            min_y = b1
        if b2 > max_x:
            max_x = b2
        if b3 > max_y:
            max_y = b3
    return py_calculate_region(frame_shape, min_x, min_y, max_x, max_y, min_region, 1.35)


def py_get_cluster_candidates(frame_shape, min_region, boxes):
    """Cluster nearby boxes together (Python reference, O(n^2))."""
    cluster_candidates = []
    used = [False] * len(boxes)

    for ci in range(len(boxes)):
        if used[ci]:
            continue
        cluster = [ci]
        used[ci] = True
        boundary = py_get_cluster_boundary(boxes[ci], min_region)
        for cj in range(len(boxes)):
            if used[cj]:
                continue
            bx = boxes[cj]
            if not py_box_inside(boundary, bx):
                continue
            potential = cluster + [cj]
            region = py_get_cluster_region(frame_shape, min_region, potential, boxes)
            should_cluster = True
            if (region[2] - region[0]) > min_region:
                for bi in potential:
                    if py_area(boxes[bi]) / py_area(region) < 0.05:
                        should_cluster = False
                        break
            if should_cluster:
                cluster.append(cj)
                used[cj] = True
        cluster_candidates.append(cluster)

    # Deduplicate
    seen = set()
    unique = []
    for cluster in cluster_candidates:
        key = tuple(sorted(cluster))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique


# ── Import Cython implementations ──────────────────────────────────────────────

from frigate.util.image_cython import (
    intersection as cy_intersection,
    area as cy_area,
    intersection_over_union as cy_iou,
)
from frigate.util.object_cython import (
    cython_box_inside as cy_box_inside,
    cython_reduce_boxes as cy_reduce_boxes,
    cython_get_cluster_region as cy_get_cluster_region,
    cython_get_cluster_candidates as cy_get_cluster_candidates,
)


class BenchmarkObjectCython(unittest.TestCase):
    """Performance benchmarks comparing Python vs Cython."""

    def test_area(self):
        iterations = 1_000_000
        box = (10, 10, 100, 100)
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_area(box)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_area(box)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_area(box), cy_area(box))
        print(f"  area:         Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations)")

    def test_intersection(self):
        iterations = 1_000_000
        box_a, box_b = (10, 10, 50, 50), (30, 30, 80, 80)
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_intersection(box_a, box_b)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_intersection(box_a, box_b)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_intersection(box_a, box_b), cy_intersection(box_a, box_b))
        print(f"  intersection: Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations)")

    def test_iou(self):
        iterations = 1_000_000
        box_a, box_b = (10, 10, 50, 50), (30, 30, 80, 80)
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_iou(box_a, box_b)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_iou(box_a, box_b)
        cy_t = time.perf_counter() - t0
        self.assertAlmostEqual(py_iou(box_a, box_b), cy_iou(box_a, box_b), places=10)
        print(f"  IoU:          Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations)")

    def test_box_inside(self):
        iterations = 1_000_000
        b1, b2 = (0, 0, 100, 100), (10, 10, 50, 50)
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_box_inside(b1, b2)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_box_inside(b1, b2)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_box_inside(b1, b2), cy_box_inside(b1, b2))
        print(f"  box_inside:   Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations)")

    def test_reduce_boxes(self):
        iterations = 10_000
        np.random.seed(42)
        boxes = [[int(np.random.randint(0, 500)), int(np.random.randint(0, 500)),
                  int(np.random.randint(0, 500)), int(np.random.randint(0, 500))]
                 for _ in range(100)]
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_reduce_boxes(boxes, 0.0)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_reduce_boxes(boxes, 0.0)
        cy_t = time.perf_counter() - t0
        self.assertEqual(len(py_reduce_boxes(boxes, 0.0)), len(cy_reduce_boxes(boxes, 0.0)))
        print(f"  reduce_boxes: Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations, 100 boxes)")

    def test_cluster_region(self):
        iterations = 100_000
        frame_shape = (1080, 1920)
        boxes = [[100, 100, 200, 200], [150, 150, 250, 250], [300, 300, 400, 400]]
        cluster = [0, 1]
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_get_cluster_region(frame_shape, 100, cluster, boxes)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_get_cluster_region(frame_shape, 100, cluster, boxes)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_get_cluster_region(frame_shape, 100, cluster, boxes),
                         cy_get_cluster_region(frame_shape, 100, cluster, boxes))
        print(f"  cluster_region: Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations)")

    def test_cluster_candidates(self):
        iterations = 1_000
        frame_shape = (1080, 1920)
        boxes = []
        for _ in range(50):
            x, y = np.random.randint(50, 150), np.random.randint(50, 150)
            boxes.append([x, y, x + np.random.randint(20, 60), y + np.random.randint(20, 60)])
        for _ in range(50):
            x, y = np.random.randint(800, 950), np.random.randint(600, 750)
            boxes.append([x, y, x + np.random.randint(20, 60), y + np.random.randint(20, 60)])
        t0 = time.perf_counter()
        for _ in range(iterations):
            py_get_cluster_candidates(frame_shape, 50, boxes)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iterations):
            cy_get_cluster_candidates(frame_shape, 50, boxes)
        cy_t = time.perf_counter() - t0
        self.assertEqual(len(py_get_cluster_candidates(frame_shape, 50, boxes)),
                         len(cy_get_cluster_candidates(frame_shape, 50, boxes)))
        print(f"  cluster_candidates: Python={py_t:.4f}s  Cython={cy_t:.4f}s  "
              f"Speedup={py_t/cy_t:.1f}x  ({iterations:,} iterations, {len(boxes)} boxes)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
