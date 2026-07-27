"""Tests for frigate/util/object_cython module."""

from unittest import TestCase, main

from frigate.util.object_cython import (
    cython_area,
    cython_box_inside,
    cython_get_cluster_boundary,
    cython_get_cluster_candidates,
    cython_get_cluster_region,
    cython_intersection,
    cython_intersection_over_union,
    cython_reduce_boxes,
)


class TestObjectCython(TestCase):
    def test_intersection_with_overlap(self):
        result = cython_intersection([10, 10, 30, 30], [20, 20, 40, 40])
        self.assertEqual(result, (20, 20, 30, 30))

    def test_intersection_no_overlap_x(self):
        result = cython_intersection([10, 10, 20, 20], [30, 30, 40, 40])
        self.assertIsNone(result)

    def test_intersection_no_overlap_y(self):
        result = cython_intersection([10, 10, 20, 20], [5, 30, 25, 50])
        self.assertIsNone(result)

    def test_intersection_touched_edges(self):
        result = cython_intersection([10, 10, 20, 20], [20, 10, 30, 20])
        self.assertIsNone(result)

    def test_area(self):
        result = cython_area([0, 0, 10, 10])
        self.assertEqual(result, 121)

    def test_area_nonzero_origin(self):
        result = cython_area([10, 10, 20, 20])
        self.assertEqual(result, 121)

    def test_box_inside(self):
        self.assertTrue(cython_box_inside([0, 0, 100, 100], [10, 10, 50, 50]))

    def test_box_outside(self):
        self.assertFalse(cython_box_inside([0, 0, 100, 100], [0, 0, 150, 150]))

    def test_box_on_boundary(self):
        self.assertTrue(cython_box_inside([0, 0, 100, 100], [0, 0, 100, 100]))

    def test_iou_identical_boxes(self):
        result = cython_intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10])
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_iou_no_overlap(self):
        result = cython_intersection_over_union([0, 0, 10, 10], [100, 100, 110, 110])
        self.assertAlmostEqual(result, 0.0, places=3)

    def test_iou_partial_overlap(self):
        result = cython_intersection_over_union([0, 0, 50, 50], [25, 25, 75, 75])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_reduce_boxes_separate(self):
        boxes = [[0, 0, 50, 50], [100, 100, 150, 150]]
        result = cython_reduce_boxes(boxes, 0.99)
        self.assertEqual(len(result), 2)

    def test_reduce_boxes_overlapping_merge(self):
        boxes = [[0, 0, 50, 50], [40, 40, 90, 90], [100, 100, 150, 150]]
        result = cython_reduce_boxes(boxes, 0.0)
        self.assertEqual(len(result), 2)

    def test_reduce_boxes_no_overlap(self):
        boxes = [[0, 0, 50, 50], [100, 100, 150, 150], [200, 200, 250, 250]]
        result = cython_reduce_boxes(boxes, 0.5)
        self.assertEqual(len(result), 3)

    def test_reduce_boxes_all_merge(self):
        boxes = [[0, 0, 100, 100], [10, 10, 110, 110], [20, 20, 120, 120]]
        result = cython_reduce_boxes(boxes, 0.0)
        self.assertEqual(len(result), 1)

    def test_get_cluster_region(self):
        frame_shape = (1080, 1920)
        boxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
        result = cython_get_cluster_region(frame_shape, 100, [0, 1], boxes)
        self.assertEqual(len(result), 4)

    def test_get_cluster_boundary(self):
        box = [10, 10, 50, 50]
        result = cython_get_cluster_boundary(box, 50)
        self.assertEqual(len(result), 4)

    def test_get_cluster_candidates_overlapping(self):
        frame_shape = (1080, 1920)
        boxes = [[10, 10, 50, 50], [40, 40, 80, 80], [100, 100, 140, 140]]
        result = cython_get_cluster_candidates(frame_shape, 50, boxes)
        self.assertEqual(len(result), 2)

    def test_get_cluster_candidates_separate(self):
        frame_shape = (1080, 1920)
        boxes = [[10, 10, 30, 30], [100, 100, 140, 140], [200, 200, 240, 240]]
        result = cython_get_cluster_candidates(frame_shape, 50, boxes)
        self.assertEqual(len(result), 3)

    def test_get_cluster_candidates_single(self):
        frame_shape = (1080, 1920)
        boxes = [[50, 50, 100, 100]]
        result = cython_get_cluster_candidates(frame_shape, 50, boxes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0])

    def test_cluster_boundary_matches_python_logic(self):
        """Verify cython_get_cluster_boundary matches Python get_cluster_boundary logic."""
        box = [10, 10, 50, 50]
        result = cython_get_cluster_boundary(box, 50)
        # box_width=40, box_height=40, max_region_area=16000
        # max_region_size = max(50, int(sqrt(16000))) = max(50, 126) = 126
        # centroid = (30, 30)
        # max_x_dist = int(126 - 20*1.1) = int(104) = 104
        expected = [30 - 104, 30 - 104, 30 + 104, 30 + 104]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    main(verbosity=2)
