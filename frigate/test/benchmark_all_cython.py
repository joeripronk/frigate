"""
Comprehensive benchmarks for all Cython-accelerated Frigate modules.

Run: python3 -u -m unittest frigate.test.benchmark_all_cython

Modules covered:
  1. frigate/util/image_cython.pyx       - intersection, area, IoU, yuv_to_3_channel_yuv
  2. frigate/util/object_cython.pyx      - box_inside, reduce_boxes
  3. frigate/embeddings/util_cython.pyx  - normalize, normalize_inplace, update_stats
  4. frigate/events/audio_cython.pyx     - select_top_k, build_detections, detect_raw
  5. frigate/detectors/detection_runners_cython.pyx - transpose, face_normalization, copyto_inplace
"""

import unittest
import time
import sys
import importlib.util
import numpy as np


def load_cython_module(name, path):
    """Load a Cython .so file directly without triggering __init__.py imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load Cython modules directly to avoid package __init__.py imports
image_cython = load_cython_module(
    "frigate.util.image_cython",
    "frigate/util/image_cython.cpython-312-x86_64-linux-gnu.so",
)
object_cython = load_cython_module(
    "frigate.util.object_cython",
    "frigate/util/object_cython.cpython-312-x86_64-linux-gnu.so",
)
embeddings_cython = load_cython_module(
    "frigate.embeddings.util_cython",
    "frigate/embeddings/util_cython.cpython-312-x86_64-linux-gnu.so",
)
audio_cython = load_cython_module(
    "frigate.events.audio_cython",
    "frigate/events/audio_cython.cpython-312-x86_64-linux-gnu.so",
)
detection_cython = load_cython_module(
    "frigate.detectors.detection_runners_cython",
    "frigate/detectors/detection_runners_cython.cpython-312-x86_64-linux-gnu.so",
)


# ============================================================================
# PURE-PYTHON REFERENCE IMPLEMENTATIONS (no Frigate dependencies)
# ============================================================================

# --- image/object clustering references ---

def py_area(box):
    b0, b1, b2, b3 = box
    return (b2 - b0 + 1) * (b3 - b1 + 1)


def py_intersection(box_a, box_b):
    a0, a1, a2, a3 = box_a
    b0, b1, b2, b3 = box_b
    ix, iy = max(a0, b0), max(a1, b1)
    ix2, iy2 = min(a2, b2), min(a3, b3)
    if ix >= ix2 or iy >= iy2:
        return None
    return (ix, iy, ix2, iy2)


def py_iou(box_a, box_b):
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
    b10, b11, b12, b13 = b1
    b20, b21, b22, b23 = b2
    return b20 >= b10 and b21 >= b11 and b22 <= b12 and b23 <= b13


def py_reduce_boxes(boxes, iou_threshold=0.0):
    clusters = []
    for box in boxes:
        b0, b1, b2, b3 = box
        matched = 0
        for cluster in clusters:
            if py_iou(box, cluster) > iou_threshold:
                matched = 1
                if b0 < cluster[0]: cluster[0] = b0
                if b1 < cluster[1]: cluster[1] = b1
                if b2 > cluster[2]: cluster[2] = b2
                if b3 > cluster[3]: cluster[3] = b3
        if not matched:
            clusters.append([b0, b1, b2, b3])
    return [tuple(c) for c in clusters]


# --- embeddings references ---

def py_normalize(distances, mean, stddev, scale_factor, bias):
    if stddev == 0.0:
        return list(distances)
    return [(x - mean) / stddev * scale_factor + bias for x in distances]


def py_normalize_inplace(distances, mean, stddev, scale_factor, bias):
    if stddev == 0.0:
        return distances
    for i in range(len(distances)):
        distances[i] = (distances[i] - mean) / stddev * scale_factor + bias
    return distances


def py_update_stats(distances):
    n, mean, m2 = 0, 0.0, 0.0
    for x in distances:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return (n, mean, m2)


# --- audio references ---

def py_select_top_k(scores, class_ids, k, min_confidence):
    result_count = 0
    for i in range(min(k, len(class_ids))):
        if scores[class_ids[i]] < min_confidence:
            break
        result_count += 1
    if result_count == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    selected_ids = np.empty(result_count, dtype=np.int64)
    selected_scores = np.empty(result_count, dtype=np.float32)
    for i in range(result_count):
        selected_ids[i] = class_ids[i]
        selected_scores[i] = float(scores[class_ids[i]])
    return selected_ids, selected_scores


def py_build_detections(selected_ids, selected_scores, count):
    detections = np.zeros((count, 6), dtype=np.float32)
    for i in range(count):
        detections[i, 0] = selected_ids[i]
        detections[i, 1] = selected_scores[i]
    return detections


def py_detect_raw(res, count, min_confidence):
    n = len(res)
    k = min(count, n)
    if k <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    class_ids = np.argpartition(-res, k - 1)[:k]
    sorted_order = np.argsort(-res[class_ids])
    class_ids = class_ids[sorted_order]
    non_zero_mask = res > min_confidence
    class_ids = class_ids[non_zero_mask[class_ids]]
    scores = res[class_ids]
    if len(scores) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    detections = np.zeros((len(scores), 6), dtype=np.float32)
    for i in range(len(scores)):
        detections[i, 0] = class_ids[i]
        detections[i, 1] = scores[i]
    return detections


# --- detection runners references ---

def py_face_normalization(face_data):
    return ((face_data + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def py_transpose_nchw_to_nhwc(data):
    if data.ndim != 4 or data.shape[1] == 1:
        return data
    return np.transpose(data, (0, 2, 3, 1))


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

class BenchmarkImageCython(unittest.TestCase):
    """Benchmarks for frigate/util/image_cython.pyx"""

    def test_area(self):
        iters = 1_000_000
        box = (10, 10, 100, 100)
        t0 = time.perf_counter()
        for _ in range(iters): py_area(box)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): image_cython.area(box)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_area(box), image_cython.area(box))
        print(f"  [image] area:          Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_intersection(self):
        iters = 1_000_000
        box_a, box_b = (10, 10, 50, 50), (30, 30, 80, 80)
        t0 = time.perf_counter()
        for _ in range(iters): py_intersection(box_a, box_b)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): image_cython.intersection(box_a, box_b)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_intersection(box_a, box_b), image_cython.intersection(box_a, box_b))
        print(f"  [image] intersection:  Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_iou(self):
        iters = 1_000_000
        box_a, box_b = (10, 10, 50, 50), (30, 30, 80, 80)
        t0 = time.perf_counter()
        for _ in range(iters): py_iou(box_a, box_b)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): image_cython.intersection_over_union(box_a, box_b)
        cy_t = time.perf_counter() - t0
        self.assertAlmostEqual(py_iou(box_a, box_b), image_cython.intersection_over_union(box_a, box_b), places=10)
        print(f"  [image] IoU:           Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_yuv_to_3_channel(self):
        iters = 1_000
        yuv = np.zeros((720*3//2, 1280), dtype=np.uint8)
        yuv[0:720, :] = 16
        yuv[720:720+180, 0:640] = 128
        yuv[720:720+180, 640:1280] = 128
        yuv[900:900+180, 0:640] = 128
        yuv[900:900+180, 640:1280] = 128
        t0 = time.perf_counter()
        for _ in range(iters):
            h, w = yuv.shape[0] // 3 * 2, yuv.shape[1]
            yuv_data = yuv.ravel()
            out = np.empty((h, w, 3), dtype=np.uint8)
            out[:, :, 0] = yuv_data[0:h*w].reshape((h, w))
            y_count = h * w
            uv_count = y_count // 4
            out[:, :, 1] = np.repeat(np.reshape(np.repeat(yuv_data[y_count:y_count+uv_count], 2, axis=0), (h//2, w)), 2, axis=0)
            out[:, :, 2] = np.repeat(np.reshape(np.repeat(yuv_data[y_count+uv_count:y_count+uv_count+uv_count], 2, axis=0), (h//2, w)), 2, axis=0)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters):
            image_cython.yuv_to_3_channel_yuv(yuv)
        cy_t = time.perf_counter() - t0
        print(f"  [image] yuv_to_3ch:    Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")


class BenchmarkObjectCython(unittest.TestCase):
    """Benchmarks for frigate/util/object_cython.pyx"""

    def test_box_inside(self):
        iters = 1_000_000
        b1, b2 = (0, 0, 100, 100), (10, 10, 50, 50)
        t0 = time.perf_counter()
        for _ in range(iters): py_box_inside(b1, b2)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): object_cython.cython_box_inside(b1, b2)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_box_inside(b1, b2), object_cython.cython_box_inside(b1, b2))
        print(f"  [object] box_inside:   Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_reduce_boxes(self):
        iters = 10_000
        np.random.seed(42)
        boxes = [[int(np.random.randint(0, 500)), int(np.random.randint(0, 500)),
                  int(np.random.randint(0, 500)), int(np.random.randint(0, 500))]
                 for _ in range(100)]
        t0 = time.perf_counter()
        for _ in range(iters): py_reduce_boxes(boxes, 0.0)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): object_cython.cython_reduce_boxes(boxes, 0.0)
        cy_t = time.perf_counter() - t0
        self.assertEqual(len(py_reduce_boxes(boxes, 0.0)), len(object_cython.cython_reduce_boxes(boxes, 0.0)))
        print(f"  [object] reduce_boxes: Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")


class BenchmarkEmbeddingsCython(unittest.TestCase):
    """Benchmarks for frigate/embeddings/util_cython.pyx"""

    def test_normalize(self):
        iters = 100_000
        distances = np.array([float(i) for i in range(1000)], dtype=np.float64)
        mean, stddev, scale, bias = 500.0, 200.0, 1.5, 0.5
        t0 = time.perf_counter()
        for _ in range(iters): py_normalize(distances.tolist(), mean, stddev, scale, bias)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): embeddings_cython.cython_normalize(distances, mean, stddev, scale, bias)
        cy_t = time.perf_counter() - t0
        py_result = py_normalize(distances.tolist(), mean, stddev, scale, bias)
        cy_result = embeddings_cython.cython_normalize(distances, mean, stddev, scale, bias)
        self.assertAlmostEqual(sum(py_result), sum(cy_result), places=5)
        print(f"  [embed] normalize:     Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_normalize_inplace(self):
        iters = 100_000
        distances = np.array([float(i) for i in range(1000)], dtype=np.float64)
        mean, stddev, scale, bias = 500.0, 200.0, 1.5, 0.5
        t0 = time.perf_counter()
        for _ in range(iters):
            d = np.array(distances, dtype=np.float64)
            py_normalize_inplace(d.tolist(), mean, stddev, scale, bias)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters):
            d = np.array(distances, dtype=np.float64)
            embeddings_cython.cython_normalize_inplace(d, mean, stddev, scale, bias)
        cy_t = time.perf_counter() - t0
        print(f"  [embed] norm_inplace:  Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_update_stats(self):
        iters = 100_000
        distances = np.array([float(i) for i in range(500)], dtype=np.float64)
        t0 = time.perf_counter()
        for _ in range(iters): py_update_stats(distances.tolist())
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): embeddings_cython.cython_update_stats(distances)
        cy_t = time.perf_counter() - t0
        self.assertEqual(py_update_stats(distances.tolist()), embeddings_cython.cython_update_stats(distances))
        print(f"  [embed] update_stats:  Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")


class BenchmarkAudioCython(unittest.TestCase):
    """Benchmarks for frigate/events/audio_cython.pyx"""

    def test_select_top_k(self):
        iters = 50_000
        scores = np.random.rand(80).astype(np.float32)
        class_ids = np.arange(80, dtype=np.int64)
        k = 20
        min_conf = 0.5
        t0 = time.perf_counter()
        for _ in range(iters): py_select_top_k(scores, class_ids, k, min_conf)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): audio_cython.cython_select_top_k(scores, class_ids, k, min_conf)
        cy_t = time.perf_counter() - t0
        print(f"  [audio] select_top_k:  Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_build_detections(self):
        iters = 50_000
        selected_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        selected_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        count = 5
        t0 = time.perf_counter()
        for _ in range(iters): py_build_detections(selected_ids, selected_scores, count)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): audio_cython.cython_build_detections(selected_ids, selected_scores, count)
        cy_t = time.perf_counter() - t0
        print(f"  [audio] build_detections: Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_detect_raw(self):
        iters = 10_000
        res = np.random.rand(80).astype(np.float32) * 0.9 + 0.1
        count = 20
        min_conf = 0.3
        t0 = time.perf_counter()
        for _ in range(iters): py_detect_raw(res, count, min_conf)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): audio_cython.cython_detect_raw(res, count, min_conf)
        cy_t = time.perf_counter() - t0
        py_result = py_detect_raw(res, count, min_conf)
        cy_result = audio_cython.cython_detect_raw(res, count, min_conf)
        self.assertEqual(py_result.shape, cy_result.shape)
        print(f"  [audio] detect_raw:    Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")


class BenchmarkDetectionRunnersCython(unittest.TestCase):
    """Benchmarks for frigate/detectors/detection_runners_cython.pyx"""

    def test_transpose_nchw_nhwc(self):
        iters = 50_000
        data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(iters): py_transpose_nchw_to_nhwc(data)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): detection_cython.transpose_nchw_to_nhwc(data)
        cy_t = time.perf_counter() - t0
        self.assertTrue(np.array_equal(py_transpose_nchw_to_nhwc(data), detection_cython.transpose_nchw_to_nhwc(data)))
        print(f"  [detect] transpose:     Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")

    def test_face_normalization(self):
        iters = 50_000
        face_data = np.random.rand(1, 112, 112, 3).astype(np.float32) * 2.0 - 1.0
        t0 = time.perf_counter()
        for _ in range(iters): py_face_normalization(face_data)
        py_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters): detection_cython.face_normalization(face_data)
        cy_t = time.perf_counter() - t0
        self.assertTrue(np.array_equal(py_face_normalization(face_data), detection_cython.face_normalization(face_data)))
        print(f"  [detect] face_norm:    Python={py_t:.4f}s  Cython={cy_t:.4f}s  Speedup={py_t/cy_t:.1f}x")


if __name__ == "__main__":
    unittest.main(verbosity=2)

