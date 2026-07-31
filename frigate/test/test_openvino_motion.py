"""Tests for OpenVINO motion detection (Phase 1).

Tests the OpenVinoMotionDetector and OpenVinoMotionModel classes,
including initialization, motion detection, mask updates, and stats.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.motion.openvino_motion import (
    DEFAULT_MOTION_MODEL_PATH,
    OpenVinoMotionDetector,
    OpenVinoMotionModel,
    build_motion_model,
)


class TestOpenVinoMotionModel(unittest.TestCase):
    """Tests for OpenVinoMotionModel class."""

    def test_init_with_existing_model(self):
        """Test initialization with existing model file."""
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.__import__") as mock_import:
                mock_ov = MagicMock()
                mock_ov.Core.return_value = MagicMock()
                mock_import.return_value = mock_ov
                model = OpenVinoMotionModel(
                    model_path="/test/model.xml", device="CPU"
                )
                self.assertEqual(model.device, "CPU")
                self.assertEqual(model.model_path, "/test/model.xml")
                self.assertIsNotNone(model.ov_core)

    def test_init_falls_back_to_gpu_on_npu_unavailable(self):
        """Test that NPU falls back to GPU when not available."""
        with patch("os.path.isfile", return_value=True):
            with patch(
                "frigate.motion.openvino_motion.get_openvino_available_devices",
                return_value=["CPU"],
            ):
                with patch("builtins.__import__") as mock_import:
                    mock_ov = MagicMock()
                    mock_ov.Core.return_value = MagicMock()
                    mock_import.return_value = mock_ov
                    model = OpenVinoMotionModel(
                        model_path="/test/model.xml", device="NPU"
                    )
                    self.assertEqual(model.device, "GPU")

    def test_run_with_no_compiled_model(self):
        """Test run returns frame when no compiled model."""
        model = OpenVinoMotionModel(
            model_path="/nonexistent.xml", device="CPU"
        )
        test_frame = np.zeros((100, 100), dtype=np.uint8)
        result = model.run(test_frame)
        self.assertIs(result, test_frame)

    def test_run_with_compiled_model(self):
        """Test run returns output tensor data."""
        with patch("os.path.isfile", return_value=True):
            mock_core = MagicMock()
            mock_core.compile_model.return_value = MagicMock()
            mock_core.compile_model.return_value.inputs = [
                MagicMock(get_shape=lambda: [1, 100, 100, 1]),
            ]
            mock_core.compile_model.return_value.outputs = [
                MagicMock(get_shape=lambda: [1, 100, 100, 1]),
            ]
            with patch("builtins.__import__") as mock_import:
                mock_ov = MagicMock()
                mock_ov.Core.return_value = mock_core
                mock_ov.Tensor = MagicMock(return_value=MagicMock())
                mock_import.return_value = mock_ov
                model = OpenVinoMotionModel(
                    model_path="/test/model.xml", device="CPU"
                )
                model.infer_request = MagicMock()
                model.input_tensor = MagicMock()
                model.input_tensor.data = np.zeros((100, 100), dtype=np.uint8)
                model.infer_request.get_output_tensor.return_value = MagicMock()
                model.infer_request.get_output_tensor.return_value.data = np.ones(
                    (1, 100, 100, 1), dtype=np.float32
                )
                test_frame = np.zeros((100, 100), dtype=np.uint8)
                result = model.run(test_frame)
                self.assertIsNotNone(result)


class TestOpenVinoMotionDetector(unittest.TestCase):
    """Tests for OpenVinoMotionDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.frame_shape = (480, 640, 3)
        self.config = MagicMock()
        self.config.enabled = True
        self.config.frame_height = 100
        self.config.improve_contrast = True
        self.config.rasterized_mask = np.full((480, 640), 255, dtype=np.uint8)

    def test_init_basic(self):
        """Test basic initialization."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
                device="CPU",
                model_path="/test/model.xml",
            )
            self.assertEqual(detector.device, "CPU")
            self.assertEqual(detector.name, "openvino")
            self.assertEqual(detector.threshold, 30)
            self.assertEqual(detector.contour_area, 10)
            self.assertEqual(detector.motion_frame_size, (100, 133))

    def test_init_with_no_openvino(self):
        """Test initialization when OpenVINO is unavailable."""
        with patch(
            "frigate.motion.openvino_motion.OpenVinoMotionModel",
            side_effect=Exception("OpenVINO error"),
        ):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
                device="CPU",
            )
            self.assertIsNone(detector.ov_model)

    def test_is_calibrating(self):
        """Test is_calibrating returns correct state."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            self.assertTrue(detector.is_calibrating())
            detector.calibrating = False
            self.assertFalse(detector.is_calibrating())

    def test_detect_disabled(self):
        """Test detect returns empty list when motion is disabled."""
        self.config.enabled = False
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.detect(frame)
            self.assertEqual(result, [])

    def test_detect_with_openvino_model(self):
        """Test detect returns motion boxes when model is available."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel") as MockModel:
            MockModel.return_value = MagicMock()
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.calibrating = False
            detector.motion_frame_count = 10

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(detector, "_extract_motion_boxes", return_value=[]):
                with patch("cv2.GaussianBlur", return_value=np.zeros((100, 133), dtype=np.uint8)):
                    with patch("cv2.threshold", return_value=(0, np.zeros((100, 133), dtype=np.uint8))):
                        with patch("cv2.dilate", return_value=np.zeros((100, 133), dtype=np.uint8)):
                            with patch("cv2.accumulateWeighted"):
                                detector.ov_model = MagicMock()
                                detector.ov_model.run.return_value = np.ones(
                                    (1, 100, 133, 1), dtype=np.float32
                                )
                                result = detector.detect(frame)
                                self.assertIsInstance(result, list)

    def test_detect_fallback_to_opencv(self):
        """Test detect falls back to OpenCV when model is unavailable."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.ov_model = None
            # Set avg_frame to match motion_frame_size shape
            detector.avg_frame = np.zeros(detector.motion_frame_size, dtype=np.float32)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(detector, "_extract_motion_boxes", return_value=[]):
                with patch("cv2.absdiff", return_value=np.zeros((100, 133), dtype=np.uint8)):
                    with patch("cv2.threshold", return_value=(0, np.zeros((100, 133), dtype=np.uint8))):
                        with patch("cv2.dilate", return_value=np.zeros((100, 133), dtype=np.uint8)):
                            with patch("cv2.accumulateWeighted"):
                                result = detector.detect(frame)
                                self.assertIsInstance(result, list)

    def test_update_mask(self):
        """Test update_mask resets state correctly."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.calibrating = False
            detector.motion_frame_count = 10
            detector.avg_frame[:] = 100.0

            detector.update_mask()
            self.assertTrue(detector.calibrating)
            self.assertEqual(detector.motion_frame_count, 0)
            self.assertTrue(
                np.all(detector.avg_frame == 0.0)
            )

    def test_update_mask_no_rasterized(self):
        """Test update_mask when no rasterized_mask in config."""
        self.config.rasterized_mask = None
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.update_mask()
            self.assertIsNotNone(detector.mask)

    def test_stop(self):
        """Test stop method."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.ov_model = MagicMock()
            detector.ov_model.infer_request = MagicMock()

            detector.stop()
            detector.ov_model.infer_request.end_inference.assert_called_once()

    def test_stop_no_model(self):
        """Test stop method when model is None."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector.ov_model = None
            detector.stop()  # Should not raise

    def test_motion_stats(self):
        """Test motion_stats property."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            detector._inference_count = 50
            detector._inference_time_ms = 5.5

            stats = detector.motion_stats
            self.assertEqual(stats["device"], "CPU")
            self.assertEqual(stats["frames_processed"], 50)
            self.assertIn("calibrating", stats)

    def test_improve_contrast(self):
        """Test _improve_contrast method."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            frame = np.full((100, 133), 128, dtype=np.uint8)
            result = detector._improve_contrast(frame)
            self.assertEqual(result.dtype, np.uint8)

    def test_extract_motion_boxes(self):
        """Test _extract_motion_boxes method."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=self.frame_shape,
                config=self.config,
            )
            thresh = np.zeros((100, 133), dtype=np.uint8)
            # Create a white rectangle
            thresh[20:50, 30:80] = 255

            boxes = detector._extract_motion_boxes(thresh)
            self.assertGreater(len(boxes), 0)
            for box in boxes:
                self.assertEqual(len(box), 4)

    def test_constructor_config_fields(self):
        """Test constructor correctly sets all config fields."""
        with patch("frigate.motion.openvino_motion.OpenVinoMotionModel"):
            detector = OpenVinoMotionDetector(
                frame_shape=(480, 640, 3),
                config=self.config,
                fps=5,
                improve_contrast=True,
                threshold=40,
                contour_area=20,
                device="GPU",
                model_path="/custom/model.xml",
                name="custom",
            )
            self.assertEqual(detector.fps, 5)
            self.assertEqual(detector.threshold, 40)
            self.assertEqual(detector.contour_area, 20)
            self.assertEqual(detector.device, "GPU")
            self.assertEqual(detector.model_path, "/custom/model.xml")
            self.assertEqual(detector.name, "custom")


class TestBuildMotionModel(unittest.TestCase):
    """Tests for build_motion_model function."""

    @patch("builtins.__import__")
    def test_build_motion_model_creates_file(self, mock_import):
        """Test build_motion_model creates the model file."""
        mock_ov = MagicMock()
        mock_core = MagicMock()
        mock_ov.Core.return_value = mock_core
        mock_ov.Model = MagicMock()
        mock_ov.save_model = MagicMock()
        mock_import.return_value = mock_ov

        with patch("os.path.isfile", return_value=False):
            result = build_motion_model(
                model_path="/test/motion.xml", frame_shape=(100, 133)
            )
            self.assertEqual(result, "/test/motion.xml")


if __name__ == "__main__":
    unittest.main()
