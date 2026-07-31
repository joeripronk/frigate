"""Tests for OpenVINO motion detector integration (Phase 4).

Tests the get_motion_detector factory function and its wiring
into CameraTracker and process_frames.
"""

import unittest
from unittest.mock import MagicMock, patch

from frigate.video.detect import get_motion_detector
from frigate.motion.openvino_motion import OpenVinoMotionDetector
from frigate.motion.improved_motion import ImprovedMotionDetector


class TestGetMotionDetector(unittest.TestCase):
    """Tests for the get_motion_detector factory function."""

    def test_factory_returns_openvino_when_configured(self):
        """Test that OpenVINO detector is returned when detector_type='openvino'."""
        mock_config = MagicMock()
        mock_config.openvino_device = "GPU"
        mock_config.detector_type = "openvino"

        with patch(
            "frigate.video.detect.OpenVinoMotionDetector"
        ) as MockOv:
            MockOv.return_value = MagicMock(spec=OpenVinoMotionDetector)
            result = get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                fps=5,
                device=None,
            )
            MockOv.assert_called_once()
            call_kwargs = MockOv.call_args
            self.assertEqual(call_kwargs.kwargs.get("device"), "GPU")

    def test_factory_returns_openvino_when_auto_and_available(self):
        """Test OpenVINO is used when detector_type='auto' and hardware available."""
        mock_config = MagicMock()
        mock_config.openvino_device = "NPU"
        mock_config.detector_type = "auto"

        with patch(
            "frigate.video.detect.is_openvino_gpu_npu_available",
            return_value=True,
        ):
            with patch(
                "frigate.video.detect.OpenVinoMotionDetector"
            ) as MockOv:
                MockOv.return_value = MagicMock(spec=OpenVinoMotionDetector)
                result = get_motion_detector(
                    frame_shape=(480, 640),
                    config=mock_config,
                    fps=5,
                )
                self.assertIsInstance(result, OpenVinoMotionDetector)

    def test_factory_falls_back_to_opencv_when_auto_no_hardware(self):
        """Test fallback to OpenCV when auto mode and no OpenVINO hardware."""
        mock_config = MagicMock()
        mock_config.openvino_device = "CPU"
        mock_config.detector_type = "auto"

        with patch(
            "frigate.video.detect.is_openvino_gpu_npu_available",
            return_value=False,
        ):
            with patch(
                "frigate.video.detect.ImprovedMotionDetector"
            ) as MockOp:
                MockOp.return_value = MagicMock(spec=ImprovedMotionDetector)
                result = get_motion_detector(
                    frame_shape=(480, 640),
                    config=mock_config,
                    fps=5,
                )
                MockOp.assert_called_once()
                self.assertIsInstance(result, ImprovedMotionDetector)

    def test_factory_returns_opencv_when_explicit(self):
        """Test OpenCV detector is returned when detector_type='opencv'."""
        mock_config = MagicMock()
        mock_config.openvino_device = "GPU"
        mock_config.detector_type = "opencv"

        with patch(
            "frigate.video.detect.ImprovedMotionDetector"
        ) as MockOp:
            MockOp.return_value = MagicMock(spec=ImprovedMotionDetector)
            result = get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                fps=5,
            )
            MockOp.assert_called_once()
            self.assertIsInstance(result, ImprovedMotionDetector)

    def test_factory_passes_device_from_config(self):
        """Test that openvino_device from config is passed through."""
        mock_config = MagicMock()
        mock_config.openvino_device = "NPU"
        mock_config.detector_type = "openvino"

        with patch(
            "frigate.video.detect.OpenVinoMotionDetector"
        ) as MockOv:
            MockOv.return_value = MagicMock(spec=OpenVinoMotionDetector)
            get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                device=None,
            )
            call_kwargs = MockOv.call_args
            self.assertEqual(call_kwargs.kwargs.get("device"), "NPU")

    def test_factory_passes_name(self):
        """Test that camera name is passed through."""
        mock_config = MagicMock()
        mock_config.openvino_device = "CPU"
        mock_config.detector_type = "openvino"

        with patch(
            "frigate.video.detect.OpenVinoMotionDetector"
        ) as MockOv:
            MockOv.return_value = MagicMock()
            get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                name="front_door",
            )
            call_kwargs = MockOv.call_args
            self.assertEqual(call_kwargs.kwargs.get("name"), "front_door")

    def test_factory_with_override_device(self):
        """Test that explicit device parameter overrides config."""
        mock_config = MagicMock()
        mock_config.openvino_device = "GPU"
        mock_config.detector_type = "openvino"

        with patch(
            "frigate.video.detect.OpenVinoMotionDetector"
        ) as MockOv:
            MockOv.return_value = MagicMock()
            get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                device="CPU",
            )
            call_kwargs = MockOv.call_args
            self.assertEqual(call_kwargs.kwargs.get("device"), "CPU")

    def test_factory_with_none_name(self):
        """Test that None name is handled correctly."""
        mock_config = MagicMock()
        mock_config.openvino_device = "CPU"
        mock_config.detector_type = "openvino"

        with patch(
            "frigate.video.detect.OpenVinoMotionDetector"
        ) as MockOv:
            MockOv.return_value = MagicMock()
            get_motion_detector(
                frame_shape=(480, 640),
                config=mock_config,
                name=None,
            )
            call_kwargs = MockOv.call_args
            self.assertEqual(call_kwargs.kwargs.get("name"), "openvino")

    def test_factory_handles_openvino_exception(self):
        """Test fallback to OpenCV when OpenVINO detector creation fails."""
        mock_config = MagicMock()
        mock_config.openvino_device = "CPU"
        mock_config.detector_type = "auto"

        with patch(
            "frigate.video.detect.is_openvino_gpu_npu_available",
            return_value=True,
        ):
            with patch(
                "frigate.video.detect.OpenVinoMotionDetector",
                side_effect=Exception("OpenVINO error"),
            ):
                with patch(
                    "frigate.video.detect.ImprovedMotionDetector"
                ) as MockOp:
                    MockOp.return_value = MagicMock(spec=ImprovedMotionDetector)
                    result = get_motion_detector(
                        frame_shape=(480, 640),
                        config=mock_config,
                    )
                    self.assertIsInstance(result, ImprovedMotionDetector)


if __name__ == "__main__":
    unittest.main()
