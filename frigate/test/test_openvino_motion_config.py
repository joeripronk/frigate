"""Tests for OpenVINO motion config (Phase 3).

Tests the detector_type and openvino_device fields in MotionConfig
and their propagation through RuntimeMotionConfig and FrigateConfig.
"""

import unittest
from typing import Any

from frigate.config.camera.motion import MotionConfig
from frigate.config.config import RuntimeMotionConfig


class TestMotionConfigOpenVINO(unittest.TestCase):
    """Tests for OpenVINO config fields in MotionConfig."""

    def test_default_values(self):
        """Test default values for detector_type and openvino_device."""
        config = MotionConfig()
        self.assertEqual(config.detector_type, "auto")
        self.assertEqual(config.openvino_device, "CPU")

    def test_explicit_values(self):
        """Test explicit values for detector_type and openvino_device."""
        config = MotionConfig(
            detector_type="openvino",
            openvino_device="GPU",
        )
        self.assertEqual(config.detector_type, "openvino")
        self.assertEqual(config.openvino_device, "GPU")

    def test_detector_type_values(self):
        """Test all valid detector_type values."""
        for value in ("auto", "opencv", "openvino"):
            config = MotionConfig(detector_type=value)  # type: ignore[arg-type]
            self.assertEqual(config.detector_type, value)

    def test_openvino_device_values(self):
        """Test valid openvino_device values."""
        for value in ("CPU", "GPU", "NPU"):
            config = MotionConfig(openvino_device=value)
            self.assertEqual(config.openvino_device, value)

    def test_serialization(self):
        """Test serialization of detector_type and openvino_device."""
        config = MotionConfig(
            detector_type="openvino",
            openvino_device="NPU",
        )
        dumped = config.model_dump()
        self.assertEqual(dumped["detector_type"], "openvino")
        self.assertEqual(dumped["openvino_device"], "NPU")

    def test_deserialization(self):
        """Test deserialization of detector_type and openvino_device."""
        data: dict[str, Any] = {
            "detector_type": "opencv",
            "openvino_device": "GPU",
        }
        config = MotionConfig.model_validate(data)
        self.assertEqual(config.detector_type, "opencv")
        self.assertEqual(config.openvino_device, "GPU")

    def test_runtime_motion_config(self):
        """Test RuntimeMotionConfig inherits the fields."""
        config = RuntimeMotionConfig(
            frame_shape=(480, 640, 3),
            detector_type="openvino",
            openvino_device="NPU",
        )
        self.assertEqual(config.detector_type, "openvino")
        self.assertEqual(config.openvino_device, "NPU")

    def test_runtime_motion_config_defaults(self):
        """Test RuntimeMotionConfig default values."""
        config = RuntimeMotionConfig(frame_shape=(480, 640, 3))
        self.assertEqual(config.detector_type, "auto")
        self.assertEqual(config.openvino_device, "CPU")


if __name__ == "__main__":
    unittest.main()
