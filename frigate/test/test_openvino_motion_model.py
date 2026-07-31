"""Tests for OpenVINO motion model (Phase 2).

Tests the model definition and build process in
frigate/motion/openvino_motion_model.py.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate.motion.openvino_motion_model import (
    build_motion_model,
    build_motion_model_from_config,
    get_motion_model_path,
)


class TestBuildMotionModel(unittest.TestCase):
    """Tests for build_motion_model function."""

    def test_build_motion_model_creates_graph(self):
        """Test that build_motion_model creates the computational graph."""
        mock_ov = MagicMock()
        mock_ov.Type = MagicMock()
        mock_ov.Type.u8 = "u8"
        mock_ov.Type.f32 = "f32"
        mock_ov.PartialShape = MagicMock(return_value=[1, 100, 133, 1])
        mock_ov.Model.return_value = MagicMock()
        mock_ov.Core.return_value = MagicMock()

        mock_ops = MagicMock()
        mock_ops.parameter.side_effect = [MagicMock(), MagicMock()]
        mock_ops.convert.return_value = MagicMock()
        mock_ops.expand.return_value = MagicMock()
        mock_ops.absdiff.return_value = MagicMock()
        mock_ops.threshold.return_value = MagicMock()

        mock_ppp_class = MagicMock()
        mock_ppp_instance = MagicMock()
        mock_ppp_instance.build.return_value = mock_ov.Model.return_value
        mock_ppp_class.return_value = mock_ppp_instance

        def mock_import(name, *args, **kwargs):
            if name == "openvino" or name == "openvino.preprocess":
                result = mock_ov
                result.PrePostProcessor = mock_ppp_class
                return result
            elif name == "opset8":
                return mock_ops
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "test_motion.xml")
                result = build_motion_model(
                    model_path=model_path,
                    frame_shape=(100, 133),
                    threshold_value=30.0,
                    compress_to_fp16=True,
                )

                self.assertEqual(result, model_path)
                mock_ov.Model.assert_called_once()
                mock_ppp_class.assert_called()
                mock_ppp_instance.build.assert_called()

    def test_build_motion_model_creates_directory(self):
        """Test that build_motion_model creates output directory."""
        mock_ov = MagicMock()
        mock_ov.Type = MagicMock()
        mock_ov.Type.u8 = "u8"
        mock_ov.Type.f32 = "f32"
        mock_ov.PartialShape = MagicMock(return_value=[1, 100, 133, 1])
        mock_ov.Model.return_value = MagicMock()
        mock_ov.parameter = MagicMock(
            side_effect=[MagicMock(), MagicMock()]
        )

        mock_ops = MagicMock()
        mock_ops.convert.return_value = MagicMock()
        mock_ops.expand.return_value = MagicMock()
        mock_ops.absdiff.return_value = MagicMock()
        mock_ops.threshold.return_value = MagicMock()

        mock_ppp_class = MagicMock()
        mock_ppp_instance = MagicMock()
        mock_ppp_instance.build.return_value = mock_ov.Model.return_value
        mock_ppp_class.return_value = mock_ppp_instance

        def mock_import(name, *args, **kwargs):
            if name == "openvino":
                return mock_ov
            elif name == "opset8":
                return mock_ops
            elif name == "PrePostProcessor":
                return mock_ppp_class
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "subdir", "model.xml")
                result = build_motion_model(
                    model_path=model_path,
                    frame_shape=(100, 133),
                )

                self.assertEqual(result, model_path)

    def test_build_motion_model_from_config(self):
        """Test build_motion_model_from_config with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "custom_model.xml")
            config = {
                "frame_height": 150,
                "frame_width": 200,
                "threshold": 40,
                "compress_to_fp16": False,
            }

            with patch(
                "frigate.motion.openvino_motion_model.build_motion_model"
            ) as mock_build:
                mock_build.return_value = model_path
                result = build_motion_model_from_config(
                    model_path=model_path, config=config
                )

                mock_build.assert_called_once_with(
                    model_path=model_path,
                    frame_shape=(150, 200),
                    threshold_value=40,
                    compress_to_fp16=False,
                )
                self.assertEqual(result, model_path)

    def test_build_motion_model_from_config_defaults(self):
        """Test build_motion_model_from_config with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "default_model.xml")

            with patch(
                "frigate.motion.openvino_motion_model.build_motion_model"
            ) as mock_build:
                mock_build.return_value = model_path
                result = build_motion_model_from_config(
                    model_path=model_path, config={}
                )

                mock_build.assert_called_once_with(
                    model_path=model_path,
                    frame_shape=(100, 133),
                    threshold_value=30,
                    compress_to_fp16=True,
                )
                self.assertEqual(result, model_path)

    def test_get_motion_model_path_existing(self):
        """Test get_motion_model_path when file exists."""
        with patch(
            "frigate.motion.openvino_motion_model.os.path.isfile",
            return_value=True,
        ):
            path = get_motion_model_path()
            self.assertEqual(path, "/openvino-model/motion_detect.xml")

    def test_get_motion_model_path_fallback(self):
        """Test get_motion_model_path fallback when file does not exist."""
        with patch(
            "frigate.motion.openvino_motion_model.os.path.isfile",
            return_value=False,
        ):
            path = get_motion_model_path()
            self.assertEqual(path, "/openvino-model/motion_detect.xml")


if __name__ == "__main__":
    unittest.main()
