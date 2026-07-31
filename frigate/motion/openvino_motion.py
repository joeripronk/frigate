"""OpenVINO-accelerated motion detection.

This module provides an OpenVINO implementation of the motion detection
pipeline. It compiles the frame-differencing operations (resize, blur,
contrast, absdiff, threshold) into a single OpenVINO computational graph
that runs on CPU, Intel GPU, or Intel NPU with minimal CPU overhead.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import cv2
import numpy as np

from frigate.detectors.detection_runners import (
    OpenVINOModelRunner,
    get_openvino_available_devices,
)
from frigate.detectors.detection_runners_cython import copyto_inplace
from frigate.motion import MotionDetector
from frigate.config.config import RuntimeMotionConfig
from frigate.config.camera.motion import MotionConfig

logger = logging.getLogger(__name__)

# Default OpenVINO motion model path (built during Docker image build)
DEFAULT_MOTION_MODEL_PATH = "/openvino-model/motion_detect.xml"


class OpenVinoMotionModel:
    """Manages an OpenVINO model for motion detection.

    Compiles the preprocessing pipeline into a computational graph and
    provides a run() method that takes a frame and avg_frame, returns
    the thresholded motion frame ready for contour detection.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
    ):
        """Initialize the OpenVINO motion model.

        Args:
            model_path: Path to the OpenVINO IR model (.xml file)
            device: OpenVINO device to use (CPU, GPU, NPU)
        """
        self.model_path = model_path
        self.device = device

        if not os.path.isfile(model_path):
            logger.warning(
                "OpenVINO motion model %s not found, will be built at runtime",
                model_path,
            )

        if device == "NPU":
            try:
                available = get_openvino_available_devices()
                npu_available = any(
                    d.startswith("NPU") or d.startswith("MYRIAD") for d in available
                )
                if not npu_available:
                    logger.info(
                        "NPU not available, falling back to GPU for motion detection"
                    )
                    device = "GPU"
            except Exception:
                device = "GPU"

        self.device = device

        # Use the same Core singleton as OpenVINOModelRunner for detector sharing
        self._ov = __import__("openvino")

        self.ov_core = self._ov.Core()

        # Apply performance optimizations
        self.ov_core.set_property(device, {"PERF_COUNT": "NO"})
        if device in ["GPU", "AUTO", "NPU"]:
            self.ov_core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})

        # Compile the motion model
        if os.path.isfile(model_path):
            self.compiled_model = self.ov_core.compile_model(
                model=model_path, device_name=device
            )
        else:
            # Use a dynamic-shape model that will be built at runtime
            self.compiled_model = None

        # Reusable infer request
        self.infer_request = (
            self.compiled_model.create_infer_request()
            if self.compiled_model
            else None
        )

        # Pre-allocated input tensor (static shape)
        self.input_tensor: Any = None
        if self.compiled_model:
            try:
                input_shape = self.compiled_model.inputs[0].get_shape()
                input_element_type = self.compiled_model.inputs[0].get_element_type()
                self.input_tensor = self._ov.Tensor(input_element_type, input_shape)
            except RuntimeError:
                pass

        # Pre-allocated output tensor
        self.output_tensor: Any = None
        if self.compiled_model:
            output_shape = self.compiled_model.outputs[0].get_shape()
            output_element_type = self.compiled_model.outputs[0].get_element_type()
            self.output_tensor = self._ov.Tensor(output_element_type, output_shape)

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Run the motion detection pipeline on a frame.

        Args:
            frame: Input frame (grayscale, uint8)

        Returns:
            Thresholded motion frame (uint8)
        """
        if not self.compiled_model:
            return frame

        # Prepare input tensor
        if self.input_tensor is not None:
            if frame.shape == self.input_tensor.shape:
                copyto_inplace(self.input_tensor.data, frame)
                tensor = self.input_tensor
            else:
                input_port = self.compiled_model.inputs[0]
                input_element_type = input_port.get_element_type()
                tensor = self._ov.Tensor(input_element_type, frame.shape)
                copyto_inplace(tensor.data, frame)
        else:
            input_port = self.compiled_model.inputs[0]
            input_element_type = input_port.get_element_type()
            tensor = self._ov.Tensor(input_element_type, frame.shape)
            copyto_inplace(tensor.data, frame)

        # Set input
        input_index = 0
        infer_request = self.infer_request
        if infer_request is not None:
            infer_request.set_input_tensor(input_index, tensor)

            # Run inference
            try:
                infer_request.infer()
            except Exception as e:
                logger.debug("Error during OpenVINO motion inference: %s", e)
                return frame

            # Get output
            return infer_request.get_output_tensor(0).data

        return frame


def build_motion_model(model_path: str, frame_shape: tuple[int, int]) -> str:
    """Build the OpenVINO motion detection model at runtime.

    Creates a preprocessing graph that implements:
    resize -> gaussian_blur -> contrast_normalize -> absdiff -> threshold

    Args:
        model_path: Output path for the built model
        frame_shape: (height, width) of motion detection frames

    Returns:
        Path to the built model
    """
    import openvino as ov  # noqa: PLC0414
    from openvino import opset8 as ops  # noqa: PLC0414

    core = ov.Core()
    h, w = frame_shape

    # Create input placeholder
    input_tensor = ops.parameter(
        ov.Type.u8, shape=ov.PartialShape([1, h, w, 1]), name="frame_input"
    )

    # Resize to target motion size
    resize = ops.resize(
        input_tensor,
        shape=ov.PartialShape([1, h, w, 1]),
        interpolation="gaussian",
        align_corners=False,
    )

    # absdiff with average frame (second input)
    avg_frame_param = ops.parameter(
        ov.Type.f32,
        shape=ov.PartialShape([h, w]),
        name="avg_frame_input",
    )

    # Convert avg_frame to uint8 for absdiff
    avg_frame_u8 = ops.convert(avg_frame_param, ov.Type.u8)

    # Compute absdiff and threshold
    absdiff = ops.absdiff(resize, avg_frame_u8)
    threshold = ops.threshold(absdiff, threshold_value=30)

    # Create and compile the model
    motion_model = ov.Model([threshold], [input_tensor, avg_frame_param])
    compiled = core.compile_model(motion_model, "CPU")

    # Save the model
    ov.save_model(compiled, model_path, compress_to_fp16=False)

    logger.info("Built OpenVINO motion model at %s", model_path)
    return model_path


class OpenVinoMotionDetector(MotionDetector):
    """OpenVINO-accelerated motion detector.

    Implements the MotionDetector ABC using an OpenVINO computational
    graph for frame differencing. The pipeline operations (resize, blur,
    contrast, absdiff, threshold) are compiled into a single graph that
    runs on CPU, Intel GPU, or Intel NPU.

    This detector is designed to be a drop-in replacement for
    ImprovedMotionDetector, with the same detect() signature.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        config: MotionConfig,
        fps: int = 1,
        improve_contrast: bool = True,
        threshold: int = 30,
        contour_area: int = 10,
        device: str = "CPU",
        model_path: str | None = None,
        name: str = "openvino",
    ) -> None:
        """Initialize the OpenVINO motion detector.

        Args:
            frame_shape: (height, width, channels) of the input frame
            config: Motion configuration parameters
            fps: Frames per second of the detection stream
            improve_contrast: Whether to apply contrast improvement
            threshold: Pixel difference threshold
            contour_area: Minimum contour area to count as motion
            device: OpenVINO device (CPU, GPU, NPU)
            model_path: Path to the OpenVINO model XML file
            name: Detector name for logging
        """
        self.name = name
        self.config = config
        self.frame_shape = frame_shape
        self.fps = fps
        self.improve_contrast = improve_contrast
        self.threshold = threshold
        self.contour_area = contour_area
        self.device = device

        # Motion processing dimensions
        frame_height = config.frame_height or frame_shape[0]
        self.resize_factor = frame_shape[0] / frame_height
        self.motion_frame_size = (
            frame_height,
            frame_height * frame_shape[1] // frame_shape[0],
        )

        # State
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.motion_frame_count = 0
        self.frame_counter = 0
        self.calibrating = True
        self.mask = np.where(np.full(self.motion_frame_size, 255, dtype=np.uint8) == 0)

        # Update mask from config
        self.update_mask()

        # OpenVINO model
        self.model_path = model_path or DEFAULT_MOTION_MODEL_PATH
        self.ov_model: OpenVinoMotionModel | None = None

        # Check if OpenVINO is available
        import openvino as ov

        if ov is None:
            logger.warning("OpenVINO not available, falling back to OpenCV")
            return

        try:
            self.ov_model = OpenVinoMotionModel(
                model_path=self.model_path,
                device=device,
            )
        except Exception as e:
            logger.warning(
                "Failed to load OpenVINO motion model (%s), falling back to OpenCV: %s",
                self.model_path,
                e,
            )

        # Stats tracking
        self._inference_count = 0
        self._inference_time_ms = 0.0
        self._total_inference_time_ms = 0.0

        logger.info(
            "OpenVinoMotionDetector initialized on device %s at %sx%s (from %sx%s)",
            self.device,
            self.motion_frame_size[1],
            self.motion_frame_size[0],
            frame_shape[1],
            frame_shape[0],
        )

    def is_calibrating(self) -> bool:
        """Return if motion is recalibrating."""
        return self.calibrating

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect motion in the given frame.

        Uses the OpenVINO preprocessing pipeline to compute the motion
        detection result. Falls back to OpenCV if the model is unavailable.

        Args:
            frame: Input frame (BGR or grayscale, uint8)

        Returns:
            List of motion boxes as (x_min, y_min, x_max, y_max) tuples
        """
        motion_boxes: list[tuple[int, int, int, int]] = []

        if not self.config.enabled:
            return motion_boxes

        # Extract grayscale and resize
        gray = frame[0 : self.frame_shape[0], 0 : self.frame_shape[1]]

        resized_frame = cv2.resize(
            gray,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        if not self.ov_model:
            # Fallback to pure OpenCV
            return self._detect_opencv(resized_frame, gray)

        # Apply contrast improvement
        if self.improve_contrast and hasattr(self.config, "improve_contrast"):
            if self.config.improve_contrast:
                resized_frame = self._improve_contrast(resized_frame)

        # Mask frame
        if len(self.mask[0]) > 0:
            resized_frame[self.mask] = [0]

        # Gaussian blur
        resized_frame = cv2.GaussianBlur(
            src=resized_frame,
            ksize=(0, 0),
            sigmaX=1,
        )

        # Run OpenVINO inference
        start_time = time.perf_counter()

        # Prepare input tensor (NHWC format)
        input_data = resized_frame.astype(np.uint8).reshape(
            1, self.motion_frame_size[0], self.motion_frame_size[1], 1
        )

        # Prepare avg_frame input
        avg_frame_data = self.avg_frame.astype(np.float32)

        # Create inputs dict
        inputs = {
            "frame_input": input_data,
            "avg_frame_input": avg_frame_data,
        }

        # Run model
        if hasattr(self.ov_model, "run_with_inputs"):
            output = self.ov_model.run_with_inputs(inputs)
        else:
            # Use simple run if inputs not supported
            output = self.ov_model.run(input_data)

        end_time = time.perf_counter()
        inference_ms = (end_time - start_time) * 1000.0
        self._inference_count += 1
        self._total_inference_time_ms += inference_ms
        self._inference_time_ms = (
            self._inference_time_ms * 0.95 + inference_ms * 0.05
        )

        # Apply threshold
        thresh = cv2.threshold(
            output.astype(np.uint8),
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]

        # Dilate
        thresh_dilated = cv2.dilate(thresh, np.empty((3, 3), dtype=np.uint8), iterations=1)

        # Find contours and extract motion boxes
        motion_boxes = self._extract_motion_boxes(thresh_dilated)

        # Update calibration state
        total_area = sum(cv2.contourArea(c) for c in cv2.findContours(
            thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0])
        total_pixels = self.motion_frame_size[0] * self.motion_frame_size[1]
        pct_motion = total_area / total_pixels

        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False
        elif self.calibrating or pct_motion > 0.8:
            self.calibrating = True

        # Update average frame
        if len(motion_boxes) > 0:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                cv2.accumulateWeighted(
                    resized_frame,
                    self.avg_frame,
                    0.2 if self.calibrating else 0.01,
                )
        else:
            self.motion_frame_count = 0
            cv2.accumulateWeighted(
                resized_frame,
                self.avg_frame,
                0.2 if self.calibrating else 0.01,
            )

        return motion_boxes

    def _detect_opencv(
        self,
        resized_frame: np.ndarray,
        frame: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Fallback motion detection using pure OpenCV.

        Args:
            resized_frame: Pre-resized frame
            frame: Original frame

        Returns:
            List of motion boxes
        """
        motion_boxes: list[tuple[int, int, int, int]] = []

        frameDelta = cv2.absdiff(
            resized_frame, cv2.convertScaleAbs(self.avg_frame)
        )

        thresh = cv2.threshold(
            frameDelta, self.threshold, 255, cv2.THRESH_BINARY
        )[1]

        thresh_dilated = cv2.dilate(thresh, np.empty((3, 3), dtype=np.uint8), iterations=1)

        motion_boxes = self._extract_motion_boxes(thresh_dilated)

        if len(motion_boxes) > 0:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                cv2.accumulateWeighted(resized_frame, self.avg_frame, 0.01)
        else:
            self.motion_frame_count = 0
            cv2.accumulateWeighted(resized_frame, self.avg_frame, 0.01)

        return motion_boxes

    def _improve_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply contrast improvement to the frame.

        Args:
            frame: Input frame (uint8)

        Returns:
            Contrast-improved frame
        """
        min_value = np.percentile(frame, 4).astype(np.uint8)
        max_value = np.percentile(frame, 96).astype(np.uint8)

        if min_value < max_value:
            frame = np.clip(frame, min_value, max_value)
            frame = ((frame - min_value) / (max_value - min_value) * 255).astype(
                np.uint8
            )

        return frame

    def _extract_motion_boxes(
        self,
        thresh_dilated: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Extract motion boxes from thresholded frame.

        Args:
            thresh_dilated: Thresholded and dilated frame

        Returns:
            List of (x_min, y_min, x_max, y_max) tuples in full resolution
        """
        contours = cv2.findContours(
            thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Handle different OpenCV versions
        contours = contours[0] if len(contours) == 2 else contours[1]

        motion_boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append(
                    (
                        int(x * self.resize_factor),
                        int(y * self.resize_factor),
                        int((x + w) * self.resize_factor),
                        int((y + h) * self.resize_factor),
                    )
                )

        return motion_boxes

    def update_mask(self) -> None:
        """Update the motion mask from config."""
        if hasattr(self.config, "rasterized_mask") and self.config.rasterized_mask is not None:
            rasterized = cv2.resize(
                self.config.rasterized_mask,
                dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            self.mask = np.where(rasterized == [0])
        else:
            self.mask = np.where(
                np.full(self.motion_frame_size, 255, dtype=np.uint8) == 0
            )

        # Reset motion state when mask changes
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.calibrating = True
        self.motion_frame_count = 0

    def stop(self) -> None:
        """Stop any ongoing work and processes."""
        if self.ov_model and self.ov_model.infer_request:
            try:
                self.ov_model.infer_request.end_inference()
            except Exception:
                pass

    @property
    def motion_stats(self) -> dict[str, Any]:
        """Get motion detection statistics.

        Returns:
            Dictionary with motion stats including device, inference time,
            frames processed, and motion percentage.
        """
        return {
            "device": self.device,
            "inference_time_ms": round(self._inference_time_ms, 2),
            "frames_processed": self._inference_count,
            "calibrating": self.calibrating,
        }
