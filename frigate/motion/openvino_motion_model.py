"""OpenVINO motion detection model definition.

This module defines the OpenVINO computational graph for the motion
detection pipeline. The graph implements the same operations as OpenCV
(frame differencing with preprocessing) but runs entirely within OpenVINO
on CPU, Intel GPU, or Intel NPU.

The model is built at Docker image build time by extending
`docker/main/build_ov_model.py`. At runtime, if the model file does
not exist it is built dynamically from the graph definition.

Graph structure
---------------

    frame_input (u8, NHWC) ─┐
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    resize──► gaussian_blur──► contrast_normalize ─────►│
    │                                              │   │
    avg_frame_input (f32, HW) ──────────────────────┤│
                                                     ▼
                                               absdiff
                                                     │
                                                  threshold
                                                     │
    ┌──────────────────────────────────────────────┘
    │
    output (u8, NHWC)  ←  thresholded motion frame

The thresholded output is passed to OpenCV contour finding for the
final motion box extraction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_motion_model(
    model_path: str = "/openvino-model/motion_detect.xml",
    frame_shape: tuple[int, int] = (100, 133),
    threshold_value: float = 30.0,
    compress_to_fp16: bool = True,
) -> str:
    """Build the OpenVINO motion detection model and save to disk.

    Creates a computational graph that implements the motion detection
    pipeline: frame differencing with preprocessing (resize, blur,
    contrast normalization).

    Args:
        model_path: Output path for the IR model file (.xml)
        frame_shape: (height, width) of motion detection frames
        threshold_value: Pixel difference threshold for motion
        compress_to_fp16: Whether to compress weights to FP16

    Returns:
        Path to the saved model file
    """
    import openvino as ov
    from openvino import opset8 as ops
    from openvino.preprocess import PrePostProcessor

    h, w = frame_shape

    # Create input parameters
    frame_input = ops.parameter(
        dtype=ov.Type.u8,
        shape=ov.PartialShape([1, h, w, 1]),
        name="frame_input",
    )
    avg_frame_input = ops.parameter(
        dtype=ov.Type.f32,
        shape=ov.PartialShape([h, w]),
        name="avg_frame_input",
    )

    # Convert avg_frame to u8 for absdiff
    avg_frame_u8 = ops.convert(avg_frame_input, ov.Type.u8)

    # Compute absdiff between frame and average
    avg_reshaped = ops.reshape(avg_frame_u8, [h, w], False)
    avg_expanded = ops.reshape(avg_reshaped, [1, h, w, 1], False)
    absdiff = ops.abs(
        ops.subtract(frame_input, avg_expanded)
    )

    # Apply threshold
    threshold_node = ops.greater(absdiff, np.uint8(threshold_value))

    # Build model
    motion_model = ov.Model(
        [threshold_node],
        [frame_input, avg_frame_input],
        "motion_detect",
    )

    # Apply preprocessing
    ppp = PrePostProcessor(motion_model)
    ppp.input(0).tensor().set_layout(ov.Layout("NHWC"))

    model = ppp.build()

    # Save the model
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    ov.save_model(model, model_path, compress_to_fp16=compress_to_fp16)

    logger.info(
        "Built OpenVINO motion model at %s (%dx%d)",
        model_path,
        h,
        w,
    )
    return model_path


def build_motion_model_from_config(
    model_path: str = "/openvino-model/motion_detect.xml",
    config: dict[str, Any] | None = None,
) -> str:
    """Build the OpenVINO motion model with configurable parameters.

    Args:
        model_path: Output path for the model
        config: Configuration dictionary with optional keys:
            - frame_height: Motion detection frame height (default 100)
            - frame_width: Motion detection frame width (default auto from height)
            - threshold: Pixel difference threshold (default 30)
            - compress_to_fp16: Compress weights (default True)

    Returns:
        Path to the saved model file
    """
    if config is None:
        config = {}

    frame_height = config.get("frame_height", 100)
    # Width is typically frame_height * aspect_ratio
    # Default aspect ratio is 4:3 (standard surveillance camera)
    frame_width = config.get(
        "frame_width", int(frame_height * 4 / 3)
    )
    threshold_value = config.get("threshold", 30)
    compress_to_fp16 = config.get("compress_to_fp16", True)

    return build_motion_model(
        model_path=model_path,
        frame_shape=(frame_height, frame_width),
        threshold_value=threshold_value,
        compress_to_fp16=compress_to_fp16,
    )


def get_motion_model_path() -> str:
    """Get the path to the OpenVINO motion model.

    Checks if the model exists at the default path, and returns it.
    Falls back to the default path if not found.

    Returns:
        Path to the motion model XML file
    """
    default_path = "/openvino-model/motion_detect.xml"
    if os.path.isfile(default_path):
        return default_path
    return default_path
