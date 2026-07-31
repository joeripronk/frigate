"""Convert the default SSDLite MobileNet v2 model to OpenVINO IR.

Replaces the legacy openvino-dev Model Optimizer conversion. The TensorFlow
frontend converts the Object Detection API frozen graph natively; the four TF
outputs are then repacked into the single [1, 1, 100, 7] DetectionOutput-style
tensor that Frigate's OpenVINO detector expects, and the input is flipped to
BGR to match the legacy reverse_input_channels behavior.
"""

import numpy as np
import openvino as ov
from openvino import opset8 as ops
from openvino.preprocess import PrePostProcessor

model = ov.convert_model(
    "/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb",
    input=[("image_tensor:0", [1, 300, 300, 3])],
)

# rows of (image_id, class_id, score, xmin, ymin, xmax, ymax)
boxes = model.output("detection_boxes:0").get_node().input_value(0)
classes = model.output("detection_classes:0").get_node().input_value(0)
scores = model.output("detection_scores:0").get_node().input_value(0)

# (ymin,xmin,ymax,xmax) -> (xmin,ymin,xmax,ymax)
boxes = ops.gather(boxes, [1, 0, 3, 2], 2)
classes = ops.unsqueeze(classes, 2)
scores = ops.unsqueeze(scores, 2)
image_id = ops.multiply(scores, np.float32(0.0))

detections = ops.concat([image_id, classes, scores, boxes], 2)
detections = ops.unsqueeze(detections, 1)
detections.output(0).get_tensor().set_names({"detection_out"})

model = ov.Model([detections], model.get_parameters(), "ssdlite_mobilenet_v2")

ppp = PrePostProcessor(model)
ppp.input().tensor().set_layout(ov.Layout("NHWC"))
ppp.input().preprocess().reverse_channels()
model = ppp.build()

ov.save_model(model, "/models/ssdlite_mobilenet_v2.xml", compress_to_fp16=True)

# Build motion detection model
try:
    from frigate.motion.openvino_motion_model import (
        build_motion_model as build_motion_model_main,
    )
except ModuleNotFoundError:
    # ov-converter stage: frigate deps (py3nvml, zmq) not installed
    # Fall back to the same logic that only needs openvino + numpy
    def build_motion_model_main(
        model_path: str = "/openvino-model/motion_detect.xml",
        frame_shape: tuple = (100, 133),
        threshold_value: float = 30.0,
        compress_to_fp16: bool = True,
    ) -> str:
        import openvino as ov
        from openvino import opset8 as ops
        from openvino.preprocess import PrePostProcessor

        h, w = frame_shape
        frame_input = ops.parameter(dtype=ov.Type.u8, shape=ov.PartialShape([1, h, w, 1]), name="frame_input")
        avg_frame_input = ops.parameter(dtype=ov.Type.f32, shape=ov.PartialShape([h, w]), name="avg_frame_input")
        avg_frame_u8 = ops.convert(avg_frame_input, ov.Type.u8)
        avg_reshaped = ops.reshape(avg_frame_u8, [h, w], False)
        avg_expanded = ops.reshape(avg_reshaped, [1, h, w, 1], False)
        absdiff = ops.abs(
            ops.subtract(frame_input, avg_expanded)
        )
        threshold_node = ops.greater(absdiff, np.uint8(threshold_value))
        motion_model = ov.Model([threshold_node], [frame_input, avg_frame_input], "motion_detect")
        ppp = PrePostProcessor(motion_model)
        ppp.input(0).tensor().set_layout(ov.Layout("NHWC"))
        model = ppp.build()
        import os
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        ov.save_model(model, model_path, compress_to_fp16=compress_to_fp16)
        return model_path

build_motion_model_main(
    model_path="/openvino-model/motion_detect.xml",
    frame_shape=(100, 133),
    threshold_value=30.0,
    compress_to_fp16=True,
)
