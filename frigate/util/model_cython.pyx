"""
Cython-accelerated YOLO and model post-processing operations.

Provides Cython-optimized implementations of YOLO post-processing loops
including the triple-nested grid scan for multipart YOLO models and
standard NMS-based post-processing for single-output YOLO models.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cv2
import numpy as np

cimport numpy

from libc.stdlib cimport malloc, free
from libc.math cimport fmax, fmin, sqrt, pow, fabs


def cython_post_process_multipart_yolo(
    object output_list,
    double width,
    double height,
):
    """Cython version of YOLO multipart post-processing.

    The hottest loop in the entire codebase - processes 3 YOLO feature map
    outputs with anchor boxes, confidence thresholding, and NMS.

    Args:
        output_list: List of 3 numpy arrays (one per feature map scale)
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization

    Returns:
        Detections array of shape (20, 6) with [class_id, confidence, y1, x1, y2, x2]
    """
    cdef:
        list anchors = [
            [(12, 16), (19, 36), (40, 28)],
            [(36, 75), (76, 55), (72, 146)],
            [(142, 110), (192, 243), (459, 401)],
        ]
        list stride_map = [8, 16, 32]
        list all_boxes = []
        list all_scores = []
        list all_class_ids = []
        int i, bs, ny, nx, num_anchors, a_idx, x, y, class_id, idx_count
        double stride, anchor_w, anchor_h, output_0
        double pred_0, pred_1, pred_2, pred_3, pred_4
        double class_conf, conf, dx, dy, dw, dh
        double bx, by, bw, bh, x1, y1, x2, y2
        double conf_threshold = 0.4
        object output, class_probs
        object pred

    for i in range(min(len(output_list), 3)):
        output = output_list[i]
        bs, _, ny, nx = output.shape
        stride = stride_map[i]
        anchor_set = anchors[i]
        num_anchors = len(anchor_set)

        output = output.reshape(bs, num_anchors, 85, ny, nx)
        output = output.transpose(0, 1, 3, 4, 2)
        output = output[0]

        for a_idx in range(num_anchors):
            anchor_w, anchor_h = anchor_set[a_idx]
            for y in range(ny):
                for x in range(nx):
                    pred = output[a_idx, y, x]
                    class_probs = pred[5:]
                    class_id = int(np.argmax(class_probs))
                    class_conf = class_probs[class_id]
                    conf = class_conf * pred[4]

                    if conf < conf_threshold:
                        continue

                    dx = pred[0]
                    dy = pred[1]
                    dw = pred[2]
                    dh = pred[3]

                    bx = ((dx * 2.0 - 0.5) + x) * stride
                    by = ((dy * 2.0 - 0.5) + y) * stride
                    bw = ((dw * 2.0) ** 2) * anchor_w
                    bh = ((dh * 2.0) ** 2) * anchor_h

                    x1 = fmax(0.0, bx - bw / 2.0)
                    y1 = fmax(0.0, by - bh / 2.0)
                    x2 = fmin(width, bx + bw / 2.0)
                    y2 = fmin(height, by + bh / 2.0)

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        bboxes=all_boxes,
        scores=all_scores,
        score_threshold=0.4,
        nms_threshold=0.4,
    )

    results = np.zeros((20, 6), np.float32)

    if len(indices) > 0:
        flat_indices = indices.flatten()
        idx_count = min(len(flat_indices), 20)
        for idx in range(idx_count):
            i = flat_indices[idx]
            class_id = all_class_ids[i]
            conf = all_scores[i]
            x1, y1, x2, y2 = all_boxes[i]
            results[idx] = [
                class_id,
                conf,
                y1 / height,
                x1 / width,
                y2 / height,
                x2 / width,
            ]

    return results


def cython_post_process_nms_yolo(
    object predictions,
    double width,
    double height,
):
    """Cython version of YOLO NMS post-processing.

    Handles single-output YOLO models with predictions in [x_center, y_center, width, height, confidence, class_probs] format.

    Args:
        predictions: Numpy array of shape (N, 85) or (85, N)
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization

    Returns:
        Detections array of shape (20, 6) with [class_id, confidence, y1, x1, y2, x2]
    """
    cdef:
        object scores
        object class_ids
        object filtered_preds
        object boxes
        object boxes_xyxy
        int n, i
        double bbox_0, bbox_1, bbox_2, bbox_3
        double confidence
        int class_id

    predictions = np.squeeze(predictions)

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > 0.4, :]
    scores = scores[scores > 0.4]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = predictions[:, :4]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes = boxes_xyxy

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 0.4, 0.4
    )

    detections = np.zeros((20, 6), np.float32)
    for i in range(min(len(indices), 20)):
        bbox = boxes[indices[i]]
        confidence = float(scores[indices[i]])
        class_id = int(class_ids[indices[i]])

        detections[i] = [
            class_id,
            confidence,
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ]

    return detections


def cython_post_process_dfine(
    object tensor_output,
    double width,
    double height,
):
    """Cython version of D-Fine post-processing.

    Args:
        tensor_output: List/tuple [class_ids, boxes, scores]
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization

    Returns:
        Detections array of shape (20, 6)
    """
    cdef:
        object class_ids, boxes, scores
        object indices
        object input_shape
        int i, idx
        double confidence
        int class_id

    # tensor_output is [class_ids, boxes, scores] - use tensor_output[2] (scores) for threshold
    scores = tensor_output[2]
    mask = scores > 0.4
    class_ids = tensor_output[0][mask]
    boxes = tensor_output[1][mask]
    scores = scores[mask]

    input_shape = np.array([height, width, height, width])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.4, 0.4)
    detections = np.zeros((20, 6), np.float32)

    for i in range(min(len(indices), 20)):
        idx = indices[i]
        confidence = float(scores[idx])
        class_id = int(class_ids[idx])

        detections[i] = [
            class_id,
            confidence,
            float(boxes[idx, 1]),
            float(boxes[idx, 0]),
            float(boxes[idx, 3]),
            float(boxes[idx, 2]),
        ]

    return detections


def cython_post_process_rfdetr(
    object tensor_output,
    double width,
    double height,
):
    """Cython version of RF-DETR post-processing.

    Args:
        tensor_output: List [boxes, raw_scores]
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization

    Returns:
        Detections array of shape (20, 6)
    """
    cdef:
        object boxes, raw_scores, exp, all_scores
        object scores, labels
        object idxs, filtered_boxes, filtered_scores, filtered_labels
        object x_center, y_center, w, h, x_min, y_min, x_max, y_max
        object indices
        int i, idx
        double bbox_0, bbox_1, bbox_2, bbox_3
        double confidence
        int class_id

    boxes = tensor_output[0]
    raw_scores = tensor_output[1]

    exp = np.exp(raw_scores - np.max(raw_scores, axis=-1, keepdims=True))
    all_scores = exp / np.sum(exp, axis=-1, keepdims=True)

    scores = np.max(all_scores[0, :, 1:], axis=-1)
    labels = np.argmax(all_scores[0, :, 1:], axis=-1)

    idxs = scores > 0.4
    filtered_boxes = boxes[0, idxs]
    filtered_scores = scores[idxs]
    filtered_labels = labels[idxs]

    x_center, y_center, w, h = (
        filtered_boxes[:, 0],
        filtered_boxes[:, 1],
        filtered_boxes[:, 2],
        filtered_boxes[:, 3],
    )
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    filtered_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

    indices = cv2.dnn.NMSBoxes(
        filtered_boxes.tolist(), filtered_scores.tolist(), 0.4, 0.4
    )
    detections = np.zeros((20, 6), np.float32)

    for i in range(min(len(indices), 20)):
        idx = indices[i]
        bbox = filtered_boxes[idx]
        confidence = float(filtered_scores[idx])
        class_id = int(filtered_labels[idx])

        detections[i] = [
            class_id,
            confidence,
            bbox[1],
            bbox[0],
            bbox[3],
            bbox[2],
        ]

    return detections


def cython_post_process_yolox(
    object predictions,
    double width,
    double height,
    object grids,
    object expanded_strides,
):
    """Cython version of YOLOX post-processing.

    Args:
        predictions: Numpy array of predictions
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization
        grids: Grid arrays for prediction positioning
        expanded_strides: Stride values for scaling

    Returns:
        Detections array of shape (20, 6)
    """
    cdef:
        object boxes, boxes_xyxy
        object cls_inds, scores
        object indices
        int i, idx, cls_ind
        double bbox_0, bbox_1, bbox_2, bbox_3
        double confidence

    predictions[..., :2] = (predictions[..., :2] + grids) * expanded_strides
    predictions[..., 2:4] = np.exp(predictions[..., 2:4]) * expanded_strides

    predictions = predictions[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    cls_inds = scores.argmax(1)
    scores = scores[np.arange(len(cls_inds)), cls_inds]

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), 0.4, 0.4
    )

    detections = np.zeros((20, 6), np.float32)
    for i in range(min(len(indices), 20)):
        idx = indices[i]
        bbox = boxes_xyxy[idx]
        cls_ind = cls_inds[idx]
        confidence = float(scores[idx])

        detections[i] = [
            cls_ind,
            confidence,
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ]

    return detections
