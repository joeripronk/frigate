import datetime
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from multiprocessing import Queue, Value
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import numpy as np
import zmq
from frigate.detectors.detection_cython import (
    filter_from_shared_memory,
    filter_raw_detections,
)

from frigate.comms.object_detector_signaler import (
    ObjectDetectorPublisher,
    ObjectDetectorSubscriber,
)
from frigate.config import FrigateConfig
from frigate.const import PROCESS_PRIORITY_HIGH
from frigate.detectors import create_detector
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    InputDTypeEnum,
    ModelConfig,
)
from frigate.util.builtin import EventsPerSecond, load_labels
from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory
from frigate.util.object import create_tensor_input
from frigate.util.process import FrigateProcess

from .util import tensor_transform

logger = logging.getLogger(__name__)


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, tensor_input: np.ndarray, threshold: float = 0.4) -> list:
        pass


class BaseLocalDetector(ObjectDetector):
    def __init__(
        self,
        detector_config: BaseDetectorConfig | None = None,
        labels: str | None = None,
        stop_event: MpEvent | None = None,
    ) -> None:
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels: dict[int, str] = {}
        else:
            self.labels = load_labels(labels)

        if detector_config and detector_config.model:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)
            self.dtype = detector_config.model.input_dtype
        else:
            self.input_transform = None
            self.dtype = InputDTypeEnum.int

        self.detect_api = create_detector(detector_config)

        # If the detector supports stop_event, pass it
        if hasattr(self.detect_api, "set_stop_event") and stop_event:
            self.detect_api.set_stop_event(stop_event)

    def _transform_input(self, tensor_input: np.ndarray) -> np.ndarray:
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)

        if self.dtype == InputDTypeEnum.float:
            tensor_input = tensor_input.astype(np.float32)
            tensor_input /= 255
        elif self.dtype == InputDTypeEnum.float_denorm:
            tensor_input = tensor_input.astype(np.float32)

        return tensor_input

    def detect(self, tensor_input: np.ndarray, threshold: float = 0.4) -> list:
        raw_detections = self.detect_raw(tensor_input)  # type: ignore[attr-defined]

        detections = filter_raw_detections(raw_detections, self.labels, threshold)
        self.fps.update()
        return detections


def _filter_raw_detections_keep_id(
    raw_detections: np.ndarray, labels: dict[int, str], threshold: float
) -> list:
    """Filter raw detections by threshold, preserving label_id format.

    Returns (label_id, score, y0, x0, y1, x1) tuples suitable for
    convert_detection_boxes.
    """
    results: list = []
    label_count = len(labels)
    for i in range(len(raw_detections)):
        label_id = int(raw_detections[i, 0])
        score = float(raw_detections[i, 1])

        if label_id < 0 or label_id >= label_count:
            continue
        if score < threshold:
            break

        results.append((label_id, score, tuple(raw_detections[i, 2:6])))

    return results


class LocalObjectDetector(BaseLocalDetector):
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.detect_raw(tensor_input=tensor_input)  # type: ignore[no-any-return]


class AsyncLocalObjectDetector(BaseLocalDetector):
    def async_send_input(self, tensor_input: np.ndarray, connection_id: str) -> None:
        tensor_input = self._transform_input(tensor_input)
        self.detect_api.send_input(connection_id, tensor_input)

    def async_receive_output(self) -> Any:
        return self.detect_api.receive_output()


BATCH_PREFIX = "-"
DETECTOR_BATCH_SIZE = 16


class DetectorRunner(FrigateProcess):
    def __init__(
        self,
        name: str,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Any,
        start_time: Any,
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict[str, Any] = {}
        self.batch_outputs: dict[str, Any] = {}

    def create_output_shm(self, name: str) -> None:
        out_shm = UntrackedSharedMemory(name=f"out-{name}", create=False)
        out_np: np.ndarray = np.ndarray((20, 6), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def create_batch_output_shm(self, name: str) -> None:
        batch_size = DETECTOR_BATCH_SIZE * 20
        out_shm = UntrackedSharedMemory(name=f"batch-out-{name}", create=True)
        out_np: np.ndarray = np.ndarray(
            (batch_size, 6), dtype=np.float32, buffer=out_shm.buf
        )
        self.batch_outputs[name] = {"shm": out_shm, "np": out_np}

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        frame_manager = SharedMemoryFrameManager()
        object_detector = LocalObjectDetector(detector_config=self.detector_config)
        detector_publisher = ObjectDetectorPublisher()

        for name in self.cameras:
            self.create_output_shm(name)

        while not self.stop_event.is_set():
            try:
                queue_message = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Check if this is a batch message (starts with BATCH_PREFIX)
            if queue_message.startswith(BATCH_PREFIX):
                try:
                    batch_str = queue_message[len(BATCH_PREFIX) :]
                    camera_name, batch_count = batch_str.rsplit(":", 1)
                    batch_count = int(batch_count)
                except ValueError:
                    logger.warning(f"Failed to parse batch message: {queue_message}")
                    continue

                if camera_name not in self.batch_outputs:
                    self.create_batch_output_shm(camera_name)

                batch_start = time.monotonic()
                offset = 0
                for region_idx in range(batch_count):
                    input_frame = frame_manager.get(
                        camera_name,
                        (
                            batch_count,
                            self.detector_config.model.height,  # type: ignore[union-attr]
                            self.detector_config.model.width,  # type: ignore[union-attr]
                            3,
                        ),
                    )

                    if input_frame is None:
                        logger.warning(
                            f"Failed to get batch frame for {camera_name} region {region_idx}"
                        )
                        break

                    mono_start = time.monotonic()
                    detections = object_detector.detect_raw(input_frame)
                    duration = time.monotonic() - mono_start

                    frame_manager.close(camera_name)
                    self.batch_outputs[camera_name]["np"][
                        offset : offset + len(detections)
                    ] = detections
                    offset += len(detections)

                duration = time.monotonic() - batch_start
                detector_publisher.publish(f"{camera_name}/batch")
                self.start_time.value = 0.0
                self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10
                continue

            # Single-region detection (existing behavior)
            connection_id = queue_message
            input_frame = frame_manager.get(
                connection_id,
                (
                    1,
                    self.detector_config.model.height,  # type: ignore[union-attr]
                    self.detector_config.model.width,  # type: ignore[union-attr]
                    3,
                ),
            )

            if input_frame is None:
                logger.warning(f"Failed to get frame {connection_id} from SHM")
                continue

            # detect and send the output
            self.start_time.value = datetime.datetime.now().timestamp()
            mono_start = time.monotonic()
            detections = object_detector.detect_raw(input_frame)
            duration = time.monotonic() - mono_start
            frame_manager.close(connection_id)

            if connection_id not in self.outputs:
                self.create_output_shm(connection_id)

            self.outputs[connection_id]["np"][:] = detections[:]
            detector_publisher.publish(connection_id)
            self.start_time.value = 0.0

            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10

        detector_publisher.stop()
        logger.info("Exited detection process...")


class AsyncDetectorRunner(FrigateProcess):
    def __init__(
        self,
        name: str,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Any,
        start_time: Any,
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict[str, Any] = {}
        self._frame_manager: SharedMemoryFrameManager | None = None
        self._publisher: ObjectDetectorPublisher | None = None
        self._detector: AsyncLocalObjectDetector | None = None
        self.send_times: deque[float] = deque()

    def create_output_shm(self, name: str) -> None:
        out_shm = UntrackedSharedMemory(name=f"out-{name}", create=False)
        out_np: np.ndarray = np.ndarray((20, 6), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def _detect_worker(self) -> None:
        logger.info("Starting Detect Worker Thread")
        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            assert self._frame_manager is not None
            input_frame = self._frame_manager.get(
                connection_id,
                (
                    1,
                    self.detector_config.model.height,  # type: ignore[union-attr]
                    self.detector_config.model.width,  # type: ignore[union-attr]
                    3,
                ),
            )

            if input_frame is None:
                logger.warning(f"Failed to get frame {connection_id} from SHM")
                continue

            # mark start time and send to accelerator
            self.send_times.append(time.perf_counter())
            assert self._detector is not None
            self._detector.async_send_input(input_frame, connection_id)

    def _result_worker(self) -> None:
        logger.info("Starting Result Worker Thread")
        while not self.stop_event.is_set():
            assert self._detector is not None
            connection_id, detections = self._detector.async_receive_output()

            # Handle timeout case (queue.Empty) - just continue
            if connection_id is None:
                continue

            if not self.send_times:
                # guard; shouldn't happen if send/recv are balanced
                continue
            ts = self.send_times.popleft()
            duration = time.perf_counter() - ts

            # release input buffer
            assert self._frame_manager is not None
            self._frame_manager.close(connection_id)

            if connection_id not in self.outputs:
                self.create_output_shm(connection_id)

            # write results and publish
            if detections is not None:
                self.outputs[connection_id]["np"][:] = detections[:]
            assert self._publisher is not None
            self._publisher.publish(connection_id)

            # update timers
            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10
            self.start_time.value = 0.0

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        self._frame_manager = SharedMemoryFrameManager()
        self._publisher = ObjectDetectorPublisher()
        self._detector = AsyncLocalObjectDetector(
            detector_config=self.detector_config, stop_event=self.stop_event
        )

        for name in self.cameras:
            self.create_output_shm(name)

        t_detect = threading.Thread(target=self._detect_worker, daemon=False)
        t_result = threading.Thread(target=self._result_worker, daemon=False)
        t_detect.start()
        t_result.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)

            logger.info(
                "Stop event detected, waiting for detector threads to finish..."
            )

            # Wait for threads to finish processing
            t_detect.join(timeout=5)
            t_result.join(timeout=5)

            # Shutdown the AsyncDetector
            self._detector.detect_api.shutdown()

            self._publisher.stop()
        except Exception as e:
            logger.error(f"Error during async detector shutdown: {e}")
        finally:
            logger.info("Exited Async detection process...")


class ObjectDetectProcess:
    def __init__(
        self,
        name: str,
        detection_queue: Queue,
        cameras: list[str],
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ):
        self.name = name
        self.cameras = cameras
        self.detection_queue = detection_queue
        self.avg_inference_speed = Value("d", 0.01)
        self.detection_start = Value("d", 0.0)
        self.detect_process: FrigateProcess | None = None
        self.config = config
        self.detector_config = detector_config
        self.stop_event = stop_event
        self.start_or_restart()

    def stop(self) -> None:
        # if the process has already exited on its own, just return
        if self.detect_process and self.detect_process.exitcode:
            return

        if self.detect_process is None:
            return

        logging.info("Waiting for detection process to exit gracefully...")
        self.detect_process.join(timeout=30)
        if self.detect_process.exitcode is None:
            logging.info("Detection process didn't exit. Force killing...")
            self.detect_process.kill()
            self.detect_process.join()
        logging.info("Detection process has exited...")

    def start_or_restart(self) -> None:
        self.detection_start.value = 0.0  # type: ignore[attr-defined]
        if (self.detect_process is not None) and self.detect_process.is_alive():
            self.stop()

        # Async path for MemryX
        if self.detector_config.type == "memryx":
            self.detect_process = AsyncDetectorRunner(
                f"frigate.detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        else:
            self.detect_process = DetectorRunner(
                f"frigate.detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        self.detect_process.start()


class RemoteObjectDetector:
    def __init__(
        self,
        name: str,
        labels: dict[int, str],
        detection_queue: Queue,
        model_config: ModelConfig,
        stop_event: MpEvent,
    ):
        self.labels = labels
        self.name = name
        self.fps = EventsPerSecond()
        self.detection_queue = detection_queue
        self.stop_event = stop_event
        self.shm = UntrackedSharedMemory(name=self.name, create=False)
        self.np_shm: np.ndarray = np.ndarray(
            (DETECTOR_BATCH_SIZE, model_config.height, model_config.width, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        self.out_shm = UntrackedSharedMemory(name=f"out-{self.name}", create=False)
        self.out_np_shm: np.ndarray = np.ndarray(
            (20, 6), dtype=np.float32, buffer=self.out_shm.buf
        )
        batch_out_size = DETECTOR_BATCH_SIZE * 20
        self.batch_out_shm = UntrackedSharedMemory(
            name=f"batch-out-{self.name}", create=False
        )
        self.batch_out_np_shm: np.ndarray = np.ndarray(
            (batch_out_size, 6), dtype=np.float32, buffer=self.batch_out_shm.buf
        )
        self.detector_subscriber = ObjectDetectorSubscriber(name)

    def detect(self, tensor_input: np.ndarray, threshold: float = 0.4) -> list:
        detections: list = []

        if self.stop_event.is_set():
            return detections

        # Drain any stale detection results from the ZMQ buffer before making a new request
        # This prevents reading detection results from a previous request
        # NOTE: This should never happen, but can in some rare cases
        while True:
            try:
                self.detector_subscriber.socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        # copy input to shared memory
        self.np_shm[0] = tensor_input
        self.detection_queue.put(self.name)
        result = self.detector_subscriber.check_for_update()

        # if it timed out
        if result is None:
            return detections

        detections = filter_from_shared_memory(self.out_np_shm, self.labels, threshold)
        self.fps.update()
        return detections

    def detect_batch(
        self, regions: list[tuple[int, int, int, int]], frame, model_config: ModelConfig
    ) -> list[list]:
        """Run detection on multiple regions in a single IPC round-trip.

        Args:
            regions: List of (x0, y0, x1, y1) region coordinates
            frame: Full YUV frame
            model_config: Model configuration

        Returns:
            List of raw detection lists per region, each containing
            (label_id, score, y0, x0, y1, x1) tuples
        """
        results: list = []

        if self.stop_event.is_set() or not regions:
            return results

        # Pack region tensors into batch input SHM
        for i, region in enumerate(regions):
            tensor_input = create_tensor_input(frame, model_config, region)
            self.np_shm[i] = tensor_input

        # Send batch message: "-<camera>:<count>"
        batch_message = f"-{self.name}:{len(regions)}"
        self.detection_queue.put(batch_message)

        # Wait for batch completion signal
        result = self.detector_subscriber.check_for_update()
        if result is None:
            return results

        # Read raw detections for each region from batch output SHM
        offset = 0
        for _ in regions:
            region_detections = self.batch_out_np_shm[offset : offset + 20]
            raw = _filter_raw_detections_keep_id(region_detections, self.labels, 0.4)
            results.append(raw)
            offset += len(raw)

        self.fps.update()
        return results

    def cleanup(self) -> None:
        self.detector_subscriber.stop()
        self.shm.unlink()
        self.out_shm.unlink()
        self.batch_out_shm.unlink()
