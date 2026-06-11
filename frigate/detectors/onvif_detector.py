"""ONVIF and Reolink camera detection for motion, person, and vehicle.

This module provides detection via ONVIF camera event streaming/subscription
(motion) and Reolink smart detection API (person, vehicle), allowing cameras
with built-in AI detection to be used without TensorFlow inference on the
Frigate server.

The ONVIF event subsystem uses Subscribe/Notify for push-based event delivery,
eliminating the need for constant polling. Events arrive in real-time via HTTP
POST notifications from the camera to a local Frigate endpoint.
"""

import asyncio
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Optional, Union

import httpx
from onvif import ONVIFError
from onvif.client import ONVIFCamera
from zeep.exceptions import Fault, TransportError

from frigate.config import CameraConfig
from frigate.ptz.onvif import OnvifController

logger = logging.getLogger(__name__)


async def _create_onvif_camera(
    onvif_host: str,
    onvif_port: int,
    user: str,
    password: str,
    wsdl_base: Optional[str],
    ignore_time_mismatch: bool,
    tls_insecure: bool,
) -> ONVIFCamera:
    """Create an ONVIF camera instance inside a running event loop."""
    return ONVIFCamera(
        onvif_host,
        onvif_port,
        user,
        password,
        wsdl_dir=wsdl_base,
        adjust_time=ignore_time_mismatch,
        encrypt=not tls_insecure,
    )


class DetectionType(str, Enum):
    """Types of detections provided by the ONVIF/Reolink detector."""

    MOTION = "motion"
    PERSON = "person"
    VEHICLE = "vehicle"


@dataclass
class OnvifDetection:
    """A detection result from ONVIF/Reolink smart detection."""

    type: DetectionType
    label: str
    score: float
    # Normalized bounding box (ymin, xmin, ymax, xmax) relative to frame size
    box: tuple[float, float, float, float]
    # Frame dimensions the box is relative to
    frame_width: int
    frame_height: int
    # Timestamp of detection
    frame_time: float = 0.0
    # Optional metadata from the camera
    metadata: dict[str, Any] = field(default_factory=dict)


class ReolinkSmartDetector:
    """Polls Reolink camera API for smart detection results (person, vehicle).

    This runs in parallel with ONVIF event streaming as a fallback for person/vehicle
    detection when the camera supports smart detection via API but not ONVIF events.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        detection_queue: queue.Queue,
        stop_event: Union[threading.Event, MpEvent],
    ) -> None:
        self.camera_config = camera_config
        self.camera_name = camera_config.name
        self.detection_queue = detection_queue
        self.stop_event = stop_event

        self.host = camera_config.onvif.host
        if self.host.startswith("http://") or self.host.startswith("https://"):
            self.host = self.host.split("://")[-1]
        self.port = 80
        user = camera_config.onvif.user
        password = camera_config.onvif.password

        # Extract host and port if format is host:port
        if ":" in self.host:
            host_part, port_str = self.host.rsplit(":", 1)
            if port_str.isdigit():
                self.host = host_part
                self.port = int(port_str)

        self.username = user if user else ""
        self.password = password if password else ""
        self.poll_interval = camera_config.onvif.detection.poll_interval
        self.enabled = camera_config.onvif.detection.enabled
        self.detect_person = camera_config.onvif.detection.person
        self.detect_vehicle = camera_config.onvif.detection.vehicle

    def start(self) -> None:
        """Start polling for smart detection in a background thread."""
        if not self.enabled:
            return
        self.thread = threading.Thread(
            target=self._poll_loop, name=f"onvif_reolink_{self.camera_name}", daemon=True
        )
        self.thread.start()
        logger.info(
            "Reolink smart detection started for %s (interval=%ds, person=%s, vehicle=%s)",
            self.camera_name,
            self.poll_interval,
            self.detect_person,
            self.detect_vehicle,
        )

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _poll_loop(self) -> None:
        """Main polling loop for Reolink smart detection."""
        session = httpx.Client(timeout=5.0)
        last_motion_time: dict[str, float] = {}
        motion_cooldown = 2.0  # seconds cooldown between same-type detections

        try:
            while not self.stop_event.is_set():
                if self.poll_interval <= 0:
                    # Polling disabled, skip
                    time.sleep(1)
                    continue

                try:
                    detections = self._get_smart_detections(session)
                    frame_time = time.time()

                    for det in detections:
                        # Apply cooldown to reduce duplicate detections
                        detection_key = f"{det.type.value}_{det.label}"
                        last_time = last_motion_time.get(detection_key, 0)
                        if frame_time - last_time < motion_cooldown:
                            continue
                        last_motion_time[detection_key] = frame_time

                        det.frame_time = frame_time
                        try:
                            self.detection_queue.put_nowait(det)
                        except queue.Full:
                            pass

                except httpx.RequestError as e:
                    logger.debug(
                        "Reolink API request failed for %s: %s", self.camera_name, e
                    )
                except Exception:
                    logger.exception(
                        "Error getting smart detections from %s", self.camera_name
                    )

                time.sleep(self.poll_interval)
        finally:
            session.close()

    def _get_smart_detections(
        self, session: httpx.Client
    ) -> list[OnvifDetection]:
        """Query Reolink API for smart detection (person, vehicle)."""
        api_url = (
            f"http://{self.host}:{self.port}/api.cgi?cmd=GetSmartInfo"
            f"&user={self.username}&password={self.password}"
        )

        detections: list[OnvifDetection] = []

        try:
            response = session.get(
                api_url,
                auth=(self.username, self.password),
                timeout=5.0,
            )

            if response.status_code >= 400:
                logger.debug(
                    "Reolink API returned status %d for %s", response.status_code,
                    self.camera_name,
                )
                return detections

            data = response.json()
            if not isinstance(data, list) or len(data) < 2:
                return detections

            # Parse the smart detection results
            smart_data = data[1] if isinstance(data[1], dict) else {}
            smart_value = smart_data.get("value", {}) if smart_data else {}
            smart_list = smart_value.get("smart", {}) if smart_value else {}

            # Person detection
            if self.detect_person:
                person_list = smart_list.get("person", [])
                if isinstance(person_list, list):
                    for person in person_list:
                        if not isinstance(person, dict):
                            continue
                        box = person.get("box", {})
                        if not box or not isinstance(box, dict):
                            continue
                        x_min = box.get("left", 0)
                        y_min = box.get("top", 0)
                        x_max = box.get("right", 0)
                        y_max = box.get("bottom", 0)
                        frame_width = box.get("width", 1920)
                        frame_height = box.get("height", 1080)

                        detections.append(OnvifDetection(
                            type=DetectionType.PERSON,
                            label="person",
                            score=0.9,
                            box=(y_min, x_min, y_max, x_max),
                            frame_width=frame_width,
                            frame_height=frame_height,
                        ))

            # Vehicle detection
            if self.detect_vehicle:
                vehicle_list = smart_list.get("vehicle", [])
                if isinstance(vehicle_list, list):
                    for vehicle in vehicle_list:
                        if not isinstance(vehicle, dict):
                            continue
                        box = vehicle.get("box", {})
                        if not box or not isinstance(box, dict):
                            continue
                        x_min = box.get("left", 0)
                        y_min = box.get("top", 0)
                        x_max = box.get("right", 0)
                        y_max = box.get("bottom", 0)
                        frame_width = box.get("width", 1920)
                        frame_height = box.get("height", 1080)

                        detections.append(OnvifDetection(
                            type=DetectionType.VEHICLE,
                            label="vehicle",
                            score=0.9,
                            box=(y_min, x_min, y_max, x_max),
                            frame_width=frame_width,
                            frame_height=frame_height,
                        ))

        except Exception:
            logger.exception("Error parsing Reolink smart detection response")

        return detections


class OnvifEventSubscriber:
    """ONVIF event subscriber using Subscribe/Notify streaming.

    Creates an ONVIF subscription and runs a local HTTP server to receive
    push-based event notifications from the camera. This eliminates polling
    and delivers events in real-time.
    """

    def __init__(
        self,
        onvif_camera: ONVIFCamera,
        camera_config: CameraConfig,
        detection_queue: queue.Queue,
        stop_event: Union[threading.Event, MpEvent],
    ) -> None:
        self.onvif_camera = onvif_camera
        self.camera_config = camera_config
        self.camera_name = camera_config.name
        self.detection_queue = detection_queue
        self.stop_event = stop_event

        self._subscribed = False
        self._subscription_id: Optional[str] = None
        self._subscription_ttl: int = 60  # seconds
        self._notification_server: Optional[HTTPServer] = None
        self._notification_thread: Optional[threading.Thread] = None
        self._notification_port: int = 8085
        self._renewal_timer: Optional[threading.Timer] = None
        self._last_motion_time: float = 0.0
        self._motion_cooldown: float = 2.0
        self._supports_renew: bool = False
        self._recreation_timer: Optional[threading.Timer] = None

    async def start(self) -> None:
        """Start the ONVIF event subscription."""
        self._start_notification_server()
        await self._subscribe()
        if self._subscribed:
            self._start_renewal_timer()
            logger.info(
                "ONVIF event subscription started for %s (streaming)",
                self.camera_name,
            )

    async def stop(self) -> None:
        """Stop the ONVIF event subscription and close the notification server."""
        if self._renewal_timer:
            self._renewal_timer.cancel()
        if self._recreation_timer:
            self._recreation_timer.cancel()

        await self._unsubscribe()
        self._stop_notification_server()

    def _start_notification_server(self) -> None:
        """Start a local HTTP server to receive ONVIF event notifications."""
        for attempt_port in range(self._notification_port, self._notification_port + 20):
            try:
                self._notification_server = HTTPServer(
                    ("0.0.0.0", attempt_port),
                    self._create_notification_handler(),
                )
                self._notification_thread = threading.Thread(
                    target=self._notification_server.serve_forever,
                    name=f"onvif_notify_{self.camera_name}",
                    daemon=True,
                )
                self._notification_thread.start()
                self._notification_port = attempt_port
                logger.debug(
                    "ONVIF notification server started on port %d for %s",
                    attempt_port,
                    self.camera_name,
                )
                return
            except OSError as e:
                if attempt_port >= self._notification_port + 19:
                    logger.warning(
                        "Failed to start ONVIF notification server for %s: %s",
                        self.camera_name,
                        e,
                    )
                    raise
                continue

    def _create_notification_handler(self) -> type[BaseHTTPRequestHandler]:
        """Create a notification handler class with closure over detector state."""
        detector = self

        class NotificationHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                """Handle incoming ONVIF event notifications."""
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = (
                        self.rfile.read(content_length).decode("utf-8")
                        if content_length
                        else ""
                    )

                    if not body:
                        self.send_response(200)
                        self.end_headers()
                        return

                    detector._process_notification(body)
                    self.send_response(200)
                    self.end_headers()
                except Exception:
                    logger.exception(
                        "Error processing ONVIF notification for %s",
                        detector.camera_name,
                    )
                    self.send_response(500)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default HTTP logging."""
                pass

        return NotificationHandler

    def _process_notification(self, xml_body: str) -> None:
        """Parse ONVIF Event notification and extract motion/person/vehicle events."""
        frame_time = time.time()

        # Extract topic from the XML notification
        topic_match = re.search(r"<tt:Topic[^>]*>(.*?)</tt:Topic>", xml_body)
        topic_str = topic_match.group(1) if topic_match else ""

        # Check for motion events
        is_motion = "motion" in topic_str.lower()
        if "VideoAnalytics" in topic_str:
            is_motion = True
        if "motion" in xml_body.lower():
            is_motion = True

        # Check for Reolink smart detection events (person, vehicle)
        is_smart = "SmartDetection" in topic_str or "smart" in topic_str.lower()
        detection_type: Optional[DetectionType] = None
        if is_smart:
            if re.search(r"<tt:Name[^>]*>(Person)</tt:Name>", xml_body):
                detection_type = DetectionType.PERSON
            elif re.search(r"<tt:Name[^>]*>(Vehicle)</tt:Name>", xml_body):
                detection_type = DetectionType.VEHICLE

        # Process motion events with cooldown
        if is_motion and (frame_time - self._last_motion_time) > self._motion_cooldown:
            self._last_motion_time = frame_time
            self.detection_queue.put_nowait(OnvifDetection(
                type=DetectionType.MOTION,
                label="motion",
                score=0.8,
                box=(0.0, 0.0, 1.0, 1.0),
                frame_width=1920,
                frame_height=1080,
                frame_time=frame_time,
                metadata={"topic": topic_str},
            ))

        # Process smart detection events
        if is_smart and detection_type:
            self.detection_queue.put_nowait(OnvifDetection(
                type=detection_type,
                label=detection_type.value,
                score=0.9,
                box=(0.0, 0.0, 1.0, 1.0),
                frame_width=1920,
                frame_height=1080,
                frame_time=frame_time,
                metadata={"topic": topic_str},
            ))

    def _stop_notification_server(self) -> None:
        """Stop the notification HTTP server."""
        if self._notification_server:
            self._notification_server.shutdown()
            self._notification_server.server_close()

    async def _subscribe(self) -> None:
        """Create ONVIF event subscription using Subscribe method."""
        try:
            event_service = await self.onvif_camera.create_events_service()

            # Check if Renew is supported
            try:
                capabilities = await event_service.GetServiceCapabilities()
                if capabilities:
                    self._supports_renew = (
                        getattr(capabilities, "SubscriptionManagerSupport", None) is True
                    )
            except Exception:
                self._supports_renew = False

            # Build the notification endpoint address (Frigate's HTTP server)
            camera_host = self.camera_config.onvif.host.split("://")[-1].split(":")[0]
            notify_addr = f"http://{camera_host}:{self._notification_port}/"

            # Try pure Subscribe first (true streaming)
            try:
                subscribe_req = event_service.create_type("Subscribe")
                subscribe_req.Address = notify_addr

                subscription = event_service.Subscribe(subscribe_req)
                logger.info(
                    "ONVIF Subscribe (streaming) successful for %s",
                    self.camera_name,
                )
            except (Fault, TransportError, Exception):
                # Fallback to PullPointSubscription for cameras that don't support Subscribe
                logger.debug(
                    "ONVIF Subscribe not supported, using PullPointSubscription for %s",
                    self.camera_name,
                )
                pull_req = event_service.create_type("CreatePullPointSubscription")
                subscription = event_service.CreatePullPointSubscription(pull_req)

            self._subscribed = True
            self._subscription_ttl = 60

            # Extract subscription ID
            try:
                sub_ref = getattr(subscription, "SubscriptionReference", None)
                if sub_ref:
                    self._subscription_id = getattr(sub_ref, "Address", None)
                if not self._subscription_id:
                    self._subscription_id = str(uuid.uuid4())
            except Exception:
                self._subscription_id = str(uuid.uuid4())

            logger.info(
                "ONVIF subscription created for %s (id=%s, ttl=%ds, addr=%s)",
                self.camera_name,
                self._subscription_id,
                self._subscription_ttl,
                notify_addr,
            )

        except Exception:
            logger.exception(
                "Failed to create ONVIF event subscription for %s",
                self.camera_name,
            )

    async def _renew_subscription(self) -> None:
        """Renew the ONVIF subscription before it expires."""
        try:
            event_service = await self.onvif_camera.create_events_service()
            from zeep import xsd

            renew_req = event_service.create_type("Renew")
            ref = xsd.Element(
                "tns:SubscriptionReference",
                xsd.ComplexType(
                    xsd.ArrayOf(xsd.AnyType(type_name="tns:ReferenceParameters"))
                ),
            )
            renew_req.SubscriptionReference = ref
            renew_req.SubscriptionReference.any = [self._subscription_id]

            event_service.Renew(renew_req)
            logger.info("ONVIF subscription renewed for %s", self.camera_name)

            # Schedule next renewal
            self._renewal_timer = threading.Timer(
                self._subscription_ttl - 5,
                self._renew_subscription,
            )
            self._renewal_timer.daemon = True
            self._renewal_timer.start()

        except (Fault, TransportError, Exception):
            logger.debug(
                "Failed to renew ONVIF subscription for %s, recreating",
                self.camera_name,
            )
            self._subscribed = False
            if self._renewal_timer:
                self._renewal_timer.cancel()
            asyncio.run(self._subscribe())

    async def _unsubscribe(self) -> None:
        """Unsubscribe from ONVIF events."""
        if not self._subscription_id:
            return

        try:
            event_service = await self.onvif_camera.create_events_service()
            from zeep import xsd

            close_req = event_service.create_type("Unsubscribe")
            close_req.SubscriptionReference = xsd.Element(
                "tns:SubscriptionReference",
                xsd.ComplexType(
                    xsd.ArrayOf(xsd.AnyType(type_name="tns:ReferenceParameters"))
                ),
            ).get_element()

            event_service.Unsubscribe(close_req)
            logger.info("ONVIF subscription closed for %s", self.camera_name)
        except Exception:
            logger.debug(
                "Error closing ONVIF subscription for %s (may already be closed)",
                self.camera_name,
            )

        self._subscription_id = None

    def _start_renewal_timer(self) -> None:
        """Start a timer to renew the subscription before it expires."""
        if self._supports_renew:
            self._renewal_timer = threading.Timer(
                self._subscription_ttl - 5,
                lambda: asyncio.run(self._renew_subscription()),
            )
            self._renewal_timer.daemon = True
            self._renewal_timer.start()
        else:
            self._recreation_timer = threading.Timer(
                self._subscription_ttl - 5,
                lambda: asyncio.run(self._subscribe()),
            )
            self._recreation_timer.daemon = True
            self._recreation_timer.start()


class OnvifDetector:
    """Main ONVIF/Reolink detector that combines motion events and smart detection.

    This detector uses ONVIF camera event subscription/streaming for motion
    detection and Reolink smart detection API for person/vehicle detection.
    It can be activated by adding an input with role 'onvif' to the camera config.

    Example config:
    cameras:
      driveway:
        ffmpeg:
          inputs:
            - path: rtsp://camera/video
              roles: [detect, record]
            - path: onvif://camera
              roles: [onvif]
        onvif:
          host: camera_ip
          port: 8000
          user: admin
          password: password
          detection:
            enabled: true
            motion: true
            person: true
            vehicle: true
            poll_interval: 5
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        detection_queue: queue.Queue,
        stop_event: Union[threading.Event, MpEvent],
    ) -> None:
        self.camera_config = camera_config
        self.camera_name = camera_config.name
        self.detection_queue = detection_queue
        self.stop_event = stop_event

        self.reolink_detector: Optional[ReolinkSmartDetector] = None
        self.event_subscriber: Optional[OnvifEventSubscriber] = None
        self.onvif_camera: Optional[ONVIFCamera] = None
        self._running = False

    def start(self) -> None:
        """Start the ONVIF detector and connect to the camera."""
        if not self.camera_config.onvif.detection.enabled:
            return

        if not self.camera_config.onvif.host:
            logger.warning(
                "ONVIF detection enabled for %s but no host configured",
                self.camera_name,
            )
            return

        self._running = True

        # Start Reolink smart detection polling as fallback for person/vehicle
        if self.camera_config.onvif.detection.person or self.camera_config.onvif.detection.vehicle:
            self.reolink_detector = ReolinkSmartDetector(
                self.camera_config, self.detection_queue, self.stop_event
            )
            self.reolink_detector.start()

        if self.camera_config.onvif.detection.motion:
            try:
                wsdl_base: str | None = None
                try:
                    from importlib.util import find_spec
                    from pathlib import Path

                    spec = find_spec("onvif")
                    origin = getattr(spec, "origin", None) if spec else None
                    if origin:
                        wsdl_base = str(Path(origin).parent / "wsdl")
                except Exception:
                    wsdl_base = None

                onvif_host = self.camera_config.onvif.host.split("://")[-1].split(":")[0]
                onvif_port = (
                    self.camera_config.onvif.port
                    if self.camera_config.onvif.port
                    else 8000
                )

                # onvif-zeep-plugin uses asyncio.get_running_loop() internally
                # so we need a running event loop for ONVIFCamera construction
                self.onvif_camera = asyncio.run(
                    _create_onvif_camera(
                        onvif_host,
                        onvif_port,
                        self.camera_config.onvif.user or "",
                        self.camera_config.onvif.password or "",
                        wsdl_base,
                        self.camera_config.onvif.ignore_time_mismatch,
                        self.camera_config.onvif.tls_insecure,
                    )
                )

                # Try to update X-Addrs
                if self.onvif_camera is not None:
                    try:
                        asyncio.run(self.onvif_camera.update_xaddrs())
                    except Exception:
                        pass

                self.event_subscriber = OnvifEventSubscriber(
                    self.onvif_camera,
                    self.camera_config,
                    self.detection_queue,
                    self.stop_event,
                )
                asyncio.run(self.event_subscriber.start())

            except Exception as e:
                logger.warning(
                    "Failed to initialize ONVIF event subscription for %s: %s",
                    self.camera_name,
                    e,
                )

        logger.info(
            "ONVIF detector started for %s (motion=%s, person=%s, vehicle=%s, streaming=%s)",
            self.camera_name,
            self.camera_config.onvif.detection.motion,
            self.camera_config.onvif.detection.person,
            self.camera_config.onvif.detection.vehicle,
            self.event_subscriber is not None and self.event_subscriber._subscribed,
        )

    async def stop(self) -> None:
        """Stop the ONVIF detector."""
        self._running = False
        if self.reolink_detector:
            self.reolink_detector.stop()
        if self.event_subscriber:
            await self.event_subscriber.stop()

    def get_detections(self) -> list[OnvifDetection]:
        """Get pending detections from the queue."""
        detections = []
        while not self.detection_queue.empty():
            try:
                det = self.detection_queue.get_nowait()
                detections.append(det)
            except queue.Empty:
                break
        return detections
