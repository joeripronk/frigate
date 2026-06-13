"""ONVIF and Reolink camera detection for motion, person, vehicle, pet, and doorbell.

This module provides detection via three complementary methods:

1. ONVIF event subscription (Subscribe/Notify) for motion events - push-based
   event delivery via HTTP POST notifications from the camera.

2. Reolink Baichuan TCP push (port 9000) for real-time AI detection of
   persons, vehicles, pets, and doorbell presses via the reolink_aio library.

3. Reolink smart detection API polling as a fallback for person/vehicle
   detection when TCP push is unavailable.

This allows cameras with built-in AI detection to be used without TensorFlow
inference on the Frigate server.
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
from onvif.client import ONVIFCamera
from reolink_aio.api import Host as ReolinkHost
from reolink_aio.const import AI_DETECT_CONVERSION
from zeep.exceptions import Fault, TransportError

from frigate.config import CameraConfig
from frigate.config.camera.onvif import OnvifVendorEnum

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
    PET = "pet"
    DOORBELL = "doorbell"


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
        self.poll_interval = camera_config.onvif.detect.poll_interval
        self.enabled = camera_config.onvif.detect.enabled
        self.detect_person = camera_config.onvif.detect.person
        self.detect_vehicle = camera_config.onvif.detect.vehicle

    def start(self) -> None:
        """Start polling for smart detection in a background thread."""
        if not self.enabled:
            return
        self.thread = threading.Thread(
            target=self._poll_loop,
            name=f"onvif_reolink_{self.camera_name}",
            daemon=True,
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

    def _get_smart_detections(self, session: httpx.Client) -> list[OnvifDetection]:
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
                    "Reolink API returned status %d for %s",
                    response.status_code,
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

                        detections.append(
                            OnvifDetection(
                                type=DetectionType.PERSON,
                                label="person",
                                score=0.9,
                                box=(y_min, x_min, y_max, x_max),
                                frame_width=frame_width,
                                frame_height=frame_height,
                            )
                        )

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

                        detections.append(
                            OnvifDetection(
                                type=DetectionType.VEHICLE,
                                label="vehicle",
                                score=0.9,
                                box=(y_min, x_min, y_max, x_max),
                                frame_width=frame_width,
                                frame_height=frame_height,
                            )
                        )

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
        for attempt_port in range(
            self._notification_port, self._notification_port + 20
        ):
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
            self.detection_queue.put_nowait(
                OnvifDetection(
                    type=DetectionType.MOTION,
                    label="motion",
                    score=0.8,
                    box=(0.0, 0.0, 1.0, 1.0),
                    frame_width=1920,
                    frame_height=1080,
                    frame_time=frame_time,
                    metadata={"topic": topic_str},
                )
            )

        # Process smart detection events
        if is_smart and detection_type:
            self.detection_queue.put_nowait(
                OnvifDetection(
                    type=detection_type,
                    label=detection_type.value,
                    score=0.9,
                    box=(0.0, 0.0, 1.0, 1.0),
                    frame_width=1920,
                    frame_height=1080,
                    frame_time=frame_time,
                    metadata={"topic": topic_str},
                )
            )

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
                        getattr(capabilities, "SubscriptionManagerSupport", None)
                        is True
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


class ReolinkTcpPushClient:
    """Reolink Baichuan TCP push client for real-time event detection.

    Uses the reolink_aio library to connect to Reolink cameras via the
    Baichuan protocol (TCP port 9000) and subscribe to real-time push
    events for motion, person, vehicle, pet, and doorbell detection.

    Events are pushed as OnvifDetection objects into the shared detection
    queue for processing by the main object tracking pipeline.
    """

    # Internal mapping: Frigate detection type -> Reolink AI type
    _FRIGATE_TO_REOLINK: dict[str, str] = {
        "person": "people",
        "pet": "dog_cat",
        "vehicle": "vehicle",
        "motion": "motion",
        "doorbell": "visitor",
    }

    # Reverse mapping for AI detections (Reolink -> Frigate)
    _REOLINK_TO_FRIGATE: dict[str, DetectionType] = {
        "people": DetectionType.PERSON,
        "vehicle": DetectionType.VEHICLE,
        "dog_cat": DetectionType.PET,
        "visitor": DetectionType.DOORBELL,
    }

    # Frigate detection types that map to Reolink AI detection
    AI_DETECTION_TYPES = frozenset({"person", "vehicle", "pet"})

    # Detection cooldown per type to prevent duplicate events
    _COOLDOWN_SECONDS: float = 2.0

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

        # Enabled detection types from config
        self.detect_motion = camera_config.onvif.detect.motion
        self.detect_person = camera_config.onvif.detect.person
        self.detect_vehicle = camera_config.onvif.detect.vehicle
        self.detect_pet = getattr(camera_config.onvif.detect, "pet", False)
        self.detect_doorbell = getattr(camera_config.onvif.detect, "doorbell", False)

        self._reolink: Optional[ReolinkHost] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._last_detection_times: dict[str, float] = {}

    def start(self) -> None:
        """Start the TCP push connection in a background thread with its own event loop."""
        if not self._should_run():
            return

        self._running = True
        self._loop = asyncio.new_event_loop()
        self._connect_task = self._loop.create_task(self._run_loop())
        self.thread = threading.Thread(
            target=self._run_event_loop,
            name=f"onvif_reolink_push_{self.camera_name}",
            daemon=True,
        )
        self.thread.start()
        logger.info(
            "Reolink TCP push started for %s (motion=%s, person=%s, vehicle=%s, pet=%s, doorbell=%s)",
            self.camera_name,
            self.detect_motion,
            self.detect_person,
            self.detect_vehicle,
            self.detect_pet,
            self.detect_doorbell,
        )

    def stop(self) -> None:
        """Signal the TCP push client to stop and cleanup."""
        self._running = False
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        thread = getattr(self, "thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=5)
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        if self._loop:
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                self._loop.stop()

    def _should_run(self) -> bool:
        """Check if any detection types are enabled for TCP push."""
        return (
            self.detect_motion
            or self.detect_person
            or self.detect_vehicle
            or self.detect_pet
            or self.detect_doorbell
        )

    async def _run_loop(self) -> None:
        """Main async loop: connect, subscribe, and handle events."""
        while self._running and not self.stop_event.is_set():
            try:
                await self._connect_and_subscribe()
            except Exception:
                logger.exception(
                    "Error in Reolink TCP push for %s, retrying in 5s",
                    self.camera_name,
                )

            if not self._running:
                break

            # Wait before retrying connection
            for _ in range(50):
                if self._running and not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                else:
                    break

    async def _connect_and_subscribe(self) -> None:
        """Connect to camera and subscribe to events."""
        self._reolink = ReolinkHost(
            host=self.host,
            username=self.username,
            password=self.password,
            port=self.port,
            bc_port=9000,
        )

        try:
            await asyncio.wait_for(self._reolink.login(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout logging into Reolink camera %s via TCP push",
                self.camera_name,
            )
            return
        except Exception:
            logger.debug(
                "Failed to log into Reolink camera %s via TCP push",
                self.camera_name,
            )
            return

        # Subscribe to Baichuan push events (cmd_id=33 for motion/AI/visitor)
        try:
            await self._reolink.baichuan.subscribe_events()
            await asyncio.sleep(2)  # Give time for events to start flowing
        except Exception:
            logger.debug(
                "Failed to subscribe to Baichuan events for %s",
                self.camera_name,
            )

        # Keep connection alive while running, processing events
        try:
            while self._running and not self.stop_event.is_set():
                await asyncio.sleep(1)
                self._process_events()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Error during TCP push event processing for %s", self.camera_name
            )
        finally:
            try:
                await self._reolink.baichuan.unsubscribe_events()
                await self._reolink.logout()
            except Exception:
                logger.debug(
                    "Error cleaning up Reolink TCP push for %s", self.camera_name
                )
            finally:
                self._reolink = None

    def _process_events(self) -> None:
        """Check current detection states and push new detections to the queue."""
        if not self._reolink:
            return

        frame_time = time.time()

        # Motion detection
        if self.detect_motion:
            channel = 0
            if self._reolink.num_channels > 0:
                channel = self._reolink.channels[0] if self._reolink.channels else 0

            motion_state = self._reolink.motion_detected(channel)
            key = "motion"
            last_time = self._last_detection_times.get(key, 0)

            if motion_state and (frame_time - last_time) > self._COOLDOWN_SECONDS:
                self._last_detection_times[key] = frame_time
                self._queue_detection(
                    DetectionType.MOTION, "motion", 0.8, channel, frame_time
                )

        # AI detections (person, vehicle, pet)
        ai_types_to_check: list[tuple[DetectionType, str, bool]] = []
        if self.detect_person:
            ai_types_to_check.append((DetectionType.PERSON, "people", True))
        if self.detect_vehicle:
            ai_types_to_check.append((DetectionType.VEHICLE, "vehicle", True))
        if self.detect_pet:
            ai_types_to_check.append((DetectionType.PET, "dog_cat", True))

        channel = 0
        if self._reolink.num_channels > 0:
            channel = self._reolink.channels[0] if self._reolink.channels else 0

        for det_type, ai_type, _ in ai_types_to_check:
            key = ai_type
            last_time = self._last_detection_times.get(key, 0)

            try:
                detected = self._reolink.ai_detected(channel, ai_type)
            except Exception:
                detected = False

            if detected and (frame_time - last_time) > self._COOLDOWN_SECONDS:
                self._last_detection_times[key] = frame_time
                self._queue_detection(det_type, ai_type, 0.9, channel, frame_time)

        # Doorbell (visitor) detection
        if self.detect_doorbell:
            key = "visitor"
            last_time = self._last_detection_times.get(key, 0)

            try:
                visitor_state = self._reolink.visitor_detected(channel)
            except Exception:
                visitor_state = False

            if visitor_state and (frame_time - last_time) > self._COOLDOWN_SECONDS:
                self._last_detection_times[key] = frame_time
                self._queue_detection(
                    DetectionType.DOORBELL, "visitor", 0.9, channel, frame_time
                )

    def _queue_detection(
        self,
        det_type: DetectionType,
        label: str,
        score: float,
        channel: int,
        frame_time: float,
    ) -> None:
        """Queue a detection result for processing by the main pipeline."""
        type_name = det_type.value
        logger.info(
            "%s: Reolink TCP push detection: %s (score=%.2f, channel=%s)",
            self.camera_name,
            type_name,
            score,
            channel,
        )
        try:
            self.detection_queue.put_nowait(
                OnvifDetection(
                    type=det_type,
                    label=label,
                    score=score,
                    box=(0.0, 0.0, 1.0, 1.0),
                    frame_width=1920,
                    frame_height=1080,
                    frame_time=frame_time,
                    metadata={
                        "source": "baichuan_tcp",
                        "channel": channel,
                    },
                )
            )
        except queue.Full:
            pass

    def get_ai_detected_sync(self, object_type: str, channel: int = 0) -> bool:
        """Synchronously check if an AI object type is currently detected.

        This allows the ONVIF event subscriber to cross-reference with TCP push
        state when processing ONVIF events.
        """
        if not self._reolink:
            return False

        # Convert Frigate type to Reolink internal type
        reolink_type = AI_DETECT_CONVERSION.get(object_type, object_type)

        try:
            return self._reolink.ai_detected(channel, reolink_type)
        except Exception:
            return False

    def get_doorbell_detected_sync(self, channel: int = 0) -> bool:
        """Synchronously check if a doorbell (visitor) event is active."""
        if not self._reolink:
            return False

        try:
            return self._reolink.visitor_detected(channel)
        except Exception:
            return False

    def get_motion_detected_sync(self, channel: int = 0) -> bool:
        """Synchronously check if motion is currently detected."""
        if not self._reolink:
            return False

        try:
            return self._reolink.motion_detected(channel)
        except Exception:
            return False


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
        self.tcp_push_client: Optional[ReolinkTcpPushClient] = None
        self.event_subscriber: Optional[OnvifEventSubscriber] = None
        self.onvif_camera: Optional[ONVIFCamera] = None
        self._running = False

    def start(self) -> None:
        """Start the ONVIF detector and connect to the camera."""
        if not self.camera_config.onvif.detect.enabled:
            return
        if not self.camera_config.onvif.detect.motion:
            return

        if not self.camera_config.onvif.host:
            logger.warning(
                "ONVIF detection enabled for %s but no host configured",
                self.camera_name,
            )
            return

        self._running = True

        # Determine which vendor implementation to use
        vendor = self.camera_config.onvif.vendor

        # Start AI detection based on vendor
        if vendor == OnvifVendorEnum.reolink:
            # Use Reolink TCP push for real-time detection
            has_ai_detection = (
                self.camera_config.onvif.detect.person
                or self.camera_config.onvif.detect.vehicle
                or getattr(self.camera_config.onvif.detect, "pet", False)
                or getattr(self.camera_config.onvif.detect, "doorbell", False)
            )

            if has_ai_detection:
                self.tcp_push_client = ReolinkTcpPushClient(
                    self.camera_config, self.detection_queue, self.stop_event
                )
                self.tcp_push_client.start()
        else:
            # Default: use polling for person/vehicle detection
            #if (
            #    self.camera_config.onvif.detect.person
            #    or self.camera_config.onvif.detect.vehicle
            #):
            #    self.reolink_detector = ReolinkSmartDetector(
            #        self.camera_config, self.detection_queue, self.stop_event
            #    )
            #    self.reolink_detector.start()

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

                onvif_host = self.camera_config.onvif.host.split("://")[-1].split(":")[
                    0
                ]
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

        has_pet = getattr(self.camera_config.onvif.detect, "pet", False)
        has_doorbell = getattr(self.camera_config.onvif.detect, "doorbell", False)
        has_tcp_push = (
            self.tcp_push_client is not None and self.tcp_push_client._running
        )
        has_polling = (
            self.reolink_detector is not None
            and self.reolink_detector.thread is not None
        )

        logger.info(
            "ONVIF detector started for %s (vendor=%s, motion=%s, person=%s, vehicle=%s, pet=%s, doorbell=%s, tcp_push=%s, polling=%s, streaming=%s)",
            self.camera_name,
            vendor.value,
            self.camera_config.onvif.detect.motion,
            self.camera_config.onvif.detect.person,
            self.camera_config.onvif.detect.vehicle,
            has_pet,
            has_doorbell,
            has_tcp_push,
            has_polling,
            self.event_subscriber is not None and self.event_subscriber._subscribed,
        )

    async def stop(self) -> None:
        """Stop the ONVIF detector."""
        self._running = False
        if self.tcp_push_client:
            self.tcp_push_client.stop()
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
